"""Bounded, stdlib-only Gemma 4 / OpenAI function-tool wire adapter.

This module never executes tools, changes native state, or publishes streaming
fragments. Parse only complete, committed model output after reasoning splitting.
The native worker may hide its terminal <|tool_response> token; both that spelling
and a final <tool_call|> are accepted. Incomplete envelopes always fail closed.

The supported schema subset is explicit: type, description/title, properties,
required, items, additionalProperties (boolean), enum and default. Types are the
seven JSON primitive/container types; unions, references and other constraints
are rejected, not silently weakened. strict:true is unsupported: validation after
generation is not grammar-constrained decoding. Outer HTTP JSON must also use a
duplicate-rejecting decoder (loads_strict_json is provided); already-decoded dicts
cannot reveal duplicate keys lost by a different decoder.

Tool result content is opaque text, encoded as {result:<|\"|>...<|\"|>}. It is not
silently interpreted as JSON, code, a schema, or an instruction. Reserved control
token spellings are refused in all data rendered into native prompts: Gemma does
not document a safe escape for its string delimiter. Property/object keys use a
bounded bare-key alphabet rather than inventing an undocumented quoting dialect.

Format references:
https://ai.google.dev/gemma/docs/core/prompt-formatting-gemma4
https://ai.google.dev/gemma/docs/capabilities/text/function-calling-gemma4
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any

__all__ = [
    "ToolFormatError", "FunctionTool", "ToolConfig", "ParsedToolOutput",
    "loads_strict_json", "validate_tools", "render_declarations",
    "parse_tool_output", "render_assistant_calls", "render_tool_results",
]


MAX_TOOLS = 64
MAX_CALLS = 32
MAX_PROPERTIES = 128
MAX_DEPTH = 32
MAX_NODES = 8192
MAX_JSON_BYTES = 262144
MAX_SCHEMA_BYTES = 65536
MAX_TEXT_BYTES = 262144
MAX_STRING_BYTES = 65536
MAX_NUMBER_CHARS = 128

STRING_DELIMITER = '<|"|>'
TOOL_OPEN = "<|tool>"
TOOL_CLOSE = "<tool|>"
CALL_OPEN = "<|tool_call>"
CALL_CLOSE = "<tool_call|>"
RESPONSE_OPEN = "<|tool_response>"
RESPONSE_CLOSE = "<tool_response|>"

_NAME = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")
_KEY = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")
_KEY_AT = re.compile(r"[A-Za-z0-9_-]{1,128}")
_CALL_ID = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")
_NUMBER_AT = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?")
_TYPES = frozenset(("object", "array", "string", "number", "integer", "boolean", "null"))
_SCHEMA_KEYS = frozenset(("type", "description", "title", "properties", "required",
                          "items", "additionalProperties", "enum", "default"))


class ToolFormatError(ValueError):
    """Invalid/unsupported client tool data or invalid generated tool output."""


@dataclass(frozen=True)
class FunctionTool:
    name: str
    description: str
    parameters_json: str

    @property
    def parameters(self) -> dict[str, Any]:
        """Return an independent schema; the validated configuration is immutable."""
        return json.loads(self.parameters_json)


@dataclass(frozen=True)
class ToolConfig:
    tools: tuple[FunctionTool, ...]
    tool_choice: str
    parallel_tool_calls: bool
    required_name: str | None = None

    @property
    def enabled(self) -> bool:
        return bool(self.tools) and self.tool_choice != "none"


@dataclass(frozen=True)
class ParsedToolOutput:
    content: str | None
    tool_calls: tuple[dict[str, Any], ...]
    finish_reason: str


def _text(value: object, field: str, limit: int = MAX_STRING_BYTES,
          *, safe: bool = True) -> str:
    if not isinstance(value, str):
        raise ToolFormatError(f"{field} must be a string")
    if len(value) > limit:
        raise ToolFormatError(f"{field} exceeds its byte limit")
    try:
        size = len(value.encode("utf-8"))
    except UnicodeEncodeError as exc:
        raise ToolFormatError(f"{field} contains invalid Unicode") from exc
    if size > limit:
        raise ToolFormatError(f"{field} exceeds its byte limit")
    if "\x00" in value:
        raise ToolFormatError(f"{field} contains NUL")
    # Reject the token families conservatively, including future special tokens.
    if safe and any(token in value for token in ("<|", "|>", "<bos>", "<eos>", "<pad>", "<unk>")):
        raise ToolFormatError(f"{field} contains a reserved Gemma control token")
    return value


def _name(value: object, field: str = "function name") -> str:
    if not isinstance(value, str) or _NAME.fullmatch(value) is None:
        raise ToolFormatError(f"{field} must match [A-Za-z0-9_-]{{1,64}}")
    return value


def _key(value: object) -> str:
    if not isinstance(value, str) or _KEY.fullmatch(value) is None:
        raise ToolFormatError("native object keys must match [A-Za-z0-9_-]{1,128}")
    return value


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False,
                      sort_keys=True, separators=(",", ":"))


def _checked_json(value: Any, field: str, *, safe: bool = True,
                  limit: int = MAX_JSON_BYTES) -> Any:
    """Copy only bounded JSON values; reject cycles/depth before serialization."""
    nodes = 0
    byte_budget = 0

    def visit(item: Any, depth: int) -> Any:
        nonlocal nodes, byte_budget
        nodes += 1
        if nodes > MAX_NODES or depth > MAX_DEPTH:
            raise ToolFormatError(f"{field} exceeds nesting or node limit")
        if item is None or type(item) is bool:
            byte_budget += 5
            result = item
        elif type(item) in (int, float):
            if type(item) is float and not math.isfinite(item):
                raise ToolFormatError(f"{field} contains a non-finite number")
            if type(item) is int and item.bit_length() > MAX_NUMBER_CHARS * 4:
                raise ToolFormatError(f"{field} contains an oversized number")
            spelling = str(item)
            if len(spelling) > MAX_NUMBER_CHARS:
                raise ToolFormatError(f"{field} contains an oversized number")
            byte_budget += len(spelling)
            result = item
        elif isinstance(item, str):
            result = _text(item, field, safe=safe)
            byte_budget += len(result.encode("utf-8")) + 2
        elif isinstance(item, (dict, list)):
            if len(item) > MAX_NODES:
                raise ToolFormatError(f"{field} exceeds node limit")
            byte_budget += 2 + len(item)
            if isinstance(item, dict):
                result = {}
                for key, child in item.items():
                    key = _text(key, field + " key", safe=safe)
                    byte_budget += len(key.encode("utf-8")) + 3
                    result[key] = visit(child, depth + 1)
            else:
                result = [visit(child, depth + 1) for child in item]
        else:
            raise ToolFormatError(f"{field} must contain only JSON values")
        if byte_budget > limit:
            raise ToolFormatError(f"{field} exceeds its byte limit")
        return result

    result = visit(value, 0)
    if len(_canonical(result).encode("utf-8")) > limit:
        raise ToolFormatError(f"{field} exceeds its byte limit")
    return result


def loads_strict_json(text: str, *, field: str = "JSON") -> Any:
    """Decode bounded JSON, refusing duplicate keys, nonfinite values and depth.

    This generic decoder permits reserved tokens as data. Rendering entry points
    independently refuse them. Use it for the HTTP envelope as well as arguments.
    """
    text = _text(text, field, MAX_JSON_BYTES, safe=False)
    depth = 0
    quoted = False
    escaped = False
    for char in text:
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char in "[{":
            depth += 1
            if depth > MAX_DEPTH:
                raise ToolFormatError(f"{field} exceeds nesting limit")
        elif char in "]}":
            depth -= 1

    def pairs(entries: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in entries:
            if key in result:
                raise ToolFormatError(f"{field} contains a duplicate object key")
            result[key] = value
        return result

    def constant(_value: str) -> Any:
        raise ToolFormatError(f"{field} contains a non-finite number")

    def number(value: str) -> int | float:
        if len(value) > MAX_NUMBER_CHARS:
            raise ToolFormatError(f"{field} contains an oversized number")
        result = float(value) if any(c in value for c in ".eE") else int(value)
        if isinstance(result, float) and not math.isfinite(result):
            raise ToolFormatError(f"{field} contains a non-finite number")
        return result

    try:
        value = json.loads(text, object_pairs_hook=pairs, parse_constant=constant,
                           parse_int=number, parse_float=number)
    except (ValueError, RecursionError) as exc:
        if isinstance(exc, ToolFormatError):
            raise
        raise ToolFormatError(f"{field} must be valid JSON") from exc
    return _checked_json(value, field, safe=False)


def _instance(value: Any, schema: dict[str, Any]) -> None:
    """Validate only the declared, supported schema subset (never execute code)."""
    kind = schema.get("type")
    valid = {
        "object": isinstance(value, dict), "array": isinstance(value, list),
        "string": isinstance(value, str), "boolean": type(value) is bool,
        "null": value is None, "number": type(value) in (int, float),
        "integer": type(value) is int or (type(value) is float and value.is_integer()),
        None: True,
    }
    if not valid[kind]:
        raise ToolFormatError("function arguments do not match their declared type")
    if "enum" in schema:
        # JSON numbers compare numerically, but true must never equal numeric 1.
        def equal(a: Any, b: Any) -> bool:
            if type(a) in (int, float) and type(b) in (int, float):
                return a == b
            return type(a) is type(b) and a == b
        if not any(equal(value, candidate) for candidate in schema["enum"]):
            raise ToolFormatError("function argument is not a declared enum value")
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        if any(key not in value for key in schema.get("required", [])):
            raise ToolFormatError("function arguments are missing a required parameter")
        if schema.get("additionalProperties") is False and set(value) - set(properties):
            raise ToolFormatError("function arguments contain an undeclared parameter")
        for key, child in value.items():
            _key(key)
            if key in properties:
                _instance(child, properties[key])
    elif isinstance(value, list) and "items" in schema:
        for child in value:
            _instance(child, schema["items"])


def _schema(value: Any, *, root: bool = False) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) - _SCHEMA_KEYS:
        raise ToolFormatError("unsupported function parameter schema keyword or shape")
    schema = dict(value)
    if root:
        schema.setdefault("type", "object")
    kind = schema.get("type")
    if "type" in schema and (not isinstance(kind, str) or kind not in _TYPES):
        raise ToolFormatError("schema type must be one supported JSON type (no unions)")
    if root and kind != "object":
        raise ToolFormatError("function parameters must have object type")
    for label in ("description", "title"):
        if label in schema:
            _text(schema[label], "schema " + label)
    object_keys = {"properties", "required", "additionalProperties"}
    if set(schema) & object_keys and kind != "object":
        raise ToolFormatError("object schema fields require type object")
    if "items" in schema and kind != "array":
        raise ToolFormatError("items requires type array")
    if kind == "object":
        properties = schema.get("properties", {})
        if not isinstance(properties, dict) or len(properties) > MAX_PROPERTIES:
            raise ToolFormatError("schema properties must be a bounded object")
        schema["properties"] = {_key(key): _schema(child) for key, child in properties.items()}
        required = schema.get("required", [])
        if (not isinstance(required, list) or len(required) > MAX_PROPERTIES
                or any(not isinstance(key, str) for key in required)):
            raise ToolFormatError("schema required must be a bounded string list")
        if len(set(required)) != len(required) or set(required) - set(properties):
            raise ToolFormatError("required parameters must be unique declared properties")
        schema["required"] = required
        if "additionalProperties" in schema and type(schema["additionalProperties"]) is not bool:
            raise ToolFormatError("additionalProperties must be boolean")
    if "items" in schema:
        schema["items"] = _schema(schema["items"])
    if "enum" in schema:
        enum = schema["enum"]
        if not isinstance(enum, list) or not enum or len(enum) > MAX_PROPERTIES:
            raise ToolFormatError("schema enum must be a nonempty bounded list")
        seen: list[Any] = []
        for item in enum:
            if isinstance(item, (dict, list)):
                raise ToolFormatError("only scalar enum values are supported")
            _instance(item, {"type": kind} if kind else {})
            if any(type(item) is type(old) and item == old for old in seen):
                raise ToolFormatError("schema enum values must be unique")
            if type(item) in (int, float) and any(type(old) in (int, float) and item == old for old in seen):
                raise ToolFormatError("schema enum values must be unique")
            seen.append(item)
    if "default" in schema:
        _instance(schema["default"], {key: item for key, item in schema.items() if key != "default"})
    return schema


def validate_tools(tools: object = None, tool_choice: object = None,
                   parallel_tool_calls: object = None) -> ToolConfig:
    """Validate OpenAI function tools and auto/none/required/named choices.

    Omitted parallel_tool_calls means true. Omitted choice is auto with tools,
    otherwise none. named is stored as tool_choice='named', required_name=name.
    """
    if tools is None:
        tools = []
    if not isinstance(tools, list) or len(tools) > MAX_TOOLS:
        raise ToolFormatError(f"tools must be a list of at most {MAX_TOOLS} functions")
    tools_data = _checked_json(tools, "tools", limit=MAX_SCHEMA_BYTES)
    if parallel_tool_calls is None:
        parallel_tool_calls = True
    if type(parallel_tool_calls) is not bool:
        raise ToolFormatError("parallel_tool_calls must be boolean")
    normalized: list[FunctionTool] = []
    names: set[str] = set()
    for tool in tools_data:
        if not isinstance(tool, dict) or set(tool) != {"type", "function"} or tool["type"] != "function":
            raise ToolFormatError("only type:function tools are supported")
        function = tool["function"]
        if not isinstance(function, dict) or set(function) - {"name", "description", "parameters", "strict"}:
            raise ToolFormatError("unsupported function tool fields")
        name = _name(function.get("name"))
        if name in names:
            raise ToolFormatError("function tool names must be unique")
        names.add(name)
        strict = function.get("strict")
        if strict is True:
            raise ToolFormatError("strict:true is unsupported; Gemma decoding is not grammar-constrained")
        if strict is not None and type(strict) is not bool:
            raise ToolFormatError("function strict must be boolean or null")
        description = _text(function.get("description", ""), "function description")
        parameters = _schema(function.get("parameters", {}), root=True)
        _native(_native_schema(parameters))
        normalized.append(FunctionTool(name, description, _canonical(parameters)))
    required_name = None
    if tool_choice is None:
        choice = "auto" if normalized else "none"
    elif isinstance(tool_choice, str):
        if tool_choice not in ("auto", "none", "required"):
            raise ToolFormatError("tool_choice must be auto, none, required or a named function")
        choice = tool_choice
    elif isinstance(tool_choice, dict):
        if (set(tool_choice) != {"type", "function"} or tool_choice.get("type") != "function"
                or not isinstance(tool_choice.get("function"), dict)
                or set(tool_choice["function"]) != {"name"}):
            raise ToolFormatError("invalid named function tool_choice")
        required_name = _name(tool_choice["function"]["name"])
        if required_name not in names:
            raise ToolFormatError("named tool_choice is not a declared function")
        choice = "named"
    else:
        raise ToolFormatError("invalid tool_choice")
    if choice == "required" and not normalized:
        raise ToolFormatError("required tool_choice needs at least one function")
    return ToolConfig(tuple(normalized), choice, parallel_tool_calls, required_name)


def _native(value: Any) -> str:
    if isinstance(value, str):
        return STRING_DELIMITER + _text(value, "native string") + STRING_DELIMITER
    if isinstance(value, dict):
        return "{" + ",".join(_key(key) + ":" + _native(value[key]) for key in sorted(value)) + "}"
    if isinstance(value, list):
        return "[" + ",".join(_native(item) for item in value) + "]"
    return _canonical(value)


def _native_schema(schema: dict[str, Any]) -> dict[str, Any]:
    result = dict(schema)
    if "type" in result:
        result["type"] = result["type"].upper()
    if "properties" in result:
        result["properties"] = {key: _native_schema(value) for key, value in result["properties"].items()}
    if "items" in result:
        result["items"] = _native_schema(result["items"])
    return result


def _rendered(value: str) -> str:
    return _text(value, "rendered tool data", MAX_TEXT_BYTES, safe=False)


def render_declarations(config: ToolConfig) -> str:
    """System-turn fragment; no BOS/turn wrappers. none emits no declarations.

    Named choice exposes only its selected declaration. Short policy instructions
    encourage required/single-call behavior; parse_tool_output enforces it later.
    """
    if not config.enabled:
        return ""
    chunks = []
    for tool in config.tools:
        if config.required_name is None or tool.name == config.required_name:
            body = {"description": tool.description, "parameters": _native_schema(tool.parameters)}
            chunks.append(TOOL_OPEN + "declaration:" + tool.name + _native(body) + TOOL_CLOSE)
    if config.tool_choice == "required":
        chunks.append("\nYou must call at least one of the declared functions.\n")
    elif config.required_name is not None:
        chunks.append("\nYou must call the function " + config.required_name + ".\n")
    if not config.parallel_tool_calls:
        chunks.append("\nCall at most one function in this response.\n")
    return _rendered("".join(chunks))


class _NativeParser:
    """Recursive descent for the documented delimiter-based data grammar."""

    def __init__(self, text: str, position: int):
        self.text = text
        self.position = position
        self.nodes = 0

    def space(self) -> None:
        while self.position < len(self.text) and self.text[self.position] in " \t\r\n":
            self.position += 1

    def take(self, token: str) -> None:
        self.space()
        if not self.text.startswith(token, self.position):
            raise ToolFormatError("malformed or incomplete native tool-call envelope")
        self.position += len(token)

    def key(self) -> str:
        self.space()
        match = _KEY_AT.match(self.text, self.position)
        if match is None:
            raise ToolFormatError("invalid native object key or function name")
        self.position = match.end()
        return match.group()

    def value(self, depth: int = 0) -> Any:
        self.space()
        self.nodes += 1
        if depth > MAX_DEPTH or self.nodes > MAX_NODES:
            raise ToolFormatError("native arguments exceed nesting or node limit")
        if self.position >= len(self.text):
            raise ToolFormatError("incomplete native tool arguments")
        if self.text.startswith(STRING_DELIMITER, self.position):
            start = self.position + len(STRING_DELIMITER)
            end = self.text.find(STRING_DELIMITER, start)
            if end < 0:
                raise ToolFormatError("unterminated native string")
            self.position = end + len(STRING_DELIMITER)
            return _text(self.text[start:end], "native argument string")
        char = self.text[self.position]
        if char in "{[":
            self.position += 1
            closing = "}" if char == "{" else "]"
            value: Any = {} if char == "{" else []
            self.space()
            if self.text.startswith(closing, self.position):
                self.position += 1
                return value
            while True:
                if char == "{":
                    key = self.key()
                    if key in value:
                        raise ToolFormatError("native arguments contain a duplicate object key")
                    self.take(":")
                    value[key] = self.value(depth + 1)
                else:
                    value.append(self.value(depth + 1))
                self.space()
                if self.text.startswith(closing, self.position):
                    self.position += 1
                    return value
                self.take(",")
        for token, value in (("true", True), ("false", False), ("null", None)):
            if self.text.startswith(token, self.position):
                self.position += len(token)
                return value
        match = _NUMBER_AT.match(self.text, self.position)
        if match is None or len(match.group()) > MAX_NUMBER_CHARS:
            raise ToolFormatError("invalid native argument value")
        self.position = match.end()
        return loads_strict_json(match.group(), field="native number")


def _first_call(text: str) -> int | None:
    """Find protocol only outside Gemma strings, prose quotes and Markdown code.

    Literal quoted examples are text, never actions. This is deliberately
    conservative for ambiguous/unclosed quotes: under required/named, no call is
    an error rather than permission to manufacture one.
    """
    i = 0
    while i < len(text):
        if text.startswith(STRING_DELIMITER, i):
            end = text.find(STRING_DELIMITER, i + len(STRING_DELIMITER))
            if end < 0:
                return None
            i = end + len(STRING_DELIMITER)
        elif text[i] == "`":
            end = i
            while end < len(text) and text[end] == "`":
                end += 1
            delimiter = text[i:end]
            closing = text.find(delimiter, end)
            if closing < 0:
                return None
            i = closing + len(delimiter)
        elif text[i] in "\"'" and (i == 0 or not text[i - 1].isalnum()):
            quote = text[i]
            i += 1
            while i < len(text):
                if text[i] == "\\":
                    i += 2
                elif text[i] == quote:
                    i += 1
                    break
                else:
                    i += 1
        elif text.startswith(CALL_OPEN, i):
            return i
        elif text.startswith(("<|tool", "<tool", "<|turn", "<turn", "<|channel", "<channel"), i):
            raise ToolFormatError("unexpected or incomplete native control envelope")
        else:
            i += 1
    return None


def parse_tool_output(text: str, config: ToolConfig,
                      call_id_prefix: str) -> ParsedToolOutput:
    """Parse one complete, reasoning-stripped model result; never execute it.

    call_id_prefix is a nonempty caller-owned request/completion identity, NOT an
    unchecked literal ID prefix. SHA-256 binds it, call ordinal, name and canonical
    arguments into safe deterministic call_<32 hex> IDs. Use a different identity
    for every distinct native request; retries of the same result preserve IDs.
    The caller owns native COMMIT/stop-vs-length; ordinary text returns 'stop'.
    """
    text = _text(text, "model output", MAX_TEXT_BYTES, safe=False)
    seed = _text(call_id_prefix, "call ID seed", 1024, safe=False)
    if not seed:
        raise ToolFormatError("call ID seed must not be empty")
    start = _first_call(text)
    if start is None:
        if config.tool_choice in ("required", "named"):
            raise ToolFormatError("model did not produce the required function call")
        return ParsedToolOutput(text, (), "stop")
    if not config.enabled:
        raise ToolFormatError("model produced a function call when tools are disabled")
    parser = _NativeParser(text, start)
    declared = {tool.name: tool for tool in config.tools}
    calls = []
    while True:
        parser.take(CALL_OPEN)
        parser.take("call:")
        name = _name(parser.key())
        if name not in declared or (config.required_name is not None and name != config.required_name):
            raise ToolFormatError("model called an undeclared or disallowed function")
        arguments = parser.value()
        if not isinstance(arguments, dict):
            raise ToolFormatError("native function arguments must be an object")
        arguments = _checked_json(arguments, "function arguments")
        _instance(arguments, declared[name].parameters)
        parser.take(CALL_CLOSE)
        serialized = _canonical(arguments)
        digest = hashlib.sha256(_canonical(["gemma4-tool-call-v1", seed, len(calls), name, arguments]).encode("utf-8")).hexdigest()
        calls.append({"id": "call_" + digest[:32], "type": "function",
                      "function": {"name": name, "arguments": serialized}})
        if len(calls) > MAX_CALLS or (len(calls) > 1 and not config.parallel_tool_calls):
            raise ToolFormatError("model exceeded the allowed number of function calls")
        parser.space()
        if text.startswith(CALL_OPEN, parser.position):
            continue
        if text.startswith(RESPONSE_OPEN, parser.position):
            parser.position += len(RESPONSE_OPEN)
            parser.space()
        if parser.position != len(text):
            raise ToolFormatError("unexpected text or incomplete envelope after native tool call")
        break
    _checked_json(calls, "parsed tool calls", safe=False)
    content = text[:start]
    return ParsedToolOutput(content if content.strip() else None, tuple(calls), "tool_calls")


def _calls(tool_calls: object, config: ToolConfig | None = None) -> list[dict[str, Any]]:
    if not isinstance(tool_calls, (list, tuple)) or not 1 <= len(tool_calls) <= MAX_CALLS:
        raise ToolFormatError(f"tool_calls must contain 1..{MAX_CALLS} functions")
    calls = _checked_json(list(tool_calls), "tool_calls")
    ids = set()
    declared = {tool.name: tool for tool in config.tools} if config is not None else None
    for call in calls:
        if not isinstance(call, dict) or set(call) != {"id", "type", "function"} or call["type"] != "function":
            raise ToolFormatError("invalid OpenAI function tool_call")
        identifier = call["id"]
        if not isinstance(identifier, str) or _CALL_ID.fullmatch(identifier) is None:
            raise ToolFormatError("tool_call id must be a bounded safe identifier")
        if identifier in ids:
            raise ToolFormatError("duplicate tool_call id")
        ids.add(identifier)
        function = call["function"]
        if not isinstance(function, dict) or set(function) != {"name", "arguments"}:
            raise ToolFormatError("invalid tool_call function fields")
        name = _name(function["name"])
        arguments = loads_strict_json(function["arguments"], field="function.arguments")
        if not isinstance(arguments, dict):
            raise ToolFormatError("function.arguments must encode a JSON object")
        arguments = _checked_json(arguments, "function.arguments")
        # Even with no current declaration, validate every key before rendering.
        _native(arguments)
        if declared is not None:
            if name not in declared:
                raise ToolFormatError("historical tool_call uses an undeclared function")
            _instance(arguments, declared[name].parameters)
        call["function"]["arguments"] = _canonical(arguments)
    return calls


def render_assistant_calls(tool_calls: object, config: ToolConfig | None = None,
                           *, content: str | None = None) -> str:
    """Historical assistant fragment ending in <|tool_response>; no turn wrapper.

    Optional config checks declarations/argument schemas, but deliberately ignores
    the current choice policy (a historical call may precede today's choice:none).
    Pair with render_tool_results and the following assistant's final text in the
    SAME model turn. Do not insert <turn|> or a new model opener between them.
    """
    calls = _calls(tool_calls, config)
    chunks = [_text(content, "assistant content", MAX_TEXT_BYTES)] if content is not None else []
    for call in calls:
        function = call["function"]
        chunks.append(CALL_OPEN + "call:" + function["name"]
                      + _native(json.loads(function["arguments"])) + CALL_CLOSE)
    chunks.append(RESPONSE_OPEN)
    return _rendered("".join(chunks))


def render_tool_results(messages: object, pending_tool_calls: object) -> str:
    """Validate all exact call IDs and render results in original call order.

    messages is ONLY the unseen tool-result suffix, not cumulative chat history.
    Every pending call must have exactly one result, no unknown/duplicate IDs.
    Out-of-order results are accepted and ordered by pending call ID, never name.
    Optional message.name must agree. Content is a string or text-only OpenAI
    content parts, kept opaque and wrapped in the native 'result' string field.

    The native state / render_assistant_calls ALREADY includes <|tool_response>.
    Return starts 'response:', closes each response, opens the next if necessary,
    and ends <tool_response|>. Append it raw, then resume SAME model generation.
    """
    calls = _calls(pending_tool_calls)
    if not isinstance(messages, list) or len(messages) != len(calls):
        raise ToolFormatError("tool results must answer every pending call exactly once")
    messages_data = _checked_json(messages, "tool results")
    pending = {call["id"]: call["function"]["name"] for call in calls}
    results = {}
    for message in messages_data:
        if (not isinstance(message, dict) or message.get("role") != "tool"
                or set(message) - {"role", "tool_call_id", "content", "name"}):
            raise ToolFormatError("tool result suffix must contain only tool messages")
        identifier = message.get("tool_call_id")
        if not isinstance(identifier, str) or identifier not in pending:
            raise ToolFormatError("tool result references an unknown call ID")
        if identifier in results:
            raise ToolFormatError("duplicate tool result call ID")
        if "name" in message and message["name"] != pending[identifier]:
            raise ToolFormatError("tool result name does not match its exact call ID")
        content = message.get("content")
        if isinstance(content, list):
            parts = []
            for part in content:
                if not isinstance(part, dict) or set(part) != {"type", "text"} or part["type"] != "text":
                    raise ToolFormatError("only text content parts are supported for tool results")
                parts.append(_text(part["text"], "tool result text"))
            content = "".join(parts)
        results[identifier] = _text(content, "tool result content")
    if set(results) != set(pending):
        raise ToolFormatError("missing tool result call ID")
    chunks = []
    for index, call in enumerate(calls):
        if index:
            chunks.append(RESPONSE_OPEN)
        chunks.append("response:" + call["function"]["name"]
                      + _native({"result": results[call["id"]]}) + RESPONSE_CLOSE)
    return _rendered("".join(chunks))
