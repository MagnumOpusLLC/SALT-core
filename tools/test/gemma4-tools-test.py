#!/usr/bin/env python3
"""No-model, stdlib-only gate for the Gemma/OpenAI tool wire adapter."""
from __future__ import annotations

import copy
from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import re
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "server" / "model"))
import gemma4_tools as g

S = g.STRING_DELIMITER


def function(name="weather", parameters=None, **fields):
    body = {"name": name, "description": 'Weather, with {braces}, quotes " and Unicode 東京.'}
    if parameters is not None:
        body["parameters"] = parameters
    body.update(fields)
    return {"type": "function", "function": body}


def native(name="weather", arguments="{}"):
    return g.CALL_OPEN + "call:" + name + arguments + g.CALL_CLOSE


def api_call(identifier="call_a", name="weather", arguments="{}"):
    return {"id": identifier, "type": "function",
            "function": {"name": name, "arguments": arguments}}


def result(identifier="call_a", content="ok", **fields):
    return {"role": "tool", "tool_call_id": identifier, "content": content, **fields}


class ToolAdapterTests(unittest.TestCase):
    def setUp(self):
        self.config = g.validate_tools([function()])
        self.required = g.validate_tools([function()], "required")

    def reject(self, callback, *args, **kwargs):
        with self.assertRaises(g.ToolFormatError):
            callback(*args, **kwargs)

    def test_default_and_none_choices(self):
        empty = g.validate_tools(None, None, None)
        self.assertFalse(empty.enabled)
        self.assertEqual(empty.tool_choice, "none")
        self.assertTrue(empty.parallel_tool_calls)
        self.assertEqual(g.render_declarations(empty), "")
        self.assertTrue(self.config.enabled)
        self.assertEqual(self.config.tool_choice, "auto")
        none = g.validate_tools([function()], "none", False)
        self.assertFalse(none.enabled)
        self.assertEqual(g.render_declarations(none), "")
        self.assertEqual(g.parse_tool_output("plain text", none, "r1").tool_calls, ())
        self.reject(g.parse_tool_output, native(), none, "r1")
        self.reject(g.parse_tool_output, native(), empty, "r1")

    def test_choice_shapes_and_named_selection(self):
        named = {"type": "function", "function": {"name": "weather"}}
        config = g.validate_tools([function(), function("clock")], named, False)
        self.assertEqual(config.required_name, "weather")
        self.assertEqual(config.tool_choice, "named")
        declarations = g.render_declarations(config)
        self.assertIn("declaration:weather", declarations)
        self.assertNotIn("declaration:clock", declarations)
        self.assertIn("at most one", declarations)
        self.reject(g.parse_tool_output, native("clock"), config, "r")
        self.reject(g.parse_tool_output, "No tool needed.", config, "r")
        self.assertEqual(g.parse_tool_output(native(), config, "r").finish_reason, "tool_calls")
        for choice in (True, [], 1, "any", {}, {"type": "function"},
                       {"type": "function", "function": {"name": "missing"}},
                       {"type": "function", "function": {"name": "weather", "extra": 1}}):
            with self.subTest(choice=choice):
                self.reject(g.validate_tools, [function()], choice)
        self.reject(g.validate_tools, [], "required")
        for parallel in (0, 1, "false", [], {}):
            self.reject(g.validate_tools, [function()], "auto", parallel)

    def test_required_fails_closed_no_fabricated_calls(self):
        self.reject(g.parse_tool_output, "I cannot do that.", self.required, "r")
        self.reject(g.parse_tool_output, "", self.required, "r")
        self.assertIn("must call", g.render_declarations(self.required))

    def test_tool_shapes_names_duplicates_and_strict(self):
        invalid = ["[]", {}, [None], [{"type": "custom", "function": {"name": "f"}}],
                   [{"type": "function", "function": "f"}],
                   [{"type": "function", "function": {}}],
                   [function(), function()], [function(strict=True)],
                   [function(strict=1)], [function(strict="false")],
                   [function(extra="not supported")]]
        for tools in invalid:
            with self.subTest(tools=tools):
                self.reject(g.validate_tools, tools)
        with self.assertRaisesRegex(g.ToolFormatError, "not grammar-constrained"):
            g.validate_tools([function(strict=True)])
        self.assertTrue(g.validate_tools([function(strict=False)]).enabled)
        self.assertTrue(g.validate_tools([function(strict=None)]).enabled)
        for name in ("", "a" * 65, "f()", "a.b", "a:b", "a b", "tool\n", "東京", 1):
            self.reject(g.validate_tools, [function(name)])
        for name in ("a-b_2", "2", "A" * 64):
            self.assertTrue(g.validate_tools([function(name)]).enabled)

    def test_schema_render_nested_types_without_touching_defaults(self):
        parameters = {"type": "object", "properties": {
            "type": {"type": "string", "enum": ["object"], "default": "object"},
            "settings": {"type": "object", "properties": {
                "labels": {"type": "array", "items": {"type": "string"}}},
                "required": ["labels"], "additionalProperties": False}},
            "required": ["settings"], "additionalProperties": False}
        config = g.validate_tools([function(parameters=parameters)])
        rendered = g.render_declarations(config)
        self.assertTrue(rendered.startswith(g.TOOL_OPEN + "declaration:weather{"))
        self.assertTrue(rendered.endswith(g.TOOL_CLOSE))
        for type_name in ("OBJECT", "ARRAY", "STRING"):
            self.assertIn("type:" + S + type_name + S, rendered)
        self.assertIn("default:" + S + "object" + S, rendered)
        self.assertIn("enum:[" + S + "object" + S + "]", rendered)
        self.assertIn("東京", rendered)
        self.assertNotIn("<bos>", rendered)
        self.assertNotIn("<|turn>", rendered)

    def test_configuration_copied_and_frozen(self):
        raw = [function(parameters={"type": "object", "properties": {"x": {"type": "string"}}})]
        config = g.validate_tools(raw)
        raw[0]["function"]["parameters"]["properties"]["x"]["type"] = "number"
        copy_schema = config.tools[0].parameters
        copy_schema["properties"]["x"]["type"] = "boolean"
        self.assertEqual(config.tools[0].parameters["properties"]["x"]["type"], "string")
        with self.assertRaises(FrozenInstanceError):
            config.tool_choice = "none"

    def test_invalid_or_unsupported_schema_refused(self):
        invalid = [[], True, {"type": "array"}, {"type": "OBJECT"}, {"type": None},
                   {"type": ["object", "null"]}, {"$ref": "#/defs/x"},
                   {"type": "object", "anyOf": []},
                   {"type": "object", "properties": []},
                   {"type": "object", "properties": {"x.y": {"type": "string"}}},
                   {"type": "object", "properties": {"x": {"type": "unknown"}}},
                   {"type": "object", "properties": {"x": {"type": "string", "minimum": 1}}},
                   {"type": "object", "required": ["missing"]},
                   {"type": "object", "required": "x"},
                   {"type": "object", "required": [1]},
                   {"type": "object", "properties": {"x": {}}, "required": ["x", "x"]},
                   {"type": "object", "additionalProperties": {}},
                   {"type": "object", "description": []},
                   {"type": "object", "items": {}},
                   {"type": "object", "properties": {"x": {"properties": {}}}},
                   {"type": "object", "properties": {"x": {"type": "array", "items": []}}},
                   {"type": "object", "properties": {"x": {"enum": []}}},
                   {"type": "object", "properties": {"x": {"enum": [1, 1.0]}}},
                   {"type": "object", "properties": {"x": {"enum": [{}]}}},
                   {"type": "object", "properties": {"x": {"type": "string", "enum": [1]}}},
                   {"type": "object", "default": []}]
        for schema in invalid:
            with self.subTest(schema=schema):
                self.reject(g.validate_tools, [function(parameters=schema)])

    def test_recursive_native_values_unicode_punctuation(self):
        args = ('{city:' + S + '東京, {district}: "湾" \\ literal\nline' + S
                + ',nested:{items:[1,-2.5e2,true,false,null,{x:' + S + 'a,b}c' + S + '}]},empty:{},list:[]}')
        output = g.parse_tool_output("Checking now. " + native(arguments=args) + g.RESPONSE_OPEN,
                                     self.config, "completion_42")
        self.assertEqual(output.content, "Checking now. ")
        self.assertEqual(output.finish_reason, "tool_calls")
        self.assertEqual(len(output.tool_calls), 1)
        call = output.tool_calls[0]
        self.assertEqual(call["type"], "function")
        self.assertIsInstance(call["function"]["arguments"], str)
        decoded = json.loads(call["function"]["arguments"])
        self.assertEqual(decoded["city"], '東京, {district}: "湾" \\ literal\nline')
        self.assertEqual(decoded["nested"]["items"], [1, -250.0, True, False, None, {"x": "a,b}c"}])
        self.assertEqual(decoded["empty"], {})
        self.assertEqual(decoded["list"], [])
        self.assertRegex(call["id"], r"^call_[0-9a-f]{32}$")

    def test_optional_hidden_yield_and_parallel_envelopes(self):
        text = native() + "\n" + native(arguments="{n:2}")
        no_yield = g.parse_tool_output(text, self.config, "r")
        yield_visible = g.parse_tool_output(text + g.RESPONSE_OPEN + "\n", self.config, "r")
        self.assertEqual(no_yield, yield_visible)
        self.assertEqual(len(no_yield.tool_calls), 2)
        self.assertIsNone(no_yield.content)
        self.assertNotEqual(no_yield.tool_calls[0]["id"], no_yield.tool_calls[1]["id"])
        one = g.validate_tools([function()], "auto", False)
        self.reject(g.parse_tool_output, text, one, "r")
        self.reject(g.parse_tool_output, native() + g.RESPONSE_OPEN + native(), self.config, "r")

    def test_deterministic_ids_bind_seed_arguments_and_ordinal(self):
        pair = native() + native()
        one = g.parse_tool_output(pair, self.config, 'unsafe identity / 東京 <ignored>').tool_calls
        self.assertEqual(one, g.parse_tool_output(pair, self.config, 'unsafe identity / 東京 <ignored>').tool_calls)
        self.assertNotEqual(one[0]["id"], one[1]["id"])
        self.assertNotEqual(one[0]["id"], g.parse_tool_output(pair, self.config, "r2").tool_calls[0]["id"])
        self.assertNotEqual(one[0]["id"], g.parse_tool_output(native(arguments="{x:1}"), self.config,
                                                              'unsafe identity / 東京 <ignored>').tool_calls[0]["id"])
        reordered = g.parse_tool_output(native(arguments="{b:2,a:1}"), self.config, "r")
        ordered = g.parse_tool_output(native(arguments="{a:1,b:2}"), self.config, "r")
        self.assertEqual(reordered.tool_calls, ordered.tool_calls)
        for seed in ("", "x" * 1025, None, "\ud800"):
            self.reject(g.parse_tool_output, native(), self.config, seed)

    def test_quoted_examples_are_not_actions(self):
        envelope = native()
        examples = ['"' + envelope + '"', "'" + envelope + "'", '`' + envelope + '`',
                    '```example\n' + envelope + '\n```', S + envelope + S,
                    'Example: "escaped \\" quote ' + envelope + '"',
                    '```' + envelope]
        for example in examples:
            with self.subTest(example=example):
                parsed = g.parse_tool_output(example, self.config, "r")
                self.assertEqual(parsed.content, example)
                self.assertEqual(parsed.tool_calls, ())
                self.reject(g.parse_tool_output, example, self.required, "r")
        self.assertEqual(g.parse_tool_output("I'll check now. " + envelope, self.config, "r").finish_reason,
                         "tool_calls")
        self.assertEqual(g.parse_tool_output('"example" then ' + envelope, self.config, "r").finish_reason,
                         "tool_calls")

    def test_incomplete_envelope_at_every_split_fails_closed(self):
        text = native(arguments="{x:" + S + "nested {,}" + S + ",y:[{z:true}]}")
        for split in range(1, len(text)):
            with self.subTest(split=split):
                self.reject(g.parse_tool_output, text[:split], self.required, "r")
        # Reassembled streaming chunks are parsed only after the complete result.
        self.assertEqual(g.parse_tool_output("".join(text[i:i + 3] for i in range(0, len(text), 3)),
                                            self.required, "r").finish_reason, "tool_calls")

    def test_native_syntax_and_trailing_control_refused(self):
        for args in ("[]", "null", '{x:"not native"}', "{x:bare}", "{x:NaN}", "{x:Infinity}",
                     "{x:1e999}", "{x:01}", "{x:+1}", "{x:.1}", "{x:True}", "{x:1,}",
                     "{x:[1,]}", "{x:1,x:2}", "{x:{y:1,y:2}}", "{x:" + S + "open}",
                     "{x.y:1}", "{x:__import__(x)}"):
            with self.subTest(args=args):
                self.reject(g.parse_tool_output, native(arguments=args), self.config, "r")
        for suffix in (" stray", "<turn|>", g.RESPONSE_OPEN + "pretend result", "<|tool_call", "<|tool_response"):
            self.reject(g.parse_tool_output, native() + suffix, self.config, "r")
        for text in (g.CALL_CLOSE, g.RESPONSE_OPEN, "<|tool_call", "<|tool_call>wrong:weather{}<tool_call|>"):
            self.reject(g.parse_tool_output, text, self.config, "r")
        self.reject(g.parse_tool_output, native("unknown"), self.config, "r")

    def test_schema_checks_generated_arguments(self):
        schema = {"type": "object", "properties": {
            "city": {"type": "string", "enum": ["東京", "London"]},
            "n": {"type": "integer"},
            "nested": {"type": "array", "items": {"type": "object", "properties": {
                "ok": {"type": "boolean"}}, "required": ["ok"], "additionalProperties": False}}},
            "required": ["city", "n"], "additionalProperties": False}
        config = g.validate_tools([function(parameters=schema)])
        valid = "{city:" + S + "東京" + S + ",n:1.0,nested:[{ok:true}]}"
        self.assertEqual(g.parse_tool_output(native(arguments=valid), config, "r").finish_reason, "tool_calls")
        for args in ("{}", "{city:" + S + "Paris" + S + ",n:1}",
                     "{city:" + S + "東京" + S + ",n:true}",
                     "{city:" + S + "東京" + S + ",n:1.5}",
                     "{city:" + S + "東京" + S + ",n:1,extra:2}",
                     "{city:" + S + "東京" + S + ",n:1,nested:[{}]}",
                     "{city:" + S + "東京" + S + ",n:1,nested:[{ok:1}]}"):
            self.reject(g.parse_tool_output, native(arguments=args), config, "r")
        bool_schema = {"type": "object", "properties": {"x": {"enum": [True]}}}
        self.reject(g.parse_tool_output, native(arguments="{x:1}"),
                    g.validate_tools([function(parameters=bool_schema)]), "r")

    def test_reserved_tokens_refused_as_prompt_data(self):
        for token in (S, g.CALL_OPEN, g.CALL_CLOSE, g.RESPONSE_OPEN, "<|turn>", "<turn|>",
                      "<|channel>", "<channel|>", "<|image|>", "<|think|>", "<bos>", "<eos>"):
            with self.subTest(token=token):
                self.reject(g.validate_tools, [function(description="bad " + token)])
                self.reject(g.validate_tools, [function(parameters={"type": "object", "properties": {
                    "x": {"type": "string", "default": token}}})])
                self.reject(g.render_tool_results, [result(content="bad " + token)], [api_call()])
                self.reject(g.render_assistant_calls, [api_call(arguments=json.dumps({"x": token}))])
                self.reject(g.render_assistant_calls, [api_call()], content=token)
                self.reject(g.parse_tool_output, native(arguments="{x:" + S + token + S + "}"), self.config, "r")
        self.reject(g.render_assistant_calls, [api_call(arguments='{"x":"\\u003c|turn>"}')])

    def test_strict_json_duplicates_nonfinite_unicode_and_limits(self):
        self.assertEqual(g.loads_strict_json('{"x":[true,null,1.5],"y":"東京"}'),
                         {"x": [True, None, 1.5], "y": "東京"})
        for text in ('{"a":1,"a":2}', '{"x":{"a":1,"a":2}}', '{"a":1,"\\u0061":2}',
                     '{"a":NaN}', '{"a":Infinity}', '{"a":-Infinity}', '{"a":1e999}',
                     '{"a":1,}', '{"a":"\\ud800"}', '{"a":"\\u0000"}',
                     "[" * (g.MAX_DEPTH + 1) + "]" * (g.MAX_DEPTH + 1),
                     "1" * (g.MAX_NUMBER_CHARS + 1), " " * (g.MAX_JSON_BYTES + 1)):
            with self.subTest(text=text[:80]):
                self.reject(g.loads_strict_json, text)
        # Reserved literals are legal envelope data; native renderers reject them.
        self.assertEqual(g.loads_strict_json('{"x":"<|turn>"}')["x"], "<|turn>")

    def test_historical_call_roundtrip_and_schema_validation(self):
        parsed = g.parse_tool_output(native(arguments="{x:[{y:" + S + "東京,{,}" + S + "}]}") + native(),
                                     self.config, "roundtrip")
        historical = g.render_assistant_calls(parsed.tool_calls, self.config, content="Checking. ")
        roundtrip = g.parse_tool_output(historical, self.config, "roundtrip")
        self.assertEqual(roundtrip.tool_calls, parsed.tool_calls)
        self.assertEqual(roundtrip.content, "Checking. ")
        self.assertTrue(historical.endswith(g.RESPONSE_OPEN))
        none = g.validate_tools([function()], "none")
        self.assertTrue(g.render_assistant_calls([api_call()], none).endswith(g.RESPONSE_OPEN))
        self.reject(g.render_assistant_calls, [api_call(name="other")], self.config)
        required = g.validate_tools([function(parameters={"type": "object", "properties": {
            "x": {"type": "string"}}, "required": ["x"]})])
        self.reject(g.render_assistant_calls, [api_call()], required)

    def test_historical_calls_must_be_exact_safe_json_objects(self):
        for arguments in ({}, "[]", "null", '{"x":1,"x":2}', '{"x":NaN}', '{"x":"\\ud800"}', '{bad}',
                          '{"spaced key":1}', '{"nested":{"bad.key":1}}'):
            self.reject(g.render_assistant_calls, [api_call(arguments=arguments)])
        for calls in ([], [api_call(), api_call()], [None], [{"id": "a"}],
                      [api_call(identifier="bad/id")], [api_call(name="bad.name")],
                      [dict(api_call(), extra=True)], [dict(api_call(), type="custom")],
                      [{"id": "a", "type": "function", "function": {"name": "weather"}}]):
            self.reject(g.render_assistant_calls, calls)

    def test_results_ordered_by_exact_ids_even_same_function(self):
        pending = [api_call("call_a"), api_call("call_b")]
        messages = [result("call_b", "second"), result("call_a", "first", name="weather")]
        original = copy.deepcopy(messages)
        suffix = g.render_tool_results(messages, pending)
        self.assertEqual(suffix, "response:weather{result:" + S + "first" + S + "}" + g.RESPONSE_CLOSE
                         + g.RESPONSE_OPEN + "response:weather{result:" + S + "second" + S + "}" + g.RESPONSE_CLOSE)
        self.assertEqual(messages, original)
        self.assertNotIn("<|turn>", suffix)
        self.assertNotIn("<turn|>", suffix)
        full_turn = g.render_assistant_calls(pending) + suffix + "The answer."
        self.assertEqual(full_turn.count(g.RESPONSE_OPEN), 2)
        self.assertTrue(full_turn.endswith(g.RESPONSE_CLOSE + "The answer."))

    def test_results_missing_unknown_duplicate_and_role_rejected(self):
        pending = [api_call("call_a"), api_call("call_b")]
        invalid = [[], [result()], [result(), result("call_unknown")], [result(), result()],
                   [result(), {"role": "user", "tool_call_id": "call_b", "content": "x"}],
                   [result(), result("call_b", "x", name="other")],
                   [result(), result("call_b", "x", extra=True)],
                   [result(), result("call_b", None)],
                   [result(), {"role": "tool", "content": "x"}],
                   [result(), result("call_b", {"x": 1})]]
        for messages in invalid:
            self.reject(g.render_tool_results, messages, pending)
        self.reject(g.render_tool_results, [result(), result()], [api_call(), api_call()])
        self.reject(g.render_tool_results, [], [])

    def test_results_preserve_opaque_text_and_text_parts(self):
        text = '{"arbitrary key":{"quoted":"東京,{}"}}\nnot code \\ value'
        suffix = g.render_tool_results([result(content=text)], [api_call()])
        self.assertIn(S + text + S, suffix)
        parts = [{"type": "text", "text": "first\n"}, {"type": "text", "text": "second"}]
        self.assertIn(S + "first\nsecond" + S, g.render_tool_results([result(content=parts)], [api_call()]))
        self.assertIn(S + S, g.render_tool_results([result(content=[])], [api_call()]))
        for parts in ([{"type": "image_url", "image_url": "x"}], [{"type": "text"}], [None],
                      [{"type": "text", "text": 1}]):
            self.reject(g.render_tool_results, [result(content=parts)], [api_call()])

    def test_count_depth_node_and_byte_bounds(self):
        self.assertEqual(len(g.validate_tools([function("f" + str(i)) for i in range(g.MAX_TOOLS)]).tools), g.MAX_TOOLS)
        self.reject(g.validate_tools, [function("f" + str(i)) for i in range(g.MAX_TOOLS + 1)])
        schema = {"type": "object", "properties": {"p" + str(i): {} for i in range(g.MAX_PROPERTIES)},
                  "required": ["p" + str(i) for i in range(g.MAX_PROPERTIES)]}
        self.assertTrue(g.validate_tools([function(parameters=schema)]).enabled)
        schema["properties"]["overflow"] = {}
        self.reject(g.validate_tools, [function(parameters=schema)])
        self.reject(g.validate_tools, [function(parameters={"type": "object", "required": ["x"] * (g.MAX_PROPERTIES + 1)})])
        self.assertEqual(len(g.parse_tool_output(native() * g.MAX_CALLS, self.config, "r").tool_calls), g.MAX_CALLS)
        self.reject(g.parse_tool_output, native() * (g.MAX_CALLS + 1), self.config, "r")
        self.reject(g.render_assistant_calls, [api_call("c" + str(i)) for i in range(g.MAX_CALLS + 1)])
        self.reject(g.parse_tool_output, "x" * (g.MAX_TEXT_BYTES + 1), self.config, "r")
        self.reject(g.parse_tool_output, "東京" * (g.MAX_TEXT_BYTES // 4), self.config, "r")
        self.reject(g.validate_tools, [function(description="x" * (g.MAX_STRING_BYTES + 1))])
        self.reject(g.validate_tools, [function("f" + str(i), description="x" * 20000) for i in range(4)])
        self.reject(g.render_tool_results, [result(content="x" * (g.MAX_STRING_BYTES + 1))], [api_call()])
        self.reject(g.parse_tool_output, native(arguments="{x:" + "[" * (g.MAX_DEPTH + 1) + "0"
                                                + "]" * (g.MAX_DEPTH + 1) + "}"), self.config, "r")
        self.reject(g.parse_tool_output, native(arguments="{x:[" + ",".join("0" for _ in range(g.MAX_NODES + 1)) + "]}"),
                    self.config, "r")
        self.reject(g.loads_strict_json, "[" + ",".join("0" for _ in range(g.MAX_NODES + 1)) + "]")
        cyclic = {}
        cyclic["x"] = cyclic
        self.reject(g.validate_tools, [function(parameters=cyclic)])
        self.reject(g.validate_tools, [function(parameters={"type": "object", "default": {"x": float("nan")}})])
        self.reject(g.validate_tools, [function(parameters={"type": "object", "default": {"x": 10 ** 1000}})])

    def test_plain_output_unchanged_and_unicode_refused(self):
        text = " No tools needed.\n東京, punctuation {} and quotes. "
        parsed = g.parse_tool_output(text, self.config, "r")
        self.assertEqual(parsed.content, text)
        self.assertEqual(parsed.finish_reason, "stop")
        self.assertEqual(parsed.tool_calls, ())
        for text in ("\ud800", "x\x00y", 1, None):
            self.reject(g.parse_tool_output, text, self.config, "r")
        self.reject(g.validate_tools, [function(description="\ud800")])


if __name__ == "__main__":
    unittest.main(verbosity=2)
