# KV Cache and Persistent Agent State Statement

**Status: public risk disclosure and responsible-use principles.** This is not
an additional software-license condition, an executed state-use agreement, or
certification that the safeguards described below are implemented.

> [!WARNING]
> A persistent KV/state artifact is more than a saved prompt or a disposable
> performance cache. It can preserve accumulated model-computed state from
> human interactions and an agent's own automated activity. A compatible system
> may use it to continue or branch an already-developed computational
> trajectory. This enables valuable continuity, portability, and reuse, while
> creating serious risks of disclosure, unauthorized reuse, and behavioral
> misrepresentation. These risks exist with or without SALT. Treat person-linked,
> group-linked, and confidential operational state as sensitive data, not as
> ownerless material or proof of a human mind.

This notice is intentionally prominent. The risk is not removed by the facts
that KV rows are opaque tensors, that source text is absent, or that no current
tool is known to reconstruct a particular input.

This statement describes SALT's safety, honesty, and rights-preserving position.
Its requirements express the responsible-use standard advocated here; they do
not amend the software license. It is not consent, a privacy policy, or a grant
of ownership or commercial rights in anyone's state. Separate agreements and
applicable law determine enforceable rights and obligations. Software licensing
and permission to use a particular state artifact are separate questions.

## Value and risk are inseparable

Preserving state can avoid repeated computation, support continuity across
sessions and compatible machines, and make accumulated computational work
reusable. The same persistence can extend the lifetime of sensitive information
and harmful conditioning; the same portability can enable unauthorized copies;
the same reuse can enable purposes the contributors never agreed to.

SALT makes preservation and reuse explicit. It does not originate these general
risks, and neither this disclosure nor use of SALT removes them.

**Open computation does not mean unrestricted access to human-derived state.
Possession, technical compatibility, and permission are different things.**

## Beyond the prompt: what is being preserved

| Object | Meaning and limit |
|---|---|
| Prompt | A particular input supplied to the model. A prompt is not the whole subsequent execution history. |
| Persistent model state | Numerical state accumulated from processed inputs and generated continuations, including tool observations, feedback, plans, and revisions insofar as they have entered the model's computation. |
| Complete agent state | Model state plus any external memory, task progress, pending actions, orchestration, and other runtime state needed by that agent. KV alone does not necessarily contain these components. |

An agent may generate a plan, request a tool action, receive an observation,
revise its approach, and continue without another human prompt. When these
developments enter the model's context, preserved state can carry their
computational effects into later execution. This includes effects of the
agent's own outputs, not only material explicitly authored by a human.

The protected interest is therefore not limited to recovering what a user
typed. It also concerns who may resume, branch, interrogate, or repurpose the
preserved computational trajectory, including behavior and conclusions the
user never explicitly authored or approved. Permission to retain a prompt is
not permission to retain or exploit that trajectory.

Preservation does not establish that every past event, internal activation, or
external action is present, independently retrievable, or reproducible. Model
geometry, context eviction or compression, the exported state boundary, model
compatibility, sampling, and the external environment matter. Exact state-byte
preservation is not a promise of identical behavior under changed conditions.

## Scope

In this document, **state artifact** includes committed KV rows, recurrent or
mindset tensors, complete portable state, state snapshots, derived state
catalogs, and any other durable representation that allows model computation
to resume from prior human or group interactions or automated agent activity.
An artifact may be personal, collective, organizational, or nonpersonal; not
every cache represents an identifiable person. Sensitivity depends on its
contents, context, uses, and affected people, not merely its filename.

**Preservation** means retaining state beyond the transient processing needed
for the active, user-requested session. It includes export, snapshots, backups,
replication, inclusion in a cache or DPR catalog, analytics copies, retained
diagnostics, and transfer to another operator or device.

## Five non-negotiable disclosures

### 1. Specific state may permit reversible reconstruction

A state artifact may permit exact or approximate reconstruction, extraction,
or inference of source information under a sufficiently specific combination
of model, tokenizer, artifact bytes, prompts, auxiliary information, and
analysis method. The applicable conditions and attainable fidelity may vary by
model and state geometry.

The absence of a known reconstruction method is not proof of irreversibility.
Future methods may extract information that present methods cannot. No SALT
operator or distributor may describe a state artifact as anonymous, safely
irreversible, or free of source information without evidence that supports
that claim for the exact artifact and threat model.

Research has demonstrated direct input reconstruction from KV caches in tested
model configurations.[1] This establishes a real class of risk, not universal
reversibility or a demonstrated exploit against every SALT package. Numerical
opacity, compression, and a checksum are not encryption.

### 2. KV rows store actual computational information

Committed KV rows contain the actual numerical key/value state used by the
model. They are not merely indexes, references, hashes, or disposable hints.
When SALT exports state losslessly, the persisted bytes preserve those rows
exactly.

Exact preservation of row bytes does not prove that every original token can
always be reconstructed. It does mean that the information represented by the
committed rows has been durably retained rather than replaced by a harmless
label or summary. Opaque encoding is not erasure.

### 3. Honesty rule: model state is not evidence of a user's mental state

A KV/state artifact is model state produced from model weights, model
arithmetic, prompts, documents, system instructions, sampling, and one or more
interactions. It is not a measurement of a person's mind.

Neither the artifact nor any reconstruction, classification, response, or
inference produced from it may be presented as evidence of a user's actual
beliefs, intentions, emotions, memories, diagnosis, honesty, consent,
culpability, capacity, or present mental condition. It must not be used as
clinical, legal, employment, insurance, credit, policing, disciplinary, or
similar evidence about that person.

This rule does not change the status of an independently authenticated source
record. It forbids substituting model state or a model's interpretation of that
state for such a record, and it forbids attributing model-generated content or
automated agent behavior to the person as testimony, admission, or approval.
It states a responsible-use boundary, not a guarantee about what a court or
third party will accept as evidence.

### 4. State must not be preserved without informed user agreement

A runtime may create transient live KV as necessary to perform the active
inference requested by the user. It must not preserve that state beyond the
agreed session boundary without the user's separate, affirmative, informed
agreement.

Agreement must identify, in clear language:

- what state will be retained;
- the purpose and expected benefit;
- duration and deletion policy;
- storage locations, custodians, and recipients;
- whether the state may be exported, replicated, merged, or transferred;
- whether and for what purposes it may be resumed, branched, queried, compared,
  profiled, distilled, or used to produce derived artifacts;
- whether any commercial use is requested;
- how the user can inspect, export, revoke future use, and request deletion;
- limits on deletion after an authorized or unauthorized copy leaves the
  operator's control.

Agreement must not be inferred from use of the software, hidden in the
software license, bundled with unrelated terms, obtained through a preselected
option, or inferred from silence. A new purpose, recipient, collective merge,
or commercial use requires new agreement.

For group-derived state, every affected participant must be identified and the
collective consent rule must be established before preservation. No individual
participant, operator, or custodian may silently convert collective state into
an individually controlled or commercially usable artifact.

Consent to preservation is not consent to every subsequent use. Transient
processing is not a loophole for unauthorized profiling or extraction. A
refusal of optional retention should not be disguised as consent, and a
session boundary or retention period must not be silently extended.

### 5. State must not be treated as a copy of a person

A state artifact may preserve person-linked inputs and may cause a compatible
model to reproduce or approximate prior knowledge, preferences, roles, or
response patterns. That capability creates serious identity and impersonation
risk.

It does not make the artifact the person. A state artifact must not be
represented or treated as the person's consciousness, identity, legal
continuation, testimony, presence, consent, signature, authorization, or
substitute decision-maker. A system executing the artifact must not claim to
be that person. Any authority to act on the person's behalf must come from a
separate, current authorization, never from possession of the artifact.

Terms such as *mental state* in technical documentation refer only to
authenticated native model state. They do not assert consciousness, human
identity, or a verified account of a person's mind.

## Individual rights and collective governance

SALT's policy position is that individually generated, person-linked state
should remain under the individual's control, with ownership of transferable
rights where legally available and clearly agreed control where ownership is
not recognized. Group-generated state should be collectively governed rather
than silently appropriated by an operator or participant.

Commercial exploitation should require prior, specific, recorded permission
from the applicable rights-holders and the individual or collective whose
authorization is required. A collective agreement should name the participants,
steward, approval rule, licensing authority, compensation arrangements, and
procedures for departure and disputes. If unanimous permission is promised,
it must not be replaced by an undisclosed majority rule. Paid hosting of the
authorized user's own session should be distinguished from independently
commercializing that user's state.

Creation, possession, payment for computation, or automated generation does
not by itself settle legal ownership or grant unlimited reuse rights. Artifacts
may include third-party content and information about people who never joined
the session. Individual or collective permission must not be treated as a
waiver of those people's rights or of applicable model and source-material
terms. Group governance must not extinguish individual privacy rights.

These are principles to implement through appropriate agreements and controls,
not a claim that raw tensors always qualify for copyright or that a repository
notice binds every downstream recipient. Employment, compulsory disclosure,
acquisition, insolvency, and cross-border use require jurisdiction-specific
legal review. A warning is not a waiver of responsibility or statutory rights.

## Restoring state is not authorization to act

Restoring a computational trajectory does not grant or renew tool permissions,
financial authority, account access, or user consent. An old plan, pending action,
or model-generated approval must not be treated as present authorization.
Applications must independently check authority when executing an action and
reconcile recorded actions with the current environment to avoid duplicate or
stale effects. KV alone is not proof that an external action occurred.

The inference engine preserves and executes model state. The application or
governed action system remains responsible for whether model output is allowed
to cause an external effect.

## Additional risks that must be disclosed

- **Unauthorized secondary use:** an operator might query or branch state to
  extract confidential information or construct profiles without exporting
  the file. Restricting sale alone does not address this risk.
- **Persistent contamination:** malicious instructions, mistaken assumptions,
  and fabricated conclusions can influence preserved state and its later
  continuation. Integrity checks do not establish truth, safety, or endorsement.
- **Cross-user leakage and correlation:** shared-cache behavior can reveal
  information without direct access to cache bytes; this class of prompt-leakage
  attack has been demonstrated in multi-tenant serving.[2] Metadata and identical
  artifact digests can also expose relationships between copies. These are
  threats to assess, not findings that a particular SALT deployment is vulnerable.
- **Incomplete deletion:** deleting an artifact does not necessarily delete
  backups, recipient copies, generated profiles, distilled state, synthetic
  datasets, or information incorporated into another model. An operator must
  disclose what deletion reaches and must not promise universal erasure.
- **Hidden persistence and runtime exposure:** swap, hibernation, crash dumps,
  diagnostic logs, replicas, and infrastructure snapshots may retain data.
  Encryption at rest or in transit does not by itself protect state while an
  authorized or compromised runtime is processing it.
- **Coercion and lifecycle changes:** employment or other power imbalances can
  undermine meaningful choice. Death, incapacity, a change of custodian, or a
  company acquisition must not be presumed to authorize new uses. Agreements
  should address these events before they occur, subject to applicable law.
- **Outdated or misleading attribution:** a state reflects a particular
  computational history, not a person's current beliefs or consent. Autonomous
  output must not be retroactively presented as something the user said or did.

## Required handling boundary

State covered by this notice should be handled as highly sensitive
confidential data:

- private by default and durably retained only after explicit agreement;
- encrypted in storage and transport, with access limited to authorized
  custodians;
- accompanied by authenticated provenance, controller, consent, purpose,
  retention, and permitted-use metadata;
- never used for model training, profiling, advertising, sale, or unrelated
  analytics without separate explicit authority;
- never shared or merged merely because two artifacts are technically
  compatible;
- deleted from operator-controlled locations when the agreed retention ends,
  subject to clearly disclosed legal and technical limits;
- treated as still sensitive after de-identification unless anonymity has been
  demonstrated for the exact state and realistic attack methods.

Before deriving or combining artifacts, confirm that every source's permitted
purposes and terms allow the operation. Preserve all applicable restrictions;
where terms conflict or cannot be reconciled, do not proceed without resolving
authority and obtaining the required agreement. Technical compatibility is not
legal permission, and derivation does not automatically transfer ownership.

Provenance and consent records should themselves minimize exposed personal
information. A signed manifest can support integrity and attribution within
its defined scope; it cannot prevent copying, establish truthful content, or
prove that consent was informed and voluntary.

## Implementation and enforcement boundary

Every deployment must distinguish **required safeguards**, **implemented
controls**, **tested guarantees**, and **unresolved risks**. This statement does
not certify consent enforcement, encryption, tenant isolation, deletion,
revocation, or downstream compliance in any build or service. Those claims
require separate evidence for the exact deployment.

Software-license permissions, including Apache-2.0 where applicable, are not
amended by this statement. This notice alone does not impose an enforceable
noncommercial condition on software users or every recipient of generated
state. Binding state-use terms require a valid legal basis and appropriate
agreement; technical and legal safeguards both have limits. Operators should
obtain qualified legal review rather than represent this disclosure as a
complete legal agreement.

## Known, credible, and unproven claims

| Class | Boundary |
|---|---|
| Technical fact | Committed KV rows are model-computed numerical state, including effects of generated continuations that have been processed; preserved state supports compatible continuation within its actual coverage. Digital bytes can be copied exactly. |
| Demonstrated research, scoped to tested systems | Input reconstruction from KV and prompt leakage through shared-cache behavior have been demonstrated.[1][2] These are not SALT-specific qualification results. |
| Credible risk requiring assessment | Unauthorized continuation, behavioral approximation, profiling, impersonation, persistent contamination, and derived-information reuse. |
| Not established | State is a human consciousness, a complete copy of a mind or whole agent, a legal person, a truthful measurement of a user's mental condition, or universally reversible to exact source text. |

Safety claims must preserve these distinctions. Unknown capability is not a
basis for claiming safety, and serious risk is not a basis for claiming
personhood or verified access to another person's mind.

## Sources

[1] https://www.ndss-symposium.org/wp-content/uploads/2026-f258-paper.pdf — Shadow in the Cache — NDSS 2026
[2] https://www.ndss-symposium.org/wp-content/uploads/2025-1772-paper.pdf — I Know What You Asked — NDSS 2025
