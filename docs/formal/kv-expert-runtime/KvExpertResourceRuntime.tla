---- MODULE KvExpertResourceRuntime ----
EXTENDS Integers, Naturals, FiniteSets

CONSTANTS
    NumLayers,
    ExpertsPerLayer,
    TopK,
    MaxSelectedExperts,
    ExpertWeightBytes,
    SlotBytes,
    Capacity,
    ResourceBudgetBytes,
    MaxTokens

NumExperts == NumLayers * ExpertsPerLayer
ImmutableExpertBytes == NumExperts * ExpertWeightBytes

ASSUME ModelBounds ==
    /\ NumLayers \in Nat \ {0}
    /\ ExpertsPerLayer \in Nat \ {0}
    /\ TopK \in 1..ExpertsPerLayer
    /\ MaxSelectedExperts \in TopK..ExpertsPerLayer
    /\ ExpertWeightBytes \in Nat \ {0}
    /\ SlotBytes \in Nat \ {0}
    /\ ExpertWeightBytes <= SlotBytes
    /\ Capacity \in Nat \ {0}
    /\ MaxSelectedExperts <= Capacity
    /\ Capacity <= ExpertsPerLayer
    /\ ResourceBudgetBytes \in Nat \ {0}
    /\ Capacity * SlotBytes <= ResourceBudgetBytes
    /\ MaxTokens \in Nat \ {0}

Experts == 1..NumExperts
Phases == {"READY", "RESERVE", "LOAD", "LEASE", "EXECUTE", "RELEASE",
           "COMMIT", "DONE", "FAILED", "CLOSED"}

ExpertFold(input, selectedExperts) ==
    (input * 3 + Cardinality(selectedExperts)) % 7

VARIABLES
    phase,
    selected,
    ready,
    loading,
    leased,
    pinned,
    bound,
    position,
    operation,
    inputDigest,
    runtimeResult,
    canonicalResult,
    kvDigest,
    canonicalKv,
    resourcesAlive,
    resourceObjects,
    objectCreates,
    objectDestroys,
    bindCount,
    publishCount,
    fenceCount,
    unbindCount

vars == <<phase, selected, ready, loading, leased, pinned, bound,
          position, operation, inputDigest, runtimeResult, canonicalResult,
          kvDigest, canonicalKv, resourcesAlive, resourceObjects,
          objectCreates, objectDestroys, bindCount, publishCount,
          fenceCount, unbindCount>>

Init ==
    /\ phase = "READY"
    /\ selected = {}
    /\ ready = {}
    /\ loading = {}
    /\ leased = {}
    /\ pinned = {}
    /\ bound = {}
    /\ position = 0
    /\ operation = 0
    /\ inputDigest = 0
    /\ runtimeResult = 0
    /\ canonicalResult = 0
    /\ kvDigest = 0
    /\ canonicalKv = 0
    /\ resourcesAlive = TRUE
    /\ resourceObjects = Capacity
    /\ objectCreates = Capacity
    /\ objectDestroys = 0
    /\ bindCount = 0
    /\ publishCount = 0
    /\ fenceCount = 0
    /\ unbindCount = 0

BeginOperation(sel, digest) ==
    /\ phase = "READY"
    /\ position < MaxTokens
    /\ sel \subseteq Experts
    /\ sel # {}
    /\ Cardinality(sel) <= MaxSelectedExperts
    /\ Cardinality(ready \cup sel) <= Capacity
    /\ digest \in 0..2
    /\ phase' = "RESERVE"
    /\ selected' = sel
    /\ operation' = operation + 1
    /\ inputDigest' = digest
    /\ runtimeResult' = 0
    /\ canonicalResult' = 0
    /\ UNCHANGED <<ready, loading, leased, pinned, bound, position,
                    kvDigest, canonicalKv, resourcesAlive, resourceObjects,
                    objectCreates, objectDestroys, bindCount, publishCount,
                    fenceCount, unbindCount>>

ReserveMissing ==
    /\ phase = "RESERVE"
    /\ loading' = selected \ ready
    /\ phase' = IF selected \subseteq ready THEN "LEASE" ELSE "LOAD"
    /\ UNCHANGED <<selected, ready, leased, pinned, bound, position,
                    operation, inputDigest, runtimeResult, canonicalResult,
                    kvDigest, canonicalKv, resourcesAlive, resourceObjects,
                    objectCreates, objectDestroys, bindCount, publishCount,
                    fenceCount, unbindCount>>

LoadBindPublish ==
    /\ phase = "LOAD"
    /\ loading # {}
    /\ loading \cap ready = {}
    /\ Cardinality(ready \cup loading) <= Capacity
    /\ ready' = ready \cup loading
    /\ bound' = bound \cup loading
    /\ bindCount' = bindCount + Cardinality(loading)
    /\ publishCount' = publishCount + Cardinality(loading)
    /\ loading' = {}
    /\ phase' = "LEASE"
    /\ UNCHANGED <<selected, leased, pinned, position, operation,
                    inputDigest, runtimeResult, canonicalResult,
                    kvDigest, canonicalKv, resourcesAlive, resourceObjects,
                    objectCreates, objectDestroys, fenceCount, unbindCount>>

AcquireLeases ==
    /\ phase = "LEASE"
    /\ selected \subseteq ready
    /\ leased' = selected
    /\ phase' = "EXECUTE"
    /\ UNCHANGED <<selected, ready, loading, pinned, bound, position,
                    operation, inputDigest, runtimeResult, canonicalResult,
                    kvDigest, canonicalKv, resourcesAlive, resourceObjects,
                    objectCreates, objectDestroys, bindCount, publishCount,
                    fenceCount, unbindCount>>

ExecuteCanonical ==
    /\ phase = "EXECUTE"
    /\ leased = selected
    /\ selected # {}
    /\ runtimeResult' = ExpertFold(inputDigest, selected)
    /\ canonicalResult' = ExpertFold(inputDigest, selected)
    /\ phase' = "RELEASE"
    /\ UNCHANGED <<selected, ready, loading, leased, pinned, bound, position,
                    operation, inputDigest, kvDigest, canonicalKv,
                    resourcesAlive, resourceObjects, objectCreates,
                    objectDestroys, bindCount, publishCount, fenceCount,
                    unbindCount>>

ReleaseLeases ==
    /\ phase = "RELEASE"
    /\ leased = selected
    /\ leased' = {}
    /\ phase' = "COMMIT"
    /\ UNCHANGED <<selected, ready, loading, pinned, bound, position,
                    operation, inputDigest, runtimeResult, canonicalResult,
                    kvDigest, canonicalKv, resourcesAlive, resourceObjects,
                    objectCreates, objectDestroys, bindCount, publishCount,
                    fenceCount, unbindCount>>

CommitKv ==
    /\ phase = "COMMIT"
    /\ leased = {}
    /\ runtimeResult = canonicalResult
    /\ position' = position + 1
    /\ kvDigest' = runtimeResult
    /\ canonicalKv' = canonicalResult
    /\ selected' = {}
    /\ inputDigest' = 0
    /\ phase' = IF position' = MaxTokens THEN "DONE" ELSE "READY"
    /\ UNCHANGED <<ready, loading, leased, pinned, bound, operation,
                    runtimeResult, canonicalResult, resourcesAlive,
                    resourceObjects, objectCreates, objectDestroys,
                    bindCount, publishCount, fenceCount, unbindCount>>

EvictBatch(victims) ==
    /\ phase = "READY"
    /\ victims \subseteq ready
    /\ victims # {}
    /\ victims \cap pinned = {}
    /\ victims \cap leased = {}
    /\ Cardinality(ready \ victims) <= Capacity
    /\ ready' = ready \ victims
    /\ bound' = bound \ victims
    /\ pinned' = pinned \ victims
    /\ fenceCount' = fenceCount + 1
    /\ unbindCount' = unbindCount + Cardinality(victims)
    /\ UNCHANGED <<phase, selected, loading, leased, position, operation,
                    inputDigest, runtimeResult, canonicalResult, kvDigest,
                    canonicalKv, resourcesAlive, resourceObjects,
                    objectCreates, objectDestroys, bindCount, publishCount>>

Pin(expert) ==
    /\ phase = "READY"
    /\ expert \in ready
    /\ pinned' = pinned \cup {expert}
    /\ UNCHANGED <<phase, selected, ready, loading, leased, bound, position,
                    operation, inputDigest, runtimeResult, canonicalResult,
                    kvDigest, canonicalKv, resourcesAlive, resourceObjects,
                    objectCreates, objectDestroys, bindCount, publishCount,
                    fenceCount, unbindCount>>

Unpin(expert) ==
    /\ phase = "READY"
    /\ expert \in pinned
    /\ pinned' = pinned \ {expert}
    /\ UNCHANGED <<phase, selected, ready, loading, leased, bound, position,
                    operation, inputDigest, runtimeResult, canonicalResult,
                    kvDigest, canonicalKv, resourcesAlive, resourceObjects,
                    objectCreates, objectDestroys, bindCount, publishCount,
                    fenceCount, unbindCount>>

Fail ==
    /\ phase \in {"RESERVE", "LOAD", "LEASE", "EXECUTE", "RELEASE", "COMMIT"}
    /\ phase' = "FAILED"
    /\ selected' = {}
    /\ loading' = {}
    /\ leased' = {}
    /\ UNCHANGED <<ready, pinned, bound, position, operation, inputDigest,
                    runtimeResult, canonicalResult, kvDigest, canonicalKv,
                    resourcesAlive, resourceObjects, objectCreates,
                    objectDestroys, bindCount, publishCount, fenceCount,
                    unbindCount>>

Shutdown ==
    /\ phase \in {"DONE", "FAILED"}
    /\ leased = {}
    /\ loading = {}
    /\ phase' = "CLOSED"
    /\ selected' = {}
    /\ loading' = {}
    /\ leased' = {}
    /\ ready' = {}
    /\ pinned' = {}
    /\ bound' = {}
    /\ resourcesAlive' = FALSE
    /\ resourceObjects' = 0
    /\ objectDestroys' = Capacity
    /\ fenceCount' = fenceCount + IF bound = {} THEN 0 ELSE 1
    /\ unbindCount' = unbindCount + Cardinality(bound)
    /\ UNCHANGED <<position, operation, inputDigest, runtimeResult,
                    canonicalResult, kvDigest, canonicalKv, objectCreates,
                    bindCount, publishCount>>

TerminalStutter ==
    /\ phase = "CLOSED"
    /\ UNCHANGED vars

Next ==
    \/ \E sel \in SUBSET Experts : \E digest \in 0..2 :
         BeginOperation(sel, digest)
    \/ ReserveMissing
    \/ LoadBindPublish
    \/ AcquireLeases
    \/ ExecuteCanonical
    \/ ReleaseLeases
    \/ CommitKv
    \/ \E victims \in SUBSET Experts : EvictBatch(victims)
    \/ \E expert \in Experts : Pin(expert)
    \/ \E expert \in Experts : Unpin(expert)
    \/ Fail
    \/ Shutdown
    \/ TerminalStutter

Spec == Init /\ [][Next]_vars

TypeOK ==
    /\ phase \in Phases
    /\ selected \subseteq Experts
    /\ ready \subseteq Experts
    /\ loading \subseteq Experts
    /\ leased \subseteq Experts
    /\ pinned \subseteq Experts
    /\ bound \subseteq Experts
    /\ position \in 0..MaxTokens
    /\ operation \in Nat
    /\ inputDigest \in 0..2
    /\ runtimeResult \in 0..6
    /\ canonicalResult \in 0..6
    /\ kvDigest \in 0..6
    /\ canonicalKv \in 0..6
    /\ resourcesAlive \in BOOLEAN
    /\ resourceObjects \in 0..Capacity
    /\ objectCreates \in 0..Capacity
    /\ objectDestroys \in 0..Capacity
    /\ bindCount \in Nat
    /\ publishCount \in Nat
    /\ fenceCount \in Nat
    /\ unbindCount \in Nat

CapacityBound ==
    /\ Cardinality(ready) <= Capacity
    /\ pinned \subseteq ready

ResidentBytes == Cardinality(ready) * SlotBytes
ResidentByteBound == ResidentBytes <= ResourceBudgetBytes

LoadingSafety ==
    /\ loading \subseteq selected
    /\ loading \cap ready = {}
    /\ (phase # "LOAD" => loading = {})

BoundBeforeReady == bound = ready

LeaseSafety ==
    /\ leased \subseteq bound
    /\ leased \subseteq selected
    /\ (phase \in {"RESERVE", "LOAD", "LEASE", "EXECUTE", "RELEASE",
                    "COMMIT"} => selected # {})
    /\ (phase \in {"EXECUTE", "RELEASE"} =>
           leased = selected /\ selected # {})
    /\ (phase \in {"READY", "RESERVE", "LOAD", "LEASE", "COMMIT",
                    "DONE", "FAILED", "CLOSED"} => leased = {})

ResultExact == runtimeResult = canonicalResult
KvExact == kvDigest = canonicalKv

NoPerOperationResourceCreation ==
    /\ objectCreates = Capacity
    /\ (resourcesAlive =>
           resourceObjects = Capacity /\ objectDestroys = 0)
    /\ (~resourcesAlive =>
           resourceObjects = 0 /\ objectDestroys = Capacity)

RetirementAccounting ==
    /\ unbindCount <= bindCount
    /\ publishCount = bindCount
    /\ fenceCount <= unbindCount

ResourceLifetime == resourcesAlive <=> phase # "CLOSED"

KvExpertResourceInvariants ==
    /\ ModelBounds
    /\ TypeOK
    /\ CapacityBound
    /\ ResidentByteBound
    /\ LoadingSafety
    /\ BoundBeforeReady
    /\ LeaseSafety
    /\ ResultExact
    /\ KvExact
    /\ NoPerOperationResourceCreation
    /\ RetirementAccounting
    /\ ResourceLifetime

====
