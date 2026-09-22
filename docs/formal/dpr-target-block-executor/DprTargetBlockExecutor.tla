---- MODULE DprTargetBlockExecutor ----
EXTENDS Integers, Naturals, FiniteSets, TLC

CONSTANTS
    MaxLayers,
    MaxHorizon,
    MaxResources,
    ExpertsPerLayer,
    TopK,
    ExpertWeightBytes,
    ResourceSlotBytes,
    ResourceBudgetBytes,
    HiddenWidth,
    ScalarBytes,
    MaxToken,
    StopToken

ImmutableExpertBytes == MaxLayers * ExpertsPerLayer * ExpertWeightBytes

ASSUME ModelBounds ==
    /\ MaxLayers \in Nat \ {0}
    /\ MaxHorizon \in Nat \ {0}
    /\ MaxResources \in Nat \ {0}
    /\ ExpertsPerLayer \in Nat \ {0}
    /\ TopK \in 1..ExpertsPerLayer
    /\ ExpertWeightBytes \in Nat \ {0}
    /\ ResourceSlotBytes \in Nat \ {0}
    /\ ExpertWeightBytes <= ResourceSlotBytes
    /\ TopK <= MaxResources
    /\ ResourceBudgetBytes \in Nat \ {0}
    /\ MaxResources * ResourceSlotBytes <= ResourceBudgetBytes
    /\ HiddenWidth \in Nat \ {0}
    /\ ScalarBytes \in Nat \ {0}
    /\ MaxToken \in Nat \ {0}
    /\ StopToken \in 1..MaxToken

VARIABLE st

vars == <<st>>

Layers == 1..MaxLayers
Rows == 1..MaxHorizon
Slots == 1..MaxResources
TokenValues == 1..MaxToken

ExecutorModes == {"CPU_ONLY", "GPU_ONLY"}
Phases == {
    "READY", "EMBED", "ATTENTION", "RESERVE", "LOAD", "LEASE",
    "FFN", "RELEASE", "HEAD", "VALIDATE", "PENDING", "DONE", "FAILED"
}
ComputePhases == {
    "EMBED", "ATTENTION", "RESERVE", "LOAD", "LEASE",
    "FFN", "RELEASE", "HEAD", "VALIDATE"
}
KvStates == {"EMPTY", "TENTATIVE", "COMMITTED", "SCRUBBED"}
SlotStates == {"EMPTY", "READY"}
ConsumeKinds == {"NONE", "NEXT_CYCLE", "FINAL_COMMIT"}

DigestAdvance(digest, amount) == (digest * 11 + amount) % 103
DigestToken(digest, token) == (digest * 13 + token) % 103

EmbedDestination(row) == row
AttentionDestination(layer, row) ==
    MaxHorizon + (layer - 1) * MaxHorizon + row
FfnDestination(layer, row) ==
    MaxHorizon + MaxLayers * MaxHorizon +
    (layer - 1) * MaxHorizon + row
HeadDestination(row) ==
    MaxHorizon + 2 * MaxLayers * MaxHorizon + row

EmbedDestinations(horizon) ==
    {EmbedDestination(row) : row \in 1..horizon}
AttentionDestinations(layer, horizon) ==
    {AttentionDestination(layer, row) : row \in 1..horizon}
FfnDestinations(layer, horizon) ==
    {FfnDestination(layer, row) : row \in 1..horizon}
HeadDestinations(horizon) ==
    {HeadDestination(row) : row \in 1..horizon}

AllDestinations ==
    {EmbedDestination(row) : row \in Rows}
    \cup {AttentionDestination(layer, row) :
            layer \in Layers, row \in Rows}
    \cup {FfnDestination(layer, row) :
            layer \in Layers, row \in Rows}
    \cup {HeadDestination(row) : row \in Rows}
ExpectedDestinationCount == (2 + 2 * MaxLayers) * MaxHorizon

AttentionPc(layer) == 2 + (layer - 1) * 6
ReservePc(layer) == AttentionPc(layer) + 1
LoadPc(layer) == AttentionPc(layer) + 2
LeasePc(layer) == AttentionPc(layer) + 3
FfnPc(layer) == AttentionPc(layer) + 4
ReleasePc(layer) == AttentionPc(layer) + 5
HeadPc == 2 + MaxLayers * 6
ValidatePc == HeadPc + 1
PendingPc == ValidatePc + 1
DonePc == PendingPc + 1
FailedPc == DonePc + 1

Init ==
    \E mode \in ExecutorModes :
    st = [
        phase |-> "READY",
        compiledExecutorMode |-> mode,
        executorMode |-> mode,
        activeHorizon |-> 0,
        verifyResult |-> 0,
        pendingChoice |-> 0,
        pendingValid |-> FALSE,
        pendingToken |-> 0,
        consumeKind |-> "NONE",
        currentLayer |-> 0,
        programCounter |-> 0,
        programEpoch |-> 1,
        blockStart |-> 0,
        runtimePosition |-> 0,
        canonicalPosition |-> 0,
        runtimeDigest |-> 0,
        canonicalDigest |-> 0,
        kvState |-> [layer \in Layers |->
            [row \in Rows |-> "EMPTY"]],
        slotState |-> [slot \in Slots |-> "EMPTY"],
        slotKey |-> [slot \in Slots |-> 0],
        leaseOwner |-> [slot \in Slots |-> 0],
        reservedSlot |-> 0,
        writtenDestinations |-> {},
        writeCount |-> 0,
        submissions |-> 0,
        completions |-> 0,
        producedCount |-> 0,
        committedCount |-> 0,
        verificationSteps |-> 0,
        pendingProduced |-> 0,
        pendingConsumed |-> 0,
        mapCount |-> 0,
        unmapCount |-> 0,
        resourceObjects |-> MaxResources,
        resourceCreates |-> 0,
        resourceDestroys |-> 0,
        intermediatePublications |-> 0,
        failureCleanups |-> 0
    ]

BeginBlock(horizon, accepted, pending) ==
    /\ st.phase = "READY"
    /\ st.submissions = 0
    /\ horizon \in Rows
    /\ accepted \in 0..horizon
    /\ pending \in TokenValues
    /\ st' = [st EXCEPT
        !.phase = IF accepted = 0 THEN "PENDING" ELSE "EMBED",
        !.activeHorizon = horizon,
        !.verifyResult = accepted,
        !.pendingChoice = pending,
        !.pendingValid = (accepted = 0),
        !.pendingToken = IF accepted = 0 THEN pending ELSE 0,
        !.consumeKind = "NONE",
        !.currentLayer = 0,
        !.programCounter = IF accepted = 0 THEN PendingPc ELSE 1,
        !.blockStart = st.runtimePosition,
        !.kvState = [layer \in Layers |->
            [row \in Rows |-> "EMPTY"]],
        !.writtenDestinations = {},
        !.writeCount = 0,
        !.submissions = @ + (IF accepted = 0 THEN 0 ELSE 1),
        !.producedCount = IF accepted = 0 THEN 1 ELSE 0,
        !.committedCount = 0,
        !.verificationSteps = 0,
        !.pendingProduced = @ + (IF accepted = 0 THEN 1 ELSE 0)]

EmbedRows ==
    /\ st.phase = "EMBED"
    /\ st' = [st EXCEPT
        !.phase = "ATTENTION",
        !.currentLayer = 1,
        !.programCounter = AttentionPc(1),
        !.writtenDestinations =
            @ \cup EmbedDestinations(st.activeHorizon),
        !.writeCount = @ + st.activeHorizon]

RunAttention ==
    /\ st.phase = "ATTENTION"
    /\ st.currentLayer \in Layers
    /\ st' = [st EXCEPT
        !.phase = "RESERVE",
        !.programCounter = ReservePc(st.currentLayer),
        !.kvState = [@ EXCEPT
            ![st.currentLayer] =
                [row \in Rows |->
                    IF row <= st.activeHorizon
                    THEN "TENTATIVE"
                    ELSE st.kvState[st.currentLayer][row]]],
        !.writtenDestinations = @ \cup
            AttentionDestinations(st.currentLayer, st.activeHorizon),
        !.writeCount = @ + st.activeHorizon]

ReserveExpert(slot) ==
    /\ st.phase = "RESERVE"
    /\ slot \in Slots
    /\ st.leaseOwner[slot] = 0
    /\ st' = [st EXCEPT
        !.phase = "LOAD",
        !.programCounter = LoadPc(st.currentLayer),
        !.reservedSlot = slot]

LoadExpert ==
    /\ st.phase = "LOAD"
    /\ st.reservedSlot \in Slots
    /\ st.leaseOwner[st.reservedSlot] = 0
    /\ st' = [st EXCEPT
        !.phase = "LEASE",
        !.programCounter = LeasePc(st.currentLayer),
        !.slotState = [@ EXCEPT ![st.reservedSlot] = "READY"],
        !.slotKey = [@ EXCEPT ![st.reservedSlot] = st.currentLayer],
        !.mapCount = @ + 1,
        !.unmapCount = @ +
            IF st.slotState[st.reservedSlot] = "READY" THEN 1 ELSE 0]

AcquireLease ==
    /\ st.phase = "LEASE"
    /\ st.reservedSlot \in Slots
    /\ st.slotState[st.reservedSlot] = "READY"
    /\ st.slotKey[st.reservedSlot] = st.currentLayer
    /\ st.leaseOwner[st.reservedSlot] = 0
    /\ st' = [st EXCEPT
        !.phase = "FFN",
        !.programCounter = FfnPc(st.currentLayer),
        !.leaseOwner = [@ EXCEPT
            ![st.reservedSlot] = st.currentLayer]]

RunFfn ==
    /\ st.phase = "FFN"
    /\ st.reservedSlot \in Slots
    /\ st.leaseOwner[st.reservedSlot] = st.currentLayer
    /\ st.slotState[st.reservedSlot] = "READY"
    /\ st.slotKey[st.reservedSlot] = st.currentLayer
    /\ st' = [st EXCEPT
        !.phase = "RELEASE",
        !.programCounter = ReleasePc(st.currentLayer),
        !.writtenDestinations = @ \cup
            FfnDestinations(st.currentLayer, st.activeHorizon),
        !.writeCount = @ + st.activeHorizon]

ReleaseExpert ==
    /\ st.phase = "RELEASE"
    /\ st.reservedSlot \in Slots
    /\ st.leaseOwner[st.reservedSlot] = st.currentLayer
    /\ st' = [st EXCEPT
        !.phase = IF st.currentLayer < MaxLayers
            THEN "ATTENTION" ELSE "HEAD",
        !.programCounter = IF st.currentLayer < MaxLayers
            THEN AttentionPc(st.currentLayer + 1) ELSE HeadPc,
        !.currentLayer = IF st.currentLayer < MaxLayers
            THEN st.currentLayer + 1 ELSE 0,
        !.leaseOwner = [@ EXCEPT ![st.reservedSlot] = 0],
        !.reservedSlot = 0]

RunHead ==
    /\ st.phase = "HEAD"
    /\ st' = [st EXCEPT
        !.phase = "VALIDATE",
        !.programCounter = ValidatePc,
        !.writtenDestinations =
            @ \cup HeadDestinations(st.activeHorizon),
        !.writeCount = @ + st.activeHorizon]

CompleteVerification ==
    /\ st.phase = "VALIDATE"
    /\ st.verifyResult > 0
    /\ \A layer \in Layers, row \in 1..st.activeHorizon :
        st.kvState[layer][row] = "TENTATIVE"
    /\ st' = [st EXCEPT
        !.phase = "PENDING",
        !.programCounter = PendingPc,
        !.kvState = [layer \in Layers |->
            [row \in Rows |->
                IF row <= st.verifyResult THEN "COMMITTED"
                ELSE IF row <= st.activeHorizon THEN "SCRUBBED"
                ELSE "EMPTY"]],
        !.runtimePosition = st.blockStart + st.verifyResult,
        !.canonicalPosition = st.blockStart + st.verifyResult,
        !.runtimeDigest = DigestAdvance(
            st.runtimeDigest, st.verifyResult),
        !.canonicalDigest = DigestAdvance(
            st.canonicalDigest, st.verifyResult),
        !.pendingValid = TRUE,
        !.pendingToken = st.pendingChoice,
        !.producedCount = st.verifyResult + 1,
        !.committedCount = st.verifyResult,
        !.verificationSteps = st.activeHorizon,
        !.pendingProduced = @ + 1,
        !.completions = @ + 1]

ConsumePending(finalCommit) ==
    /\ finalCommit \in BOOLEAN
    /\ st.phase = "PENDING"
    /\ st.pendingValid
    /\ st.pendingToken \in TokenValues
    /\ (finalCommit = (st.pendingToken = StopToken))
    /\ st' = [st EXCEPT
        !.phase = "DONE",
        !.programCounter = DonePc,
        !.runtimePosition = @ + 1,
        !.canonicalPosition = @ + 1,
        !.runtimeDigest = DigestToken(@, st.pendingToken),
        !.canonicalDigest = DigestToken(@, st.pendingToken),
        !.pendingValid = FALSE,
        !.pendingToken = 0,
        !.consumeKind = IF finalCommit
            THEN "FINAL_COMMIT" ELSE "NEXT_CYCLE",
        !.committedCount = @ + 1,
        !.verificationSteps = @ + 1,
        !.pendingConsumed = @ + 1]

FailExecution ==
    /\ st.phase \in ComputePhases
    /\ st' = [st EXCEPT
        !.phase = "FAILED",
        !.programCounter = FailedPc,
        !.currentLayer = 0,
        !.reservedSlot = 0,
        !.leaseOwner = [slot \in Slots |-> 0],
        !.kvState = [layer \in Layers |->
            [row \in Rows |->
                IF st.kvState[layer][row] = "TENTATIVE"
                THEN "SCRUBBED"
                ELSE st.kvState[layer][row]]],
        !.completions = @ + 1,
        !.failureCleanups = @ + 1]

TerminalStutter ==
    /\ st.phase \in {"DONE", "FAILED"}
    /\ UNCHANGED st

Next ==
    \/ \E horizon \in Rows :
        \E accepted \in 0..horizon :
            \E pending \in TokenValues :
                BeginBlock(horizon, accepted, pending)
    \/ EmbedRows
    \/ RunAttention
    \/ \E slot \in Slots : ReserveExpert(slot)
    \/ LoadExpert
    \/ AcquireLease
    \/ RunFfn
    \/ ReleaseExpert
    \/ RunHead
    \/ CompleteVerification
    \/ \E finalCommit \in BOOLEAN : ConsumePending(finalCommit)
    \/ FailExecution
    \/ TerminalStutter

Spec == Init /\ [][Next]_vars

TypeOK ==
    /\ st.phase \in Phases
    /\ st.compiledExecutorMode \in ExecutorModes
    /\ st.executorMode \in ExecutorModes
    /\ st.activeHorizon \in 0..MaxHorizon
    /\ st.verifyResult \in 0..st.activeHorizon
    /\ st.pendingChoice \in 0..MaxToken
    /\ st.pendingValid \in BOOLEAN
    /\ st.pendingToken \in 0..MaxToken
    /\ st.consumeKind \in ConsumeKinds
    /\ st.currentLayer \in 0..MaxLayers
    /\ st.programCounter \in 0..FailedPc
    /\ st.programEpoch = 1
    /\ st.blockStart \in 0..(MaxHorizon + 1)
    /\ st.runtimePosition \in 0..(MaxHorizon + 1)
    /\ st.canonicalPosition \in 0..(MaxHorizon + 1)
    /\ st.runtimeDigest \in 0..102
    /\ st.canonicalDigest \in 0..102
    /\ st.kvState \in [Layers -> [Rows -> KvStates]]
    /\ st.slotState \in [Slots -> SlotStates]
    /\ st.slotKey \in [Slots -> 0..MaxLayers]
    /\ st.leaseOwner \in [Slots -> 0..MaxLayers]
    /\ st.reservedSlot \in 0..MaxResources
    /\ st.writtenDestinations \subseteq AllDestinations
    /\ st.writeCount \in Nat
    /\ st.submissions \in 0..1
    /\ st.completions \in 0..1
    /\ st.producedCount \in 0..(MaxHorizon + 1)
    /\ st.committedCount \in 0..(MaxHorizon + 1)
    /\ st.verificationSteps \in 0..(MaxHorizon + 1)
    /\ st.pendingProduced \in 0..1
    /\ st.pendingConsumed \in 0..1
    /\ st.mapCount \in Nat
    /\ st.unmapCount \in Nat
    /\ st.resourceObjects = MaxResources
    /\ st.resourceCreates \in Nat
    /\ st.resourceDestroys \in Nat
    /\ st.intermediatePublications \in Nat
    /\ st.failureCleanups \in 0..1

ResidentResourceBytes ==
    Cardinality({slot \in Slots : st.slotState[slot] = "READY"}) *
        ResourceSlotBytes

ResourceByteBound == ResidentResourceBytes <= ResourceBudgetBytes

ProgramOrderExact ==
    CASE st.phase = "READY" ->
            st.programCounter = 0 /\ st.currentLayer = 0
      [] st.phase = "EMBED" ->
            st.programCounter = 1 /\ st.currentLayer = 0
      [] st.phase = "ATTENTION" ->
            st.currentLayer \in Layers /\
            st.programCounter = AttentionPc(st.currentLayer)
      [] st.phase = "RESERVE" ->
            st.currentLayer \in Layers /\
            st.programCounter = ReservePc(st.currentLayer)
      [] st.phase = "LOAD" ->
            st.currentLayer \in Layers /\
            st.programCounter = LoadPc(st.currentLayer)
      [] st.phase = "LEASE" ->
            st.currentLayer \in Layers /\
            st.programCounter = LeasePc(st.currentLayer)
      [] st.phase = "FFN" ->
            st.currentLayer \in Layers /\
            st.programCounter = FfnPc(st.currentLayer)
      [] st.phase = "RELEASE" ->
            st.currentLayer \in Layers /\
            st.programCounter = ReleasePc(st.currentLayer)
      [] st.phase = "HEAD" ->
            st.programCounter = HeadPc /\ st.currentLayer = 0
      [] st.phase = "VALIDATE" ->
            st.programCounter = ValidatePc /\ st.currentLayer = 0
      [] st.phase = "PENDING" ->
            st.programCounter = PendingPc /\ st.currentLayer = 0
      [] st.phase = "DONE" ->
            st.programCounter = DonePc /\ st.currentLayer = 0
      [] st.phase = "FAILED" ->
            st.programCounter = FailedPc /\ st.currentLayer = 0
      [] OTHER -> FALSE

ProgramImmutable == st.programEpoch = 1

ExecutorPolicyImmutable ==
    st.executorMode = st.compiledExecutorMode

CanonicalDestinationsDisjoint ==
    /\ st.programEpoch = 1
    /\ Cardinality(AllDestinations) = ExpectedDestinationCount

NoDestinationRewrite ==
    st.writeCount = Cardinality(st.writtenDestinations)

NoIntermediatePublication == st.intermediatePublications = 0

TentativeKvIsolated ==
    /\ (~(st.phase \in {"PENDING", "DONE"}) =>
        \A layer \in Layers, row \in Rows :
            st.kvState[layer][row] # "COMMITTED")
    /\ ((st.phase \in ComputePhases) \/ st.phase = "FAILED" =>
        st.runtimePosition = st.blockStart)

AcceptedPrefixOnly ==
    st.phase \in {"PENDING", "DONE"} /\ st.verifyResult > 0 =>
        \A layer \in Layers, row \in Rows :
            st.kvState[layer][row] =
                IF row <= st.verifyResult THEN "COMMITTED"
                ELSE IF row <= st.activeHorizon THEN "SCRUBBED"
                ELSE "EMPTY"

RejectedSuffixScrubbed ==
    st.phase \in {"PENDING", "DONE"} /\ st.verifyResult > 0 =>
        \A layer \in Layers, row \in Rows :
            (row > st.verifyResult /\ row <= st.activeHorizon) =>
                st.kvState[layer][row] = "SCRUBBED"

PendingNotStepped ==
    /\ (st.pendingValid <=> st.phase = "PENDING")
    /\ (st.phase = "PENDING" =>
        /\ st.pendingToken = st.pendingChoice
        /\ st.pendingToken \in TokenValues
        /\ st.verificationSteps =
            IF st.verifyResult = 0 THEN 0 ELSE st.activeHorizon
        /\ st.pendingConsumed < st.pendingProduced)
    /\ (st.phase # "PENDING" => st.pendingToken = 0)

PendingConsumedOnce ==
    /\ st.pendingProduced = st.pendingConsumed +
        IF st.pendingValid THEN 1 ELSE 0
    /\ (st.phase = "DONE" =>
        /\ st.pendingProduced = 1
        /\ st.pendingConsumed = 1
        /\ st.consumeKind = IF st.pendingChoice = StopToken
            THEN "FINAL_COMMIT" ELSE "NEXT_CYCLE")

RuntimeEqualsCanonical ==
    /\ st.runtimePosition = st.canonicalPosition
    /\ st.runtimeDigest = st.canonicalDigest

LeaseOutlivesConsumer ==
    /\ \A slot \in Slots :
        st.leaseOwner[slot] # 0 =>
            /\ st.slotState[slot] = "READY"
            /\ st.slotKey[slot] = st.leaseOwner[slot]
            /\ st.phase \in {"FFN", "RELEASE"}
            /\ st.reservedSlot = slot
    /\ (st.phase \in {"FFN", "RELEASE"} =>
        /\ st.reservedSlot \in Slots
        /\ st.leaseOwner[st.reservedSlot] = st.currentLayer)
    /\ (~(st.phase \in {"FFN", "RELEASE"}) =>
        \A slot \in Slots : st.leaseOwner[slot] = 0)

BoundedResidency ==
    /\ st.unmapCount <= st.mapCount
    /\ st.mapCount - st.unmapCount =
        Cardinality({slot \in Slots : st.slotState[slot] = "READY"})
    /\ \A slot \in Slots :
        /\ (st.slotState[slot] = "EMPTY" <=> st.slotKey[slot] = 0)
        /\ (st.slotState[slot] = "READY" => st.slotKey[slot] \in Layers)

NoProcessResourceCreation ==
    /\ st.resourceObjects = MaxResources
    /\ st.resourceCreates = 0
    /\ st.resourceDestroys = 0

OneSubmissionOneCompletion ==
    /\ (st.phase \in ComputePhases =>
        st.submissions = st.completions + 1)
    /\ (~(st.phase \in ComputePhases) =>
        st.submissions = st.completions)

ZeroAcceptedSkipsExecutor ==
    st.phase \in {"PENDING", "DONE"} /\ st.verifyResult = 0 =>
        /\ st.submissions = 0
        /\ st.completions = 0
        /\ st.writeCount = 0

ZeroAcceptedKvUntouched ==
    st.phase \in {"PENDING", "DONE"} /\ st.verifyResult = 0 =>
        /\ st.writtenDestinations = {}
        /\ \A layer \in Layers, row \in Rows :
            st.kvState[layer][row] = "EMPTY"

ExecutorRefinesDprTarget ==
    /\ ((st.phase = "READY" \/ st.phase \in ComputePhases \/
         st.phase = "FAILED") =>
        /\ st.runtimePosition = st.blockStart
        /\ st.producedCount = 0
        /\ st.committedCount = 0
        /\ st.verificationSteps = 0)
    /\ (st.phase = "PENDING" =>
        /\ st.runtimePosition = st.blockStart + st.verifyResult
        /\ st.producedCount = st.verifyResult + 1
        /\ st.committedCount = st.verifyResult
        /\ st.verificationSteps =
            IF st.verifyResult = 0 THEN 0 ELSE st.activeHorizon)
    /\ (st.phase = "DONE" =>
        /\ st.runtimePosition = st.blockStart + st.verifyResult + 1
        /\ st.producedCount = st.verifyResult + 1
        /\ st.committedCount = st.verifyResult + 1
        /\ st.verificationSteps =
            IF st.verifyResult = 0 THEN 1 ELSE st.activeHorizon + 1)

====
