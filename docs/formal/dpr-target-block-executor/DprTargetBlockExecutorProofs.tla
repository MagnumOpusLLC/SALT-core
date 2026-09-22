---- MODULE DprTargetBlockExecutorProofs ----
EXTENDS DprTargetBlockExecutor, TLAPS

(* Proof-only induction support for the unchanged executable model.
   The theorem retains every semantic conjunct from the earlier proof target.
   RecordShape establishes the domain needed to reason about record EXCEPT;
   ControlFacts and Reachability exclude unreachable counterexamples to induction.
   Program-order, concrete destination cardinality, full per-row KV and residency
   invariants remain checked by TLC, not silently claimed as TLAPS theorems. *)

RecordShape == st = [
    phase |-> st.phase,
    compiledExecutorMode |-> st.compiledExecutorMode,
    executorMode |-> st.executorMode,
    activeHorizon |-> st.activeHorizon,
    verifyResult |-> st.verifyResult,
    pendingChoice |-> st.pendingChoice,
    pendingValid |-> st.pendingValid,
    pendingToken |-> st.pendingToken,
    consumeKind |-> st.consumeKind,
    currentLayer |-> st.currentLayer,
    programCounter |-> st.programCounter,
    programEpoch |-> st.programEpoch,
    blockStart |-> st.blockStart,
    runtimePosition |-> st.runtimePosition,
    canonicalPosition |-> st.canonicalPosition,
    runtimeDigest |-> st.runtimeDigest,
    canonicalDigest |-> st.canonicalDigest,
    kvState |-> st.kvState,
    slotState |-> st.slotState,
    slotKey |-> st.slotKey,
    leaseOwner |-> st.leaseOwner,
    reservedSlot |-> st.reservedSlot,
    writtenDestinations |-> st.writtenDestinations,
    writeCount |-> st.writeCount,
    submissions |-> st.submissions,
    completions |-> st.completions,
    producedCount |-> st.producedCount,
    committedCount |-> st.committedCount,
    verificationSteps |-> st.verificationSteps,
    pendingProduced |-> st.pendingProduced,
    pendingConsumed |-> st.pendingConsumed,
    mapCount |-> st.mapCount,
    unmapCount |-> st.unmapCount,
    resourceObjects |-> st.resourceObjects,
    resourceCreates |-> st.resourceCreates,
    resourceDestroys |-> st.resourceDestroys,
    intermediatePublications |-> st.intermediatePublications,
    failureCleanups |-> st.failureCleanups]

ControlFacts ==
    /\ st.phase \in Phases
    /\ st.activeHorizon \in 0..MaxHorizon
    /\ st.verifyResult \in 0..st.activeHorizon
    /\ st.pendingChoice \in 0..MaxToken
    /\ st.pendingValid \in BOOLEAN
    /\ st.pendingToken \in 0..MaxToken
    /\ st.pendingProduced \in 0..1
    /\ st.pendingConsumed \in 0..1
    /\ st.submissions \in 0..1
    /\ st.completions \in 0..1
    /\ st.blockStart = 0
    /\ st.runtimePosition \in 0..(MaxHorizon + 1)
    /\ st.canonicalPosition \in 0..(MaxHorizon + 1)
    /\ st.producedCount \in 0..(MaxHorizon + 1)
    /\ st.committedCount \in 0..(MaxHorizon + 1)
    /\ st.verificationSteps \in 0..(MaxHorizon + 1)

Reachability ==
    /\ (st.phase = "READY" =>
          /\ st.activeHorizon = 0 /\ st.verifyResult = 0
          /\ st.pendingProduced = 0 /\ st.pendingConsumed = 0
          /\ st.submissions = 0 /\ st.completions = 0)
    /\ (st.phase \in ComputePhases =>
          /\ st.verifyResult \in 1..st.activeHorizon
          /\ st.pendingChoice \in TokenValues
          /\ st.pendingProduced = 0 /\ st.pendingConsumed = 0
          /\ st.submissions = 1 /\ st.completions = 0)

Inv ==
    /\ RecordShape /\ ControlFacts /\ Reachability
    /\ ProgramImmutable
    /\ ExecutorPolicyImmutable
    /\ NoIntermediatePublication
    /\ PendingNotStepped
    /\ PendingConsumedOnce
    /\ RuntimeEqualsCanonical
    /\ NoProcessResourceCreation
    /\ OneSubmissionOneCompletion
    /\ ZeroAcceptedSkipsExecutor
    /\ ExecutorRefinesDprTarget

(* Effect operators below are proved projections of concrete actions, not
   replacements or assumptions. Semantic induction does not unfold EXCEPT. *)

ComputeActions ==
    EmbedRows \/ RunAttention \/ (\E slot \in Slots : ReserveExpert(slot))
    \/ LoadExpert \/ AcquireLease \/ RunFfn \/ ReleaseExpert \/ RunHead

BeginEffect(horizon, accepted, pending) ==
    /\ RecordShape'
    /\ st.phase = "READY"
    /\ st.submissions = 0
    /\ horizon \in Rows
    /\ accepted \in 0..horizon
    /\ pending \in TokenValues
    /\ st'.phase = (IF accepted = 0 THEN "PENDING" ELSE "EMBED")
    /\ st'.activeHorizon = (horizon)
    /\ st'.verifyResult = (accepted)
    /\ st'.pendingChoice = (pending)
    /\ st'.pendingValid = (accepted = 0)
    /\ st'.pendingToken = (IF accepted = 0 THEN pending ELSE 0)
    /\ st'.consumeKind = ("NONE")
    /\ st'.blockStart = (st.runtimePosition)
    /\ st'.writeCount = (0)
    /\ st'.submissions = (st.submissions + (IF accepted = 0 THEN 0 ELSE 1))
    /\ st'.producedCount = (IF accepted = 0 THEN 1 ELSE 0)
    /\ st'.committedCount = (0)
    /\ st'.verificationSteps = (0)
    /\ st'.pendingProduced = (st.pendingProduced + (IF accepted = 0 THEN 1 ELSE 0))
    /\ UNCHANGED <<st.compiledExecutorMode, st.executorMode, st.programEpoch, st.runtimePosition, st.canonicalPosition, st.runtimeDigest, st.canonicalDigest, st.completions, st.pendingConsumed, st.resourceObjects, st.resourceCreates, st.resourceDestroys, st.intermediatePublications>>

LEMMA BeginZeroProjection ==
    \A horizon \in Rows, pending \in TokenValues :
      RecordShape /\ BeginBlock(horizon, 0, pending) => BeginEffect(horizon, 0, pending)
BY Z3T(15)
 DEF RecordShape, BeginBlock, BeginEffect, ComputePhases

LEMMA BeginPositiveProjection ==
    \A horizon \in Rows : \A accepted \in 0..horizon, pending \in TokenValues :
      accepted # 0 /\ RecordShape /\ BeginBlock(horizon, accepted, pending)
        => BeginEffect(horizon, accepted, pending)
BY Z3T(15)
 DEF RecordShape, BeginBlock, BeginEffect, ComputePhases

LEMMA BeginProjection ==
    \A horizon \in Rows : \A accepted \in 0..horizon, pending \in TokenValues :
      RecordShape /\ BeginBlock(horizon, accepted, pending) => BeginEffect(horizon, accepted, pending)
BY Z3, BeginZeroProjection, BeginPositiveProjection

LEMMA BeginPreservesInv ==
    \A horizon \in Rows : \A accepted \in 0..horizon, pending \in TokenValues :
      Inv /\ BeginEffect(horizon, accepted, pending) => Inv'
BY Z3T(15), ModelBounds
 DEF BeginEffect, Inv, ControlFacts, Reachability, ProgramImmutable,
     ExecutorPolicyImmutable, NoIntermediatePublication, PendingNotStepped,
     PendingConsumedOnce, RuntimeEqualsCanonical, NoProcessResourceCreation,
     OneSubmissionOneCompletion, ZeroAcceptedSkipsExecutor, ExecutorRefinesDprTarget,
     Phases, ComputePhases, Rows, TokenValues

ComputeEffect ==
    /\ RecordShape'
    /\ st.phase \in ComputePhases
    /\ st'.phase \in ComputePhases
    /\ UNCHANGED <<st.compiledExecutorMode, st.executorMode, st.activeHorizon, st.verifyResult, st.pendingChoice, st.pendingValid, st.pendingToken, st.consumeKind, st.programEpoch, st.blockStart, st.runtimePosition, st.canonicalPosition, st.runtimeDigest, st.canonicalDigest, st.submissions, st.completions, st.producedCount, st.committedCount, st.verificationSteps, st.pendingProduced, st.pendingConsumed, st.resourceObjects, st.resourceCreates, st.resourceDestroys, st.intermediatePublications>>

LEMMA ComputeProjection ==
    RecordShape /\ ComputeActions => ComputeEffect
BY Z3T(15)
 DEF RecordShape, ComputeActions, EmbedRows, RunAttention, ReserveExpert, LoadExpert,
     AcquireLease, RunFfn, ReleaseExpert, RunHead, ComputeEffect, ComputePhases

LEMMA ComputePreservesInv ==
    Inv /\ ComputeEffect => Inv'
BY Z3T(15), ModelBounds
 DEF ComputeEffect, Inv, ControlFacts, Reachability, ProgramImmutable,
     ExecutorPolicyImmutable, NoIntermediatePublication, PendingNotStepped,
     PendingConsumedOnce, RuntimeEqualsCanonical, NoProcessResourceCreation,
     OneSubmissionOneCompletion, ZeroAcceptedSkipsExecutor, ExecutorRefinesDprTarget,
     Phases, ComputePhases, Rows, TokenValues

VerificationEffect ==
    /\ RecordShape'
    /\ st.phase = "VALIDATE"
    /\ st.verifyResult > 0
    /\ st'.phase = ("PENDING")
    /\ st'.runtimePosition = (st.blockStart + st.verifyResult)
    /\ st'.canonicalPosition = (st.blockStart + st.verifyResult)
    /\ st'.runtimeDigest = (DigestAdvance(st.runtimeDigest, st.verifyResult))
    /\ st'.canonicalDigest = (DigestAdvance(st.canonicalDigest, st.verifyResult))
    /\ st'.pendingValid = (TRUE)
    /\ st'.pendingToken = (st.pendingChoice)
    /\ st'.producedCount = (st.verifyResult + 1)
    /\ st'.committedCount = (st.verifyResult)
    /\ st'.verificationSteps = (st.activeHorizon)
    /\ st'.pendingProduced = (st.pendingProduced + 1)
    /\ st'.completions = (st.completions + 1)
    /\ UNCHANGED <<st.compiledExecutorMode, st.executorMode, st.activeHorizon, st.verifyResult, st.pendingChoice, st.consumeKind, st.programEpoch, st.blockStart, st.writeCount, st.submissions, st.pendingConsumed, st.resourceObjects, st.resourceCreates, st.resourceDestroys, st.intermediatePublications>>

LEMMA VerificationProjection ==
    RecordShape /\ CompleteVerification => VerificationEffect
BY Z3T(15)
 DEF RecordShape, CompleteVerification, VerificationEffect, ComputePhases

LEMMA VerificationPreservesInv ==
    Inv /\ VerificationEffect => Inv'
BY Z3T(15), ModelBounds
 DEF VerificationEffect, Inv, ControlFacts, Reachability, ProgramImmutable,
     ExecutorPolicyImmutable, NoIntermediatePublication, PendingNotStepped,
     PendingConsumedOnce, RuntimeEqualsCanonical, NoProcessResourceCreation,
     OneSubmissionOneCompletion, ZeroAcceptedSkipsExecutor, ExecutorRefinesDprTarget,
     Phases, ComputePhases, Rows, TokenValues

ConsumeEffect(finalCommit) ==
    /\ RecordShape'
    /\ finalCommit \in BOOLEAN
    /\ st.phase = "PENDING"
    /\ st.pendingValid
    /\ st.pendingToken \in TokenValues
    /\ finalCommit = (st.pendingToken = StopToken)
    /\ st'.phase = ("DONE")
    /\ st'.runtimePosition = (st.runtimePosition + 1)
    /\ st'.canonicalPosition = (st.canonicalPosition + 1)
    /\ st'.runtimeDigest = (DigestToken(st.runtimeDigest, st.pendingToken))
    /\ st'.canonicalDigest = (DigestToken(st.canonicalDigest, st.pendingToken))
    /\ st'.pendingValid = (FALSE)
    /\ st'.pendingToken = (0)
    /\ st'.consumeKind = (IF finalCommit THEN "FINAL_COMMIT" ELSE "NEXT_CYCLE")
    /\ st'.committedCount = (st.committedCount + 1)
    /\ st'.verificationSteps = (st.verificationSteps + 1)
    /\ st'.pendingConsumed = (st.pendingConsumed + 1)
    /\ UNCHANGED <<st.compiledExecutorMode, st.executorMode, st.activeHorizon, st.verifyResult, st.pendingChoice, st.programEpoch, st.blockStart, st.writeCount, st.submissions, st.completions, st.producedCount, st.pendingProduced, st.resourceObjects, st.resourceCreates, st.resourceDestroys, st.intermediatePublications>>

LEMMA ConsumeProjection ==
    \A finalCommit \in BOOLEAN :
      RecordShape /\ ConsumePending(finalCommit) => ConsumeEffect(finalCommit)
BY Z3T(15)
 DEF RecordShape, ConsumePending, ConsumeEffect, ComputePhases

LEMMA ConsumePreservesInv ==
    \A finalCommit \in BOOLEAN :
      Inv /\ ConsumeEffect(finalCommit) => Inv'
BY Z3T(15), ModelBounds
 DEF ConsumeEffect, Inv, ControlFacts, Reachability, ProgramImmutable,
     ExecutorPolicyImmutable, NoIntermediatePublication, PendingNotStepped,
     PendingConsumedOnce, RuntimeEqualsCanonical, NoProcessResourceCreation,
     OneSubmissionOneCompletion, ZeroAcceptedSkipsExecutor, ExecutorRefinesDprTarget,
     Phases, ComputePhases, Rows, TokenValues

FailureEffect ==
    /\ RecordShape'
    /\ st.phase \in ComputePhases
    /\ st'.phase = ("FAILED")
    /\ st'.completions = (st.completions + 1)
    /\ UNCHANGED <<st.compiledExecutorMode, st.executorMode, st.activeHorizon, st.verifyResult, st.pendingChoice, st.pendingValid, st.pendingToken, st.consumeKind, st.programEpoch, st.blockStart, st.runtimePosition, st.canonicalPosition, st.runtimeDigest, st.canonicalDigest, st.writeCount, st.submissions, st.producedCount, st.committedCount, st.verificationSteps, st.pendingProduced, st.pendingConsumed, st.resourceObjects, st.resourceCreates, st.resourceDestroys, st.intermediatePublications>>

LEMMA FailureProjection ==
    RecordShape /\ FailExecution => FailureEffect
BY Z3T(15)
 DEF RecordShape, FailExecution, FailureEffect, ComputePhases

LEMMA FailurePreservesInv ==
    Inv /\ FailureEffect => Inv'
BY Z3T(15), ModelBounds
 DEF FailureEffect, Inv, ControlFacts, Reachability, ProgramImmutable,
     ExecutorPolicyImmutable, NoIntermediatePublication, PendingNotStepped,
     PendingConsumedOnce, RuntimeEqualsCanonical, NoProcessResourceCreation,
     OneSubmissionOneCompletion, ZeroAcceptedSkipsExecutor, ExecutorRefinesDprTarget,
     Phases, ComputePhases, Rows, TokenValues

InitialEffect ==
    /\ RecordShape
    /\ st.phase = "READY"
    /\ st.compiledExecutorMode = st.executorMode
    /\ st.activeHorizon = 0 /\ st.verifyResult = 0
    /\ st.pendingChoice = 0 /\ st.pendingValid = FALSE
    /\ st.pendingToken = 0 /\ st.consumeKind = "NONE"
    /\ st.programEpoch = 1 /\ st.blockStart = 0
    /\ st.runtimePosition = 0 /\ st.canonicalPosition = 0
    /\ st.runtimeDigest = 0 /\ st.canonicalDigest = 0
    /\ st.writeCount = 0 /\ st.submissions = 0 /\ st.completions = 0
    /\ st.producedCount = 0 /\ st.committedCount = 0
    /\ st.verificationSteps = 0
    /\ st.pendingProduced = 0 /\ st.pendingConsumed = 0
    /\ st.resourceObjects = MaxResources
    /\ st.resourceCreates = 0 /\ st.resourceDestroys = 0
    /\ st.intermediatePublications = 0

LEMMA InitProjection == Init => InitialEffect
BY Z3T(15) DEF Init, InitialEffect, RecordShape, ExecutorModes

LEMMA InitialEffectEstablishesInv == InitialEffect => Inv
BY Z3T(15), ModelBounds
 DEF InitialEffect, Inv, ControlFacts, Reachability,
     ProgramImmutable, ExecutorPolicyImmutable, NoIntermediatePublication,
     PendingNotStepped, PendingConsumedOnce, RuntimeEqualsCanonical,
     NoProcessResourceCreation, OneSubmissionOneCompletion,
     ZeroAcceptedSkipsExecutor, ExecutorRefinesDprTarget,
     ExecutorModes, Phases, ComputePhases, Layers, Rows, Slots, TokenValues

LEMMA InitEstablishesInv == Init => Inv
BY Z3, InitProjection, InitialEffectEstablishesInv

LEMMA StutterPreservesInv == Inv /\ UNCHANGED vars => Inv'
BY Z3 DEF vars, Inv, RecordShape, ControlFacts, Reachability,
          ProgramImmutable, ExecutorPolicyImmutable, NoIntermediatePublication,
          PendingNotStepped, PendingConsumedOnce, RuntimeEqualsCanonical,
          NoProcessResourceCreation, OneSubmissionOneCompletion,
          ZeroAcceptedSkipsExecutor, ExecutorRefinesDprTarget

LEMMA NextPreservesInv == Inv /\ Next => Inv'
BY Z3, BeginProjection, BeginPreservesInv,
       ComputeProjection, ComputePreservesInv,
       VerificationProjection, VerificationPreservesInv,
       ConsumeProjection, ConsumePreservesInv,
       FailureProjection, FailurePreservesInv, StutterPreservesInv
 DEF Inv, Next, ComputeActions, TerminalStutter, vars

THEOREM InductiveInvariant == Inv /\ [Next]_vars => Inv'
BY Z3, NextPreservesInv, StutterPreservesInv

THEOREM DprTargetBlockExecutorSafety == Spec => []Inv
<1>1. Init => Inv
       BY InitEstablishesInv
<1>2. Inv /\ [Next]_vars => Inv'
       BY InductiveInvariant
<1>3. QED
       BY <1>1, <1>2, PTL DEF Spec

====
