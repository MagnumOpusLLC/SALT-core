---- MODULE DprTargetRuntime ----
EXTENDS Integers, Naturals

CONSTANTS MaxPosition, MaxHorizon, MaxCycles

ASSUME DprConstantsOK ==
       /\ MaxPosition \in Nat \ {0}
       /\ MaxHorizon \in Nat \ {0}
       /\ MaxCycles \in Nat \ {0}

Phases == {"READY", "PARENT_LOOKUP", "NOMOGRAM_LOOKUP", "COLD_SEARCH",
           "VERIFYING", "PENDING", "DONE", "FAILED"}
TokenValues == 1..3
BoolOK(value) == value = TRUE \/ value = FALSE

DigestAdvance(d, amount) == (d * 11 + amount) % 103
DigestToken(d, token) == (d * 13 + token) % 103

VARIABLES
    phase,
    dprEnabled,
    position,
    cycleStart,
    committedCount,
    cycle,
    parentChecked,
    parentHit,
    nomogramChecked,
    proposalCount,
    acceptedCount,
    pendingValid,
    pendingToken,
    targetSteps,
    nomogramCommitted,
    strictRequested,
    proofComputed,
    runtimeDigest,
    canonicalDigest

vars == <<phase, dprEnabled, position, cycleStart, committedCount, cycle,
          parentChecked, parentHit, nomogramChecked, proposalCount,
          acceptedCount, pendingValid, pendingToken, targetSteps,
          nomogramCommitted, strictRequested, proofComputed,
          runtimeDigest, canonicalDigest>>

Init ==
    /\ phase = "READY"
    /\ dprEnabled \in BOOLEAN
    /\ position = 0
    /\ cycleStart = 0
    /\ committedCount = 0
    /\ cycle = 0
    /\ parentChecked = FALSE
    /\ parentHit = FALSE
    /\ nomogramChecked = FALSE
    /\ proposalCount = 0
    /\ acceptedCount = 0
    /\ pendingValid = FALSE
    /\ pendingToken = 0
    /\ targetSteps = 0
    /\ nomogramCommitted = 0
    /\ strictRequested = FALSE
    /\ proofComputed = FALSE
    /\ runtimeDigest = 0
    /\ canonicalDigest = 0

BeginCycle(strict) ==
    /\ phase = "READY"
    /\ (~strictRequested \/ proofComputed)
    /\ cycle < MaxCycles
    /\ position < MaxPosition
    /\ BoolOK(strict)
    /\ phase' = IF dprEnabled THEN "PARENT_LOOKUP" ELSE "COLD_SEARCH"
    /\ cycleStart' = position
    /\ cycle' = cycle + 1
    /\ parentChecked' = FALSE
    /\ parentHit' = FALSE
    /\ nomogramChecked' = FALSE
    /\ proposalCount' = 0
    /\ acceptedCount' = 0
    /\ strictRequested' = strict
    /\ proofComputed' = FALSE
    /\ UNCHANGED <<dprEnabled, position, committedCount, pendingValid, pendingToken,
                    targetSteps, nomogramCommitted, runtimeDigest,
                    canonicalDigest>>

ParentExact(horizon) ==
    /\ phase = "PARENT_LOOKUP"
    /\ horizon \in 1..MaxHorizon
    /\ position + horizon <= MaxPosition
    /\ parentChecked' = TRUE
    /\ parentHit' = TRUE
    /\ position' = position + horizon
    /\ committedCount' = committedCount + horizon
    /\ runtimeDigest' = DigestAdvance(runtimeDigest, horizon)
    /\ canonicalDigest' = DigestAdvance(canonicalDigest, horizon)
    /\ phase' = IF position' = MaxPosition THEN "DONE" ELSE "READY"
    /\ UNCHANGED <<dprEnabled, cycleStart, cycle, nomogramChecked, proposalCount,
                    acceptedCount, pendingValid, pendingToken, targetSteps,
                    nomogramCommitted, strictRequested, proofComputed>>

ParentMiss ==
    /\ phase = "PARENT_LOOKUP"
    /\ phase' = "NOMOGRAM_LOOKUP"
    /\ parentChecked' = TRUE
    /\ parentHit' = FALSE
    /\ UNCHANGED <<dprEnabled, position, cycleStart, committedCount, cycle,
                    nomogramChecked, proposalCount, acceptedCount,
                    pendingValid, pendingToken, targetSteps,
                    nomogramCommitted, strictRequested, proofComputed,
                    runtimeDigest, canonicalDigest>>

NomogramMiss ==
    /\ phase = "NOMOGRAM_LOOKUP"
    /\ parentChecked
    /\ ~parentHit
    /\ phase' = "COLD_SEARCH"
    /\ nomogramChecked' = TRUE
    /\ proposalCount' = 0
    /\ UNCHANGED <<dprEnabled, position, cycleStart, committedCount, cycle,
                    parentChecked, parentHit, acceptedCount,
                    pendingValid, pendingToken, targetSteps,
                    nomogramCommitted, strictRequested, proofComputed,
                    runtimeDigest, canonicalDigest>>

NomogramHit(horizon) ==
    /\ phase = "NOMOGRAM_LOOKUP"
    /\ parentChecked
    /\ ~parentHit
    /\ horizon \in 1..MaxHorizon
    /\ position + horizon <= MaxPosition
    /\ phase' = "VERIFYING"
    /\ nomogramChecked' = TRUE
    /\ proposalCount' = horizon
    /\ UNCHANGED <<dprEnabled, position, cycleStart, committedCount, cycle,
                    parentChecked, parentHit, acceptedCount,
                    pendingValid, pendingToken, targetSteps,
                    nomogramCommitted, strictRequested, proofComputed,
                    runtimeDigest, canonicalDigest>>

TargetVerify(accepted, pending) ==
    /\ phase = "VERIFYING"
    /\ proposalCount \in 1..MaxHorizon
    /\ accepted \in 0..proposalCount
    /\ pending \in TokenValues
    /\ position + accepted < MaxPosition
    /\ phase' = "PENDING"
    /\ acceptedCount' = accepted
    /\ position' = position + accepted
    /\ committedCount' = committedCount + accepted
    /\ pendingValid' = TRUE
    /\ pendingToken' = pending
    /\ targetSteps' = targetSteps +
        IF accepted = 0 THEN 0 ELSE proposalCount
    /\ runtimeDigest' = DigestAdvance(runtimeDigest, accepted)
    /\ canonicalDigest' = DigestAdvance(canonicalDigest, accepted)
    /\ UNCHANGED <<dprEnabled, cycleStart, cycle, parentChecked, parentHit,
                    nomogramChecked, proposalCount, nomogramCommitted,
                    strictRequested, proofComputed>>

ConsumePending ==
    /\ phase = "PENDING"
    /\ pendingValid
    /\ pendingToken \in TokenValues
    /\ position < MaxPosition
    /\ position' = position + 1
    /\ committedCount' = committedCount + 1
    /\ targetSteps' = targetSteps + 1
    /\ runtimeDigest' = DigestToken(runtimeDigest, pendingToken)
    /\ canonicalDigest' = DigestToken(canonicalDigest, pendingToken)
    /\ pendingValid' = FALSE
    /\ pendingToken' = 0
    /\ phase' = IF position' = MaxPosition THEN "DONE" ELSE "READY"
    /\ UNCHANGED <<dprEnabled, cycleStart, cycle, parentChecked, parentHit,
                    nomogramChecked, proposalCount, acceptedCount,
                    nomogramCommitted, strictRequested, proofComputed>>

ColdSearchResolve(activeHorizon, accepted, pending) ==
    /\ phase = "COLD_SEARCH"
    /\ activeHorizon \in 1..MaxHorizon
    /\ accepted \in 0..activeHorizon
    /\ pending \in TokenValues
    /\ position + accepted < MaxPosition
    /\ phase' = "PENDING"
    /\ position' = position + accepted
    /\ committedCount' = committedCount + accepted
    /\ acceptedCount' = accepted
    /\ pendingValid' = TRUE
    /\ pendingToken' = pending
    /\ targetSteps' = targetSteps +
        IF accepted = 0 THEN 0 ELSE activeHorizon
    /\ runtimeDigest' = DigestAdvance(runtimeDigest, accepted)
    /\ canonicalDigest' = DigestAdvance(canonicalDigest, accepted)
    /\ UNCHANGED <<dprEnabled, cycleStart, cycle, parentChecked, parentHit,
                    nomogramChecked, proposalCount, nomogramCommitted,
                    strictRequested, proofComputed>>

StrictProof ==
    /\ phase \in {"READY", "PENDING", "DONE"}
    /\ strictRequested
    /\ ~proofComputed
    /\ proofComputed' = TRUE
    /\ UNCHANGED <<phase, dprEnabled, position, cycleStart, committedCount, cycle,
                    parentChecked, parentHit, nomogramChecked, proposalCount,
                    acceptedCount, pendingValid, pendingToken, targetSteps,
                    nomogramCommitted, strictRequested, runtimeDigest,
                    canonicalDigest>>

Fail ==
    /\ phase \notin {"DONE", "FAILED"}
    /\ phase' = "FAILED"
    /\ pendingValid' = FALSE
    /\ pendingToken' = 0
    /\ UNCHANGED <<dprEnabled, position, cycleStart, committedCount, cycle,
                    parentChecked, parentHit, nomogramChecked, proposalCount,
                    acceptedCount, targetSteps, nomogramCommitted,
                    strictRequested, proofComputed, runtimeDigest,
                    canonicalDigest>>

TerminalStutter ==
    /\ phase \in {"DONE", "FAILED"}
    /\ UNCHANGED vars

Next ==
    \/ \E strict \in BOOLEAN : BeginCycle(strict)
    \/ \E horizon \in 1..MaxHorizon : ParentExact(horizon)
    \/ ParentMiss
    \/ NomogramMiss
    \/ \E horizon \in 1..MaxHorizon : NomogramHit(horizon)
    \/ \E accepted \in 0..MaxHorizon : \E pending \in TokenValues :
         TargetVerify(accepted, pending)
    \/ ConsumePending
    \/ \E activeHorizon \in 1..MaxHorizon :
         \E accepted \in 0..activeHorizon, pending \in TokenValues :
           ColdSearchResolve(activeHorizon, accepted, pending)
    \/ StrictProof
    \/ Fail
    \/ TerminalStutter

Spec == Init /\ [][Next]_vars

ControlTypeOK ==
    /\ phase \in Phases
    /\ BoolOK(dprEnabled)
    /\ cycle \in 0..MaxCycles
    /\ BoolOK(strictRequested)
    /\ BoolOK(proofComputed)

PositionTypeOK ==
    /\ position \in 0..MaxPosition
    /\ cycleStart \in 0..MaxPosition
    /\ committedCount \in 0..MaxPosition
    /\ targetSteps \in Nat

LookupTypeOK ==
    /\ BoolOK(parentChecked)
    /\ BoolOK(parentHit)
    /\ BoolOK(nomogramChecked)

ProposalTypeOK ==
    /\ proposalCount \in 0..MaxHorizon
    /\ acceptedCount \in 0..MaxHorizon

PendingTypeOK ==
    /\ BoolOK(pendingValid)
    /\ pendingToken \in {0} \cup TokenValues

DigestTypeOK ==
    /\ nomogramCommitted = 0
    /\ runtimeDigest \in 0..102
    /\ canonicalDigest \in 0..102

TypeOK ==
    /\ ControlTypeOK
    /\ PositionTypeOK
    /\ LookupTypeOK
    /\ ProposalTypeOK
    /\ PendingTypeOK
    /\ DigestTypeOK

CommittedIdentityExact == position = committedCount
DigestExact == runtimeDigest = canonicalDigest

ParentFirst ==
    phase \in {"NOMOGRAM_LOOKUP", "VERIFYING"}
        => dprEnabled /\ parentChecked /\ ~parentHit

ProposalIsObservational ==
    phase \in {"PARENT_LOOKUP", "NOMOGRAM_LOOKUP", "COLD_SEARCH", "VERIFYING"}
        => position = cycleStart

OptionalDprOnly ==
    /\ (~dprEnabled =>
        /\ ~parentChecked
        /\ ~parentHit
        /\ ~nomogramChecked
        /\ proposalCount = 0)
    /\ (phase \in {"PARENT_LOOKUP", "NOMOGRAM_LOOKUP", "VERIFYING"}
        => dprEnabled)

NoB1Fallback == phase # "ORDINARY"

PendingNotStepped ==
    /\ (pendingValid <=> phase = "PENDING")
    /\ (phase = "PENDING" => pendingToken \in TokenValues)
    /\ (phase # "PENDING" => pendingToken = 0)

NomogramNeverCommits == nomogramCommitted = 0
StrictProofObservational == proofComputed => strictRequested

DprTargetInvariants ==
    /\ TypeOK
    /\ CommittedIdentityExact
    /\ DigestExact
    /\ ParentFirst
    /\ ProposalIsObservational
    /\ PendingNotStepped
    /\ NomogramNeverCommits
    /\ StrictProofObservational
    /\ OptionalDprOnly
    /\ NoB1Fallback

====
