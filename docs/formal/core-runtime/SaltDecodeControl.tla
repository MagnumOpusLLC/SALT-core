---- MODULE SaltDecodeControl ----
EXTENDS Integers, Naturals

(* Current normal-serving control refinement.
   Source: d3a2037f404d520998ac2d3b2d182ca58f529c82,
   2026-09-21 19:50:34 -04:00.
   src/text_verify.c:2735-3003,3417-3521;
   models/gemma4-26b-a4b/src/inference.c:2102-2440.
   Token IDs, tensor arithmetic and callback internals are abstracted.
   ResolveTarget takes the exact verifier's accepted-prefix length, not an
   untrusted proposal's claim. DPR and explicit import/clear are separate models.
   No fairness/liveness theorem or concrete byte-equivalence theorem is claimed. *)

CONSTANTS MaxPosition, MaxTurns, PromptMax, OutputCap,
          WarmupRows, WarmX, RecipeX

ASSUME ModelBounds ==
    /\ MaxPosition \in Nat \ {0}
    /\ MaxTurns \in Nat \ {0}
    /\ PromptMax \in Nat \ {0}
    /\ OutputCap \in Nat \ {0}
    /\ WarmupRows \in Nat \ {0}
    /\ WarmX \in Nat \ {0}
    /\ RecipeX \in Nat \ {0}
    /\ PromptMax + OutputCap + 1 <= MaxPosition

Min(a, b) == IF a < b THEN a ELSE b
Allowed(prior) == Min(IF prior < WarmupRows THEN 1 ELSE WarmX, RecipeX)

Phases == {"READY", "PREFILL", "BOUNDARY", "SELECT", "CONSUME",
           "PROPOSE", "TARGET", "WAIT_RESOURCE", "FINISH", "SEAL",
           "JOURNAL", "PUBLISH", "FAILED", "CLOSED"}
Active == Phases \ {"READY", "PREFILL", "FAILED", "CLOSED"}

VARIABLES phase, position, priorRows, source, outputCount, pending,
          stopKind, width, closeRows, journaled, published, turns,
          protectedEnd, targetSource, targetFloor, lease

vars == <<phase, position, priorRows, source, outputCount, pending,
          stopKind, width, closeRows, journaled, published, turns,
          protectedEnd, targetSource, targetFloor, lease>>

Init ==
    /\ phase = "READY"
    /\ position = 0 /\ priorRows = 0 /\ source = 0
    /\ outputCount = 0 /\ pending = FALSE /\ stopKind = 0
    /\ width = 0 /\ closeRows = 0
    /\ journaled = FALSE /\ published = FALSE /\ turns = 0
    /\ protectedEnd = 0 /\ targetSource = 0 /\ targetFloor = 0
    /\ lease = FALSE

Admit(promptRows) ==
    /\ phase = "READY" /\ turns < MaxTurns
    /\ promptRows \in 1..PromptMax
    /\ position + promptRows + OutputCap + 1 <= MaxPosition
    /\ phase' = "PREFILL"
    /\ priorRows' = position
    /\ source' = position + promptRows
    /\ outputCount' = 0 /\ pending' = FALSE /\ stopKind' = 0
    /\ width' = 0 /\ closeRows' = 0
    /\ journaled' = FALSE /\ published' = FALSE
    /\ protectedEnd' = position /\ targetSource' = position
    /\ targetFloor' = position /\ lease' = FALSE
    /\ UNCHANGED <<position, turns>>

Prefill ==
    /\ phase = "PREFILL"
    /\ phase' = "BOUNDARY" /\ position' = source
    /\ UNCHANGED <<priorRows, source, outputCount, pending, stopKind,
                    width, closeRows, journaled, published, turns,
                    protectedEnd, targetSource, targetFloor, lease>>

ChooseRoute ==
    /\ phase = "BOUNDARY" /\ outputCount < OutputCap /\ stopKind = 0
    /\ width' = Min(Allowed(priorRows), OutputCap - outputCount)
    /\ phase' = IF width' = 1 THEN "SELECT" ELSE "PROPOSE"
    /\ UNCHANGED <<position, priorRows, source, outputCount, pending,
                    stopKind, closeRows, journaled, published, turns,
                    protectedEnd, targetSource, targetFloor, lease>>

ProposeMiss ==
    /\ phase = "PROPOSE"
    /\ phase' = "SELECT"
    /\ UNCHANGED <<position, priorRows, source, outputCount, pending,
                    stopKind, width, closeRows, journaled, published,
                    turns, protectedEnd, targetSource, targetFloor, lease>>

ProposeHit ==
    /\ phase = "PROPOSE"
    /\ phase' = "TARGET"
    /\ targetSource' = position /\ targetFloor' = protectedEnd
    /\ UNCHANGED <<position, priorRows, source, outputCount, pending,
                    stopKind, width, closeRows, journaled, published,
                    turns, protectedEnd, lease>>

SampleOrdinary(kind) ==
    /\ phase = "SELECT" /\ kind \in 0..2
    /\ outputCount' = outputCount + 1 /\ pending' = TRUE
    /\ stopKind' = kind
    /\ phase' = IF kind # 0 \/ outputCount' = OutputCap
                THEN "FINISH" ELSE "CONSUME"
    /\ UNCHANGED <<position, priorRows, source, width, closeRows,
                    journaled, published, turns, protectedEnd,
                    targetSource, targetFloor, lease>>

ConsumeOrdinary ==
    /\ phase = "CONSUME" /\ pending
    /\ position' = position + 1 /\ pending' = FALSE
    /\ phase' = "BOUNDARY"
    /\ UNCHANGED <<priorRows, source, outputCount, stopKind, width,
                    closeRows, journaled, published, turns,
                    protectedEnd, targetSource, targetFloor, lease>>

NeedResource ==
    /\ phase = "TARGET" /\ ~lease
    /\ phase' = "WAIT_RESOURCE"
    /\ UNCHANGED <<position, priorRows, source, outputCount, pending,
                    stopKind, width, closeRows, journaled, published,
                    turns, protectedEnd, targetSource, targetFloor, lease>>

AcquireResume ==
    /\ phase = "WAIT_RESOURCE"
    /\ phase' = "TARGET" /\ lease' = TRUE
    /\ UNCHANGED <<position, priorRows, source, outputCount, pending,
                    stopKind, width, closeRows, journaled, published,
                    turns, protectedEnd, targetSource, targetFloor>>

ResolveTarget(accepted, kind) ==
    /\ phase = "TARGET" /\ accepted \in 1..width /\ kind \in 0..2
    /\ phase' = "BOUNDARY"
    /\ position' = targetSource + accepted
    /\ protectedEnd' = position'
    /\ outputCount' = outputCount + accepted
    /\ stopKind' = kind /\ lease' = FALSE
    /\ UNCHANGED <<priorRows, source, pending, width, closeRows,
                    journaled, published, turns, targetSource, targetFloor>>

BeginFinish ==
    /\ phase = "BOUNDARY"
    /\ outputCount = OutputCap \/ stopKind # 0
    /\ phase' = "FINISH"
    /\ UNCHANGED <<position, priorRows, source, outputCount, pending,
                    stopKind, width, closeRows, journaled, published,
                    turns, protectedEnd, targetSource, targetFloor, lease>>

FinishPending ==
    /\ phase = "FINISH"
    /\ phase' = "SEAL"
    /\ position' = position + (IF pending THEN 1 ELSE 0)
    /\ pending' = FALSE
    /\ UNCHANGED <<priorRows, source, outputCount, stopKind, width,
                    closeRows, journaled, published, turns,
                    protectedEnd, targetSource, targetFloor, lease>>

SealTurn ==
    /\ phase = "SEAL"
    /\ closeRows' = IF stopKind = 1 THEN 0 ELSE 1
    /\ position' = position + closeRows'
    /\ phase' = "JOURNAL"
    /\ UNCHANGED <<priorRows, source, outputCount, pending, stopKind,
                    width, journaled, published, turns, protectedEnd,
                    targetSource, targetFloor, lease>>

WriteJournal ==
    /\ phase = "JOURNAL"
    /\ journaled' = TRUE /\ turns' = turns + 1
    /\ phase' = "PUBLISH"
    /\ UNCHANGED <<position, priorRows, source, outputCount, pending,
                    stopKind, width, closeRows, published, protectedEnd,
                    targetSource, targetFloor, lease>>

Publish ==
    /\ phase = "PUBLISH" /\ journaled /\ ~pending /\ ~published
    /\ published' = TRUE /\ phase' = "READY"
    /\ UNCHANGED <<position, priorRows, source, outputCount, pending,
                    stopKind, width, closeRows, journaled, turns,
                    protectedEnd, targetSource, targetFloor, lease>>

Fail ==
    /\ phase \in Phases \ {"READY", "FAILED", "CLOSED"}
    /\ phase' = "FAILED" /\ lease' = FALSE
    /\ UNCHANGED <<position, priorRows, source, outputCount, pending,
                    stopKind, width, closeRows, journaled, published,
                    turns, protectedEnd, targetSource, targetFloor>>

Close ==
    /\ phase \in {"READY", "FAILED"}
    /\ phase' = "CLOSED"
    /\ UNCHANGED <<position, priorRows, source, outputCount, pending,
                    stopKind, width, closeRows, journaled, published,
                    turns, protectedEnd, targetSource, targetFloor, lease>>

Idle ==
    /\ phase \in {"READY", "FAILED", "CLOSED"}
    /\ UNCHANGED vars

Next ==
    \/ \E p \in 1..PromptMax : Admit(p)
    \/ Prefill \/ ChooseRoute \/ ProposeMiss \/ ProposeHit
    \/ \E k \in 0..2 : SampleOrdinary(k)
    \/ ConsumeOrdinary \/ NeedResource \/ AcquireResume
    \/ \E a \in 1..width, k \in 0..2 : ResolveTarget(a, k)
    \/ BeginFinish \/ FinishPending \/ SealTurn \/ WriteJournal
    \/ Publish \/ Fail \/ Close \/ Idle

Spec == Init /\ [][Next]_vars

TypeOK ==
    /\ phase \in Phases
    /\ position \in 0..MaxPosition /\ priorRows \in 0..MaxPosition
    /\ source \in 0..MaxPosition /\ outputCount \in 0..OutputCap
    /\ pending \in BOOLEAN /\ stopKind \in 0..2
    /\ width \in 0..RecipeX /\ closeRows \in 0..1
    /\ journaled \in BOOLEAN /\ published \in BOOLEAN
    /\ turns \in 0..MaxTurns
    /\ protectedEnd \in 0..MaxPosition
    /\ targetSource \in 0..MaxPosition /\ targetFloor \in 0..MaxPosition
    /\ lease \in BOOLEAN

ColdWidthBound == priorRows < WarmupRows => width <= 1
WidthBound == width <= Allowed(priorRows)
PrefixPreserved ==
    /\ position >= protectedEnd
    /\ targetSource >= targetFloor
PublicationSafety ==
    published => journaled /\ ~pending /\ phase \in {"READY", "CLOSED"}
JournalAfterFinish ==
    journaled => ~pending /\ position = source + outputCount + closeRows
PendingAccounting ==
    phase \in Active =>
      position + (IF pending THEN 1 ELSE 0) = source + outputCount + closeRows
ResourceIsolation == lease => phase = "TARGET"

(* Reachability strengthening. These facts are derived from the actual guards;
   they are not extra assumptions imposed on the implementation. *)
Reachability ==
    /\ (phase \in Phases \ {"READY", "FAILED", "CLOSED"} =>
          source + OutputCap + 1 <= MaxPosition)
    /\ (phase \in {"PREFILL", "BOUNDARY", "SELECT", "CONSUME", "PROPOSE",
                     "TARGET", "WAIT_RESOURCE", "FINISH", "SEAL", "JOURNAL"}
        => turns < MaxTurns)
    /\ (phase \in {"SELECT", "PROPOSE", "TARGET", "WAIT_RESOURCE"} =>
          /\ outputCount < OutputCap /\ width >= 1
          /\ width <= OutputCap - outputCount /\ ~pending)
    /\ (phase \in {"TARGET", "WAIT_RESOURCE"} => position = targetSource)
    /\ (phase \in {"PREFILL", "BOUNDARY", "PROPOSE", "TARGET",
                     "WAIT_RESOURCE", "SELECT", "SEAL", "JOURNAL", "PUBLISH"}
        => ~pending)
    /\ (phase \in {"PREFILL", "BOUNDARY", "SELECT", "CONSUME", "PROPOSE",
                     "TARGET", "WAIT_RESOURCE", "FINISH", "SEAL"}
        => closeRows = 0 /\ ~journaled)
    /\ (phase = "PREFILL" =>
          position = priorRows /\ source > position /\ outputCount = 0)
    /\ (stopKind # 0 => outputCount > 0)
    /\ (phase \in {"JOURNAL", "PUBLISH"} =>
          outputCount > 0 /\ (outputCount = OutputCap \/ stopKind # 0))
    /\ (phase = "FINISH" =>
          outputCount > 0 /\ (outputCount = OutputCap \/ stopKind # 0))
    /\ (phase = "SEAL" =>
          outputCount > 0 /\ (outputCount = OutputCap \/ stopKind # 0))
    /\ (phase = "CONSUME" => pending /\ outputCount < OutputCap)
    /\ (phase = "PUBLISH" => journaled)

Inv == TypeOK /\ ColdWidthBound /\ WidthBound /\ PrefixPreserved
       /\ PublicationSafety /\ JournalAfterFinish /\ PendingAccounting
       /\ ResourceIsolation /\ Reachability

====
