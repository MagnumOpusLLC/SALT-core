---- MODULE DprNfqTokenEpochProofs ----
EXTENDS DprNfqTokenEpoch, TLAPS

(* Proof-only refresh of the retained epoch abstraction. This model uses N
   as target depth and admits no ordinary B1 action; therefore even a successful
   proof is NOT a refinement seal for current X-separated normal serving.
   SaltDecodeControl separately models the current cold/warm decode route. *)

EpochSetup ==
    /\ (phase \in {"CACHE_SCAN", "COLD_SEARCH"} =>
          /\ ~exactHit /\ matchingRoute = 0 /\ targetWinner = 0
          /\ submissions = 0 /\ completionFences = 0
          /\ queueItems = 0 /\ invalidatedItems = 0)
    /\ (phase = "CACHE_SCAN" => ~cacheChecked /\ ~coldSearchStarted)
    /\ (phase = "COLD_SEARCH" => cacheChecked /\ coldSearchStarted)
    /\ (phase = "TARGET_VERIFY" =>
          /\ coldSearchStarted /\ ~exactHit /\ matchingRoute \in Routes
          /\ submissions = 1 /\ completionFences = 0
          /\ queueItems = TargetRows * Q)

Inv ==
    /\ TypeOK /\ EpochSetup
    /\ CacheFirst /\ ColdSearchOnlyOnMiss /\ NoSearchOnExactHit
    /\ TargetSelectsWinner /\ CompletionOrderNonAuthoritative
    /\ NoMatchSkipsTarget /\ TargetExecutionExact
    /\ PendingOnlyAfterResolution /\ PositionAuthority
    /\ AcceptedWidthIsNNotQ /\ QueueRunwayBounded /\ NoB1Fallback
    /\ PendingConsumedByTarget /\ OldEpochInvalidatedBeforeCommit
    /\ OneEpochAtATime /\ BoundaryAfterInvalidation

LEMMA InitEstablishesInv == Init => Inv
BY Z3, ModelBounds
 DEF Init, Inv, TypeOK, EpochSetup, CacheFirst, ColdSearchOnlyOnMiss,
     NoSearchOnExactHit, TargetSelectsWinner, CompletionOrderNonAuthoritative,
     NoMatchSkipsTarget, TargetExecutionExact, PendingOnlyAfterResolution,
     PositionAuthority, AcceptedWidthIsNNotQ, QueueRunwayBounded, NoB1Fallback,
     PendingConsumedByTarget, OldEpochInvalidatedBeforeCommit,
     OneEpochAtATime, BoundaryAfterInvalidation, Phases, Tokens, Routes,
     TargetRows, ActiveWorkerWidth, MaxCandidateExpertDemandBytes

LEMMA NextPreservesInv == Inv /\ Next => Inv'
BY Z3, ModelBounds
 DEF Next, BeginEpoch, CacheExact, CacheMiss, SelectRoutes,
     VerifyWinningRoute, ConsumeTargetPending, Fail, TerminalStutter, vars,
     Inv, TypeOK, EpochSetup, CacheFirst, ColdSearchOnlyOnMiss,
     NoSearchOnExactHit, TargetSelectsWinner, CompletionOrderNonAuthoritative,
     NoMatchSkipsTarget, TargetExecutionExact, PendingOnlyAfterResolution,
     PositionAuthority, AcceptedWidthIsNNotQ, QueueRunwayBounded, NoB1Fallback,
     PendingConsumedByTarget, OldEpochInvalidatedBeforeCommit,
     OneEpochAtATime, BoundaryAfterInvalidation, Phases, Tokens, Routes,
     TargetRows, ActiveWorkerWidth, MaxCandidateExpertDemandBytes

LEMMA StutterPreservesInv == Inv /\ UNCHANGED vars => Inv'
BY Z3 DEF vars, Inv, TypeOK, EpochSetup, CacheFirst, ColdSearchOnlyOnMiss,
          NoSearchOnExactHit, TargetSelectsWinner, CompletionOrderNonAuthoritative,
          NoMatchSkipsTarget, TargetExecutionExact, PendingOnlyAfterResolution,
          PositionAuthority, AcceptedWidthIsNNotQ, QueueRunwayBounded, NoB1Fallback,
          PendingConsumedByTarget, OldEpochInvalidatedBeforeCommit,
          OneEpochAtATime, BoundaryAfterInvalidation, Phases, Tokens, Routes,
          TargetRows

THEOREM InductiveInvariant == Inv /\ [Next]_vars => Inv'
BY Z3, NextPreservesInv, StutterPreservesInv

THEOREM DprNfqTokenEpochSafety == Spec => []Inv
<1>1. Init => Inv
       BY InitEstablishesInv
<1>2. Inv /\ [Next]_vars => Inv'
       BY InductiveInvariant
<1>3. QED
       BY <1>1, <1>2, PTL DEF Spec

====
