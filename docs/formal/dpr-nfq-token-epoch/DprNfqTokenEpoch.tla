---- MODULE DprNfqTokenEpoch ----
EXTENDS Integers, Naturals

CONSTANTS
    NumLayers,
    ExpertsPerLayer,
    TopK,
    ExpertWeightBytes,
    N,
    F,
    Q,
    TargetRowCapacity,
    WorkerBudget,
    MaxEpoch,
    MaxPosition,
    MaxToken

TargetRows == N * F
ActiveWorkerWidth ==
    IF WorkerBudget < TargetRows THEN WorkerBudget ELSE TargetRows
MaxCandidateExpertDemandBytes ==
    TargetRows * TopK * ExpertWeightBytes

ASSUME ModelBounds ==
    /\ NumLayers \in Nat \ {0}
    /\ ExpertsPerLayer \in Nat \ {0}
    /\ TopK \in 1..ExpertsPerLayer
    /\ ExpertWeightBytes \in Nat \ {0}
    /\ N \in Nat \ {0}
    /\ F \in Nat \ {0}
    /\ Q \in Nat \ {0}
    /\ TargetRowCapacity \in Nat \ {0}
    /\ TargetRows <= TargetRowCapacity
    /\ WorkerBudget \in Nat \ {0}
    /\ ActiveWorkerWidth \in 1..TargetRows
    /\ TargetRows % ActiveWorkerWidth = 0
    /\ MaxEpoch \in Nat \ {0}
    /\ MaxPosition \in Nat \ {0}
    /\ MaxToken \in Nat \ {0}
    /\ N + 1 <= MaxPosition

Routes == 1..F
Tokens == 1..MaxToken
Phases == {
    "BOUNDARY", "CACHE_SCAN", "COLD_SEARCH", "TARGET_VERIFY",
    "RESOLVED", "FAILED"
}

VARIABLES
    phase,
    epoch,
    lastEpoch,
    position,
    parentPosition,
    cacheChecked,
    exactHit,
    coldSearchStarted,
    targetToken,
    matchingRoute,
    completionRoute,
    targetWinner,
    acceptedCount,
    pendingValid,
    pendingToken,
    targetSteps,
    submissions,
    completionFences,
    queueItems,
    invalidatedItems,
    oldEpochInvalidated

vars == <<
    phase, epoch, lastEpoch, position, parentPosition, cacheChecked,
    exactHit, coldSearchStarted, targetToken, matchingRoute,
    completionRoute, targetWinner, acceptedCount, pendingValid,
    pendingToken, targetSteps, submissions, completionFences, queueItems,
    invalidatedItems, oldEpochInvalidated
>>

Init ==
    /\ phase = "BOUNDARY"
    /\ epoch = 0
    /\ lastEpoch = 0
    /\ position = 0
    /\ parentPosition = 0
    /\ cacheChecked = FALSE
    /\ exactHit = FALSE
    /\ coldSearchStarted = FALSE
    /\ targetToken = 0
    /\ matchingRoute = 0
    /\ completionRoute = 0
    /\ targetWinner = 0
    /\ acceptedCount = 0
    /\ pendingValid = FALSE
    /\ pendingToken = 0
    /\ targetSteps = 0
    /\ submissions = 0
    /\ completionFences = 0
    /\ queueItems = 0
    /\ invalidatedItems = 0
    /\ oldEpochInvalidated = FALSE

BeginEpoch ==
    /\ phase = "BOUNDARY"
    /\ ~pendingValid
    /\ epoch < MaxEpoch
    /\ position < MaxPosition
    /\ phase' = "CACHE_SCAN"
    /\ epoch' = epoch + 1
    /\ parentPosition' = position
    /\ cacheChecked' = FALSE
    /\ exactHit' = FALSE
    /\ coldSearchStarted' = FALSE
    /\ targetToken' = 0
    /\ matchingRoute' = 0
    /\ completionRoute' = 0
    /\ targetWinner' = 0
    /\ acceptedCount' = 0
    /\ pendingValid' = FALSE
    /\ pendingToken' = 0
    /\ submissions' = 0
    /\ completionFences' = 0
    /\ queueItems' = 0
    /\ invalidatedItems' = 0
    /\ oldEpochInvalidated' = FALSE
    /\ UNCHANGED <<lastEpoch, position, targetSteps>>

CacheExact(accepted, pending) ==
    /\ phase = "CACHE_SCAN"
    /\ accepted \in 1..N
    /\ pending \in Tokens
    /\ position + accepted < MaxPosition
    /\ phase' = "RESOLVED"
    /\ cacheChecked' = TRUE
    /\ exactHit' = TRUE
    /\ acceptedCount' = accepted
    /\ position' = position + accepted
    /\ pendingValid' = TRUE
    /\ pendingToken' = pending
    /\ oldEpochInvalidated' = TRUE
    /\ UNCHANGED <<epoch, lastEpoch, parentPosition, coldSearchStarted,
                    targetToken, matchingRoute, completionRoute, targetWinner,
                    targetSteps, submissions, completionFences, queueItems,
                    invalidatedItems>>

CacheMiss ==
    /\ phase = "CACHE_SCAN"
    /\ phase' = "COLD_SEARCH"
    /\ cacheChecked' = TRUE
    /\ exactHit' = FALSE
    /\ coldSearchStarted' = TRUE
    /\ UNCHANGED <<epoch, lastEpoch, position, parentPosition, targetToken,
                    matchingRoute, completionRoute, targetWinner,
                    acceptedCount, pendingValid, pendingToken, targetSteps,
                    submissions, completionFences, queueItems,
                    invalidatedItems, oldEpochInvalidated>>

SelectRoutes(target, matching, completedFirst) ==
    /\ phase = "COLD_SEARCH"
    /\ target \in Tokens
    /\ matching \in 0..F
    /\ completedFirst \in Routes
    /\ phase' = IF matching = 0 THEN "RESOLVED" ELSE "TARGET_VERIFY"
    /\ targetToken' = target
    /\ matchingRoute' = matching
    /\ completionRoute' = completedFirst
    /\ targetWinner' = matching
    /\ acceptedCount' = 0
    /\ pendingValid' = (matching = 0)
    /\ pendingToken' = IF matching = 0 THEN target ELSE 0
    /\ submissions' = IF matching = 0 THEN 0 ELSE 1
    /\ completionFences' = 0
    /\ queueItems' = TargetRows * Q
    /\ invalidatedItems' = IF matching = 0 THEN TargetRows * Q ELSE 0
    /\ oldEpochInvalidated' = (matching = 0)
    /\ UNCHANGED <<epoch, lastEpoch, position, parentPosition,
                    cacheChecked, exactHit, coldSearchStarted, targetSteps>>

VerifyWinningRoute(accepted, pending) ==
    /\ phase = "TARGET_VERIFY"
    /\ targetWinner \in Routes
    /\ accepted \in 1..N
    /\ pending \in Tokens
    /\ position + accepted < MaxPosition
    /\ phase' = "RESOLVED"
    /\ acceptedCount' = accepted
    /\ position' = position + accepted
    /\ pendingValid' = TRUE
    /\ pendingToken' = pending
    /\ completionFences' = 1
    /\ invalidatedItems' = TargetRows * Q
    /\ oldEpochInvalidated' = TRUE
    /\ targetSteps' = targetSteps + N
    /\ UNCHANGED <<epoch, lastEpoch, parentPosition, cacheChecked, exactHit,
                    coldSearchStarted, targetToken, matchingRoute,
                    completionRoute, targetWinner, submissions, queueItems>>

ConsumeTargetPending ==
    /\ phase = "RESOLVED"
    /\ pendingValid
    /\ pendingToken \in Tokens
    /\ position < MaxPosition
    /\ phase' = "BOUNDARY"
    /\ lastEpoch' = epoch
    /\ position' = position + 1
    /\ pendingValid' = FALSE
    /\ pendingToken' = 0
    /\ targetSteps' = targetSteps + 1
    /\ UNCHANGED <<epoch, parentPosition, cacheChecked, exactHit,
                    coldSearchStarted, targetToken, matchingRoute,
                    completionRoute, targetWinner, acceptedCount,
                    submissions, completionFences, queueItems,
                    invalidatedItems, oldEpochInvalidated>>

Fail ==
    /\ phase \in {"CACHE_SCAN", "COLD_SEARCH", "TARGET_VERIFY"}
    /\ phase' = "FAILED"
    /\ position' = parentPosition
    /\ acceptedCount' = 0
    /\ pendingValid' = FALSE
    /\ pendingToken' = 0
    /\ completionFences' = submissions
    /\ invalidatedItems' = queueItems
    /\ oldEpochInvalidated' = TRUE
    /\ UNCHANGED <<epoch, lastEpoch, parentPosition, cacheChecked, exactHit,
                    coldSearchStarted, targetToken, matchingRoute,
                    completionRoute, targetWinner, targetSteps, submissions,
                    queueItems>>

TerminalStutter ==
    /\ phase = "FAILED" \/ (phase = "BOUNDARY" /\ epoch = MaxEpoch)
    /\ UNCHANGED vars

Next ==
    \/ BeginEpoch
    \/ \E accepted \in 1..N, pending \in Tokens :
         CacheExact(accepted, pending)
    \/ CacheMiss
    \/ \E target \in Tokens, matching \in 0..F,
          completedFirst \in Routes :
         SelectRoutes(target, matching, completedFirst)
    \/ \E accepted \in 1..N, pending \in Tokens :
         VerifyWinningRoute(accepted, pending)
    \/ ConsumeTargetPending
    \/ Fail
    \/ TerminalStutter

Spec == Init /\ [][Next]_vars

TypeOK ==
    /\ phase \in Phases
    /\ epoch \in 0..MaxEpoch
    /\ lastEpoch \in 0..MaxEpoch
    /\ position \in 0..MaxPosition
    /\ parentPosition \in 0..MaxPosition
    /\ cacheChecked \in BOOLEAN
    /\ exactHit \in BOOLEAN
    /\ coldSearchStarted \in BOOLEAN
    /\ targetToken \in {0} \cup Tokens
    /\ matchingRoute \in 0..F
    /\ completionRoute \in 0..F
    /\ targetWinner \in 0..F
    /\ acceptedCount \in 0..N
    /\ pendingValid \in BOOLEAN
    /\ pendingToken \in {0} \cup Tokens
    /\ targetSteps \in Nat
    /\ submissions \in 0..1
    /\ completionFences \in 0..1
    /\ queueItems \in 0..(TargetRows * Q)
    /\ invalidatedItems \in 0..(TargetRows * Q)
    /\ oldEpochInvalidated \in BOOLEAN

FrontierGeometryBound ==
    /\ TargetRows <= TargetRowCapacity
    /\ ActiveWorkerWidth <= WorkerBudget
    /\ ActiveWorkerWidth <= TargetRows
    /\ TargetRows % ActiveWorkerWidth = 0
    /\ MaxCandidateExpertDemandBytes > 0

CacheFirst ==
    phase \in {"COLD_SEARCH", "TARGET_VERIFY", "RESOLVED"} => cacheChecked

ColdSearchOnlyOnMiss ==
    coldSearchStarted => cacheChecked /\ ~exactHit

NoSearchOnExactHit ==
    exactHit =>
        /\ ~coldSearchStarted
        /\ matchingRoute = 0
        /\ targetWinner = 0
        /\ submissions = 0
        /\ completionFences = 0
        /\ queueItems = 0

TargetSelectsWinner == targetWinner = matchingRoute

CompletionOrderNonAuthoritative ==
    targetWinner = matchingRoute

NoMatchSkipsTarget ==
    phase = "RESOLVED" /\ ~exactHit /\ matchingRoute = 0 =>
        /\ acceptedCount = 0
        /\ submissions = 0
        /\ completionFences = 0
        /\ pendingToken = targetToken

TargetExecutionExact ==
    phase = "RESOLVED" /\ targetWinner \in Routes =>
        /\ submissions = 1
        /\ completionFences = 1
        /\ acceptedCount \in 1..N
        /\ position = parentPosition + acceptedCount

PendingOnlyAfterResolution == pendingValid <=> phase = "RESOLVED"

PositionAuthority ==
    /\ (phase \in {"CACHE_SCAN", "COLD_SEARCH", "TARGET_VERIFY", "FAILED"}
        => position = parentPosition)
    /\ (phase = "RESOLVED" =>
        position = parentPosition + acceptedCount)

AcceptedWidthIsNNotQ == acceptedCount \in 0..N

QueueRunwayBounded ==
    /\ queueItems <= TargetRows * Q
    /\ invalidatedItems <= queueItems

NoB1Fallback == phase # "ORDINARY"

PendingConsumedByTarget ==
    phase = "BOUNDARY" /\ epoch > 0 => targetSteps > 0

OldEpochInvalidatedBeforeCommit ==
    phase = "RESOLVED" => oldEpochInvalidated

OneEpochAtATime ==
    /\ lastEpoch <= epoch
    /\ (phase = "BOUNDARY" => lastEpoch = epoch)
    /\ (phase = "BOUNDARY" /\ epoch > 0 =>
        /\ oldEpochInvalidated
        /\ ~pendingValid)

BoundaryAfterInvalidation ==
    phase = "BOUNDARY" /\ epoch > 0 =>
        /\ oldEpochInvalidated
        /\ lastEpoch = epoch
        /\ ~pendingValid

====
