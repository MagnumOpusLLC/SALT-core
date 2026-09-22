---- MODULE QwenRuntime ----
EXTENDS Integers, Naturals, FiniteSets, Sequences

CONSTANTS NumLayers, RingSize, MaxTokens

QwenConstantsOK ==
       /\ NumLayers \in Nat \ {0}
       /\ RingSize \in Nat \ {0}
       /\ MaxTokens \in Nat
       /\ MaxTokens > RingSize

ASSUME NumLayers \in Nat \ {0}
ASSUME RingSize \in Nat \ {0}
ASSUME MaxTokens \in Nat
ASSUME MaxTokens > RingSize

Phases == {"START", "READY", "EMBED", "LAYERS", "HEAD", "SAMPLE",
           "COMMIT", "DONE", "FAILED"}
TokenValues == 1..3

DigestStep(d, t) == (d * 5 + t) % 97

PushBounded(seq, value) ==
    IF Len(seq) < RingSize
    THEN Append(seq, value)
    ELSE Append(Tail(seq), value)

VARIABLES
    phase,
    position,
    currentToken,
    pendingToken,
    layer,
    ringHistory,
    kvRows,
    linearUpdates,
    outputHistory,
    recentHistory,
    rngCounter,
    runtimeDigest,
    canonicalDigest

vars == <<phase, position, currentToken, pendingToken, layer,
          ringHistory, kvRows, linearUpdates, outputHistory,
          recentHistory, rngCounter, runtimeDigest, canonicalDigest>>

Init ==
    /\ NumLayers \in Nat \ {0}
    /\ RingSize \in Nat \ {0}
    /\ MaxTokens \in Nat
    /\ MaxTokens > RingSize
    /\ phase = "START"
    /\ position = 0
    /\ currentToken = 0
    /\ pendingToken = 0
    /\ layer = 0
    /\ ringHistory = <<>>
    /\ kvRows = {}
    /\ linearUpdates = 0
    /\ outputHistory = <<>>
    /\ recentHistory = <<>>
    /\ rngCounter = 0
    /\ runtimeDigest = 0
    /\ canonicalDigest = 0

Start(first) ==
    /\ phase = "START"
    /\ first \in TokenValues
    /\ phase' = "READY"
    /\ currentToken' = first
    /\ UNCHANGED <<position, pendingToken, layer, ringHistory, kvRows,
                    linearUpdates, outputHistory, recentHistory,
                    rngCounter, runtimeDigest, canonicalDigest>>

BeginToken ==
    /\ phase = "READY"
    /\ position < MaxTokens
    /\ phase' = "EMBED"
    /\ layer' = 0
    /\ UNCHANGED <<position, currentToken, pendingToken, ringHistory,
                    kvRows, linearUpdates, outputHistory, recentHistory,
                    rngCounter, runtimeDigest, canonicalDigest>>

Embed ==
    /\ phase = "EMBED"
    /\ phase' = "LAYERS"
    /\ UNCHANGED <<position, currentToken, pendingToken, layer,
                    ringHistory, kvRows, linearUpdates, outputHistory,
                    recentHistory, rngCounter, runtimeDigest, canonicalDigest>>

RunLayer ==
    /\ phase = "LAYERS"
    /\ IF layer + 1 < NumLayers
       THEN /\ layer' = layer + 1
            /\ phase' = "LAYERS"
       ELSE /\ layer' = layer
            /\ phase' = "HEAD"
    /\ UNCHANGED <<position, currentToken, pendingToken, ringHistory,
                    kvRows, linearUpdates, outputHistory, recentHistory,
                    rngCounter, runtimeDigest, canonicalDigest>>

HeadProjection ==
    /\ phase = "HEAD"
    /\ phase' = "SAMPLE"
    /\ UNCHANGED <<position, currentToken, pendingToken, layer,
                    ringHistory, kvRows, linearUpdates, outputHistory,
                    recentHistory, rngCounter, runtimeDigest, canonicalDigest>>

SampleGreedy ==
    /\ phase = "SAMPLE"
    /\ pendingToken' = (runtimeDigest % 3) + 1
    /\ outputHistory' = Append(outputHistory, pendingToken')
    /\ recentHistory' = PushBounded(recentHistory, pendingToken')
    /\ phase' = "COMMIT"
    /\ UNCHANGED <<position, currentToken, layer, ringHistory, kvRows,
                    linearUpdates, rngCounter, runtimeDigest, canonicalDigest>>

SampleSeeded ==
    /\ phase = "SAMPLE"
    /\ pendingToken' = ((runtimeDigest + rngCounter) % 3) + 1
    /\ outputHistory' = Append(outputHistory, pendingToken')
    /\ recentHistory' = PushBounded(recentHistory, pendingToken')
    /\ rngCounter' = rngCounter + 1
    /\ phase' = "COMMIT"
    /\ UNCHANGED <<position, currentToken, layer, ringHistory, kvRows,
                    linearUpdates, runtimeDigest, canonicalDigest>>

CommitToken ==
    /\ phase = "COMMIT"
    /\ pendingToken \in TokenValues
    /\ position < MaxTokens
    /\ position' = position + 1
    /\ ringHistory' = PushBounded(ringHistory, currentToken)
    /\ kvRows' = kvRows \cup {position'}
    /\ linearUpdates' = linearUpdates + 1
    /\ runtimeDigest' = DigestStep(runtimeDigest, currentToken)
    /\ canonicalDigest' = DigestStep(canonicalDigest, currentToken)
    /\ currentToken' = pendingToken
    /\ pendingToken' = 0
    /\ phase' = IF position' = MaxTokens THEN "DONE" ELSE "READY"
    /\ UNCHANGED <<layer, outputHistory, recentHistory, rngCounter>>

Fail ==
    /\ phase \notin {"COMMIT", "DONE", "FAILED"}
    /\ phase' = "FAILED"
    /\ UNCHANGED <<position, currentToken, pendingToken, layer,
                    ringHistory, kvRows, linearUpdates, outputHistory,
                    recentHistory, rngCounter, runtimeDigest, canonicalDigest>>

TerminalStutter ==
    /\ phase \in {"DONE", "FAILED"}
    /\ UNCHANGED vars

Next ==
    \/ \E first \in TokenValues : Start(first)
    \/ BeginToken
    \/ Embed
    \/ RunLayer
    \/ HeadProjection
    \/ SampleGreedy
    \/ SampleSeeded
    \/ CommitToken
    \/ Fail
    \/ TerminalStutter

Spec == Init /\ [][Next]_vars

TypeOK ==
    /\ phase \in Phases
    /\ position \in 0..MaxTokens
    /\ currentToken \in {0} \cup TokenValues
    /\ pendingToken \in {0} \cup TokenValues
    /\ layer \in 0..(NumLayers - 1)
    /\ ringHistory \in Seq(TokenValues)
    /\ kvRows \subseteq 1..MaxTokens
    /\ linearUpdates \in 0..MaxTokens
    /\ outputHistory \in Seq(TokenValues)
    /\ recentHistory \in Seq(TokenValues)
    /\ rngCounter \in Nat
    /\ runtimeDigest \in 0..96
    /\ canonicalDigest \in 0..96

DigestExact == runtimeDigest = canonicalDigest

HistoryPosition ==
    Len(outputHistory) = position + IF phase = "COMMIT" THEN 1 ELSE 0

KvRowsExact == kvRows = 1..position

LinearStateExact == linearUpdates = position

RingPositionExact ==
    Len(ringHistory) = IF position < RingSize THEN position ELSE RingSize

RingCausal ==
    /\ Len(ringHistory) <= RingSize
    /\ (position >= RingSize => Len(ringHistory) = RingSize)
    /\ \A i \in 1..Len(ringHistory) : ringHistory[i] \in TokenValues

RecentBounded == Len(recentHistory) <= RingSize

PendingCoherent ==
    /\ (phase = "COMMIT" => pendingToken \in TokenValues)
    /\ (phase # "COMMIT" => pendingToken = 0)
    /\ (phase = "DONE" => position = MaxTokens)

QwenRuntimeInvariants ==
    /\ TypeOK
    /\ DigestExact
    /\ HistoryPosition
    /\ KvRowsExact
    /\ LinearStateExact
    /\ RingPositionExact
    /\ RingCausal
    /\ RecentBounded
    /\ PendingCoherent

====
