---- MODULE GemmaRuntime ----
EXTENDS Integers, Naturals, Sequences

CONSTANTS NumLayers, MindsetEnd, MaxTokens

ASSUME GemmaConstantsOK ==
       /\ NumLayers \in Nat \ {0}
       /\ MindsetEnd \in Nat
       /\ MaxTokens \in Nat \ {0}

Phases == {"START", "READY", "TENTATIVE", "VERIFIED", "CLEARING",
           "DONE", "FAILED"}
TokenValues == 1..3
Layers == 1..NumLayers

DigestStep(d, t) == (d * 7 + t) % 101

VARIABLES
    phase,
    position,
    tentativePosition,
    currentToken,
    layerPosition,
    outputHistory,
    committedTurns,
    mutationStarted,
    mindsetDigest,
    runtimeDigest,
    canonicalDigest,
    snapshotDigest,
    tentativeDigest

vars == <<phase, position, tentativePosition, currentToken, layerPosition,
          outputHistory, committedTurns, mutationStarted, mindsetDigest,
          runtimeDigest, canonicalDigest, snapshotDigest, tentativeDigest>>

Init ==
    /\ phase = "START"
    /\ position = MindsetEnd
    /\ tentativePosition = MindsetEnd
    /\ currentToken = 0
    /\ layerPosition = [l \in Layers |-> MindsetEnd]
    /\ outputHistory = <<>>
    /\ committedTurns = 0
    /\ mutationStarted = FALSE
    /\ mindsetDigest = 1
    /\ runtimeDigest = 1
    /\ canonicalDigest = 1
    /\ snapshotDigest = 1
    /\ tentativeDigest = 1

Startup ==
    /\ phase = "START"
    /\ phase' = "READY"
    /\ UNCHANGED <<position, tentativePosition, currentToken, layerPosition,
                    outputHistory, committedTurns, mutationStarted,
                    mindsetDigest, runtimeDigest, canonicalDigest,
                    snapshotDigest, tentativeDigest>>

BeginStep(token) ==
    /\ phase = "READY"
    /\ position < MindsetEnd + MaxTokens
    /\ token \in TokenValues
    /\ phase' = "TENTATIVE"
    /\ currentToken' = token
    /\ tentativePosition' = position + 1
    /\ mutationStarted' = TRUE
    /\ snapshotDigest' = runtimeDigest
    /\ tentativeDigest' = runtimeDigest
    /\ UNCHANGED <<position, layerPosition, outputHistory, committedTurns,
                    mindsetDigest, runtimeDigest, canonicalDigest>>

ComputeOk ==
    /\ phase = "TENTATIVE"
    /\ mutationStarted
    /\ phase' = "VERIFIED"
    /\ tentativeDigest' = DigestStep(runtimeDigest, currentToken)
    /\ UNCHANGED <<position, tentativePosition, currentToken, layerPosition,
                    outputHistory, committedTurns, mutationStarted,
                    mindsetDigest, runtimeDigest, canonicalDigest, snapshotDigest>>

Commit ==
    /\ phase = "VERIFIED"
    /\ mutationStarted
    /\ tentativePosition = position + 1
    /\ position' = tentativePosition
    /\ tentativePosition' = tentativePosition
    /\ layerPosition' = [l \in Layers |-> tentativePosition]
    /\ outputHistory' = Append(outputHistory, currentToken)
    /\ committedTurns' = committedTurns + 1
    /\ runtimeDigest' = tentativeDigest
    /\ canonicalDigest' = DigestStep(canonicalDigest, currentToken)
    /\ snapshotDigest' = tentativeDigest
    /\ currentToken' = 0
    /\ mutationStarted' = FALSE
    /\ phase' = IF position' = MindsetEnd + MaxTokens THEN "DONE" ELSE "READY"
    /\ UNCHANGED <<mindsetDigest, tentativeDigest>>

Rollback ==
    /\ phase \in {"TENTATIVE", "VERIFIED"}
    /\ mutationStarted
    /\ phase' = "READY"
    /\ tentativePosition' = position
    /\ currentToken' = 0
    /\ mutationStarted' = FALSE
    /\ runtimeDigest' = snapshotDigest
    /\ tentativeDigest' = snapshotDigest
    /\ UNCHANGED <<position, layerPosition, outputHistory, committedTurns,
                    mindsetDigest, canonicalDigest, snapshotDigest>>

BeginClearFacts ==
    /\ phase = "READY"
    /\ phase' = "CLEARING"
    /\ UNCHANGED <<position, tentativePosition, currentToken, layerPosition,
                    outputHistory, committedTurns, mutationStarted,
                    mindsetDigest, runtimeDigest, canonicalDigest,
                    snapshotDigest, tentativeDigest>>

ClearFactsDone ==
    /\ phase = "CLEARING"
    /\ phase' = "READY"
    /\ position' = MindsetEnd
    /\ tentativePosition' = MindsetEnd
    /\ currentToken' = 0
    /\ layerPosition' = [l \in Layers |-> MindsetEnd]
    /\ outputHistory' = <<>>
    /\ committedTurns' = 0
    /\ mutationStarted' = FALSE
    /\ runtimeDigest' = mindsetDigest
    /\ canonicalDigest' = mindsetDigest
    /\ snapshotDigest' = mindsetDigest
    /\ tentativeDigest' = mindsetDigest
    /\ UNCHANGED mindsetDigest

Fail ==
    /\ phase \notin {"DONE", "FAILED"}
    /\ phase' = "FAILED"
    /\ tentativePosition' = position
    /\ currentToken' = 0
    /\ mutationStarted' = FALSE
    /\ runtimeDigest' = snapshotDigest
    /\ tentativeDigest' = snapshotDigest
    /\ UNCHANGED <<position, layerPosition, outputHistory, committedTurns,
                    mindsetDigest, canonicalDigest, snapshotDigest>>

TerminalStutter ==
    /\ phase \in {"DONE", "FAILED"}
    /\ UNCHANGED vars

Next ==
    \/ Startup
    \/ \E token \in TokenValues : BeginStep(token)
    \/ ComputeOk
    \/ Commit
    \/ Rollback
    \/ BeginClearFacts
    \/ ClearFactsDone
    \/ Fail
    \/ TerminalStutter

Spec == Init /\ [][Next]_vars

TypeOK ==
    /\ phase \in Phases
    /\ position \in MindsetEnd..(MindsetEnd + MaxTokens)
    /\ tentativePosition \in MindsetEnd..(MindsetEnd + MaxTokens)
    /\ currentToken \in {0} \cup TokenValues
    /\ layerPosition \in [Layers -> MindsetEnd..(MindsetEnd + MaxTokens)]
    /\ outputHistory \in Seq(TokenValues)
    /\ committedTurns \in 0..MaxTokens
    /\ mutationStarted \in BOOLEAN
    /\ mindsetDigest = 1
    /\ runtimeDigest \in 0..100
    /\ canonicalDigest \in 0..100
    /\ snapshotDigest \in 0..100
    /\ tentativeDigest \in 0..100

LayerKvAligned == \A l \in Layers : layerPosition[l] = position

MindsetPreserved == /\ mindsetDigest = 1
                     /\ position >= MindsetEnd

CommittedStateExact == runtimeDigest = canonicalDigest

HistoryPosition ==
    /\ committedTurns = Len(outputHistory)
    /\ position = MindsetEnd + committedTurns

TentativeIsolation ==
    /\ (mutationStarted <=> phase \in {"TENTATIVE", "VERIFIED"})
    /\ (mutationStarted => tentativePosition = position + 1)
    /\ (~mutationStarted => tentativePosition = position)

TerminalCoherent == phase = "DONE" => position = MindsetEnd + MaxTokens

GemmaRuntimeInvariants ==
    /\ TypeOK
    /\ LayerKvAligned
    /\ MindsetPreserved
    /\ CommittedStateExact
    /\ HistoryPosition
    /\ TentativeIsolation
    /\ TerminalCoherent

====
