---- MODULE SaltPrefillControl ----
EXTENDS Integers, Naturals

CONSTANTS
    NumLayers,
    HiddenWidth,
    ScalarBytes,
    MaxTokens,
    MaxBatch,
    ScratchBudgetBytes

ASSUME ModelBounds ==
    /\ NumLayers \in Nat \ {0}
    /\ HiddenWidth \in Nat \ {0}
    /\ ScalarBytes \in Nat \ {0}
    /\ MaxTokens \in Nat \ {0}
    /\ MaxBatch \in Nat \ {0}
    /\ MaxBatch <= MaxTokens
    /\ ScratchBudgetBytes \in Nat \ {0}
    /\ 2 * MaxBatch * HiddenWidth * ScalarBytes <= ScratchBudgetBytes

Phases == {
    "CHUNK_BEGIN", "EMBED", "ATTENTION", "FEED_FORWARD", "RELEASE",
    "CHUNK_FINISH", "FINAL_FINISH", "NEXT_PREPARE", "PUBLISH",
    "DONE", "FAILED"
}
LookaheadStates == {
    "EMPTY", "PREPARED", "READY", "CLAIMED", "REFUSED", "CANCELLED"
}

VARIABLES
    phase,
    completed,
    batch,
    layer,
    attempted,
    released,
    layerFailed,
    finished,
    lookaheadStatus,
    preparedPosition,
    published,
    publishCount

vars == <<phase, completed, batch, layer, attempted, released,
          layerFailed, finished, lookaheadStatus, preparedPosition,
          published, publishCount>>

Init ==
    /\ phase = "CHUNK_BEGIN"
    /\ completed = 0
    /\ batch = 0
    /\ layer = 0
    /\ attempted = 0
    /\ released = 0
    /\ layerFailed = FALSE
    /\ finished = FALSE
    /\ lookaheadStatus = "EMPTY"
    /\ preparedPosition = 0
    /\ published = FALSE
    /\ publishCount = 0

BeginChunk ==
    /\ phase = "CHUNK_BEGIN"
    /\ completed < MaxTokens
    /\ phase' = "EMBED"
    /\ batch' = IF MaxTokens - completed < MaxBatch
                 THEN MaxTokens - completed ELSE MaxBatch
    /\ layer' = 0
    /\ layerFailed' = FALSE
    /\ UNCHANGED <<completed, attempted, released, finished,
                    lookaheadStatus, preparedPosition, published,
                    publishCount>>

BeginFinal ==
    /\ phase = "CHUNK_BEGIN"
    /\ completed = MaxTokens
    /\ phase' = "FINAL_FINISH"
    /\ UNCHANGED <<completed, batch, layer, attempted, released,
                    layerFailed, finished, lookaheadStatus,
                    preparedPosition, published, publishCount>>

EmbedOk ==
    /\ phase = "EMBED"
    /\ phase' = "ATTENTION"
    /\ UNCHANGED <<completed, batch, layer, attempted, released,
                    layerFailed, finished, lookaheadStatus,
                    preparedPosition, published, publishCount>>

EmbedFail ==
    /\ phase = "EMBED"
    /\ phase' = "FAILED"
    /\ UNCHANGED <<completed, batch, layer, attempted, released,
                    layerFailed, finished, lookaheadStatus,
                    preparedPosition, published, publishCount>>

AttentionOk ==
    /\ phase = "ATTENTION"
    /\ phase' = "FEED_FORWARD"
    /\ attempted' = attempted + 1
    /\ layerFailed' = FALSE
    /\ UNCHANGED <<completed, batch, layer, released, finished,
                    lookaheadStatus, preparedPosition, published,
                    publishCount>>

AttentionFail ==
    /\ phase = "ATTENTION"
    /\ phase' = "RELEASE"
    /\ attempted' = attempted + 1
    /\ layerFailed' = TRUE
    /\ UNCHANGED <<completed, batch, layer, released, finished,
                    lookaheadStatus, preparedPosition, published,
                    publishCount>>

FeedForwardOk ==
    /\ phase = "FEED_FORWARD"
    /\ phase' = "RELEASE"
    /\ layerFailed' = FALSE
    /\ UNCHANGED <<completed, batch, layer, attempted, released,
                    finished, lookaheadStatus, preparedPosition,
                    published, publishCount>>

FeedForwardFail ==
    /\ phase = "FEED_FORWARD"
    /\ phase' = "RELEASE"
    /\ layerFailed' = TRUE
    /\ UNCHANGED <<completed, batch, layer, attempted, released,
                    finished, lookaheadStatus, preparedPosition,
                    published, publishCount>>

ReleaseLayer ==
    /\ phase = "RELEASE"
    /\ released' = released + 1
    /\ IF layerFailed
       THEN /\ phase' = "FAILED"
            /\ layer' = layer
       ELSE IF layer + 1 = NumLayers
            THEN /\ phase' = "CHUNK_FINISH"
                 /\ layer' = layer
            ELSE /\ phase' = "ATTENTION"
                 /\ layer' = layer + 1
    /\ UNCHANGED <<completed, batch, attempted, layerFailed, finished,
                    lookaheadStatus, preparedPosition, published,
                    publishCount>>

FinishChunk ==
    /\ phase = "CHUNK_FINISH"
    /\ completed + batch <= MaxTokens
    /\ phase' = "CHUNK_BEGIN"
    /\ completed' = completed + batch
    /\ batch' = 0
    /\ layer' = 0
    /\ layerFailed' = FALSE
    /\ UNCHANGED <<attempted, released, finished, lookaheadStatus,
                    preparedPosition, published, publishCount>>

FinishChunkFail ==
    /\ phase = "CHUNK_FINISH"
    /\ phase' = "FAILED"
    /\ UNCHANGED <<completed, batch, layer, attempted, released,
                    layerFailed, finished, lookaheadStatus,
                    preparedPosition, published, publishCount>>

FinishOk ==
    /\ phase = "FINAL_FINISH"
    /\ completed = MaxTokens
    /\ phase' = "NEXT_PREPARE"
    /\ finished' = TRUE
    /\ UNCHANGED <<completed, batch, layer, attempted, released,
                    layerFailed, lookaheadStatus, preparedPosition,
                    published, publishCount>>

FinishFail ==
    /\ phase = "FINAL_FINISH"
    /\ phase' = "FAILED"
    /\ UNCHANGED <<completed, batch, layer, attempted, released,
                    layerFailed, finished, lookaheadStatus,
                    preparedPosition, published, publishCount>>

PrepareNone ==
    /\ phase = "NEXT_PREPARE"
    /\ phase' = "PUBLISH"
    /\ lookaheadStatus' = "EMPTY"
    /\ preparedPosition' = 0
    /\ UNCHANGED <<completed, batch, layer, attempted, released,
                    layerFailed, finished, published, publishCount>>

PrepareOk ==
    /\ phase = "NEXT_PREPARE"
    /\ phase' = "PUBLISH"
    /\ lookaheadStatus' = "PREPARED"
    /\ preparedPosition' = MaxTokens
    /\ UNCHANGED <<completed, batch, layer, attempted, released,
                    layerFailed, finished, published, publishCount>>

PrepareRefused ==
    /\ phase = "NEXT_PREPARE"
    /\ phase' = "PUBLISH"
    /\ lookaheadStatus' = "REFUSED"
    /\ preparedPosition' = 0
    /\ UNCHANGED <<completed, batch, layer, attempted, released,
                    layerFailed, finished, published, publishCount>>

PrepareFail ==
    /\ phase = "NEXT_PREPARE"
    /\ phase' = "FAILED"
    /\ lookaheadStatus' = "CANCELLED"
    /\ preparedPosition' = 0
    /\ UNCHANGED <<completed, batch, layer, attempted, released,
                    layerFailed, finished, published, publishCount>>

PublishOk ==
    /\ phase = "PUBLISH"
    /\ finished
    /\ completed = MaxTokens
    /\ ~published
    /\ phase' = "DONE"
    /\ published' = TRUE
    /\ publishCount' = publishCount + 1
    /\ lookaheadStatus' = IF lookaheadStatus = "PREPARED"
                           THEN "READY" ELSE lookaheadStatus
    /\ UNCHANGED <<completed, batch, layer, attempted, released,
                    layerFailed, finished, preparedPosition>>

PublishFail ==
    /\ phase = "PUBLISH"
    /\ phase' = "FAILED"
    /\ lookaheadStatus' = IF lookaheadStatus = "PREPARED"
                           THEN "CANCELLED" ELSE lookaheadStatus
    /\ UNCHANGED <<completed, batch, layer, attempted, released,
                    layerFailed, finished, preparedPosition, published,
                    publishCount>>

ClaimLookahead(matchingIdentity) ==
    /\ matchingIdentity \in BOOLEAN
    /\ phase = "DONE"
    /\ lookaheadStatus = "READY"
    /\ lookaheadStatus' = IF matchingIdentity THEN "CLAIMED" ELSE "CANCELLED"
    /\ UNCHANGED <<phase, completed, batch, layer, attempted, released,
                    layerFailed, finished, preparedPosition, published,
                    publishCount>>

TerminalStutter ==
    /\ phase \in {"DONE", "FAILED"}
    /\ UNCHANGED vars

Next ==
    \/ BeginChunk
    \/ BeginFinal
    \/ EmbedOk
    \/ EmbedFail
    \/ AttentionOk
    \/ AttentionFail
    \/ FeedForwardOk
    \/ FeedForwardFail
    \/ ReleaseLayer
    \/ FinishChunk
    \/ FinishChunkFail
    \/ FinishOk
    \/ FinishFail
    \/ PrepareNone
    \/ PrepareOk
    \/ PrepareRefused
    \/ PrepareFail
    \/ PublishOk
    \/ PublishFail
    \/ \E matchingIdentity \in BOOLEAN : ClaimLookahead(matchingIdentity)
    \/ TerminalStutter

Spec == Init /\ [][Next]_vars

TypeOK ==
    /\ phase \in Phases
    /\ completed \in 0..MaxTokens
    /\ batch \in 0..MaxBatch
    /\ layer \in 0..(NumLayers - 1)
    /\ attempted \in Nat
    /\ released \in Nat
    /\ layerFailed \in BOOLEAN
    /\ finished \in BOOLEAN
    /\ lookaheadStatus \in LookaheadStates
    /\ preparedPosition \in 0..MaxTokens
    /\ published \in BOOLEAN
    /\ publishCount \in 0..1

ReleaseAccounting ==
    /\ released <= attempted
    /\ attempted <= released + 1
    /\ (phase \notin {"FEED_FORWARD", "RELEASE"} => attempted = released)

FinishBeforePublication ==
    /\ (phase \in {"NEXT_PREPARE", "PUBLISH", "DONE"} => finished)
    /\ (finished => completed = MaxTokens)

PublicationSafety ==
    /\ (published => phase = "DONE" /\ finished /\ publishCount = 1)
    /\ (phase = "FAILED" => ~published)
    /\ (phase = "DONE" => published /\ completed = MaxTokens)

LookaheadIdentity ==
    /\ (lookaheadStatus = "PREPARED" =>
           phase = "PUBLISH" /\ preparedPosition = MaxTokens /\ ~published)
    /\ (lookaheadStatus \in {"READY", "CLAIMED"} =>
           phase = "DONE" /\ published /\
           preparedPosition = completed /\ completed = MaxTokens)
    /\ (phase = "FAILED" =>
           lookaheadStatus \notin {"PREPARED", "READY"})

LookaheadIsResourceOnly ==
    /\ (lookaheadStatus \in
           {"PREPARED", "READY", "CLAIMED", "REFUSED", "CANCELLED"}
        => completed = MaxTokens)
    /\ (phase \in {"CHUNK_BEGIN", "EMBED", "ATTENTION", "FEED_FORWARD",
                    "RELEASE", "CHUNK_FINISH", "FINAL_FINISH"} =>
        /\ ~finished
        /\ lookaheadStatus = "EMPTY"
        /\ preparedPosition = 0
        /\ ~published
        /\ publishCount = 0)
    /\ (phase \in {"EMBED", "ATTENTION", "FEED_FORWARD", "RELEASE",
                    "CHUNK_FINISH"} =>
        /\ batch \in 1..MaxBatch
        /\ completed + batch <= MaxTokens)

CoreControlInvariants ==
    /\ ModelBounds
    /\ TypeOK
    /\ ReleaseAccounting
    /\ FinishBeforePublication
    /\ PublicationSafety
    /\ LookaheadIdentity
    /\ LookaheadIsResourceOnly

====
