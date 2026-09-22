---- MODULE SaltPrefillControlProofs ----
EXTENDS SaltPrefillControl, TLAPS

(* Proof-only refresh against the unchanged executable prefill model.
   Strengthening records reachable arithmetic and ownership facts; no model
   action, runtime guard or executable safety invariant is weakened. *)

ReachabilityFacts ==
    /\ (phase \in {"FEED_FORWARD", "RELEASE"} => attempted = released + 1)
    /\ (~published => publishCount = 0)

Inv ==
    /\ TypeOK
    /\ ReachabilityFacts
    /\ ReleaseAccounting
    /\ FinishBeforePublication
    /\ PublicationSafety
    /\ LookaheadIdentity
    /\ LookaheadIsResourceOnly

LEMMA InitEstablishesInv == Init => Inv
BY Z3, ModelBounds
 DEF Init, Inv, TypeOK, ReachabilityFacts, ReleaseAccounting,
     FinishBeforePublication, PublicationSafety, LookaheadIdentity,
     LookaheadIsResourceOnly, Phases, LookaheadStates

LEMMA NextPreservesInv == Inv /\ Next => Inv'
BY Z3, ModelBounds
 DEF Next, BeginChunk, BeginFinal, EmbedOk, EmbedFail,
     AttentionOk, AttentionFail, FeedForwardOk, FeedForwardFail,
     ReleaseLayer, FinishChunk, FinishChunkFail, FinishOk, FinishFail,
     PrepareNone, PrepareOk, PrepareRefused, PrepareFail, PublishOk,
     PublishFail, ClaimLookahead, TerminalStutter, vars,
     Inv, TypeOK, ReachabilityFacts, ReleaseAccounting,
     FinishBeforePublication, PublicationSafety, LookaheadIdentity,
     LookaheadIsResourceOnly, Phases, LookaheadStates

LEMMA StutterPreservesInv == Inv /\ UNCHANGED vars => Inv'
BY Z3 DEF vars, Inv, TypeOK, ReachabilityFacts, ReleaseAccounting,
          FinishBeforePublication, PublicationSafety, LookaheadIdentity,
          LookaheadIsResourceOnly, Phases, LookaheadStates

THEOREM InductiveInvariant == Init /\ [][Next]_vars => []Inv
<1>1. Init => Inv
       BY InitEstablishesInv
<1>2. Inv /\ [Next]_vars => Inv'
       BY NextPreservesInv, StutterPreservesInv
<1>3. QED
       BY <1>1, <1>2, PTL

THEOREM CurrentCoreSafety == Spec => []Inv
BY InductiveInvariant DEF Spec

====
