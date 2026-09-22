---- MODULE QwenRuntimeProofs ----
EXTENDS QwenRuntime, TLAPS, SequenceTheorems

(* The executable module is imported directly, and PushBounded is unfolded in
   every sampling/commit preservation obligation. There is no uninterpreted
   replacement for the ring/history transition.

   tlapm 1.5.0 is not asked to discharge the complete TypeOK set/subset
   conjunction as one theorem. Instead it certifies the typed control and
   history seam below plus every semantic invariant. The original TypeOK
   remains an explicit bounded TLC invariant. *)

ActivePhases == {"READY", "EMBED", "LAYERS", "HEAD", "SAMPLE",
                  "COMMIT", "DONE"}

ControlType ==
    /\ NumLayers \in Nat \ {0}
    /\ RingSize \in Nat \ {0}
    /\ MaxTokens \in Nat
    /\ MaxTokens > RingSize
    /\ phase \in Phases
    /\ position \in 0..MaxTokens
    /\ currentToken \in {0} \cup TokenValues
    /\ layer \in 0..(NumLayers - 1)
    /\ rngCounter \in Nat
    /\ runtimeDigest \in 0..96
    /\ canonicalDigest \in 0..96

HistoryTyping ==
    /\ ringHistory \in Seq(TokenValues)
    /\ outputHistory \in Seq(TokenValues)
    /\ recentHistory \in Seq(TokenValues)

ActiveToken == phase \in ActivePhases => currentToken \in TokenValues

ProofSupport == ControlType /\ HistoryTyping /\ ActiveToken

Inv ==
    /\ ProofSupport
    /\ DigestExact
    /\ HistoryPosition
    /\ KvRowsExact
    /\ LinearStateExact
    /\ RingPositionExact
    /\ RingCausal
    /\ RecentBounded
    /\ PendingCoherent

LEMMA InitEstablishesInv == Init => Inv
<1>1. Init => ControlType
  BY Z3 DEF Init, ControlType, Phases, TokenValues
<1>2. Init => HistoryTyping
  BY DEF Init, HistoryTyping, TokenValues
<1>3. Init => ActiveToken
  BY Z3 DEF Init, ActiveToken, ActivePhases, TokenValues
<1>4. Init => DigestExact
  BY Z3 DEF Init, DigestExact
<1>5. Init => HistoryPosition
  BY Z3 DEF Init, HistoryPosition
<1>6. Init => KvRowsExact
  BY Z3 DEF Init, KvRowsExact
<1>7. Init => LinearStateExact
  BY Z3 DEF Init, LinearStateExact
<1>8. Init => RingPositionExact
  BY Z3 DEF Init, RingPositionExact
<1>9. Init => Len(ringHistory) <= RingSize
  BY DEF Init
<1>10. Init =>
       (position >= RingSize => Len(ringHistory) = RingSize)
  BY Z3 DEF Init
<1>11. Init => \A i \in 1..Len(ringHistory) :
        ringHistory[i] \in TokenValues
  BY Z3, EmptySeq DEF Init, TokenValues
<1>12. Init => RingCausal
  BY <1>9, <1>10, <1>11 DEF RingCausal
<1>13. Init => RecentBounded
  BY DEF Init, RecentBounded
<1>14. Init => PendingCoherent
  BY Z3 DEF Init, PendingCoherent
<1>15. QED
  BY <1>1, <1>2, <1>3, <1>4, <1>5, <1>6, <1>7, <1>8,
     <1>12, <1>13, <1>14
  DEF Inv, ProofSupport

LEMMA StartPreservesInv ==
  \A first \in TokenValues : Inv /\ Start(first) => Inv'
<1>1. SUFFICES ASSUME NEW first \in TokenValues
               PROVE Inv /\ Start(first) => Inv'
  OBVIOUS
<1>2. QED
  BY Z3 DEF Inv, ProofSupport, ControlType, HistoryTyping, ActiveToken,
            ActivePhases, DigestExact, HistoryPosition, KvRowsExact,
            LinearStateExact, RingPositionExact, RingCausal, RecentBounded, PendingCoherent,
            Start, Phases, TokenValues

LEMMA BeginTokenPreservesInv == Inv /\ BeginToken => Inv'
BY Z3 DEF Inv, ProofSupport, ControlType, HistoryTyping, ActiveToken,
          ActivePhases, DigestExact, HistoryPosition, KvRowsExact,
          LinearStateExact, RingPositionExact, RingCausal, RecentBounded, PendingCoherent,
          BeginToken, Phases, TokenValues

LEMMA EmbedPreservesInv == Inv /\ Embed => Inv'
BY Z3 DEF Inv, ProofSupport, ControlType, HistoryTyping, ActiveToken,
          ActivePhases, DigestExact, HistoryPosition, KvRowsExact,
          LinearStateExact, RingPositionExact, RingCausal, RecentBounded, PendingCoherent,
          Embed, Phases, TokenValues

LEMMA RunLayerPreservesInv == Inv /\ RunLayer => Inv'
BY Z3 DEF Inv, ProofSupport, ControlType, HistoryTyping, ActiveToken,
          ActivePhases, DigestExact, HistoryPosition, KvRowsExact,
          LinearStateExact, RingPositionExact, RingCausal, RecentBounded, PendingCoherent,
          RunLayer, Phases, TokenValues

LEMMA HeadProjectionPreservesInv == Inv /\ HeadProjection => Inv'
BY Z3 DEF Inv, ProofSupport, ControlType, HistoryTyping, ActiveToken,
          ActivePhases, DigestExact, HistoryPosition, KvRowsExact,
          LinearStateExact, RingPositionExact, RingCausal, RecentBounded, PendingCoherent,
          HeadProjection, Phases, TokenValues

LEMMA SampleGreedyPreservesInv == Inv /\ SampleGreedy => Inv'
<1>1. Inv /\ SampleGreedy => ControlType'
  BY Z3 DEF Inv, ProofSupport, ControlType, SampleGreedy, Phases,
            TokenValues
<1>2. Inv /\ SampleGreedy => HistoryTyping'
  BY Z3 DEF Inv, ProofSupport, ControlType, HistoryTyping, SampleGreedy,
            PushBounded, TokenValues
<1>3. Inv /\ SampleGreedy => ActiveToken'
  BY Z3 DEF Inv, ProofSupport, ControlType, ActiveToken, ActivePhases,
            SampleGreedy, TokenValues
<1>4. Inv /\ SampleGreedy => DigestExact'
  BY Z3 DEF Inv, DigestExact, SampleGreedy
<1>5. Inv /\ SampleGreedy => HistoryPosition'
  BY Z3 DEF Inv, ProofSupport, ControlType, HistoryTyping,
            HistoryPosition, SampleGreedy
<1>6. Inv /\ SampleGreedy => KvRowsExact'
  BY Z3 DEF Inv, KvRowsExact, SampleGreedy
<1>7. Inv /\ SampleGreedy => LinearStateExact'
  BY Z3 DEF Inv, LinearStateExact, SampleGreedy
<1>8. Inv /\ SampleGreedy => RingPositionExact'
  BY Z3 DEF Inv, RingPositionExact, SampleGreedy
<1>9. Inv /\ SampleGreedy => RingCausal'
  BY Z3 DEF Inv, RingCausal, SampleGreedy
<1>10. Inv /\ SampleGreedy => RecentBounded'
  BY Z3 DEF Inv, ProofSupport, ControlType, HistoryTyping, RecentBounded,
            SampleGreedy, PushBounded, TokenValues
<1>11. Inv /\ SampleGreedy => PendingCoherent'
  BY Z3 DEF Inv, ProofSupport, ControlType, PendingCoherent,
            SampleGreedy, TokenValues
<1>12. QED
  BY <1>1, <1>2, <1>3, <1>4, <1>5, <1>6, <1>7, <1>8,
     <1>9, <1>10, <1>11
  DEF Inv, ProofSupport

LEMMA SampleSeededPreservesInv == Inv /\ SampleSeeded => Inv'
<1>1. Inv /\ SampleSeeded => ControlType'
  BY Z3 DEF Inv, ProofSupport, ControlType, SampleSeeded, Phases,
            TokenValues
<1>2. Inv /\ SampleSeeded => HistoryTyping'
  BY Z3 DEF Inv, ProofSupport, ControlType, HistoryTyping, SampleSeeded,
            PushBounded, TokenValues
<1>3. Inv /\ SampleSeeded => ActiveToken'
  BY Z3 DEF Inv, ProofSupport, ControlType, ActiveToken, ActivePhases,
            SampleSeeded, TokenValues
<1>4. Inv /\ SampleSeeded => DigestExact'
  BY Z3 DEF Inv, DigestExact, SampleSeeded
<1>5. Inv /\ SampleSeeded => HistoryPosition'
  BY Z3 DEF Inv, ProofSupport, ControlType, HistoryTyping,
            HistoryPosition, SampleSeeded
<1>6. Inv /\ SampleSeeded => KvRowsExact'
  BY Z3 DEF Inv, KvRowsExact, SampleSeeded
<1>7. Inv /\ SampleSeeded => LinearStateExact'
  BY Z3 DEF Inv, LinearStateExact, SampleSeeded
<1>8. Inv /\ SampleSeeded => RingPositionExact'
  BY Z3 DEF Inv, RingPositionExact, SampleSeeded
<1>9. Inv /\ SampleSeeded => RingCausal'
  BY Z3 DEF Inv, RingCausal, SampleSeeded
<1>10. Inv /\ SampleSeeded => RecentBounded'
  BY Z3 DEF Inv, ProofSupport, ControlType, HistoryTyping, RecentBounded,
            SampleSeeded, PushBounded, TokenValues
<1>11. Inv /\ SampleSeeded => PendingCoherent'
  BY Z3 DEF Inv, ProofSupport, ControlType, PendingCoherent,
            SampleSeeded, TokenValues
<1>12. QED
  BY <1>1, <1>2, <1>3, <1>4, <1>5, <1>6, <1>7, <1>8,
     <1>9, <1>10, <1>11
  DEF Inv, ProofSupport

LEMMA CommitTokenPreservesInv == Inv /\ CommitToken => Inv'
<1>1. Inv /\ CommitToken => ControlType'
  BY Z3 DEF Inv, ProofSupport, ControlType, HistoryTyping, ActiveToken,
            ActivePhases, PendingCoherent, CommitToken, DigestStep,
            Phases, TokenValues
<1>2. Inv /\ CommitToken => HistoryTyping'
  BY Z3 DEF Inv, ProofSupport, ControlType, HistoryTyping, ActiveToken,
            ActivePhases, RingPositionExact, RingCausal, CommitToken,
            PushBounded, TokenValues
<1>3. Inv /\ CommitToken => ActiveToken'
  BY Z3 DEF Inv, ProofSupport, ControlType, HistoryTyping, ActiveToken,
            ActivePhases, PendingCoherent, CommitToken, TokenValues
<1>4. Inv /\ CommitToken => DigestExact'
  BY Z3 DEF Inv, DigestExact, CommitToken, DigestStep
<1>5. Inv /\ CommitToken => HistoryPosition'
  BY Z3 DEF Inv, ProofSupport, ControlType, HistoryTyping,
            HistoryPosition, CommitToken, Phases, TokenValues
<1>6. Inv /\ CommitToken => KvRowsExact'
  BY Z3 DEF Inv, ProofSupport, ControlType, KvRowsExact, CommitToken
<1>7. Inv /\ CommitToken => LinearStateExact'
  BY Z3 DEF Inv, LinearStateExact, CommitToken
<1>8. Inv /\ CommitToken => RingPositionExact'
  BY Z3 DEF Inv, ProofSupport, ControlType, HistoryTyping, ActiveToken,
            ActivePhases, RingPositionExact, RingCausal, CommitToken,
            PushBounded
<1>9. Inv /\ CommitToken => Len(ringHistory') <= RingSize
  BY Z3 DEF Inv, ProofSupport, ControlType, HistoryTyping, RingPositionExact,
            RingCausal, ActiveToken, ActivePhases, CommitToken, PushBounded
<1>10. Inv /\ CommitToken =>
        (position' >= RingSize => Len(ringHistory') = RingSize)
  BY Z3 DEF Inv, ProofSupport, ControlType, HistoryTyping, RingPositionExact,
            RingCausal, ActiveToken, ActivePhases, CommitToken, PushBounded
<1>11. Inv /\ CommitToken =>
        \A i \in 1..Len(ringHistory') : ringHistory'[i] \in TokenValues
  BY Z3 DEF Inv, ProofSupport, ControlType, HistoryTyping, RingPositionExact,
            RingCausal, ActiveToken, ActivePhases, CommitToken, PushBounded,
            TokenValues
<1>12. Inv /\ CommitToken => RingCausal'
  BY <1>9, <1>10, <1>11 DEF RingCausal
<1>13. Inv /\ CommitToken => RecentBounded'
  BY Z3 DEF Inv, RecentBounded, CommitToken
<1>14. Inv /\ CommitToken => PendingCoherent'
  BY Z3 DEF Inv, ProofSupport, ControlType, PendingCoherent, CommitToken,
            TokenValues
<1>15. QED
  BY <1>1, <1>2, <1>3, <1>4, <1>5, <1>6, <1>7, <1>8, <1>12,
     <1>13, <1>14
  DEF Inv, ProofSupport

LEMMA FailPreservesInv == Inv /\ Fail => Inv'
BY Z3 DEF Inv, ProofSupport, ControlType, HistoryTyping, ActiveToken,
          ActivePhases, DigestExact, HistoryPosition, KvRowsExact,
          LinearStateExact, RingPositionExact, RingCausal, RecentBounded, PendingCoherent,
          Fail, Phases, TokenValues

LEMMA TerminalStutterPreservesInv == Inv /\ TerminalStutter => Inv'
BY Z3 DEF Inv, ProofSupport, ControlType, HistoryTyping, ActiveToken,
          ActivePhases, DigestExact, HistoryPosition, KvRowsExact,
          LinearStateExact, RingPositionExact, RingCausal, RecentBounded, PendingCoherent,
          TerminalStutter, Phases, TokenValues, vars

LEMMA NextPreservesInv == Inv /\ Next => Inv'
BY Z3, StartPreservesInv, BeginTokenPreservesInv, EmbedPreservesInv,
       RunLayerPreservesInv, HeadProjectionPreservesInv,
       SampleGreedyPreservesInv, SampleSeededPreservesInv,
       CommitTokenPreservesInv, FailPreservesInv,
       TerminalStutterPreservesInv
   DEF Next

LEMMA StutterPreservesInv == Inv /\ UNCHANGED vars => Inv'
BY Z3 DEF Inv, ProofSupport, ControlType, HistoryTyping, ActiveToken,
          ActivePhases, DigestExact, HistoryPosition, KvRowsExact,
          LinearStateExact, RingPositionExact, RingCausal, RecentBounded, PendingCoherent, vars

THEOREM InductiveInvariant == Inv /\ [Next]_vars => Inv'
BY Z3, NextPreservesInv, StutterPreservesInv DEF vars

THEOREM QwenRuntimeSafety == Spec => []Inv
<1>1. Init => Inv
  BY InitEstablishesInv
<1>2. Inv /\ [Next]_vars => Inv'
  BY InductiveInvariant
<1>3. QED
  BY <1>1, <1>2, PTL DEF Spec

====
