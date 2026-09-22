---- MODULE GemmaRuntimeProofs ----
EXTENDS GemmaRuntime, TLAPS, SequenceTheorems

(* Latest Gemma 4 runtime proof at ea564711. The executable model is imported
   directly so proof obligations cannot drift from GemmaRuntime.tla.

   TLC checks TypeOK for the bounded NumLayers/MindsetEnd/MaxTokens instance.
   TLAPS certifies the six semantic invariants in SemanticInv for arbitrary
   constants satisfying GemmaConstantsOK. SnapshotCanonical,
   VerifiedDigestExact, HistoryTyped, and NumericTyped are inductive
   strengthening facts required by Commit, Fail, Append, and integer
   arithmetic; they are proof support, not weaker replacements for the
   semantic claims. Every transition obligation is discharged explicitly by
   Z3. *)

SemanticInv ==
    /\ LayerKvAligned
    /\ MindsetPreserved
    /\ CommittedStateExact
    /\ HistoryPosition
    /\ TentativeIsolation
    /\ TerminalCoherent

SnapshotCanonical == snapshotDigest = canonicalDigest

VerifiedDigestExact ==
    phase = "VERIFIED" =>
        tentativeDigest = DigestStep(runtimeDigest, currentToken)

HistoryTyped ==
    /\ outputHistory \in Seq(TokenValues)
    /\ currentToken \in {0} \cup TokenValues
    /\ (mutationStarted => currentToken \in TokenValues)
    /\ (~mutationStarted => currentToken = 0)

NumericTyped ==
    /\ position \in Int
    /\ tentativePosition \in Int
    /\ committedTurns \in Nat

ProofSupport ==
    /\ SnapshotCanonical
    /\ VerifiedDigestExact
    /\ HistoryTyped
    /\ NumericTyped

Inv ==
    /\ SemanticInv
    /\ ProofSupport

LEMMA LayerMapAt ==
    \A p : \A l \in Layers : [i \in Layers |-> p][l] = p
BY Z3

LEMMA EmptyHistoryFacts ==
    /\ <<>> \in Seq(TokenValues)
    /\ Len(<<>>) = 0
BY Z3, EmptySeq

LEMMA AppendHistoryFacts ==
    \A seq \in Seq(TokenValues), token \in TokenValues :
        /\ Append(seq, token) \in Seq(TokenValues)
        /\ Len(Append(seq, token)) = Len(seq) + 1
BY Z3, AppendProperties

LEMMA GemmaConstantTypes ==
    /\ NumLayers \in Nat \ {0}
    /\ MindsetEnd \in Nat
    /\ MaxTokens \in Nat \ {0}
BY Z3, GemmaConstantsOK DEF GemmaConstantsOK

LEMMA InitLayerKvAligned == Init => LayerKvAligned
BY Z3, LayerMapAt DEF Init, LayerKvAligned

LEMMA InitMindsetPreserved == Init => MindsetPreserved
BY Z3, GemmaConstantTypes DEF Init, MindsetPreserved

LEMMA InitCommittedStateExact == Init => CommittedStateExact
BY Z3 DEF Init, CommittedStateExact

LEMMA InitHistoryPosition == Init => HistoryPosition
BY Z3, GemmaConstantTypes, EmptyHistoryFacts DEF Init, HistoryPosition

LEMMA InitTentativeIsolation == Init => TentativeIsolation
BY Z3 DEF Init, TentativeIsolation

LEMMA InitTerminalCoherent == Init => TerminalCoherent
BY Z3 DEF Init, TerminalCoherent

LEMMA InitSnapshotCanonical == Init => SnapshotCanonical
BY Z3 DEF Init, SnapshotCanonical

LEMMA InitVerifiedDigestExact == Init => VerifiedDigestExact
BY Z3 DEF Init, VerifiedDigestExact

LEMMA InitHistoryTyped == Init => HistoryTyped
BY Z3, EmptyHistoryFacts DEF Init, HistoryTyped, TokenValues

LEMMA InitNumericTyped == Init => NumericTyped
BY Z3, GemmaConstantTypes DEF Init, NumericTyped

LEMMA InitEstablishesInv == Init => Inv
BY Z3, InitLayerKvAligned, InitMindsetPreserved,
       InitCommittedStateExact, InitHistoryPosition,
       InitTentativeIsolation, InitTerminalCoherent,
       InitSnapshotCanonical, InitVerifiedDigestExact, InitHistoryTyped,
       InitNumericTyped
   DEF Inv, SemanticInv, ProofSupport

LEMMA StartupPreservesInv == Inv /\ Startup => Inv'
BY Z3 DEF Inv, SemanticInv, ProofSupport, SnapshotCanonical,
          VerifiedDigestExact, HistoryTyped, NumericTyped, LayerKvAligned,
          MindsetPreserved, CommittedStateExact, HistoryPosition,
          TentativeIsolation, TerminalCoherent, Startup

LEMMA BeginStepPreservesInv ==
    \A token \in TokenValues : Inv /\ BeginStep(token) => Inv'
BY Z3 DEF Inv, SemanticInv, ProofSupport, SnapshotCanonical,
          VerifiedDigestExact, HistoryTyped, NumericTyped, LayerKvAligned,
          MindsetPreserved, CommittedStateExact, HistoryPosition,
          TentativeIsolation, TerminalCoherent, BeginStep

LEMMA ComputeOkPreservesInv == Inv /\ ComputeOk => Inv'
BY Z3 DEF Inv, SemanticInv, ProofSupport, SnapshotCanonical,
          VerifiedDigestExact, HistoryTyped, NumericTyped, LayerKvAligned,
          MindsetPreserved, CommittedStateExact, HistoryPosition,
          TentativeIsolation, TerminalCoherent, ComputeOk

LEMMA CommitLayerKvAligned == Inv /\ Commit => LayerKvAligned'
BY Z3, LayerMapAt DEF Inv, SemanticInv, LayerKvAligned, Commit

LEMMA CommitMindsetPreserved == Inv /\ Commit => MindsetPreserved'
BY Z3, GemmaConstantTypes
   DEF Inv, SemanticInv, ProofSupport, NumericTyped, MindsetPreserved, Commit

LEMMA CommitCommittedStateExact == Inv /\ Commit => CommittedStateExact'
BY Z3 DEF Inv, SemanticInv, ProofSupport, VerifiedDigestExact,
          CommittedStateExact, Commit

LEMMA CommitHistoryPosition == Inv /\ Commit => HistoryPosition'
BY Z3, GemmaConstantTypes, AppendHistoryFacts
   DEF Inv, SemanticInv, ProofSupport, HistoryTyped, NumericTyped,
       HistoryPosition, Commit

LEMMA CommitTentativeIsolation == Inv /\ Commit => TentativeIsolation'
BY Z3 DEF Inv, SemanticInv, TentativeIsolation, Commit

LEMMA CommitTerminalCoherent == Inv /\ Commit => TerminalCoherent'
BY Z3 DEF Inv, SemanticInv, TerminalCoherent, Commit

LEMMA CommitSnapshotCanonical == Inv /\ Commit => SnapshotCanonical'
BY Z3 DEF Inv, SemanticInv, ProofSupport, SnapshotCanonical,
          VerifiedDigestExact, CommittedStateExact, Commit

LEMMA CommitVerifiedDigestExact == Inv /\ Commit => VerifiedDigestExact'
BY Z3 DEF Inv, VerifiedDigestExact, Commit

LEMMA CommitHistoryTyped == Inv /\ Commit => HistoryTyped'
BY Z3, AppendHistoryFacts
   DEF Inv, ProofSupport, HistoryTyped, Commit, TokenValues

LEMMA CommitNumericTyped == Inv /\ Commit => NumericTyped'
BY Z3 DEF Inv, ProofSupport, NumericTyped, Commit

LEMMA CommitPreservesInv == Inv /\ Commit => Inv'
BY Z3, CommitLayerKvAligned, CommitMindsetPreserved,
       CommitCommittedStateExact, CommitHistoryPosition,
       CommitTentativeIsolation, CommitTerminalCoherent,
       CommitSnapshotCanonical, CommitVerifiedDigestExact,
       CommitHistoryTyped, CommitNumericTyped
   DEF Inv, SemanticInv, ProofSupport

LEMMA RollbackPreservesInv == Inv /\ Rollback => Inv'
BY Z3 DEF Inv, SemanticInv, ProofSupport, SnapshotCanonical,
          VerifiedDigestExact, HistoryTyped, NumericTyped, LayerKvAligned,
          MindsetPreserved, CommittedStateExact, HistoryPosition,
          TentativeIsolation, TerminalCoherent, Rollback

LEMMA BeginClearFactsPreservesInv == Inv /\ BeginClearFacts => Inv'
BY Z3 DEF Inv, SemanticInv, ProofSupport, SnapshotCanonical,
          VerifiedDigestExact, HistoryTyped, NumericTyped, LayerKvAligned,
          MindsetPreserved, CommittedStateExact, HistoryPosition,
          TentativeIsolation, TerminalCoherent, BeginClearFacts

LEMMA ClearFactsLayerKvAligned == Inv /\ ClearFactsDone => LayerKvAligned'
BY Z3, LayerMapAt DEF LayerKvAligned, ClearFactsDone

LEMMA ClearFactsMindsetPreserved == Inv /\ ClearFactsDone => MindsetPreserved'
BY Z3, GemmaConstantTypes
   DEF Inv, SemanticInv, MindsetPreserved, ClearFactsDone

LEMMA ClearFactsCommittedStateExact ==
    Inv /\ ClearFactsDone => CommittedStateExact'
BY Z3 DEF CommittedStateExact, ClearFactsDone

LEMMA ClearFactsHistoryPosition == Inv /\ ClearFactsDone => HistoryPosition'
BY Z3, GemmaConstantTypes, EmptyHistoryFacts
   DEF HistoryPosition, ClearFactsDone

LEMMA ClearFactsTentativeIsolation ==
    Inv /\ ClearFactsDone => TentativeIsolation'
BY Z3 DEF TentativeIsolation, ClearFactsDone

LEMMA ClearFactsTerminalCoherent == Inv /\ ClearFactsDone => TerminalCoherent'
BY Z3 DEF TerminalCoherent, ClearFactsDone

LEMMA ClearFactsSnapshotCanonical ==
    Inv /\ ClearFactsDone => SnapshotCanonical'
BY Z3 DEF SnapshotCanonical, ClearFactsDone

LEMMA ClearFactsVerifiedDigestExact ==
    Inv /\ ClearFactsDone => VerifiedDigestExact'
BY Z3 DEF VerifiedDigestExact, ClearFactsDone

LEMMA ClearFactsHistoryTyped == Inv /\ ClearFactsDone => HistoryTyped'
BY Z3, EmptyHistoryFacts DEF HistoryTyped, ClearFactsDone, TokenValues

LEMMA ClearFactsNumericTyped == Inv /\ ClearFactsDone => NumericTyped'
BY Z3, GemmaConstantTypes DEF NumericTyped, ClearFactsDone

LEMMA ClearFactsDonePreservesInv == Inv /\ ClearFactsDone => Inv'
BY Z3, ClearFactsLayerKvAligned, ClearFactsMindsetPreserved,
       ClearFactsCommittedStateExact, ClearFactsHistoryPosition,
       ClearFactsTentativeIsolation, ClearFactsTerminalCoherent,
       ClearFactsSnapshotCanonical, ClearFactsVerifiedDigestExact,
       ClearFactsHistoryTyped, ClearFactsNumericTyped
   DEF Inv, SemanticInv, ProofSupport

LEMMA FailPreservesInv == Inv /\ Fail => Inv'
BY Z3 DEF Inv, SemanticInv, ProofSupport, SnapshotCanonical,
          VerifiedDigestExact, HistoryTyped, NumericTyped, LayerKvAligned,
          MindsetPreserved, CommittedStateExact, HistoryPosition,
          TentativeIsolation, TerminalCoherent, Fail

LEMMA TerminalStutterPreservesInv == Inv /\ TerminalStutter => Inv'
BY Z3 DEF Inv, SemanticInv, ProofSupport, SnapshotCanonical,
          VerifiedDigestExact, HistoryTyped, NumericTyped, LayerKvAligned,
          MindsetPreserved, CommittedStateExact, HistoryPosition,
          TentativeIsolation, TerminalCoherent, TerminalStutter, vars

LEMMA NextPreservesInv == Inv /\ Next => Inv'
BY Z3, StartupPreservesInv, BeginStepPreservesInv, ComputeOkPreservesInv,
       CommitPreservesInv, RollbackPreservesInv,
       BeginClearFactsPreservesInv, ClearFactsDonePreservesInv,
       FailPreservesInv, TerminalStutterPreservesInv
   DEF Next

LEMMA StutterPreservesInv == Inv /\ UNCHANGED vars => Inv'
BY Z3 DEF Inv, SemanticInv, ProofSupport, SnapshotCanonical,
          VerifiedDigestExact, HistoryTyped, NumericTyped, LayerKvAligned,
          MindsetPreserved, CommittedStateExact, HistoryPosition,
          TentativeIsolation, TerminalCoherent, vars

THEOREM InductiveInvariant == Inv /\ [Next]_vars => Inv'
BY Z3, NextPreservesInv, StutterPreservesInv DEF vars

THEOREM GemmaRuntimeProofInvariant == Spec => []Inv
<1>1. Init => Inv
  BY Z3, InitEstablishesInv
<1>2. Inv /\ [Next]_vars => Inv'
  BY Z3, InductiveInvariant
<1>3. QED
  BY <1>1, <1>2, PTL DEF Spec

THEOREM GemmaRuntimeSemanticSafety == Spec => []SemanticInv
BY GemmaRuntimeProofInvariant, PTL DEF Inv

THEOREM LayerKvAlignedSafety == Spec => []LayerKvAligned
BY GemmaRuntimeSemanticSafety, PTL DEF SemanticInv

THEOREM MindsetPreservedSafety == Spec => []MindsetPreserved
BY GemmaRuntimeSemanticSafety, PTL DEF SemanticInv

THEOREM CommittedStateExactSafety == Spec => []CommittedStateExact
BY GemmaRuntimeSemanticSafety, PTL DEF SemanticInv

THEOREM HistoryPositionSafety == Spec => []HistoryPosition
BY GemmaRuntimeSemanticSafety, PTL DEF SemanticInv

THEOREM TentativeIsolationSafety == Spec => []TentativeIsolation
BY GemmaRuntimeSemanticSafety, PTL DEF SemanticInv

THEOREM TerminalCoherentSafety == Spec => []TerminalCoherent
BY GemmaRuntimeSemanticSafety, PTL DEF SemanticInv

====
