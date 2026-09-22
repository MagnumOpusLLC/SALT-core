---- MODULE DprTargetRuntimeProofs ----
EXTENDS DprTargetRuntime, TLAPS

(* Refined optional-DPR/base-N/F/Q proof designed against 1576df7. The older
   transmutation ledger remains pinned to 696a06d4373d2a1b696aa549925213d01b9ce4e9
   and is not a current-source seal. The executable model is imported directly.
   TypeOK is split into definitionally identical scalar
   groups to stay below the tlapm 1.5.0 tuple-expansion wall.  Every Init,
   transition, and stutter obligation is discharged explicitly by Z3. DPR is an
   optional lookup/proposal overlay over the base N/F/Q cold-search transition;
   no ordinary/B1 fallback action is admitted. *)

Inv ==
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

LEMMA DprConstantsTyped ==
    /\ MaxPosition \in Nat \ {0}
    /\ MaxHorizon \in Nat \ {0}
    /\ MaxCycles \in Nat \ {0}
BY Z3, DprConstantsOK DEF DprConstantsOK

LEMMA InitEstablishesType == Init => TypeOK
<1>1. Init => phase \in Phases /\ BoolOK(dprEnabled)
  BY Z3, DprConstantsTyped DEF Init, ControlTypeOK, BoolOK, Phases, TokenValues
<1>2. Init => cycle \in 0..MaxCycles
  BY Z3, DprConstantsTyped DEF Init, ControlTypeOK, BoolOK, Phases, TokenValues
<1>3. Init => BoolOK(strictRequested)
  BY Z3, DprConstantsTyped DEF Init, ControlTypeOK, BoolOK, Phases, TokenValues
<1>4. Init => BoolOK(proofComputed)
  BY Z3, DprConstantsTyped DEF Init, ControlTypeOK, BoolOK, Phases, TokenValues
<1>5. Init => position \in 0..MaxPosition
  BY Z3, DprConstantsTyped DEF Init, PositionTypeOK, BoolOK, Phases, TokenValues
<1>6. Init => cycleStart \in 0..MaxPosition
  BY Z3, DprConstantsTyped DEF Init, PositionTypeOK, BoolOK, Phases, TokenValues
<1>7. Init => committedCount \in 0..MaxPosition
  BY Z3, DprConstantsTyped DEF Init, PositionTypeOK, BoolOK, Phases, TokenValues, CommittedIdentityExact
<1>8. Init => targetSteps \in Nat
  BY Z3, DprConstantsTyped DEF Init, PositionTypeOK, BoolOK, Phases, TokenValues
<1>9. Init => BoolOK(parentChecked)
  BY Z3, DprConstantsTyped DEF Init, LookupTypeOK, BoolOK, Phases, TokenValues
<1>10. Init => BoolOK(parentHit)
  BY Z3, DprConstantsTyped DEF Init, LookupTypeOK, BoolOK, Phases, TokenValues
<1>11. Init => BoolOK(nomogramChecked)
  BY Z3, DprConstantsTyped DEF Init, LookupTypeOK, BoolOK, Phases, TokenValues
<1>12. Init => proposalCount \in 0..MaxHorizon
  BY Z3, DprConstantsTyped DEF Init, ProposalTypeOK, BoolOK, Phases, TokenValues
<1>13. Init => acceptedCount \in 0..MaxHorizon
  BY Z3, DprConstantsTyped DEF Init, ProposalTypeOK, BoolOK, Phases, TokenValues
<1>14. Init => BoolOK(pendingValid)
  BY Z3, DprConstantsTyped DEF Init, PendingTypeOK, BoolOK, Phases, TokenValues
<1>15. Init => pendingToken \in {0} \cup TokenValues
  BY Z3, DprConstantsTyped DEF Init, PendingTypeOK, BoolOK, Phases, TokenValues
<1>16. Init => nomogramCommitted = 0
  BY Z3, DprConstantsTyped DEF Init, DigestTypeOK, BoolOK, Phases, TokenValues
<1>17. Init => runtimeDigest \in 0..102
  BY Z3, DprConstantsTyped DEF Init, DigestTypeOK, BoolOK, Phases, TokenValues
<1>18. Init => canonicalDigest \in 0..102
  BY Z3, DprConstantsTyped DEF Init, DigestTypeOK, BoolOK, Phases, TokenValues
<1>19. QED
  BY Z3, <1>1, <1>2, <1>3, <1>4, <1>5, <1>6, <1>7, <1>8, <1>9, <1>10, <1>11, <1>12, <1>13, <1>14, <1>15, <1>16, <1>17, <1>18
     DEF TypeOK, ControlTypeOK, PositionTypeOK, LookupTypeOK, ProposalTypeOK, PendingTypeOK, DigestTypeOK

LEMMA InitEstablishesInv == Init => Inv
BY Z3, InitEstablishesType, DprConstantsTyped
   DEF Init, Inv, CommittedIdentityExact, DigestExact, ParentFirst, ProposalIsObservational, PendingNotStepped, NomogramNeverCommits, StrictProofObservational, OptionalDprOnly, NoB1Fallback, BoolOK, Phases, TokenValues

LEMMA BeginCyclePreservesType ==
    \A strict \in BOOLEAN :
        Inv /\ BeginCycle(strict) => TypeOK'
<1>1. \A strict \in BOOLEAN :
       ControlTypeOK /\ BeginCycle(strict) => phase' \in Phases /\ BoolOK(dprEnabled')
  BY Z3, DprConstantsTyped DEF ControlTypeOK, BeginCycle, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>2. \A strict \in BOOLEAN :
       ControlTypeOK /\ BeginCycle(strict) => cycle' \in 0..MaxCycles
  BY Z3, DprConstantsTyped DEF ControlTypeOK, BeginCycle, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>3. \A strict \in BOOLEAN :
       ControlTypeOK /\ BeginCycle(strict) => BoolOK(strictRequested')
  BY Z3, DprConstantsTyped DEF ControlTypeOK, BeginCycle, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>4. \A strict \in BOOLEAN :
       ControlTypeOK /\ BeginCycle(strict) => BoolOK(proofComputed')
  BY Z3, DprConstantsTyped DEF ControlTypeOK, BeginCycle, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>5. \A strict \in BOOLEAN :
       PositionTypeOK /\ BeginCycle(strict) => position' \in 0..MaxPosition
  BY Z3, DprConstantsTyped DEF PositionTypeOK, BeginCycle, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>6. \A strict \in BOOLEAN :
       PositionTypeOK /\ BeginCycle(strict) => cycleStart' \in 0..MaxPosition
  BY Z3, DprConstantsTyped DEF PositionTypeOK, BeginCycle, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>7. \A strict \in BOOLEAN :
       PositionTypeOK /\ CommittedIdentityExact /\ BeginCycle(strict) => committedCount' \in 0..MaxPosition
  BY Z3, DprConstantsTyped DEF PositionTypeOK, BeginCycle, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues, CommittedIdentityExact
<1>8. \A strict \in BOOLEAN :
       PositionTypeOK /\ BeginCycle(strict) => targetSteps' \in Nat
  BY Z3, DprConstantsTyped DEF PositionTypeOK, BeginCycle, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>9. \A strict \in BOOLEAN :
       LookupTypeOK /\ BeginCycle(strict) => BoolOK(parentChecked')
  BY Z3, DprConstantsTyped DEF LookupTypeOK, BeginCycle, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>10. \A strict \in BOOLEAN :
       LookupTypeOK /\ BeginCycle(strict) => BoolOK(parentHit')
  BY Z3, DprConstantsTyped DEF LookupTypeOK, BeginCycle, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>11. \A strict \in BOOLEAN :
       LookupTypeOK /\ BeginCycle(strict) => BoolOK(nomogramChecked')
  BY Z3, DprConstantsTyped DEF LookupTypeOK, BeginCycle, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>12. \A strict \in BOOLEAN :
       ProposalTypeOK /\ BeginCycle(strict) => proposalCount' \in 0..MaxHorizon
  BY Z3, DprConstantsTyped DEF ProposalTypeOK, BeginCycle, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>13. \A strict \in BOOLEAN :
       ProposalTypeOK /\ BeginCycle(strict) => acceptedCount' \in 0..MaxHorizon
  BY Z3, DprConstantsTyped DEF ProposalTypeOK, BeginCycle, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>14. \A strict \in BOOLEAN :
       PendingTypeOK /\ BeginCycle(strict) => BoolOK(pendingValid')
  BY Z3, DprConstantsTyped DEF PendingTypeOK, BeginCycle, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>15. \A strict \in BOOLEAN :
       PendingTypeOK /\ BeginCycle(strict) => pendingToken' \in {0} \cup TokenValues
  BY Z3, DprConstantsTyped DEF PendingTypeOK, BeginCycle, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>16. \A strict \in BOOLEAN :
       DigestTypeOK /\ BeginCycle(strict) => nomogramCommitted' = 0
  BY Z3, DprConstantsTyped DEF DigestTypeOK, BeginCycle, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>17. \A strict \in BOOLEAN :
       DigestTypeOK /\ BeginCycle(strict) => runtimeDigest' \in 0..102
  BY Z3, DprConstantsTyped DEF DigestTypeOK, BeginCycle, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>18. \A strict \in BOOLEAN :
       DigestTypeOK /\ BeginCycle(strict) => canonicalDigest' \in 0..102
  BY Z3, DprConstantsTyped DEF DigestTypeOK, BeginCycle, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>19. QED
  BY Z3, <1>1, <1>2, <1>3, <1>4, <1>5, <1>6, <1>7, <1>8, <1>9, <1>10, <1>11, <1>12, <1>13, <1>14, <1>15, <1>16, <1>17, <1>18
     DEF Inv, TypeOK, ControlTypeOK, PositionTypeOK, LookupTypeOK, ProposalTypeOK, PendingTypeOK, DigestTypeOK

LEMMA BeginCyclePreservesInv ==
    \A strict \in BOOLEAN :
        Inv /\ BeginCycle(strict) => Inv'
BY Z3, BeginCyclePreservesType, DprConstantsTyped
   DEF Inv, CommittedIdentityExact, DigestExact, ParentFirst, ProposalIsObservational, PendingNotStepped, NomogramNeverCommits, StrictProofObservational, OptionalDprOnly, NoB1Fallback, BeginCycle, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues

LEMMA ParentExactPreservesType ==
    \A horizon \in 1..MaxHorizon :
        Inv /\ ParentExact(horizon) => TypeOK'
<1>1. \A horizon \in 1..MaxHorizon :
       ControlTypeOK /\ ParentExact(horizon) => phase' \in Phases /\ BoolOK(dprEnabled')
  BY Z3, DprConstantsTyped DEF ControlTypeOK, ParentExact, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>2. \A horizon \in 1..MaxHorizon :
       ControlTypeOK /\ ParentExact(horizon) => cycle' \in 0..MaxCycles
  BY Z3, DprConstantsTyped DEF ControlTypeOK, ParentExact, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>3. \A horizon \in 1..MaxHorizon :
       ControlTypeOK /\ ParentExact(horizon) => BoolOK(strictRequested')
  BY Z3, DprConstantsTyped DEF ControlTypeOK, ParentExact, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>4. \A horizon \in 1..MaxHorizon :
       ControlTypeOK /\ ParentExact(horizon) => BoolOK(proofComputed')
  BY Z3, DprConstantsTyped DEF ControlTypeOK, ParentExact, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>5. \A horizon \in 1..MaxHorizon :
       PositionTypeOK /\ ParentExact(horizon) => position' \in 0..MaxPosition
  BY Z3, DprConstantsTyped DEF PositionTypeOK, ParentExact, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>6. \A horizon \in 1..MaxHorizon :
       PositionTypeOK /\ ParentExact(horizon) => cycleStart' \in 0..MaxPosition
  BY Z3, DprConstantsTyped DEF PositionTypeOK, ParentExact, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>7. \A horizon \in 1..MaxHorizon :
       PositionTypeOK /\ CommittedIdentityExact /\ ParentExact(horizon) => committedCount' \in 0..MaxPosition
  BY Z3, DprConstantsTyped DEF PositionTypeOK, ParentExact, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues, CommittedIdentityExact
<1>8. \A horizon \in 1..MaxHorizon :
       PositionTypeOK /\ ParentExact(horizon) => targetSteps' \in Nat
  BY Z3, DprConstantsTyped DEF PositionTypeOK, ParentExact, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>9. \A horizon \in 1..MaxHorizon :
       LookupTypeOK /\ ParentExact(horizon) => BoolOK(parentChecked')
  BY Z3, DprConstantsTyped DEF LookupTypeOK, ParentExact, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>10. \A horizon \in 1..MaxHorizon :
       LookupTypeOK /\ ParentExact(horizon) => BoolOK(parentHit')
  BY Z3, DprConstantsTyped DEF LookupTypeOK, ParentExact, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>11. \A horizon \in 1..MaxHorizon :
       LookupTypeOK /\ ParentExact(horizon) => BoolOK(nomogramChecked')
  BY Z3, DprConstantsTyped DEF LookupTypeOK, ParentExact, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>12. \A horizon \in 1..MaxHorizon :
       ProposalTypeOK /\ ParentExact(horizon) => proposalCount' \in 0..MaxHorizon
  BY Z3, DprConstantsTyped DEF ProposalTypeOK, ParentExact, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>13. \A horizon \in 1..MaxHorizon :
       ProposalTypeOK /\ ParentExact(horizon) => acceptedCount' \in 0..MaxHorizon
  BY Z3, DprConstantsTyped DEF ProposalTypeOK, ParentExact, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>14. \A horizon \in 1..MaxHorizon :
       PendingTypeOK /\ ParentExact(horizon) => BoolOK(pendingValid')
  BY Z3, DprConstantsTyped DEF PendingTypeOK, ParentExact, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>15. \A horizon \in 1..MaxHorizon :
       PendingTypeOK /\ ParentExact(horizon) => pendingToken' \in {0} \cup TokenValues
  BY Z3, DprConstantsTyped DEF PendingTypeOK, ParentExact, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>16. \A horizon \in 1..MaxHorizon :
       DigestTypeOK /\ ParentExact(horizon) => nomogramCommitted' = 0
  BY Z3, DprConstantsTyped DEF DigestTypeOK, ParentExact, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>17. \A horizon \in 1..MaxHorizon :
       DigestTypeOK /\ ParentExact(horizon) => runtimeDigest' \in 0..102
  BY Z3, DprConstantsTyped DEF DigestTypeOK, ParentExact, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>18. \A horizon \in 1..MaxHorizon :
       DigestTypeOK /\ ParentExact(horizon) => canonicalDigest' \in 0..102
  BY Z3, DprConstantsTyped DEF DigestTypeOK, ParentExact, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>19. QED
  BY Z3, <1>1, <1>2, <1>3, <1>4, <1>5, <1>6, <1>7, <1>8, <1>9, <1>10, <1>11, <1>12, <1>13, <1>14, <1>15, <1>16, <1>17, <1>18
     DEF Inv, TypeOK, ControlTypeOK, PositionTypeOK, LookupTypeOK, ProposalTypeOK, PendingTypeOK, DigestTypeOK

LEMMA ParentExactPreservesInv ==
    \A horizon \in 1..MaxHorizon :
        Inv /\ ParentExact(horizon) => Inv'
BY Z3, ParentExactPreservesType, DprConstantsTyped
   DEF Inv, CommittedIdentityExact, DigestExact, ParentFirst, ProposalIsObservational, PendingNotStepped, NomogramNeverCommits, StrictProofObservational, OptionalDprOnly, NoB1Fallback, ParentExact, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues

LEMMA ParentMissPreservesType ==
Inv /\ ParentMiss => TypeOK'
<1>1. ControlTypeOK /\ ParentMiss => phase' \in Phases /\ BoolOK(dprEnabled')
  BY Z3, DprConstantsTyped DEF ControlTypeOK, ParentMiss, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>2. ControlTypeOK /\ ParentMiss => cycle' \in 0..MaxCycles
  BY Z3, DprConstantsTyped DEF ControlTypeOK, ParentMiss, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>3. ControlTypeOK /\ ParentMiss => BoolOK(strictRequested')
  BY Z3, DprConstantsTyped DEF ControlTypeOK, ParentMiss, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>4. ControlTypeOK /\ ParentMiss => BoolOK(proofComputed')
  BY Z3, DprConstantsTyped DEF ControlTypeOK, ParentMiss, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>5. PositionTypeOK /\ ParentMiss => position' \in 0..MaxPosition
  BY Z3, DprConstantsTyped DEF PositionTypeOK, ParentMiss, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>6. PositionTypeOK /\ ParentMiss => cycleStart' \in 0..MaxPosition
  BY Z3, DprConstantsTyped DEF PositionTypeOK, ParentMiss, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>7. PositionTypeOK /\ CommittedIdentityExact /\ ParentMiss => committedCount' \in 0..MaxPosition
  BY Z3, DprConstantsTyped DEF PositionTypeOK, ParentMiss, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues, CommittedIdentityExact
<1>8. PositionTypeOK /\ ParentMiss => targetSteps' \in Nat
  BY Z3, DprConstantsTyped DEF PositionTypeOK, ParentMiss, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>9. LookupTypeOK /\ ParentMiss => BoolOK(parentChecked')
  BY Z3, DprConstantsTyped DEF LookupTypeOK, ParentMiss, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>10. LookupTypeOK /\ ParentMiss => BoolOK(parentHit')
  BY Z3, DprConstantsTyped DEF LookupTypeOK, ParentMiss, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>11. LookupTypeOK /\ ParentMiss => BoolOK(nomogramChecked')
  BY Z3, DprConstantsTyped DEF LookupTypeOK, ParentMiss, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>12. ProposalTypeOK /\ ParentMiss => proposalCount' \in 0..MaxHorizon
  BY Z3, DprConstantsTyped DEF ProposalTypeOK, ParentMiss, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>13. ProposalTypeOK /\ ParentMiss => acceptedCount' \in 0..MaxHorizon
  BY Z3, DprConstantsTyped DEF ProposalTypeOK, ParentMiss, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>14. PendingTypeOK /\ ParentMiss => BoolOK(pendingValid')
  BY Z3, DprConstantsTyped DEF PendingTypeOK, ParentMiss, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>15. PendingTypeOK /\ ParentMiss => pendingToken' \in {0} \cup TokenValues
  BY Z3, DprConstantsTyped DEF PendingTypeOK, ParentMiss, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>16. DigestTypeOK /\ ParentMiss => nomogramCommitted' = 0
  BY Z3, DprConstantsTyped DEF DigestTypeOK, ParentMiss, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>17. DigestTypeOK /\ ParentMiss => runtimeDigest' \in 0..102
  BY Z3, DprConstantsTyped DEF DigestTypeOK, ParentMiss, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>18. DigestTypeOK /\ ParentMiss => canonicalDigest' \in 0..102
  BY Z3, DprConstantsTyped DEF DigestTypeOK, ParentMiss, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>19. QED
  BY Z3, <1>1, <1>2, <1>3, <1>4, <1>5, <1>6, <1>7, <1>8, <1>9, <1>10, <1>11, <1>12, <1>13, <1>14, <1>15, <1>16, <1>17, <1>18
     DEF Inv, TypeOK, ControlTypeOK, PositionTypeOK, LookupTypeOK, ProposalTypeOK, PendingTypeOK, DigestTypeOK

LEMMA ParentMissPreservesInv ==
Inv /\ ParentMiss => Inv'
BY Z3, ParentMissPreservesType, DprConstantsTyped
   DEF Inv, CommittedIdentityExact, DigestExact, ParentFirst, ProposalIsObservational, PendingNotStepped, NomogramNeverCommits, StrictProofObservational, OptionalDprOnly, NoB1Fallback, ParentMiss, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues

LEMMA NomogramMissPreservesType ==
Inv /\ NomogramMiss => TypeOK'
<1>1. ControlTypeOK /\ NomogramMiss => phase' \in Phases /\ BoolOK(dprEnabled')
  BY Z3, DprConstantsTyped DEF ControlTypeOK, NomogramMiss, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>2. ControlTypeOK /\ NomogramMiss => cycle' \in 0..MaxCycles
  BY Z3, DprConstantsTyped DEF ControlTypeOK, NomogramMiss, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>3. ControlTypeOK /\ NomogramMiss => BoolOK(strictRequested')
  BY Z3, DprConstantsTyped DEF ControlTypeOK, NomogramMiss, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>4. ControlTypeOK /\ NomogramMiss => BoolOK(proofComputed')
  BY Z3, DprConstantsTyped DEF ControlTypeOK, NomogramMiss, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>5. PositionTypeOK /\ NomogramMiss => position' \in 0..MaxPosition
  BY Z3, DprConstantsTyped DEF PositionTypeOK, NomogramMiss, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>6. PositionTypeOK /\ NomogramMiss => cycleStart' \in 0..MaxPosition
  BY Z3, DprConstantsTyped DEF PositionTypeOK, NomogramMiss, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>7. PositionTypeOK /\ CommittedIdentityExact /\ NomogramMiss => committedCount' \in 0..MaxPosition
  BY Z3, DprConstantsTyped DEF PositionTypeOK, NomogramMiss, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues, CommittedIdentityExact
<1>8. PositionTypeOK /\ NomogramMiss => targetSteps' \in Nat
  BY Z3, DprConstantsTyped DEF PositionTypeOK, NomogramMiss, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>9. LookupTypeOK /\ NomogramMiss => BoolOK(parentChecked')
  BY Z3, DprConstantsTyped DEF LookupTypeOK, NomogramMiss, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>10. LookupTypeOK /\ NomogramMiss => BoolOK(parentHit')
  BY Z3, DprConstantsTyped DEF LookupTypeOK, NomogramMiss, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>11. LookupTypeOK /\ NomogramMiss => BoolOK(nomogramChecked')
  BY Z3, DprConstantsTyped DEF LookupTypeOK, NomogramMiss, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>12. ProposalTypeOK /\ NomogramMiss => proposalCount' \in 0..MaxHorizon
  BY Z3, DprConstantsTyped DEF ProposalTypeOK, NomogramMiss, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>13. ProposalTypeOK /\ NomogramMiss => acceptedCount' \in 0..MaxHorizon
  BY Z3, DprConstantsTyped DEF ProposalTypeOK, NomogramMiss, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>14. PendingTypeOK /\ NomogramMiss => BoolOK(pendingValid')
  BY Z3, DprConstantsTyped DEF PendingTypeOK, NomogramMiss, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>15. PendingTypeOK /\ NomogramMiss => pendingToken' \in {0} \cup TokenValues
  BY Z3, DprConstantsTyped DEF PendingTypeOK, NomogramMiss, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>16. DigestTypeOK /\ NomogramMiss => nomogramCommitted' = 0
  BY Z3, DprConstantsTyped DEF DigestTypeOK, NomogramMiss, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>17. DigestTypeOK /\ NomogramMiss => runtimeDigest' \in 0..102
  BY Z3, DprConstantsTyped DEF DigestTypeOK, NomogramMiss, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>18. DigestTypeOK /\ NomogramMiss => canonicalDigest' \in 0..102
  BY Z3, DprConstantsTyped DEF DigestTypeOK, NomogramMiss, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>19. QED
  BY Z3, <1>1, <1>2, <1>3, <1>4, <1>5, <1>6, <1>7, <1>8, <1>9, <1>10, <1>11, <1>12, <1>13, <1>14, <1>15, <1>16, <1>17, <1>18
     DEF Inv, TypeOK, ControlTypeOK, PositionTypeOK, LookupTypeOK, ProposalTypeOK, PendingTypeOK, DigestTypeOK

LEMMA NomogramMissPreservesInv ==
Inv /\ NomogramMiss => Inv'
BY Z3, NomogramMissPreservesType, DprConstantsTyped
   DEF Inv, CommittedIdentityExact, DigestExact, ParentFirst, ProposalIsObservational, PendingNotStepped, NomogramNeverCommits, StrictProofObservational, OptionalDprOnly, NoB1Fallback, NomogramMiss, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues

LEMMA NomogramHitPreservesType ==
    \A horizon \in 1..MaxHorizon :
        Inv /\ NomogramHit(horizon) => TypeOK'
<1>1. \A horizon \in 1..MaxHorizon :
       ControlTypeOK /\ NomogramHit(horizon) => phase' \in Phases /\ BoolOK(dprEnabled')
  BY Z3, DprConstantsTyped DEF ControlTypeOK, NomogramHit, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>2. \A horizon \in 1..MaxHorizon :
       ControlTypeOK /\ NomogramHit(horizon) => cycle' \in 0..MaxCycles
  BY Z3, DprConstantsTyped DEF ControlTypeOK, NomogramHit, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>3. \A horizon \in 1..MaxHorizon :
       ControlTypeOK /\ NomogramHit(horizon) => BoolOK(strictRequested')
  BY Z3, DprConstantsTyped DEF ControlTypeOK, NomogramHit, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>4. \A horizon \in 1..MaxHorizon :
       ControlTypeOK /\ NomogramHit(horizon) => BoolOK(proofComputed')
  BY Z3, DprConstantsTyped DEF ControlTypeOK, NomogramHit, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>5. \A horizon \in 1..MaxHorizon :
       PositionTypeOK /\ NomogramHit(horizon) => position' \in 0..MaxPosition
  BY Z3, DprConstantsTyped DEF PositionTypeOK, NomogramHit, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>6. \A horizon \in 1..MaxHorizon :
       PositionTypeOK /\ NomogramHit(horizon) => cycleStart' \in 0..MaxPosition
  BY Z3, DprConstantsTyped DEF PositionTypeOK, NomogramHit, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>7. \A horizon \in 1..MaxHorizon :
       PositionTypeOK /\ CommittedIdentityExact /\ NomogramHit(horizon) => committedCount' \in 0..MaxPosition
  BY Z3, DprConstantsTyped DEF PositionTypeOK, NomogramHit, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues, CommittedIdentityExact
<1>8. \A horizon \in 1..MaxHorizon :
       PositionTypeOK /\ NomogramHit(horizon) => targetSteps' \in Nat
  BY Z3, DprConstantsTyped DEF PositionTypeOK, NomogramHit, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>9. \A horizon \in 1..MaxHorizon :
       LookupTypeOK /\ NomogramHit(horizon) => BoolOK(parentChecked')
  BY Z3, DprConstantsTyped DEF LookupTypeOK, NomogramHit, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>10. \A horizon \in 1..MaxHorizon :
       LookupTypeOK /\ NomogramHit(horizon) => BoolOK(parentHit')
  BY Z3, DprConstantsTyped DEF LookupTypeOK, NomogramHit, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>11. \A horizon \in 1..MaxHorizon :
       LookupTypeOK /\ NomogramHit(horizon) => BoolOK(nomogramChecked')
  BY Z3, DprConstantsTyped DEF LookupTypeOK, NomogramHit, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>12. \A horizon \in 1..MaxHorizon :
       ProposalTypeOK /\ NomogramHit(horizon) => proposalCount' \in 0..MaxHorizon
  BY Z3, DprConstantsTyped DEF ProposalTypeOK, NomogramHit, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>13. \A horizon \in 1..MaxHorizon :
       ProposalTypeOK /\ NomogramHit(horizon) => acceptedCount' \in 0..MaxHorizon
  BY Z3, DprConstantsTyped DEF ProposalTypeOK, NomogramHit, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>14. \A horizon \in 1..MaxHorizon :
       PendingTypeOK /\ NomogramHit(horizon) => BoolOK(pendingValid')
  BY Z3, DprConstantsTyped DEF PendingTypeOK, NomogramHit, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>15. \A horizon \in 1..MaxHorizon :
       PendingTypeOK /\ NomogramHit(horizon) => pendingToken' \in {0} \cup TokenValues
  BY Z3, DprConstantsTyped DEF PendingTypeOK, NomogramHit, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>16. \A horizon \in 1..MaxHorizon :
       DigestTypeOK /\ NomogramHit(horizon) => nomogramCommitted' = 0
  BY Z3, DprConstantsTyped DEF DigestTypeOK, NomogramHit, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>17. \A horizon \in 1..MaxHorizon :
       DigestTypeOK /\ NomogramHit(horizon) => runtimeDigest' \in 0..102
  BY Z3, DprConstantsTyped DEF DigestTypeOK, NomogramHit, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>18. \A horizon \in 1..MaxHorizon :
       DigestTypeOK /\ NomogramHit(horizon) => canonicalDigest' \in 0..102
  BY Z3, DprConstantsTyped DEF DigestTypeOK, NomogramHit, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>19. QED
  BY Z3, <1>1, <1>2, <1>3, <1>4, <1>5, <1>6, <1>7, <1>8, <1>9, <1>10, <1>11, <1>12, <1>13, <1>14, <1>15, <1>16, <1>17, <1>18
     DEF Inv, TypeOK, ControlTypeOK, PositionTypeOK, LookupTypeOK, ProposalTypeOK, PendingTypeOK, DigestTypeOK

LEMMA NomogramHitPreservesInv ==
    \A horizon \in 1..MaxHorizon :
        Inv /\ NomogramHit(horizon) => Inv'
BY Z3, NomogramHitPreservesType, DprConstantsTyped
   DEF Inv, CommittedIdentityExact, DigestExact, ParentFirst, ProposalIsObservational, PendingNotStepped, NomogramNeverCommits, StrictProofObservational, OptionalDprOnly, NoB1Fallback, NomogramHit, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues

LEMMA TargetVerifyPreservesType ==
    \A accepted \in 0..MaxHorizon :
        \A pending \in TokenValues :
            Inv /\ TargetVerify(accepted, pending) => TypeOK'
<1>1. \A accepted \in 0..MaxHorizon :
       \A pending \in TokenValues :
       ControlTypeOK /\ TargetVerify(accepted, pending) => phase' \in Phases /\ BoolOK(dprEnabled')
  BY Z3, DprConstantsTyped DEF ControlTypeOK, TargetVerify, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>2. \A accepted \in 0..MaxHorizon :
       \A pending \in TokenValues :
       ControlTypeOK /\ TargetVerify(accepted, pending) => cycle' \in 0..MaxCycles
  BY Z3, DprConstantsTyped DEF ControlTypeOK, TargetVerify, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>3. \A accepted \in 0..MaxHorizon :
       \A pending \in TokenValues :
       ControlTypeOK /\ TargetVerify(accepted, pending) => BoolOK(strictRequested')
  BY Z3, DprConstantsTyped DEF ControlTypeOK, TargetVerify, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>4. \A accepted \in 0..MaxHorizon :
       \A pending \in TokenValues :
       ControlTypeOK /\ TargetVerify(accepted, pending) => BoolOK(proofComputed')
  BY Z3, DprConstantsTyped DEF ControlTypeOK, TargetVerify, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>5. \A accepted \in 0..MaxHorizon :
       \A pending \in TokenValues :
       PositionTypeOK /\ TargetVerify(accepted, pending) => position' \in 0..MaxPosition
  BY Z3, DprConstantsTyped DEF PositionTypeOK, TargetVerify, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>6. \A accepted \in 0..MaxHorizon :
       \A pending \in TokenValues :
       PositionTypeOK /\ TargetVerify(accepted, pending) => cycleStart' \in 0..MaxPosition
  BY Z3, DprConstantsTyped DEF PositionTypeOK, TargetVerify, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>7. \A accepted \in 0..MaxHorizon :
       \A pending \in TokenValues :
       PositionTypeOK /\ CommittedIdentityExact /\ TargetVerify(accepted, pending) => committedCount' \in 0..MaxPosition
  BY Z3, DprConstantsTyped DEF PositionTypeOK, TargetVerify, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues, CommittedIdentityExact
<1>8. \A accepted \in 0..MaxHorizon :
       \A pending \in TokenValues :
       PositionTypeOK /\ TargetVerify(accepted, pending) => targetSteps' \in Nat
  BY Z3, DprConstantsTyped DEF PositionTypeOK, TargetVerify, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>9. \A accepted \in 0..MaxHorizon :
       \A pending \in TokenValues :
       LookupTypeOK /\ TargetVerify(accepted, pending) => BoolOK(parentChecked')
  BY Z3, DprConstantsTyped DEF LookupTypeOK, TargetVerify, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>10. \A accepted \in 0..MaxHorizon :
       \A pending \in TokenValues :
       LookupTypeOK /\ TargetVerify(accepted, pending) => BoolOK(parentHit')
  BY Z3, DprConstantsTyped DEF LookupTypeOK, TargetVerify, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>11. \A accepted \in 0..MaxHorizon :
       \A pending \in TokenValues :
       LookupTypeOK /\ TargetVerify(accepted, pending) => BoolOK(nomogramChecked')
  BY Z3, DprConstantsTyped DEF LookupTypeOK, TargetVerify, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>12. \A accepted \in 0..MaxHorizon :
       \A pending \in TokenValues :
       ProposalTypeOK /\ TargetVerify(accepted, pending) => proposalCount' \in 0..MaxHorizon
  BY Z3, DprConstantsTyped DEF ProposalTypeOK, TargetVerify, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>13. \A accepted \in 0..MaxHorizon :
       \A pending \in TokenValues :
       ProposalTypeOK /\ TargetVerify(accepted, pending) => acceptedCount' \in 0..MaxHorizon
  BY Z3, DprConstantsTyped DEF ProposalTypeOK, TargetVerify, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>14. \A accepted \in 0..MaxHorizon :
       \A pending \in TokenValues :
       PendingTypeOK /\ TargetVerify(accepted, pending) => BoolOK(pendingValid')
  BY Z3, DprConstantsTyped DEF PendingTypeOK, TargetVerify, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>15. \A accepted \in 0..MaxHorizon :
       \A pending \in TokenValues :
       PendingTypeOK /\ TargetVerify(accepted, pending) => pendingToken' \in {0} \cup TokenValues
  BY Z3, DprConstantsTyped DEF PendingTypeOK, TargetVerify, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>16. \A accepted \in 0..MaxHorizon :
       \A pending \in TokenValues :
       DigestTypeOK /\ TargetVerify(accepted, pending) => nomogramCommitted' = 0
  BY Z3, DprConstantsTyped DEF DigestTypeOK, TargetVerify, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>17. \A accepted \in 0..MaxHorizon :
       \A pending \in TokenValues :
       DigestTypeOK /\ TargetVerify(accepted, pending) => runtimeDigest' \in 0..102
  BY Z3, DprConstantsTyped DEF DigestTypeOK, TargetVerify, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>18. \A accepted \in 0..MaxHorizon :
       \A pending \in TokenValues :
       DigestTypeOK /\ TargetVerify(accepted, pending) => canonicalDigest' \in 0..102
  BY Z3, DprConstantsTyped DEF DigestTypeOK, TargetVerify, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>19. QED
  BY Z3, <1>1, <1>2, <1>3, <1>4, <1>5, <1>6, <1>7, <1>8, <1>9, <1>10, <1>11, <1>12, <1>13, <1>14, <1>15, <1>16, <1>17, <1>18
     DEF Inv, TypeOK, ControlTypeOK, PositionTypeOK, LookupTypeOK, ProposalTypeOK, PendingTypeOK, DigestTypeOK

LEMMA TargetVerifyPreservesInv ==
    \A accepted \in 0..MaxHorizon :
        \A pending \in TokenValues :
            Inv /\ TargetVerify(accepted, pending) => Inv'
BY Z3, TargetVerifyPreservesType, DprConstantsTyped
   DEF Inv, CommittedIdentityExact, DigestExact, ParentFirst, ProposalIsObservational, PendingNotStepped, NomogramNeverCommits, StrictProofObservational, OptionalDprOnly, NoB1Fallback, TargetVerify, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues

LEMMA ConsumePendingPreservesType ==
Inv /\ ConsumePending => TypeOK'
<1>1. ControlTypeOK /\ ConsumePending => phase' \in Phases /\ BoolOK(dprEnabled')
  BY Z3, DprConstantsTyped DEF ControlTypeOK, ConsumePending, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>2. ControlTypeOK /\ ConsumePending => cycle' \in 0..MaxCycles
  BY Z3, DprConstantsTyped DEF ControlTypeOK, ConsumePending, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>3. ControlTypeOK /\ ConsumePending => BoolOK(strictRequested')
  BY Z3, DprConstantsTyped DEF ControlTypeOK, ConsumePending, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>4. ControlTypeOK /\ ConsumePending => BoolOK(proofComputed')
  BY Z3, DprConstantsTyped DEF ControlTypeOK, ConsumePending, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>5. PositionTypeOK /\ ConsumePending => position' \in 0..MaxPosition
  BY Z3, DprConstantsTyped DEF PositionTypeOK, ConsumePending, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>6. PositionTypeOK /\ ConsumePending => cycleStart' \in 0..MaxPosition
  BY Z3, DprConstantsTyped DEF PositionTypeOK, ConsumePending, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>7. PositionTypeOK /\ CommittedIdentityExact /\ ConsumePending => committedCount' \in 0..MaxPosition
  BY Z3, DprConstantsTyped DEF PositionTypeOK, ConsumePending, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues, CommittedIdentityExact
<1>8. PositionTypeOK /\ ConsumePending => targetSteps' \in Nat
  BY Z3, DprConstantsTyped DEF PositionTypeOK, ConsumePending, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>9. LookupTypeOK /\ ConsumePending => BoolOK(parentChecked')
  BY Z3, DprConstantsTyped DEF LookupTypeOK, ConsumePending, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>10. LookupTypeOK /\ ConsumePending => BoolOK(parentHit')
  BY Z3, DprConstantsTyped DEF LookupTypeOK, ConsumePending, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>11. LookupTypeOK /\ ConsumePending => BoolOK(nomogramChecked')
  BY Z3, DprConstantsTyped DEF LookupTypeOK, ConsumePending, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>12. ProposalTypeOK /\ ConsumePending => proposalCount' \in 0..MaxHorizon
  BY Z3, DprConstantsTyped DEF ProposalTypeOK, ConsumePending, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>13. ProposalTypeOK /\ ConsumePending => acceptedCount' \in 0..MaxHorizon
  BY Z3, DprConstantsTyped DEF ProposalTypeOK, ConsumePending, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>14. PendingTypeOK /\ ConsumePending => BoolOK(pendingValid')
  BY Z3, DprConstantsTyped DEF PendingTypeOK, ConsumePending, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>15. PendingTypeOK /\ ConsumePending => pendingToken' \in {0} \cup TokenValues
  BY Z3, DprConstantsTyped DEF PendingTypeOK, ConsumePending, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>16. DigestTypeOK /\ ConsumePending => nomogramCommitted' = 0
  BY Z3, DprConstantsTyped DEF DigestTypeOK, ConsumePending, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>17. DigestTypeOK /\ ConsumePending => runtimeDigest' \in 0..102
  BY Z3, DprConstantsTyped DEF DigestTypeOK, ConsumePending, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>18. DigestTypeOK /\ ConsumePending => canonicalDigest' \in 0..102
  BY Z3, DprConstantsTyped DEF DigestTypeOK, ConsumePending, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>19. QED
  BY Z3, <1>1, <1>2, <1>3, <1>4, <1>5, <1>6, <1>7, <1>8, <1>9, <1>10, <1>11, <1>12, <1>13, <1>14, <1>15, <1>16, <1>17, <1>18
     DEF Inv, TypeOK, ControlTypeOK, PositionTypeOK, LookupTypeOK, ProposalTypeOK, PendingTypeOK, DigestTypeOK

LEMMA ConsumePendingPreservesInv ==
Inv /\ ConsumePending => Inv'
BY Z3, ConsumePendingPreservesType, DprConstantsTyped
   DEF Inv, CommittedIdentityExact, DigestExact, ParentFirst, ProposalIsObservational, PendingNotStepped, NomogramNeverCommits, StrictProofObservational, OptionalDprOnly, NoB1Fallback, ConsumePending, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues

LEMMA ColdSearchResolvePreservesInv ==
    \A activeHorizon \in 1..MaxHorizon :
      \A accepted \in 0..activeHorizon, pending \in TokenValues :
          Inv /\ ColdSearchResolve(activeHorizon, accepted, pending) => Inv'
<1>1. SUFFICES ASSUME NEW activeHorizon \in 1..MaxHorizon,
                       NEW accepted \in 0..activeHorizon,
                       NEW pending \in TokenValues
       PROVE Inv /\ ColdSearchResolve(activeHorizon, accepted, pending) => Inv'
       OBVIOUS
<1>2. QED
  BY Z3, <1>1, DprConstantsTyped
     DEF ColdSearchResolve, Inv, TypeOK, ControlTypeOK, PositionTypeOK,
         LookupTypeOK, ProposalTypeOK, PendingTypeOK, DigestTypeOK,
         CommittedIdentityExact, DigestExact, ParentFirst,
         ProposalIsObservational, PendingNotStepped, NomogramNeverCommits,
         StrictProofObservational, OptionalDprOnly, NoB1Fallback,
         BoolOK, DigestAdvance, DigestToken, Phases, TokenValues

LEMMA StrictProofPreservesType ==
Inv /\ StrictProof => TypeOK'
<1>1. ControlTypeOK /\ StrictProof => phase' \in Phases /\ BoolOK(dprEnabled')
  BY Z3, DprConstantsTyped DEF ControlTypeOK, StrictProof, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>2. ControlTypeOK /\ StrictProof => cycle' \in 0..MaxCycles
  BY Z3, DprConstantsTyped DEF ControlTypeOK, StrictProof, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>3. ControlTypeOK /\ StrictProof => BoolOK(strictRequested')
  BY Z3, DprConstantsTyped DEF ControlTypeOK, StrictProof, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>4. ControlTypeOK /\ StrictProof => BoolOK(proofComputed')
  BY Z3, DprConstantsTyped DEF ControlTypeOK, StrictProof, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>5. PositionTypeOK /\ StrictProof => position' \in 0..MaxPosition
  BY Z3, DprConstantsTyped DEF PositionTypeOK, StrictProof, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>6. PositionTypeOK /\ StrictProof => cycleStart' \in 0..MaxPosition
  BY Z3, DprConstantsTyped DEF PositionTypeOK, StrictProof, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>7. PositionTypeOK /\ CommittedIdentityExact /\ StrictProof => committedCount' \in 0..MaxPosition
  BY Z3, DprConstantsTyped DEF PositionTypeOK, StrictProof, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues, CommittedIdentityExact
<1>8. PositionTypeOK /\ StrictProof => targetSteps' \in Nat
  BY Z3, DprConstantsTyped DEF PositionTypeOK, StrictProof, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>9. LookupTypeOK /\ StrictProof => BoolOK(parentChecked')
  BY Z3, DprConstantsTyped DEF LookupTypeOK, StrictProof, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>10. LookupTypeOK /\ StrictProof => BoolOK(parentHit')
  BY Z3, DprConstantsTyped DEF LookupTypeOK, StrictProof, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>11. LookupTypeOK /\ StrictProof => BoolOK(nomogramChecked')
  BY Z3, DprConstantsTyped DEF LookupTypeOK, StrictProof, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>12. ProposalTypeOK /\ StrictProof => proposalCount' \in 0..MaxHorizon
  BY Z3, DprConstantsTyped DEF ProposalTypeOK, StrictProof, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>13. ProposalTypeOK /\ StrictProof => acceptedCount' \in 0..MaxHorizon
  BY Z3, DprConstantsTyped DEF ProposalTypeOK, StrictProof, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>14. PendingTypeOK /\ StrictProof => BoolOK(pendingValid')
  BY Z3, DprConstantsTyped DEF PendingTypeOK, StrictProof, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>15. PendingTypeOK /\ StrictProof => pendingToken' \in {0} \cup TokenValues
  BY Z3, DprConstantsTyped DEF PendingTypeOK, StrictProof, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>16. DigestTypeOK /\ StrictProof => nomogramCommitted' = 0
  BY Z3, DprConstantsTyped DEF DigestTypeOK, StrictProof, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>17. DigestTypeOK /\ StrictProof => runtimeDigest' \in 0..102
  BY Z3, DprConstantsTyped DEF DigestTypeOK, StrictProof, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>18. DigestTypeOK /\ StrictProof => canonicalDigest' \in 0..102
  BY Z3, DprConstantsTyped DEF DigestTypeOK, StrictProof, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>19. QED
  BY Z3, <1>1, <1>2, <1>3, <1>4, <1>5, <1>6, <1>7, <1>8, <1>9, <1>10, <1>11, <1>12, <1>13, <1>14, <1>15, <1>16, <1>17, <1>18
     DEF Inv, TypeOK, ControlTypeOK, PositionTypeOK, LookupTypeOK, ProposalTypeOK, PendingTypeOK, DigestTypeOK

LEMMA StrictProofPreservesInv ==
Inv /\ StrictProof => Inv'
BY Z3, StrictProofPreservesType, DprConstantsTyped
   DEF Inv, CommittedIdentityExact, DigestExact, ParentFirst, ProposalIsObservational, PendingNotStepped, NomogramNeverCommits, StrictProofObservational, OptionalDprOnly, NoB1Fallback, StrictProof, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues

LEMMA FailPreservesType ==
Inv /\ Fail => TypeOK'
<1>1. ControlTypeOK /\ Fail => phase' \in Phases /\ BoolOK(dprEnabled')
  BY Z3, DprConstantsTyped DEF ControlTypeOK, Fail, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>2. ControlTypeOK /\ Fail => cycle' \in 0..MaxCycles
  BY Z3, DprConstantsTyped DEF ControlTypeOK, Fail, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>3. ControlTypeOK /\ Fail => BoolOK(strictRequested')
  BY Z3, DprConstantsTyped DEF ControlTypeOK, Fail, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>4. ControlTypeOK /\ Fail => BoolOK(proofComputed')
  BY Z3, DprConstantsTyped DEF ControlTypeOK, Fail, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>5. PositionTypeOK /\ Fail => position' \in 0..MaxPosition
  BY Z3, DprConstantsTyped DEF PositionTypeOK, Fail, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>6. PositionTypeOK /\ Fail => cycleStart' \in 0..MaxPosition
  BY Z3, DprConstantsTyped DEF PositionTypeOK, Fail, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>7. PositionTypeOK /\ CommittedIdentityExact /\ Fail => committedCount' \in 0..MaxPosition
  BY Z3, DprConstantsTyped DEF PositionTypeOK, Fail, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues, CommittedIdentityExact
<1>8. PositionTypeOK /\ Fail => targetSteps' \in Nat
  BY Z3, DprConstantsTyped DEF PositionTypeOK, Fail, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>9. LookupTypeOK /\ Fail => BoolOK(parentChecked')
  BY Z3, DprConstantsTyped DEF LookupTypeOK, Fail, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>10. LookupTypeOK /\ Fail => BoolOK(parentHit')
  BY Z3, DprConstantsTyped DEF LookupTypeOK, Fail, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>11. LookupTypeOK /\ Fail => BoolOK(nomogramChecked')
  BY Z3, DprConstantsTyped DEF LookupTypeOK, Fail, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>12. ProposalTypeOK /\ Fail => proposalCount' \in 0..MaxHorizon
  BY Z3, DprConstantsTyped DEF ProposalTypeOK, Fail, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>13. ProposalTypeOK /\ Fail => acceptedCount' \in 0..MaxHorizon
  BY Z3, DprConstantsTyped DEF ProposalTypeOK, Fail, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>14. PendingTypeOK /\ Fail => BoolOK(pendingValid')
  BY Z3, DprConstantsTyped DEF PendingTypeOK, Fail, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>15. PendingTypeOK /\ Fail => pendingToken' \in {0} \cup TokenValues
  BY Z3, DprConstantsTyped DEF PendingTypeOK, Fail, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>16. DigestTypeOK /\ Fail => nomogramCommitted' = 0
  BY Z3, DprConstantsTyped DEF DigestTypeOK, Fail, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>17. DigestTypeOK /\ Fail => runtimeDigest' \in 0..102
  BY Z3, DprConstantsTyped DEF DigestTypeOK, Fail, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>18. DigestTypeOK /\ Fail => canonicalDigest' \in 0..102
  BY Z3, DprConstantsTyped DEF DigestTypeOK, Fail, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues
<1>19. QED
  BY Z3, <1>1, <1>2, <1>3, <1>4, <1>5, <1>6, <1>7, <1>8, <1>9, <1>10, <1>11, <1>12, <1>13, <1>14, <1>15, <1>16, <1>17, <1>18
     DEF Inv, TypeOK, ControlTypeOK, PositionTypeOK, LookupTypeOK, ProposalTypeOK, PendingTypeOK, DigestTypeOK

LEMMA FailPreservesInv ==
Inv /\ Fail => Inv'
BY Z3, FailPreservesType, DprConstantsTyped
   DEF Inv, CommittedIdentityExact, DigestExact, ParentFirst, ProposalIsObservational, PendingNotStepped, NomogramNeverCommits, StrictProofObservational, OptionalDprOnly, NoB1Fallback, Fail, BoolOK, DigestAdvance, DigestToken, Phases, TokenValues

SameState ==
    /\ phase' = phase
    /\ dprEnabled' = dprEnabled
    /\ cycle' = cycle
    /\ strictRequested' = strictRequested
    /\ proofComputed' = proofComputed
    /\ position' = position
    /\ cycleStart' = cycleStart
    /\ committedCount' = committedCount
    /\ targetSteps' = targetSteps
    /\ parentChecked' = parentChecked
    /\ parentHit' = parentHit
    /\ nomogramChecked' = nomogramChecked
    /\ proposalCount' = proposalCount
    /\ acceptedCount' = acceptedCount
    /\ pendingValid' = pendingValid
    /\ pendingToken' = pendingToken
    /\ nomogramCommitted' = nomogramCommitted
    /\ runtimeDigest' = runtimeDigest
    /\ canonicalDigest' = canonicalDigest

LEMMA VarsUnchangedScalar == UNCHANGED vars => SameState
<1>1. UNCHANGED vars => phase' = phase
  BY Z3 DEF vars
<1>2. UNCHANGED vars => cycle' = cycle
  BY Z3 DEF vars
<1>3. UNCHANGED vars => strictRequested' = strictRequested
  BY Z3 DEF vars
<1>4. UNCHANGED vars => proofComputed' = proofComputed
  BY Z3 DEF vars
<1>5. UNCHANGED vars => position' = position
  BY Z3 DEF vars
<1>6. UNCHANGED vars => cycleStart' = cycleStart
  BY Z3 DEF vars
<1>7. UNCHANGED vars => committedCount' = committedCount
  BY Z3 DEF vars
<1>8. UNCHANGED vars => targetSteps' = targetSteps
  BY Z3 DEF vars
<1>9. UNCHANGED vars => parentChecked' = parentChecked
  BY Z3 DEF vars
<1>10. UNCHANGED vars => parentHit' = parentHit
  BY Z3 DEF vars
<1>11. UNCHANGED vars => nomogramChecked' = nomogramChecked
  BY Z3 DEF vars
<1>12. UNCHANGED vars => proposalCount' = proposalCount
  BY Z3 DEF vars
<1>13. UNCHANGED vars => acceptedCount' = acceptedCount
  BY Z3 DEF vars
<1>14. UNCHANGED vars => pendingValid' = pendingValid
  BY Z3 DEF vars
<1>15. UNCHANGED vars => pendingToken' = pendingToken
  BY Z3 DEF vars
<1>16. UNCHANGED vars => nomogramCommitted' = nomogramCommitted
  BY Z3 DEF vars
<1>17. UNCHANGED vars => runtimeDigest' = runtimeDigest
  BY Z3 DEF vars
<1>18. UNCHANGED vars => canonicalDigest' = canonicalDigest
  BY Z3 DEF vars
<1>19. UNCHANGED vars => dprEnabled' = dprEnabled
  BY Z3 DEF vars
<1>20. QED
  BY Z3, <1>1, <1>2, <1>3, <1>4, <1>5, <1>6, <1>7, <1>8, <1>9, <1>10, <1>11, <1>12, <1>13, <1>14, <1>15, <1>16, <1>17, <1>18, <1>19 DEF SameState

LEMMA StutterPreservesInv == Inv /\ UNCHANGED vars => Inv'
BY Z3, VarsUnchangedScalar, DprConstantsTyped
   DEF Inv, TypeOK, ControlTypeOK, PositionTypeOK, LookupTypeOK, ProposalTypeOK, PendingTypeOK, DigestTypeOK, CommittedIdentityExact, DigestExact, ParentFirst, ProposalIsObservational, PendingNotStepped, NomogramNeverCommits, StrictProofObservational, OptionalDprOnly, NoB1Fallback, BoolOK, SameState

LEMMA TerminalStutterPreservesInv == Inv /\ TerminalStutter => Inv'
BY Z3, StutterPreservesInv DEF TerminalStutter

LEMMA NextPreservesInv == Inv /\ Next => Inv'
BY Z3, BeginCyclePreservesInv, ParentExactPreservesInv, ParentMissPreservesInv, NomogramMissPreservesInv, NomogramHitPreservesInv, TargetVerifyPreservesInv, ConsumePendingPreservesInv, ColdSearchResolvePreservesInv, StrictProofPreservesInv, FailPreservesInv, TerminalStutterPreservesInv
   DEF Next

THEOREM InductiveInvariant == Inv /\ [Next]_vars => Inv'
BY Z3, NextPreservesInv, StutterPreservesInv DEF vars

THEOREM DprTargetSafety == Spec => []Inv
<1>1. Init => Inv
  BY Z3, InitEstablishesInv
<1>2. Inv /\ [Next]_vars => Inv'
  BY Z3, InductiveInvariant
<1>3. QED
  BY <1>1, <1>2, PTL DEF Spec

====
