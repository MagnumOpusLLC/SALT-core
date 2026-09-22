---- MODULE DprOrthogonalNomogramsProofs ----
EXTENDS DprOrthogonalNomograms, TLAPS

(* Current engine-level ND/NM proof mirror at source/transmutation pin
   696a06d4373d2a1b696aa549925213d01b9ce4e9 / salt-dev-696a06d4.
   The executable model retains BadNdTriggersNm for the deliberate TLC RED
   mutation. The production theorem target is explicitly conditional on
   EnableBadCoupling = FALSE, which is the admitted engine configuration; the
   RED mutation is intentionally not certifiable. The Docker proof refresh
   includes TypeOK in the induction hypothesis and proves all 57 obligations.
   This certifies this abstract model, not concrete runtime correspondence. *)

ASSUME OrthogonalProofConstants ==
    /\ MaxOps \in Nat \ {0}
    /\ EnableBadCoupling = FALSE

Inv ==
    /\ TypeOK
    /\ NdNeverRelevance
    /\ NmNeverTokens
    /\ NomogramsNeverCommit
    /\ NoForcedCoupling
    /\ NdAdmissionRequiresEligibility
    /\ NmAdmissionRequiresEligibility
    /\ CellMatchesPlans

LEMMA InitEstablishesInv == Init => Inv
BY Z3, OrthogonalProofConstants
   DEF Init, Inv, TypeOK, NdNeverRelevance, NmNeverTokens,
       NomogramsNeverCommit, NoForcedCoupling,
       NdAdmissionRequiresEligibility, NmAdmissionRequiresEligibility,
       CellMatchesPlans, Cells, Phases

LEMMA StartOperationPreservesInv ==
    Inv /\ StartOperation => Inv'
BY Z3, OrthogonalProofConstants
   DEF StartOperation, Inv, TypeOK, NdNeverRelevance, NmNeverTokens,
       NomogramsNeverCommit, NoForcedCoupling,
       NdAdmissionRequiresEligibility, NmAdmissionRequiresEligibility,
       CellMatchesPlans, Cells, Phases

LEMMA TryNdPreservesInv ==
    \A admit \in BOOLEAN : Inv /\ TryNd(admit) => Inv'
BY Z3, OrthogonalProofConstants
   DEF TryNd, Inv, TypeOK, NdNeverRelevance, NmNeverTokens,
       NomogramsNeverCommit, NoForcedCoupling,
       NdAdmissionRequiresEligibility, NmAdmissionRequiresEligibility,
       CellMatchesPlans, Cells, Phases

LEMMA TryNmPreservesInv ==
    \A admit \in BOOLEAN : Inv /\ TryNm(admit) => Inv'
BY Z3, OrthogonalProofConstants
   DEF TryNm, Inv, TypeOK, NdNeverRelevance, NmNeverTokens,
       NomogramsNeverCommit, NoForcedCoupling,
       NdAdmissionRequiresEligibility, NmAdmissionRequiresEligibility,
       CellMatchesPlans, Cells, Phases

LEMMA SkipNdPreservesInv == Inv /\ SkipNd => Inv'
BY Z3, OrthogonalProofConstants
   DEF SkipNd, Inv, TypeOK, NdNeverRelevance, NmNeverTokens,
       NomogramsNeverCommit, NoForcedCoupling,
       NdAdmissionRequiresEligibility, NmAdmissionRequiresEligibility,
       CellMatchesPlans, Cells, Phases

LEMMA SkipNmPreservesInv == Inv /\ SkipNm => Inv'
BY Z3, OrthogonalProofConstants
   DEF SkipNm, Inv, TypeOK, NdNeverRelevance, NmNeverTokens,
       NomogramsNeverCommit, NoForcedCoupling,
       NdAdmissionRequiresEligibility, NmAdmissionRequiresEligibility,
       CellMatchesPlans, Cells, Phases

LEMMA ComposePreservesInv == Inv /\ Compose => Inv'
BY Z3, OrthogonalProofConstants
   DEF Compose, Inv, TypeOK, NdNeverRelevance, NmNeverTokens,
       NomogramsNeverCommit, NoForcedCoupling,
       NdAdmissionRequiresEligibility, NmAdmissionRequiresEligibility,
       CellMatchesPlans, Cells, Phases

LEMMA TargetCommitPreservesInv == Inv /\ TargetCommit => Inv'
BY Z3, OrthogonalProofConstants
   DEF TargetCommit, Inv, TypeOK, NdNeverRelevance, NmNeverTokens,
       NomogramsNeverCommit, NoForcedCoupling,
       NdAdmissionRequiresEligibility, NmAdmissionRequiresEligibility,
       CellMatchesPlans, Cells, Phases

LEMMA FinishPreservesInv == Inv /\ Finish => Inv'
BY Z3, OrthogonalProofConstants
   DEF Finish, Inv, TypeOK, NdNeverRelevance, NmNeverTokens,
       NomogramsNeverCommit, NoForcedCoupling,
       NdAdmissionRequiresEligibility, NmAdmissionRequiresEligibility,
       CellMatchesPlans, Cells, Phases

LEMMA BadCouplingImpossible == Inv /\ BadNdTriggersNm => Inv'
BY Z3, OrthogonalProofConstants
   DEF BadNdTriggersNm, Inv, TypeOK, NdNeverRelevance, NmNeverTokens,
       NomogramsNeverCommit, NoForcedCoupling,
       NdAdmissionRequiresEligibility, NmAdmissionRequiresEligibility,
       CellMatchesPlans, Cells, Phases

LEMMA NextPreservesInv == Inv /\ Next => Inv'
BY Z3, StartOperationPreservesInv, TryNdPreservesInv, TryNmPreservesInv,
       SkipNdPreservesInv, SkipNmPreservesInv, ComposePreservesInv,
       TargetCommitPreservesInv, FinishPreservesInv, BadCouplingImpossible
   DEF Next

LEMMA StutterPreservesInv == Inv /\ UNCHANGED vars => Inv'
BY Z3 DEF Inv, TypeOK, NdNeverRelevance, NmNeverTokens,
          NomogramsNeverCommit, NoForcedCoupling,
          NdAdmissionRequiresEligibility, NmAdmissionRequiresEligibility,
          CellMatchesPlans, vars

THEOREM InductiveInvariant == Inv /\ [Next]_vars => Inv'
BY Z3, NextPreservesInv, StutterPreservesInv DEF vars

THEOREM DprOrthogonalNomogramsSafety == Spec => []Inv
<1>1. Init => Inv
  BY Z3, InitEstablishesInv
<1>2. Inv /\ [Next]_vars => Inv'
  BY Z3, InductiveInvariant
<1>3. QED
  BY <1>1, <1>2, PTL DEF Spec

====
