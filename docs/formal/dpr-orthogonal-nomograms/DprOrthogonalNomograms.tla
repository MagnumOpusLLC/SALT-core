---- MODULE DprOrthogonalNomograms ----
EXTENDS Naturals, TLC

CONSTANTS
  MaxOps,
  EnableBadCoupling

VARIABLES
  phase,
  operation,
  ndEligible,
  nmEligible,
  ndTried,
  nmTried,
  ndAdmitted,
  nmAdmitted,
  cell,
  ndOutput,
  nmOutput,
  targetCommits,
  badCoupling

vars == <<phase, operation, ndEligible, nmEligible, ndTried, nmTried,
          ndAdmitted, nmAdmitted, cell, ndOutput, nmOutput,
          targetCommits, badCoupling>>

Cells == {"UNPLANNED", "NATIVE", "ND_ONLY", "NM_ONLY", "ND_NM"}
Phases == {"READY", "PLANNING", "TARGET", "DONE"}

Init ==
    /\ phase = "READY"
    /\ operation = 0
    /\ ndEligible = FALSE
    /\ nmEligible = FALSE
    /\ ndTried = FALSE
    /\ nmTried = FALSE
    /\ ndAdmitted = FALSE
    /\ nmAdmitted = FALSE
    /\ cell = "UNPLANNED"
    /\ ndOutput = "NONE"
    /\ nmOutput = "NONE"
    /\ targetCommits = 0
    /\ badCoupling = FALSE

StartOperation ==
    /\ phase = "READY"
    /\ operation < MaxOps
    /\ phase' = "PLANNING"
    /\ operation' = operation + 1
    /\ ndEligible' \in BOOLEAN
    /\ nmEligible' \in BOOLEAN
    /\ ndTried' = FALSE
    /\ nmTried' = FALSE
    /\ ndAdmitted' = FALSE
    /\ nmAdmitted' = FALSE
    /\ cell' = "UNPLANNED"
    /\ ndOutput' = "NONE"
    /\ nmOutput' = "NONE"
    /\ UNCHANGED <<targetCommits, badCoupling>>

TryNd(admit) ==
    /\ phase = "PLANNING"
    /\ ndEligible
    /\ ~ndTried
    /\ admit \in BOOLEAN
    /\ ndTried' = TRUE
    /\ ndAdmitted' = admit
    /\ ndOutput' = IF admit THEN "TOKENS" ELSE "NONE"
    /\ UNCHANGED <<phase, operation, ndEligible, nmEligible, nmTried,
                    nmAdmitted, cell, nmOutput, targetCommits, badCoupling>>

TryNm(admit) ==
    /\ phase = "PLANNING"
    /\ nmEligible
    /\ ~nmTried
    /\ admit \in BOOLEAN
    /\ nmTried' = TRUE
    /\ nmAdmitted' = admit
    /\ nmOutput' = IF admit THEN "RELEVANCE" ELSE "NONE"
    /\ UNCHANGED <<phase, operation, ndEligible, nmEligible, ndTried,
                    ndAdmitted, cell, ndOutput, targetCommits, badCoupling>>

SkipNd ==
    /\ phase = "PLANNING"
    /\ ~ndEligible
    /\ ~ndTried
    /\ ndTried' = TRUE
    /\ UNCHANGED <<phase, operation, ndEligible, nmEligible, nmTried,
                    ndAdmitted, nmAdmitted, cell, ndOutput, nmOutput,
                    targetCommits, badCoupling>>

SkipNm ==
    /\ phase = "PLANNING"
    /\ ~nmEligible
    /\ ~nmTried
    /\ nmTried' = TRUE
    /\ UNCHANGED <<phase, operation, ndEligible, nmEligible, ndTried,
                    ndAdmitted, nmAdmitted, cell, ndOutput, nmOutput,
                    targetCommits, badCoupling>>

Compose ==
    /\ phase = "PLANNING"
    /\ ndTried
    /\ nmTried
    /\ phase' = "TARGET"
    /\ cell' = IF ndAdmitted /\ nmAdmitted THEN "ND_NM"
                ELSE IF ndAdmitted THEN "ND_ONLY"
                ELSE IF nmAdmitted THEN "NM_ONLY"
                ELSE "NATIVE"
    /\ UNCHANGED <<operation, ndEligible, nmEligible, ndTried, nmTried,
                    ndAdmitted, nmAdmitted, ndOutput, nmOutput,
                    targetCommits, badCoupling>>

TargetCommit ==
    /\ phase = "TARGET"
    /\ phase' = "DONE"
    /\ targetCommits' = targetCommits + 1
    /\ UNCHANGED <<operation, ndEligible, nmEligible, ndTried, nmTried,
                    ndAdmitted, nmAdmitted, cell, ndOutput, nmOutput,
                    badCoupling>>

Finish ==
    /\ phase = "DONE"
    /\ phase' = "READY"
    /\ UNCHANGED <<operation, ndEligible, nmEligible, ndTried, nmTried,
                    ndAdmitted, nmAdmitted, cell, ndOutput, nmOutput,
                    targetCommits, badCoupling>>

BadNdTriggersNm ==
    /\ EnableBadCoupling
    /\ phase = "PLANNING"
    /\ ndTried
    /\ ndAdmitted
    /\ ~nmEligible
    /\ ~nmTried
    /\ nmTried' = TRUE
    /\ nmAdmitted' = TRUE
    /\ nmOutput' = "RELEVANCE"
    /\ badCoupling' = TRUE
    /\ UNCHANGED <<phase, operation, ndEligible, nmEligible, ndTried,
                    ndAdmitted, cell, ndOutput, targetCommits>>

Next ==
    \/ StartOperation
    \/ \E admit \in BOOLEAN : TryNd(admit)
    \/ \E admit \in BOOLEAN : TryNm(admit)
    \/ SkipNd
    \/ SkipNm
    \/ Compose
    \/ TargetCommit
    \/ Finish
    \/ BadNdTriggersNm

Spec == Init /\ [][Next]_vars

TypeOK ==
    /\ phase \in Phases
    /\ operation \in 0..MaxOps
    /\ ndEligible \in BOOLEAN
    /\ nmEligible \in BOOLEAN
    /\ ndTried \in BOOLEAN
    /\ nmTried \in BOOLEAN
    /\ ndAdmitted \in BOOLEAN
    /\ nmAdmitted \in BOOLEAN
    /\ cell \in Cells
    /\ ndOutput \in {"NONE", "TOKENS"}
    /\ nmOutput \in {"NONE", "RELEVANCE"}
    /\ targetCommits \in 0..MaxOps
    /\ badCoupling \in BOOLEAN

NdNeverRelevance == ndOutput # "RELEVANCE"
NmNeverTokens == nmOutput # "TOKENS"
NomogramsNeverCommit ==
    /\ targetCommits <= operation
    /\ (phase \in {"PLANNING", "TARGET"} =>
           targetCommits + 1 = operation)
    /\ (phase \in {"READY", "DONE"} =>
           targetCommits = operation)
NoForcedCoupling == ~badCoupling
NdAdmissionRequiresEligibility == ndAdmitted => ndEligible
NmAdmissionRequiresEligibility == nmAdmitted => nmEligible
CellMatchesPlans ==
    phase \in {"TARGET", "DONE"} =>
      /\ (cell = "ND_NM") = (ndAdmitted /\ nmAdmitted)
      /\ (cell = "ND_ONLY") = (ndAdmitted /\ ~nmAdmitted)
      /\ (cell = "NM_ONLY") = (~ndAdmitted /\ nmAdmitted)
      /\ (cell = "NATIVE") = (~ndAdmitted /\ ~nmAdmitted)

====
