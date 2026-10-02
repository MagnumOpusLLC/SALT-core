---- MODULE LoadFailureResourcesProofs ----
EXTENDS LoadFailureResources, TLAPS

(* Scoped abstract safety proof. Rosarium origin paths were unavailable, so
   theorem success is not source-extracted correspondence or real OS cleanup. *)

LEMMA InitEstablishesInv == Init => Inv
BY Z3, SizeBounds
   DEF Init, Inv, TypeOK, StageOwnership, FailureOwnership, ExactUnmapRequest,
       NoFailedPoolPublication, SuccessTransfer, Kinds, Phases, Resources

LEMMA NextPreservesInv == Inv /\ Next => Inv'
BY Z3, SizeBounds
   DEF Next, StOpen, StOpenFailed, StHeader, StDocument, StTable,
       StReturn, StAbort, TokStart, TokStartFailed, TokIndex,
       TokMerges, TokReturn, TokAbort, PoolMap, PoolMapFailed,
       PoolReturn, PoolReject, Inv, TypeOK, StageOwnership, FailureOwnership,
       ExactUnmapRequest, NoFailedPoolPublication, SuccessTransfer,
       Kinds, Phases, Resources

LEMMA StutterPreservesInv == Inv /\ UNCHANGED vars => Inv'
BY DEF vars, Inv, TypeOK, StageOwnership, FailureOwnership, ExactUnmapRequest,
       NoFailedPoolPublication, SuccessTransfer

THEOREM LoadFailureSafety == Spec => []Inv
<1>1. Init => Inv
       BY InitEstablishesInv
<1>2. Inv /\ [Next]_vars => Inv'
       BY NextPreservesInv, StutterPreservesInv
<1>3. QED
       BY <1>1, <1>2, PTL DEF Spec

(* Conditional OS-contract thesis: if munmap returns zero, the mapped span
   was requested in full and is no longer possibly mapped in this model.
   add_pool ignores the return, so no actual C execution is certified here. *)
THEOREM OSZeroReturnConditionalRelease ==
    Inv /\ PoolReject(TRUE) =>
      /\ phase' = "FAILED" /\ unmapRequested' = extent
      /\ extent > 0 /\ ~mappedPossible' /\ ~poolSet'
BY Z3 DEF Inv, StageOwnership, PoolReject

(* On an OS error the caller still rejects the header and does not publish;
   mappedPossible is TRUE, so actual release is deliberately unproved. *)
THEOREM OSErrorNoPublication ==
    Inv /\ PoolReject(FALSE) =>
      /\ phase' = "FAILED" /\ unmapRequested' = extent
      /\ mappedPossible' /\ ~poolSet'
BY Z3 DEF Inv, StageOwnership, PoolReject

====
