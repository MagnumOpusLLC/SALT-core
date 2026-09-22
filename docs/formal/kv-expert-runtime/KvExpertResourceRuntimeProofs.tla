---- MODULE KvExpertResourceRuntimeProofs ----
EXTENDS KvExpertResourceRuntime, TLAPS

(* Current engine-owned expert-resource proof at source/transmutation pin
   696a06d4373d2a1b696aa549925213d01b9ce4e9 / salt-dev-696a06d4.

   ModelBounds is the symbolic admitted-geometry condition. TLC checks the full
   capacity, resident-byte, loading, and retirement-counter invariants. TLAPS
   certifies bind-before-ready, leases, exact result/KV, fixed resource objects,
   and lifecycle for all values satisfying ModelBounds. *)

SemanticInv ==
    /\ BoundBeforeReady
    /\ LeaseSafety
    /\ ResultExact
    /\ KvExact
    /\ NoPerOperationResourceCreation
    /\ ResourceLifetime

LEMMA InitEstablishesSemanticInv == Init => SemanticInv
BY Z3
   DEF Init, SemanticInv, BoundBeforeReady, LeaseSafety, ResultExact,
       KvExact, NoPerOperationResourceCreation, ResourceLifetime, Phases,
       Experts

LEMMA BeginOperationPreservesSemanticInv ==
    \A sel \in SUBSET Experts : \A digest \in 0..2 :
        SemanticInv /\ BeginOperation(sel, digest) => SemanticInv'
<1>1. SUFFICES ASSUME NEW sel \in SUBSET Experts, NEW digest \in 0..2
       PROVE SemanticInv /\ BeginOperation(sel, digest) => SemanticInv'
       OBVIOUS
<1>2. QED
       BY Z3, <1>1
       DEF BeginOperation, SemanticInv, BoundBeforeReady, LeaseSafety,
           ResultExact, KvExact, NoPerOperationResourceCreation,
           ResourceLifetime, Phases, Experts

LEMMA ReserveMissingPreservesSemanticInv ==
    SemanticInv /\ ReserveMissing => SemanticInv'
BY Z3 DEF ReserveMissing, SemanticInv, BoundBeforeReady, LeaseSafety,
          ResultExact, KvExact, NoPerOperationResourceCreation,
          ResourceLifetime, Phases, Experts

LEMMA LoadBindPublishPreservesSemanticInv ==
    SemanticInv /\ LoadBindPublish => SemanticInv'
BY Z3 DEF LoadBindPublish, SemanticInv, BoundBeforeReady, LeaseSafety,
          ResultExact, KvExact, NoPerOperationResourceCreation,
          ResourceLifetime, Phases, Experts

LEMMA AcquireLeasesPreservesSemanticInv ==
    SemanticInv /\ AcquireLeases => SemanticInv'
BY Z3 DEF AcquireLeases, SemanticInv, BoundBeforeReady, LeaseSafety,
          ResultExact, KvExact, NoPerOperationResourceCreation,
          ResourceLifetime, Phases, Experts

LEMMA ExecuteCanonicalPreservesSemanticInv ==
    SemanticInv /\ ExecuteCanonical => SemanticInv'
BY Z3 DEF ExecuteCanonical, SemanticInv, BoundBeforeReady, LeaseSafety,
          ResultExact, KvExact, NoPerOperationResourceCreation,
          ResourceLifetime, ExpertFold, Phases, Experts

LEMMA ReleaseLeasesPreservesSemanticInv ==
    SemanticInv /\ ReleaseLeases => SemanticInv'
BY Z3 DEF ReleaseLeases, SemanticInv, BoundBeforeReady, LeaseSafety,
          ResultExact, KvExact, NoPerOperationResourceCreation,
          ResourceLifetime, Phases, Experts

LEMMA CommitKvPreservesSemanticInv ==
    SemanticInv /\ CommitKv => SemanticInv'
BY Z3 DEF CommitKv, SemanticInv, BoundBeforeReady, LeaseSafety,
          ResultExact, KvExact, NoPerOperationResourceCreation,
          ResourceLifetime, Phases, Experts

LEMMA EvictBatchPreservesSemanticInv ==
    \A victims \in SUBSET Experts :
        SemanticInv /\ EvictBatch(victims) => SemanticInv'
<1>1. SUFFICES ASSUME NEW victims \in SUBSET Experts
       PROVE SemanticInv /\ EvictBatch(victims) => SemanticInv'
       OBVIOUS
<1>2. QED
       BY Z3, <1>1
       DEF EvictBatch, SemanticInv, BoundBeforeReady, LeaseSafety,
           ResultExact, KvExact, NoPerOperationResourceCreation,
           ResourceLifetime, Phases, Experts

LEMMA PinPreservesSemanticInv ==
    \A expert \in Experts : SemanticInv /\ Pin(expert) => SemanticInv'
<1>1. SUFFICES ASSUME NEW expert \in Experts
       PROVE SemanticInv /\ Pin(expert) => SemanticInv'
       OBVIOUS
<1>2. QED
       BY Z3, <1>1
       DEF Pin, SemanticInv, BoundBeforeReady, LeaseSafety, ResultExact,
           KvExact, NoPerOperationResourceCreation, ResourceLifetime,
           Phases, Experts

LEMMA UnpinPreservesSemanticInv ==
    \A expert \in Experts : SemanticInv /\ Unpin(expert) => SemanticInv'
<1>1. SUFFICES ASSUME NEW expert \in Experts
       PROVE SemanticInv /\ Unpin(expert) => SemanticInv'
       OBVIOUS
<1>2. QED
       BY Z3, <1>1
       DEF Unpin, SemanticInv, BoundBeforeReady, LeaseSafety, ResultExact,
           KvExact, NoPerOperationResourceCreation, ResourceLifetime,
           Phases, Experts

LEMMA FailPreservesSemanticInv == SemanticInv /\ Fail => SemanticInv'
BY Z3 DEF Fail, SemanticInv, BoundBeforeReady, LeaseSafety, ResultExact,
          KvExact, NoPerOperationResourceCreation, ResourceLifetime,
          Phases, Experts

LEMMA ShutdownPreservesSemanticInv ==
    SemanticInv /\ Shutdown => SemanticInv'
BY Z3 DEF Shutdown, SemanticInv, BoundBeforeReady, LeaseSafety,
          ResultExact, KvExact, NoPerOperationResourceCreation,
          ResourceLifetime, Phases, Experts

LEMMA TerminalStutterPreservesSemanticInv ==
    SemanticInv /\ TerminalStutter => SemanticInv'
BY Z3 DEF TerminalStutter, SemanticInv, BoundBeforeReady, LeaseSafety,
          ResultExact, KvExact, NoPerOperationResourceCreation,
          ResourceLifetime, vars, Phases, Experts

LEMMA BindPrecedesReadyPublication ==
    SemanticInv /\ LoadBindPublish => bound' = ready'
BY Z3 DEF SemanticInv, BoundBeforeReady, LoadBindPublish

LEMMA NextPreservesSemanticInv == SemanticInv /\ Next => SemanticInv'
BY Z3, BeginOperationPreservesSemanticInv,
       ReserveMissingPreservesSemanticInv,
       LoadBindPublishPreservesSemanticInv,
       AcquireLeasesPreservesSemanticInv,
       ExecuteCanonicalPreservesSemanticInv,
       ReleaseLeasesPreservesSemanticInv,
       CommitKvPreservesSemanticInv,
       EvictBatchPreservesSemanticInv,
       PinPreservesSemanticInv,
       UnpinPreservesSemanticInv,
       FailPreservesSemanticInv,
       ShutdownPreservesSemanticInv,
       TerminalStutterPreservesSemanticInv
   DEF Next

LEMMA StutterPreservesSemanticInv ==
    SemanticInv /\ UNCHANGED vars => SemanticInv'
BY Z3 DEF SemanticInv, BoundBeforeReady, LeaseSafety, ResultExact,
          KvExact, NoPerOperationResourceCreation, ResourceLifetime,
          vars, Phases, Experts

THEOREM InductiveInvariant ==
    SemanticInv /\ [Next]_vars => SemanticInv'
BY Z3, NextPreservesSemanticInv, StutterPreservesSemanticInv DEF vars

THEOREM KvExpertResourceSafety == Spec => []SemanticInv
<1>1. Init => SemanticInv
  BY Z3, InitEstablishesSemanticInv
<1>2. SemanticInv /\ [Next]_vars => SemanticInv'
  BY Z3, InductiveInvariant
<1>3. QED
  BY <1>1, <1>2, PTL DEF Spec

====
