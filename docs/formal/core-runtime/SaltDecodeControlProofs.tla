---- MODULE SaltDecodeControlProofs ----
EXTENDS SaltDecodeControl, TLAPS

(* Parameterized safety proof for the inspected normal decode/finish control.
   Inv includes reachability strengthening, not extra runtime preconditions.
   No claim of source-code extraction, arithmetic equivalence, or liveness. *)

LEMMA InitEstablishesInv == Init => Inv
BY Z3, ModelBounds
 DEF Init, Inv, TypeOK, ColdWidthBound, WidthBound, PrefixPreserved,
     PublicationSafety, JournalAfterFinish, PendingAccounting,
     ResourceIsolation, Reachability, Phases, Active, Allowed, Min

LEMMA NextPreservesInv == Inv /\ Next => Inv'
BY Z3, ModelBounds
 DEF Next, Admit, Prefill, ChooseRoute, ProposeMiss, ProposeHit,
     SampleOrdinary, ConsumeOrdinary, NeedResource, AcquireResume,
     ResolveTarget, BeginFinish, FinishPending, SealTurn, WriteJournal,
     Publish, Fail, Close, Idle, vars,
     Inv, TypeOK, ColdWidthBound, WidthBound, PrefixPreserved,
     PublicationSafety, JournalAfterFinish, PendingAccounting,
     ResourceIsolation, Reachability, Phases, Active, Allowed, Min

LEMMA StutterPreservesInv == Inv /\ UNCHANGED vars => Inv'
BY DEF vars, Inv, TypeOK, ColdWidthBound, WidthBound, PrefixPreserved,
       PublicationSafety, JournalAfterFinish, PendingAccounting,
       ResourceIsolation, Reachability, Active

THEOREM DecodeControlSafety == Spec => []Inv
<1>1. Init => Inv
       BY InitEstablishesInv
<1>2. Inv /\ [Next]_vars => Inv'
       BY NextPreservesInv, StutterPreservesInv
<1>3. QED
       BY <1>1, <1>2, PTL DEF Spec

====
