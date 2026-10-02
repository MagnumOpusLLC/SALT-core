---- MODULE LoadFailureResources ----
EXTENDS Naturals, FiniteSets

(* Scoped loader/cleanup abstraction for merged source 9f5c1e0b26059e79f557e4c5a5994579e4634aec.
   Citrinitas at e1eb51ee supplies structural flow candidates; source inspection
   of src/st.c, src/tokenizer.c and Gemma's add_pool supplies the actual actions.
   Rosarium R3 refused rose_source_untraced, so this is not a traced refinement.
   A failed munmap or close is NOT assumed successful. *)

CONSTANTS HeaderBytes, FullBytes
ASSUME SizeBounds == /\ HeaderBytes \in Nat \ {0}
                     /\ FullBytes \in Nat \ {0}
                     /\ HeaderBytes < FullBytes

Kinds == {"none", "st", "tokenizer", "pool"}
Phases == {"IDLE", "ST_FD", "ST_HDR", "ST_DOC", "ST_TABLE",
           "TOK_VOCAB", "TOK_INDEX", "TOK_MERGES", "POOL_HEADER",
           "SUCCESS", "FAILED"}
Resources == {"fd", "header", "document", "table", "vocab", "index", "merges"}

VARIABLES kind, phase, owned, extent, mappedPossible,
          unmapRequested, poolSet, closeRequested
vars == <<kind, phase, owned, extent, mappedPossible,
          unmapRequested, poolSet, closeRequested>>

Init ==
    /\ kind = "none" /\ phase = "IDLE" /\ owned = {}
    /\ extent = 0 /\ mappedPossible = FALSE /\ unmapRequested = 0
    /\ poolSet = FALSE /\ closeRequested = FALSE

StOpen ==
    /\ phase = "IDLE"
    /\ kind' = "st" /\ phase' = "ST_FD" /\ owned' = {"fd"}
    /\ UNCHANGED <<extent, mappedPossible, unmapRequested, poolSet, closeRequested>>
StOpenFailed ==
    /\ phase = "IDLE"
    /\ kind' = "st" /\ phase' = "FAILED"
    /\ UNCHANGED <<owned, extent, mappedPossible, unmapRequested,
                    poolSet, closeRequested>>
StHeader ==
    /\ phase = "ST_FD"
    /\ phase' = "ST_HDR" /\ owned' = owned \cup {"header"}
    /\ UNCHANGED <<kind, extent, mappedPossible, unmapRequested, poolSet, closeRequested>>
StDocument ==
    /\ phase = "ST_HDR"
    /\ phase' = "ST_DOC" /\ owned' = owned \cup {"document"}
    /\ UNCHANGED <<kind, extent, mappedPossible, unmapRequested, poolSet, closeRequested>>
StTable ==
    /\ phase = "ST_DOC"
    /\ phase' = "ST_TABLE" /\ owned' = owned \cup {"table"}
    /\ UNCHANGED <<kind, extent, mappedPossible, unmapRequested, poolSet, closeRequested>>
StReturn ==
    /\ phase = "ST_TABLE"
    /\ phase' = "SUCCESS" /\ owned' = owned \ {"document"}
    /\ UNCHANGED <<kind, extent, mappedPossible, unmapRequested, poolSet, closeRequested>>
StAbort ==
    /\ phase \in {"ST_FD", "ST_HDR", "ST_DOC", "ST_TABLE"}
    /\ phase' = "FAILED" /\ owned' = {}
    /\ closeRequested' = ("fd" \in owned)
    /\ UNCHANGED <<kind, extent, mappedPossible, unmapRequested, poolSet>>

TokStart ==
    /\ phase = "IDLE"
    /\ kind' = "tokenizer" /\ phase' = "TOK_VOCAB"
    /\ owned' = {"vocab"}
    /\ UNCHANGED <<extent, mappedPossible, unmapRequested, poolSet, closeRequested>>
TokStartFailed ==
    /\ phase = "IDLE"
    /\ kind' = "tokenizer" /\ phase' = "FAILED"
    /\ UNCHANGED <<owned, extent, mappedPossible, unmapRequested,
                    poolSet, closeRequested>>
TokIndex ==
    /\ phase = "TOK_VOCAB"
    /\ phase' = "TOK_INDEX" /\ owned' = owned \cup {"index"}
    /\ UNCHANGED <<kind, extent, mappedPossible, unmapRequested, poolSet, closeRequested>>
TokMerges ==
    /\ phase = "TOK_INDEX"
    /\ phase' = "TOK_MERGES" /\ owned' = owned \cup {"merges"}
    /\ UNCHANGED <<kind, extent, mappedPossible, unmapRequested, poolSet, closeRequested>>
TokReturn ==
    /\ phase = "TOK_MERGES"
    /\ phase' = "SUCCESS"
    /\ UNCHANGED <<kind, owned, extent, mappedPossible, unmapRequested,
                    poolSet, closeRequested>>
TokAbort ==
    /\ phase \in {"TOK_VOCAB", "TOK_INDEX", "TOK_MERGES"}
    /\ phase' = "FAILED" /\ owned' = {}
    /\ UNCHANGED <<kind, extent, mappedPossible, unmapRequested, poolSet, closeRequested>>

PoolMap(n) ==
    /\ phase = "IDLE" /\ n \in {HeaderBytes, FullBytes}
    /\ kind' = "pool" /\ phase' = "POOL_HEADER"
    /\ extent' = n /\ mappedPossible' = TRUE
    /\ UNCHANGED <<owned, unmapRequested, poolSet, closeRequested>>
PoolMapFailed ==
    /\ phase = "IDLE"
    /\ kind' = "pool" /\ phase' = "FAILED"
    /\ UNCHANGED <<owned, extent, mappedPossible, unmapRequested,
                    poolSet, closeRequested>>
PoolReturn ==
    /\ phase = "POOL_HEADER"
    /\ phase' = "SUCCESS" /\ poolSet' = TRUE
    /\ UNCHANGED <<kind, owned, extent, mappedPossible, unmapRequested, closeRequested>>
(* unmapSucceeded is the OS outcome, not a value read by add_pool.
   POSIX zero-return success removes the supplied span. FALSE records that
   the mapping may still exist; it is not a claim of an observed leak. *)
PoolReject(unmapSucceeded) ==
    /\ phase = "POOL_HEADER" /\ unmapSucceeded \in BOOLEAN
    /\ phase' = "FAILED" /\ unmapRequested' = extent
    /\ mappedPossible' = ~unmapSucceeded
    /\ UNCHANGED <<kind, owned, extent, poolSet, closeRequested>>

Next ==
    \/ StOpen \/ StOpenFailed \/ StHeader \/ StDocument \/ StTable
    \/ StReturn \/ StAbort
    \/ TokStart \/ TokStartFailed \/ TokIndex \/ TokMerges
    \/ TokReturn \/ TokAbort
    \/ (\E n \in {HeaderBytes, FullBytes}: PoolMap(n))
    \/ PoolMapFailed \/ PoolReturn
    \/ (\E ok \in BOOLEAN: PoolReject(ok))

Spec == Init /\ [][Next]_vars

TypeOK ==
    /\ kind \in Kinds /\ phase \in Phases /\ owned \subseteq Resources
    /\ extent \in {0, HeaderBytes, FullBytes}
    /\ mappedPossible \in BOOLEAN /\ poolSet \in BOOLEAN
    /\ closeRequested \in BOOLEAN /\ unmapRequested \in {0, HeaderBytes, FullBytes}
    /\ (mappedPossible => kind = "pool" /\ extent > 0)
    /\ (closeRequested => kind = "st")
StageOwnership ==
    /\ (phase = "IDLE" =>
          kind = "none" /\ owned = {} /\ extent = 0
          /\ ~mappedPossible /\ ~poolSet /\ ~closeRequested)
    /\ (phase = "ST_FD" => kind = "st" /\ owned = {"fd"})
    /\ (phase = "ST_HDR" => kind = "st" /\ owned = {"fd", "header"})
    /\ (phase = "ST_DOC" => kind = "st" /\ owned = {"fd", "header", "document"})
    /\ (phase = "ST_TABLE" =>
          kind = "st" /\ owned = {"fd", "header", "document", "table"})
    /\ (phase = "TOK_VOCAB" => kind = "tokenizer" /\ owned = {"vocab"})
    /\ (phase = "TOK_INDEX" =>
          kind = "tokenizer" /\ owned = {"vocab", "index"})
    /\ (phase = "TOK_MERGES" =>
          kind = "tokenizer" /\ owned = {"vocab", "index", "merges"})
    /\ (phase = "POOL_HEADER" =>
          kind = "pool" /\ owned = {} /\ extent > 0
          /\ mappedPossible /\ ~poolSet)
FailureOwnership == phase = "FAILED" => owned = {}
ExactUnmapRequest ==
    (kind = "pool" /\ phase = "FAILED" /\ extent > 0)
    => unmapRequested = extent
NoFailedPoolPublication == phase # "SUCCESS" => ~poolSet
SuccessTransfer ==
    phase = "SUCCESS" =>
      \/ (kind = "st" /\ owned = {"fd", "header", "table"})
      \/ (kind = "tokenizer" /\ owned = {"vocab", "index", "merges"})
      \/ (kind = "pool" /\ poolSet /\ mappedPossible /\ extent > 0)

Inv == TypeOK /\ StageOwnership /\ FailureOwnership /\ ExactUnmapRequest
       /\ NoFailedPoolPublication /\ SuccessTransfer

(* Deliberately NOT an invariant: the C caller ignores munmap's return. *)
ActualPoolRelease ==
    (kind = "pool" /\ phase = "FAILED" /\ extent > 0) => ~mappedPossible
ActualFullPoolRelease ==
    (kind = "pool" /\ phase = "FAILED" /\ extent = FullBytes) => ~mappedPossible

====
