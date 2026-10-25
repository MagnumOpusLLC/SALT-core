/* salt/ledger.h - OPT IN to the diagnostic ledger.
 *
 * Include this AFTER <stdio.h> in a file whose diagnostics should be recorded
 * into the ledger's performance sub-ledger instead of written to a stream.
 * Files that do NOT include it keep the real fprintf, which is what lets the
 * redirect be enabled one file at a time (a bisect by file, not a text pass).
 *
 * The parent's retained prefixes and the runtime_ready= scan pass through.
 */
#ifndef SALT_LEDGER_H
#define SALT_LEDGER_H

#include <stdio.h>

/* ---- THE LEDGER BOOK ------------------------------------------------------
 * ONE book, built at initialization, holding every sub-ledger.  Host and device
 * realizations carry IDENTICAL metadata and structure and differ only in the
 * addresses they hold; the book is the state machine's proof.
 *
 * Sub-ledgers:
 *   performance - the diagnostic/telemetry record.  Core diagnostics are written
 *                 here as plain memory writes and flushed at a designated point,
 *                 never from the operation path.
 * ------------------------------------------------------------------------- */
#define SALT_LEDGER_SLOTS  262144
#define SALT_LEDGER_BYTES  256

typedef struct {
    /* ---- sub-ledger: performance ------------------------------------- */
    struct {
        char     ring[SALT_LEDGER_SLOTS][SALT_LEDGER_BYTES];
        unsigned head;
        unsigned long long dropped;
        int      flush_registered;
    } performance;
} SaltLedgerBook;

/* The one book.  One instance, sized at compile time: nothing in the operation
 * path allocates. */
extern SaltLedgerBook salt_ledger_book;

int  salt_ledger_emit(FILE *stream, const char *fmt, ...);
void salt_ledger_flush(void);

#define fprintf(stream, ...) salt_ledger_emit((stream), __VA_ARGS__)

#endif /* SALT_LEDGER_H */
