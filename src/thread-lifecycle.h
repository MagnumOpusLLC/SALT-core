#ifndef SALT_THREAD_LIFECYCLE_H
#define SALT_THREAD_LIFECYCLE_H

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

#define SALT_THREAD_JOIN_FATAL_STATUS 125

typedef int (*SaltThreadJoinFn)(pthread_t, void **);

/* A created thread may retain pointers into caller-owned stack or mapped
 * storage. Returning after an unsuccessful join would invalidate that state.
 * Treat an unprovable join as process-terminal so OS teardown, rather than
 * ordinary object cleanup, owns the remaining thread and mappings. */
static inline void *salt_join_one_or_exit(
        pthread_t thread, SaltThreadJoinFn join_fn,
        const char *owner, int index) {
    void *result = NULL;
    if (!join_fn) {
        fprintf(stderr, "%s: invalid thread-join lifecycle\n",
                owner ? owner : "salt");
        fflush(stderr);
        _Exit(SALT_THREAD_JOIN_FATAL_STATUS);
    }
    int rc = join_fn(thread, &result);
    if (rc != 0) {
        fprintf(stderr, "%s: pthread_join failed rc=%d index=%d\n",
                owner ? owner : "salt", rc, index);
        fflush(stderr);
        _Exit(SALT_THREAD_JOIN_FATAL_STATUS);
    }
    return result;
}

static inline void salt_join_started_or_exit(
        pthread_t *threads, const int *started, int count,
        SaltThreadJoinFn join_fn, const char *owner) {
    if (!threads || !started || count < 0 || !join_fn) {
        fprintf(stderr, "%s: invalid thread-join lifecycle\n",
                owner ? owner : "salt");
        fflush(stderr);
        _Exit(SALT_THREAD_JOIN_FATAL_STATUS);
    }
    for (int i = 0; i < count; i++) {
        if (!started[i]) continue;
        (void)salt_join_one_or_exit(threads[i], join_fn, owner, i);
    }
}

#endif
