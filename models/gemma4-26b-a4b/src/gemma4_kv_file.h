#ifndef SALT_MODELS_GEMMA4_KV_FILE_H
#define SALT_MODELS_GEMMA4_KV_FILE_H

#include "gemma4_text.h"
#include "sha256.h"

#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#ifndef O_NOFOLLOW
#define O_NOFOLLOW 0
#endif

#define G4_KV_FILE_HEADER_BYTES 256u
/* Conservative V4/V5 upper bound used only for file-size admission. */
#define G4_KV_FILE_BYTES_PER_TOKEN 450560u

typedef struct G4KvSharedLease {
    int fd;
    void *mapping;
    size_t bytes;
} G4KvSharedLease;

static inline int g4_kv_file_error(char *error, size_t error_size,
                                   const char *operation, const char *path) {
    if (error && error_size)
        snprintf(error, error_size, "%s %s: %s", operation, path,
                 strerror(errno));
    return -1;
}

static inline void g4_kv_shared_release(G4KvSharedLease *lease) {
    if (!lease) return;
    if (lease->mapping != MAP_FAILED && lease->mapping)
        munmap(lease->mapping, lease->bytes);
    if (lease->fd >= 0) close(lease->fd);
    lease->fd = -1;
    lease->mapping = MAP_FAILED;
    lease->bytes = 0;
}

static inline int g4_kv_file_attach_shared(
        SaltGemma4Text *model, const char *path, int max_context,
        G4KvSharedLease *lease, int *shared_position,
        char *error, size_t error_size) {
    struct stat before, after;
    size_t max_bytes;
    if (lease) {
        lease->fd = -1;
        lease->mapping = MAP_FAILED;
        lease->bytes = 0;
    }
    if (shared_position) *shared_position = 0;
    if (!model || !path || !*path || max_context < 1 || !lease ||
        !shared_position || (size_t)max_context >
            (SIZE_MAX - G4_KV_FILE_HEADER_BYTES) /
                G4_KV_FILE_BYTES_PER_TOKEN) {
        if (error && error_size)
            snprintf(error, error_size, "invalid shared KV arguments");
        return -1;
    }
    max_bytes = G4_KV_FILE_HEADER_BYTES +
                (size_t)max_context * G4_KV_FILE_BYTES_PER_TOKEN;
    lease->fd = open(path, O_RDONLY | O_NOFOLLOW);
    if (lease->fd < 0)
        return g4_kv_file_error(error, error_size, "open", path);
    if (fstat(lease->fd, &before) != 0 || !S_ISREG(before.st_mode) ||
        before.st_nlink != 1 || before.st_size < 0 ||
        (uint64_t)before.st_size < G4_KV_FILE_HEADER_BYTES ||
        (uint64_t)before.st_size > (uint64_t)max_bytes ||
        (uint64_t)(size_t)before.st_size != (uint64_t)before.st_size) {
        if (error && error_size)
            snprintf(error, error_size, "shared KV is not a bounded private file");
        g4_kv_shared_release(lease);
        return -1;
    }
    lease->bytes = (size_t)before.st_size;
    lease->mapping = mmap(NULL, lease->bytes, PROT_READ, MAP_PRIVATE,
                          lease->fd, 0);
    if (lease->mapping == MAP_FAILED ||
        salt_gemma4_text_kv_attach_shared(
            model, lease->mapping, lease->bytes) != 0 ||
        fstat(lease->fd, &after) != 0 || before.st_dev != after.st_dev ||
        before.st_ino != after.st_ino || before.st_size != after.st_size) {
        if (error && error_size)
            snprintf(error, error_size, "shared KV authentication/attach failed");
        g4_kv_shared_release(lease);
        return -1;
    }
    *shared_position = salt_gemma4_text_shared_position(model);
    if (*shared_position < 1 || *shared_position > max_context) {
        g4_kv_shared_release(lease);
        return -1;
    }
    return 0;
}

static inline int g4_kv_file_load(SaltGemma4Text *model, const char *path,
                                  int max_context, int *loaded_position,
                                  char *error, size_t error_size) {
    int fd = -1, rc = -1;
    SaltStateTransaction transaction;
    struct stat before, after;
    void *mapping = MAP_FAILED;
    size_t max_bytes;
    memset(&transaction, 0, sizeof transaction);
    if (loaded_position) *loaded_position = 0;
    if (!model || !path || !*path || max_context < 1 || !loaded_position ||
        (size_t)max_context >
            (SIZE_MAX - G4_KV_FILE_HEADER_BYTES) /
                G4_KV_FILE_BYTES_PER_TOKEN) {
        if (error && error_size) snprintf(error, error_size, "invalid KV load arguments");
        return -1;
    }
    if (salt_gemma4_text_state_transaction_begin(model,
            SALT_STATE_TRANSACTION_IMPORT, SALT_STATE_ARTIFACT_FULL,
            SALT_STATE_REASON_EXPLICIT, &transaction) != 0) {
        if (error && error_size)
            snprintf(error, error_size, "KV import transaction rejected");
        return -1;
    }
    max_bytes = G4_KV_FILE_HEADER_BYTES +
                (size_t)max_context * G4_KV_FILE_BYTES_PER_TOKEN;
    fd = open(path, O_RDONLY | O_NOFOLLOW);
    if (fd < 0) {
        g4_kv_file_error(error, error_size, "open", path);
        goto done;
    }
    if (fstat(fd, &before) != 0) {
        g4_kv_file_error(error, error_size, "stat", path);
        goto done;
    }
    if (!S_ISREG(before.st_mode) || before.st_nlink != 1 || before.st_size < 0 ||
        (uint64_t)before.st_size < G4_KV_FILE_HEADER_BYTES ||
        (uint64_t)before.st_size > (uint64_t)max_bytes ||
        (uint64_t)(size_t)before.st_size != (uint64_t)before.st_size) {
        if (error && error_size)
            snprintf(error, error_size, "KV cache is not a bounded private regular file");
        goto done;
    }
    mapping = mmap(NULL, (size_t)before.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapping == MAP_FAILED) {
        g4_kv_file_error(error, error_size, "mmap", path);
        goto done;
    }
    if (salt_gemma4_text_kv_import(model, mapping, (size_t)before.st_size) != 0) {
        if (error && error_size)
            snprintf(error, error_size, "KV cache authentication/import failed");
        goto done;
    }
    if (fstat(fd, &after) != 0 || before.st_dev != after.st_dev ||
        before.st_ino != after.st_ino || before.st_size != after.st_size) {
        if (error && error_size)
            snprintf(error, error_size, "KV cache identity changed while loading");
        goto done;
    }
    *loaded_position = salt_gemma4_text_position(model);
    if (*loaded_position < 1 || *loaded_position > max_context) {
        if (error && error_size)
            snprintf(error, error_size, "KV cache restored an invalid position");
        goto done;
    }
    if (salt_gemma4_text_state_transaction_finish(model, &transaction) != 0) {
        if (error && error_size)
            snprintf(error, error_size, "KV import transaction failed");
        goto done;
    }
    rc = 0;

done:
    if (mapping != MAP_FAILED) munmap(mapping, (size_t)before.st_size);
    if (fd >= 0) close(fd);
    salt_state_transaction_abort(&transaction);
    return rc;
}

static inline int g4_kv_fd_load_stream(
        SaltGemma4Text *model, int source_fd, int max_context,
        size_t expected_file_bytes,
        const unsigned char expected_file_sha256[32], int *loaded_position,
        char *error, size_t error_size) {
    int fd = -1, rc = -1;
    struct stat before, after;
    FILE *stream = NULL;
    size_t max_bytes;
    unsigned char actual_sha256[32];
    if (loaded_position) *loaded_position = 0;
    if (!model || source_fd < 0 || max_context < 1 ||
        expected_file_bytes < G4_KV_FILE_HEADER_BYTES ||
        !expected_file_sha256 || !loaded_position ||
        (size_t)max_context >
            (SIZE_MAX - G4_KV_FILE_HEADER_BYTES) /
                G4_KV_FILE_BYTES_PER_TOKEN) {
        if (error && error_size)
            snprintf(error, error_size, "invalid streaming KV fd arguments");
        return -1;
    }
    max_bytes = G4_KV_FILE_HEADER_BYTES +
                (size_t)max_context * G4_KV_FILE_BYTES_PER_TOKEN;
    if (expected_file_bytes > max_bytes) {
        if (error && error_size)
            snprintf(error, error_size, "streaming KV exceeds context budget");
        return -1;
    }
    fd = dup(source_fd);
    if (fd < 0) {
        if (error && error_size) snprintf(error, error_size, "dup KV fd failed");
        return -1;
    }
    if (fstat(fd, &before) != 0 || !S_ISREG(before.st_mode) ||
        before.st_nlink != 1 || before.st_size < 0 ||
        (uint64_t)before.st_size != (uint64_t)expected_file_bytes ||
        (uint64_t)(size_t)before.st_size != (uint64_t)before.st_size) {
        if (error && error_size)
            snprintf(error, error_size,
                     "streaming KV fd is not the admitted private file");
        goto done_fd_stream;
    }
    stream = fdopen(fd, "rb");
    if (!stream) {
        if (error && error_size) snprintf(error, error_size, "fdopen KV failed");
        goto done_fd_stream;
    }
    fd = -1;
    if (salt_gemma4_text_kv_import_stream_sha256(
            model, stream, actual_sha256) != 0 ||
        memcmp(actual_sha256, expected_file_sha256, sizeof actual_sha256) != 0 ||
        fstat(fileno(stream), &after) != 0 || before.st_dev != after.st_dev ||
        before.st_ino != after.st_ino || before.st_size != after.st_size ||
        before.st_mtime != after.st_mtime || before.st_ctime != after.st_ctime) {
        if (error && error_size)
            snprintf(error, error_size,
                     "streaming KV fd authentication/import failed");
        goto done_fd_stream;
    }
    *loaded_position = salt_gemma4_text_position(model);
    if (*loaded_position < 1 || *loaded_position > max_context)
        goto done_fd_stream;
    rc = 0;

done_fd_stream:
    if (stream) fclose(stream);
    if (fd >= 0) close(fd);
    return rc;
}

static inline int g4_kv_fd_load_delta(
        SaltGemma4Text *model, int source_fd, int max_context,
        size_t expected_file_bytes, int source_position, int token_count,
        int *loaded_position, char *error, size_t error_size) {
    int fd = -1, rc = -1;
    struct stat before, after;
    FILE *stream = NULL;
    size_t max_bytes;
    if (loaded_position) *loaded_position = 0;
    if (!model || source_fd < 0 || max_context < 1 ||
        expected_file_bytes < G4_KV_FILE_HEADER_BYTES ||
        source_position < 0 || token_count < 1 ||
        source_position > max_context - token_count || !loaded_position ||
        (size_t)max_context >
            (SIZE_MAX - G4_KV_FILE_HEADER_BYTES) /
                G4_KV_FILE_BYTES_PER_TOKEN) {
        if (error && error_size)
            snprintf(error, error_size, "invalid delta KV fd arguments");
        return -1;
    }
    max_bytes = G4_KV_FILE_HEADER_BYTES +
                (size_t)max_context * G4_KV_FILE_BYTES_PER_TOKEN;
    if (expected_file_bytes > max_bytes) {
        if (error && error_size)
            snprintf(error, error_size, "delta KV exceeds context budget");
        return -1;
    }
    fd = dup(source_fd);
    if (fd < 0) {
        if (error && error_size) snprintf(error, error_size, "dup delta KV fd failed");
        return -1;
    }
    if (fstat(fd, &before) != 0 || !S_ISREG(before.st_mode) ||
        before.st_nlink != 1 || before.st_size < 0 ||
        (uint64_t)before.st_size != (uint64_t)expected_file_bytes ||
        (uint64_t)(size_t)before.st_size != (uint64_t)before.st_size) {
        if (error && error_size)
            snprintf(error, error_size,
                     "delta KV fd is not the admitted private file");
        goto done_fd_delta;
    }
    stream = fdopen(fd, "rb");
    if (!stream) {
        if (error && error_size) snprintf(error, error_size, "fdopen delta KV failed");
        goto done_fd_delta;
    }
    fd = -1;
    if (salt_gemma4_text_kv_import_delta_stream(
            model, stream, source_position, token_count) != 0 ||
        fstat(fileno(stream), &after) != 0 || before.st_dev != after.st_dev ||
        before.st_ino != after.st_ino || before.st_size != after.st_size ||
        before.st_mtime != after.st_mtime || before.st_ctime != after.st_ctime) {
        if (error && error_size)
            snprintf(error, error_size,
                     "delta KV fd identity/import failed");
        goto done_fd_delta;
    }
    *loaded_position = salt_gemma4_text_position(model);
    if (*loaded_position != source_position + token_count)
        goto done_fd_delta;
    rc = 0;

done_fd_delta:
    if (stream) fclose(stream);
    if (fd >= 0) close(fd);
    return rc;
}

static inline int g4_kv_file_load_stream(
        SaltGemma4Text *model, const char *path, int max_context,
        size_t expected_file_bytes,
        const char expected_file_sha256[65], int *loaded_position,
        char *error, size_t error_size) {
    int fd = -1, rc = -1;
    struct stat before, after;
    FILE *stream = NULL;
    size_t max_bytes;
    unsigned char expected_sha256[32], actual_sha256[32];
    if (loaded_position) *loaded_position = 0;
    if (!model || !path || !*path || max_context < 1 ||
        expected_file_bytes < G4_KV_FILE_HEADER_BYTES ||
        !expected_file_sha256 ||
        salt_sha256_hex_parse(expected_file_sha256, expected_sha256) != 0 ||
        !loaded_position ||
        (size_t)max_context >
            (SIZE_MAX - G4_KV_FILE_HEADER_BYTES) /
                G4_KV_FILE_BYTES_PER_TOKEN) {
        if (error && error_size)
            snprintf(error, error_size, "invalid streaming KV load arguments");
        return -1;
    }
    max_bytes = G4_KV_FILE_HEADER_BYTES +
                (size_t)max_context * G4_KV_FILE_BYTES_PER_TOKEN;
    if (expected_file_bytes > max_bytes) {
        if (error && error_size)
            snprintf(error, error_size, "streaming KV exceeds context budget");
        return -1;
    }
    fd = open(path, O_RDONLY | O_NOFOLLOW);
    if (fd < 0) return g4_kv_file_error(error, error_size, "open", path);
    if (fstat(fd, &before) != 0 || !S_ISREG(before.st_mode) ||
        before.st_nlink != 1 || before.st_size < 0 ||
        (uint64_t)before.st_size != (uint64_t)expected_file_bytes ||
        (uint64_t)(size_t)before.st_size != (uint64_t)before.st_size) {
        if (error && error_size)
            snprintf(error, error_size,
                     "streaming KV is not the admitted private file");
        goto done_stream;
    }
    stream = fdopen(fd, "rb");
    if (!stream) {
        g4_kv_file_error(error, error_size, "fdopen", path);
        goto done_stream;
    }
    fd = -1;
    if (salt_gemma4_text_kv_import_stream_sha256(
            model, stream, actual_sha256) != 0 ||
        memcmp(actual_sha256, expected_sha256, sizeof actual_sha256) != 0 ||
        fstat(fileno(stream), &after) != 0 || before.st_dev != after.st_dev ||
        before.st_ino != after.st_ino || before.st_size != after.st_size ||
        before.st_mtime != after.st_mtime || before.st_ctime != after.st_ctime) {
        if (error && error_size)
            snprintf(error, error_size,
                     "streaming KV authentication/import failed");
        goto done_stream;
    }
    *loaded_position = salt_gemma4_text_position(model);
    if (*loaded_position < 1 || *loaded_position > max_context) goto done_stream;
    rc = 0;

done_stream:
    if (stream) fclose(stream);
    if (fd >= 0) close(fd);
    return rc;
}

static inline int g4_kv_file_save(SaltGemma4Text *model, const char *path,
                                  int *saved_position,
                                  char *error, size_t error_size) {
    int fd = -1, rc = -1;
    SaltStateTransaction transaction;
    void *mapping = MAP_FAILED;
    size_t bytes = 0;
    off_t file_bytes;
    memset(&transaction, 0, sizeof transaction);
    if (saved_position) *saved_position = 0;
    if (!model || !path || !*path || !saved_position) {
        if (error && error_size) snprintf(error, error_size, "invalid KV save arguments");
        return -1;
    }
    if (salt_gemma4_text_state_transaction_begin(model,
            SALT_STATE_TRANSACTION_EXPORT, SALT_STATE_ARTIFACT_FULL,
            SALT_STATE_REASON_EXPLICIT, &transaction) != 0) {
        if (error && error_size)
            snprintf(error, error_size, "KV export transaction rejected");
        return -1;
    }
    bytes = salt_gemma4_text_kv_export_size(model);
    file_bytes = (off_t)bytes;
    if (bytes <= G4_KV_FILE_HEADER_BYTES || file_bytes < 0 ||
        (size_t)file_bytes != bytes) {
        if (error && error_size) snprintf(error, error_size, "invalid KV export size");
        goto done;
    }
    fd = open(path, O_RDWR | O_CREAT | O_EXCL | O_NOFOLLOW, 0600);
    if (fd < 0) {
        g4_kv_file_error(error, error_size, "create", path);
        goto done;
    }
    if (ftruncate(fd, file_bytes) != 0) {
        g4_kv_file_error(error, error_size, "size", path);
        goto done;
    }
    mapping = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (mapping == MAP_FAILED) {
        g4_kv_file_error(error, error_size, "mmap", path);
        goto done;
    }
    if (salt_gemma4_text_kv_export(model, mapping, bytes) != 0) {
        if (error && error_size) snprintf(error, error_size, "KV cache export failed");
        goto done;
    }
    if (msync(mapping, bytes, MS_SYNC) != 0 || fsync(fd) != 0) {
        g4_kv_file_error(error, error_size, "sync", path);
        goto done;
    }
    *saved_position = salt_gemma4_text_position(model);
    if (*saved_position > 0 &&
        salt_gemma4_text_state_transaction_finish(model, &transaction) == 0)
        rc = 0;

done:
    if (mapping != MAP_FAILED) munmap(mapping, bytes);
    if (fd >= 0) close(fd);
    salt_state_transaction_abort(&transaction);
    if (rc != 0) unlink(path);
    return rc;
}

#endif /* SALT_MODELS_GEMMA4_KV_FILE_H */
