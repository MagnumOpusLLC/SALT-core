#if defined(__APPLE__) && !defined(_DARWIN_C_SOURCE)
#define _DARWIN_C_SOURCE
#endif
#if !defined(__APPLE__) && !defined(_GNU_SOURCE)
#define _POSIX_C_SOURCE 200809L
#endif

#include "salt/dpr_store.h"
#include "sha256.h"

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#ifndef O_NOFOLLOW
#define O_NOFOLLOW 0
#endif

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

#define DPR_REFERENCE_MAX_BYTES \
    (SALT_DPR_REFERENCE_HEADER_BYTES + SALT_DPR_PATH_MAX)

static const uint8_t EDGE_MAGIC[8] = {'S','A','L','T','D','P','R','E'};
static const uint8_t FAMILY_MAGIC[8] = {'S','A','L','T','D','P','F','M'};
static const uint8_t BINDING_MAGIC[8] = {'S','A','L','T','D','P','F','B'};
static const uint8_t ATTENTION_MAGIC[8] = {'S','A','L','T','D','P','N','M'};
static const uint8_t REFERENCE_MAGIC[8] = {'S','A','L','T','D','P','R','R'};

static void le32_store(uint8_t *out, uint32_t value) {
    out[0] = (uint8_t)value;
    out[1] = (uint8_t)(value >> 8);
    out[2] = (uint8_t)(value >> 16);
    out[3] = (uint8_t)(value >> 24);
}

static void le64_store(uint8_t *out, uint64_t value) {
    for (unsigned i = 0; i < 8; i++) out[i] = (uint8_t)(value >> (8u * i));
}

static uint32_t le32_load(const uint8_t *in) {
    return (uint32_t)in[0] |
           ((uint32_t)in[1] << 8) |
           ((uint32_t)in[2] << 16) |
           ((uint32_t)in[3] << 24);
}

static uint64_t le64_load(const uint8_t *in) {
    uint64_t value = 0;
    for (unsigned i = 0; i < 8; i++) value |= (uint64_t)in[i] << (8u * i);
    return value;
}

static int digest_is_zero(const uint8_t value[32]) {
    uint8_t total = 0;
    if (!value) return 1;
    for (size_t i = 0; i < 32; i++) total |= value[i];
    return total == 0;
}

static uint64_t stat_mtime_ns(const struct stat *value) {
#if defined(__APPLE__)
    return (uint64_t)value->st_mtimespec.tv_sec * UINT64_C(1000000000) +
           (uint64_t)value->st_mtimespec.tv_nsec;
#else
    return (uint64_t)value->st_mtim.tv_sec * UINT64_C(1000000000) +
           (uint64_t)value->st_mtim.tv_nsec;
#endif
}

static uint64_t stat_ctime_ns(const struct stat *value) {
#if defined(__APPLE__)
    return (uint64_t)value->st_ctimespec.tv_sec * UINT64_C(1000000000) +
           (uint64_t)value->st_ctimespec.tv_nsec;
#else
    return (uint64_t)value->st_ctim.tv_sec * UINT64_C(1000000000) +
           (uint64_t)value->st_ctim.tv_nsec;
#endif
}

static int stat_private_regular(const struct stat *value) {
    return value && S_ISREG(value->st_mode) && value->st_nlink == 1 &&
           value->st_uid == geteuid() && (value->st_mode & 077) == 0 &&
           value->st_size >= 0;
}

static int stat_private_directory(const struct stat *value) {
    return value && S_ISDIR(value->st_mode) && value->st_uid == geteuid() &&
           (value->st_mode & 077) == 0;
}

static int same_identity(const struct stat *left, const struct stat *right) {
    return left && right && left->st_dev == right->st_dev &&
           left->st_ino == right->st_ino && left->st_size == right->st_size &&
           stat_mtime_ns(left) == stat_mtime_ns(right) &&
           stat_ctime_ns(left) == stat_ctime_ns(right);
}

static int ensure_private_directory(const char *path, int create) {
    struct stat value;
    if (!path || !*path) return -1;
    if (lstat(path, &value) != 0) {
        if (!create || errno != ENOENT || mkdir(path, 0700) != 0 ||
            lstat(path, &value) != 0)
            return -1;
    }
    return stat_private_directory(&value) ? 0 : -1;
}

static int join_path(const char *root, const char *suffix,
                     char *out, size_t out_size) {
    int count;
    if (!root || !*root || !suffix || !*suffix || !out || out_size == 0)
        return -1;
    count = snprintf(out, out_size, "%s/%s", root, suffix);
    return count > 0 && (size_t)count < out_size ? 0 : -1;
}

static void hex_encode(const uint8_t value[32], char out[65]) {
    static const char HEX[] = "0123456789abcdef";
    for (size_t i = 0; i < 32; i++) {
        out[2 * i] = HEX[value[i] >> 4];
        out[2 * i + 1] = HEX[value[i] & 15u];
    }
    out[64] = 0;
}

static int hex_parse(const char *text, uint8_t out[32]) {
    if (!text || !out || strlen(text) != 64) return -1;
    for (size_t i = 0; i < 32; i++) {
        unsigned hi, lo;
        char a = text[2 * i], b = text[2 * i + 1];
        if (a >= '0' && a <= '9') hi = (unsigned)(a - '0');
        else if (a >= 'a' && a <= 'f') hi = (unsigned)(a - 'a' + 10);
        else return -1;
        if (b >= '0' && b <= '9') lo = (unsigned)(b - '0');
        else if (b >= 'a' && b <= 'f') lo = (unsigned)(b - 'a' + 10);
        else return -1;
        out[i] = (uint8_t)((hi << 4) | lo);
    }
    return 0;
}

static int write_all(int fd, const uint8_t *bytes, size_t count) {
    size_t offset = 0;
    while (offset < count) {
        ssize_t written = write(fd, bytes + offset, count - offset);
        if (written <= 0) return -1;
        offset += (size_t)written;
    }
    return 0;
}

static int read_exact_private(const char *path, uint8_t *bytes,
                              size_t expected_bytes) {
    int fd = -1, rc = -1;
    struct stat before, after;
    size_t offset = 0;
    if (!path || !bytes || expected_bytes == 0) return -1;
    fd = open(path, O_RDONLY | O_NOFOLLOW);
    if (fd < 0 || fstat(fd, &before) != 0 ||
        !stat_private_regular(&before) ||
        (uint64_t)before.st_size != (uint64_t)expected_bytes)
        goto done;
    while (offset < expected_bytes) {
        ssize_t count = read(fd, bytes + offset, expected_bytes - offset);
        if (count <= 0) goto done;
        offset += (size_t)count;
    }
    if (read(fd, bytes, 1) != 0 || fstat(fd, &after) != 0 ||
        !same_identity(&before, &after))
        goto done;
    rc = 0;
done:
    if (fd >= 0) close(fd);
    return rc;
}

static int publish_atomic(const char *directory, const char *target,
                          const uint8_t *bytes, size_t count) {
    char staging[PATH_MAX];
    int fd = -1, dir_fd = -1, rc = -1;
    uint8_t existing[DPR_REFERENCE_MAX_BYTES];
    int length;
    if (!directory || !target || !bytes || count == 0 ||
        count > sizeof existing)
        return -1;
    length = snprintf(staging, sizeof staging, "%s/.pending-XXXXXX", directory);
    if (length < 0 || (size_t)length >= sizeof staging) return -1;
    fd = mkstemp(staging);
    if (fd < 0 || fchmod(fd, 0600) != 0 ||
        write_all(fd, bytes, count) != 0 || fsync(fd) != 0)
        goto done;
    if (close(fd) != 0) {
        fd = -1;
        goto done;
    }
    fd = -1;
    if (link(staging, target) != 0) {
        if (errno != EEXIST ||
            read_exact_private(target, existing, count) != 0 ||
            memcmp(existing, bytes, count) != 0)
            goto done;
    }
    dir_fd = open(directory, O_RDONLY);
    if (dir_fd < 0 || fsync(dir_fd) != 0) goto done;
    rc = 0;
done:
    if (dir_fd >= 0) close(dir_fd);
    if (fd >= 0) close(fd);
    unlink(staging);
    return rc;
}

static int edge_structural_validate(const SaltDprEdge *edge) {
    uint64_t maximum_accepted;
    int reference_absent, next_absent;
    if (!edge) return -1;
    reference_absent = digest_is_zero(edge->reference_sha256);
    next_absent = digest_is_zero(edge->next_node_sha256);
    if (edge->schema_version != SALT_DPR_EDGE_VERSION ||
        (edge->mode != SALT_DPR_PERSIST && edge->mode != SALT_DPR_DYNAMIC) ||
        edge->horizon == 0 || edge->horizon > SALT_DPR_MAX_HORIZON ||
        edge->candidate_count != edge->horizon ||
        digest_is_zero(edge->source_node_sha256) ||
        edge->chart.horizon != edge->horizon ||
        salt_dpr_chart_validate(&edge->chart) != 0 ||
        edge->chart.rounds > UINT64_MAX / edge->horizon)
        return -1;
    if (edge->kind == SALT_DPR_EDGE_PARENT_EXACT ||
        edge->kind == SALT_DPR_EDGE_PARENT_PREFILL) {
        if (reference_absent || edge->reference_bytes == 0 ||
            edge->reference_position == 0 || !next_absent)
            return -1;
        if ((edge->kind == SALT_DPR_EDGE_PARENT_EXACT &&
             edge->bonus_token_id < 0) ||
            (edge->kind == SALT_DPR_EDGE_PARENT_PREFILL &&
             edge->bonus_token_id != -1))
            return -1;
    } else if (edge->kind == SALT_DPR_EDGE_NOMOGRAM_DRAFT ||
               edge->kind == SALT_DPR_EDGE_NOMOGRAM_PREFILL) {
        if (!reference_absent || edge->reference_bytes != 0 ||
            edge->reference_position != 0 || next_absent ||
            edge->horizon != 1 || edge->candidate_count != 1 ||
            edge->bonus_token_id != -1)
            return -1;
    } else {
        return -1;
    }
    maximum_accepted = edge->chart.rounds * edge->horizon;
    if (edge->chart.accepted_tokens_total > maximum_accepted) return -1;
    for (uint32_t i = 0; i < edge->candidate_count; i++)
        if (edge->candidate_token_ids[i] < 0) return -1;
    for (uint32_t i = edge->candidate_count; i < SALT_DPR_MAX_HORIZON; i++)
        if (edge->candidate_token_ids[i] != 0) return -1;
    return 0;
}

int salt_dpr_store_prepare(const char *root) {
    char edges[PATH_MAX], references[PATH_MAX], families[PATH_MAX];
    char bindings[PATH_MAX], attention[PATH_MAX];
    if (!root || !*root || ensure_private_directory(root, 1) != 0 ||
        join_path(root, "edges", edges, sizeof edges) != 0 ||
        join_path(root, "references", references, sizeof references) != 0 ||
        join_path(root, "families", families, sizeof families) != 0 ||
        join_path(root, "bindings", bindings, sizeof bindings) != 0 ||
        join_path(root, "attention", attention, sizeof attention) != 0 ||
        ensure_private_directory(edges, 1) != 0 ||
        ensure_private_directory(references, 1) != 0 ||
        ensure_private_directory(families, 1) != 0 ||
        ensure_private_directory(bindings, 1) != 0 ||
        ensure_private_directory(attention, 1) != 0)
        return -1;
    return 0;
}

int salt_dpr_edge_path(const char *root, const uint8_t edge_sha256[32],
                       char *out, size_t out_size) {
    char hex[65], suffix[80];
    int count;
    if (!root || !edge_sha256 || !out || out_size == 0) return -1;
    hex_encode(edge_sha256, hex);
    count = snprintf(suffix, sizeof suffix, "edges/%s.dpr", hex);
    if (count < 0 || (size_t)count >= sizeof suffix) return -1;
    return join_path(root, suffix, out, out_size);
}

int salt_dpr_reference_receipt_path(
        const char *root, const uint8_t edge_sha256[32],
        char *out, size_t out_size) {
    char hex[65], suffix[80];
    int count;
    if (!root || !edge_sha256 || !out || out_size == 0) return -1;
    hex_encode(edge_sha256, hex);
    count = snprintf(suffix, sizeof suffix, "references/%s.ref", hex);
    if (count < 0 || (size_t)count >= sizeof suffix) return -1;
    return join_path(root, suffix, out, out_size);
}

int salt_dpr_mentor_family_path(
        const char *root, const uint8_t family_sha256[32],
        char *out, size_t out_size) {
    char hex[65], suffix[96];
    int count;
    if (!root || !family_sha256 || digest_is_zero(family_sha256) ||
        !out || out_size == 0)
        return -1;
    hex_encode(family_sha256, hex);
    count = snprintf(suffix, sizeof suffix, "families/%s.family", hex);
    if (count < 0 || (size_t)count >= sizeof suffix) return -1;
    return join_path(root, suffix, out, out_size);
}

static int mentor_material_validate(
        const SaltDprMentorFamilyMaterial *material) {
    if (!material ||
        material->schema_version != SALT_DPR_MENTOR_FAMILY_VERSION ||
        digest_is_zero(material->mindset_sha256) ||
        digest_is_zero(material->state_compatibility_sha256) ||
        digest_is_zero(material->policy_sha256) ||
        digest_is_zero(material->nomogram_root_sha256) ||
        digest_is_zero(material->provenance_sha256))
        return -1;
    return 0;
}

static int mentor_family_file_sha256(
        const uint8_t bytes[SALT_DPR_MENTOR_FAMILY_FILE_BYTES],
        uint8_t out[32]) {
    SaltSha256 hasher;
    if (!bytes || !out) return -1;
    salt_sha256_init(&hasher);
    salt_sha256_update(&hasher, bytes, 184);
    salt_sha256_update(&hasher, bytes + 216,
                       SALT_DPR_MENTOR_FAMILY_FILE_BYTES - 216u);
    salt_sha256_final(&hasher, out);
    return 0;
}

int salt_dpr_mentor_family_encode(
        const SaltDprMentorFamilyMaterial *material,
        uint8_t out[SALT_DPR_MENTOR_FAMILY_FILE_BYTES],
        uint8_t family_sha256[32]) {
    if (!out || !family_sha256 || mentor_material_validate(material) != 0)
        return -1;
    memset(out, 0, SALT_DPR_MENTOR_FAMILY_FILE_BYTES);
    memcpy(out, FAMILY_MAGIC, sizeof FAMILY_MAGIC);
    le32_store(out + 8, SALT_DPR_MENTOR_FAMILY_VERSION);
    le32_store(out + 12, SALT_DPR_MENTOR_FAMILY_FILE_BYTES);
    le32_store(out + 16, material->schema_version);
    memcpy(out + 24, material->mindset_sha256, 32);
    memcpy(out + 56, material->state_compatibility_sha256, 32);
    memcpy(out + 88, material->policy_sha256, 32);
    memcpy(out + 120, material->nomogram_root_sha256, 32);
    memcpy(out + 152, material->provenance_sha256, 32);
    if (mentor_family_file_sha256(out, family_sha256) != 0) return -1;
    memcpy(out + 184, family_sha256, 32);
    return 0;
}

int salt_dpr_mentor_family_decode(
        const uint8_t bytes[SALT_DPR_MENTOR_FAMILY_FILE_BYTES],
        SaltDprMentorFamily *family) {
    uint8_t actual[32];
    if (!bytes || !family ||
        memcmp(bytes, FAMILY_MAGIC, sizeof FAMILY_MAGIC) != 0 ||
        le32_load(bytes + 8) != SALT_DPR_MENTOR_FAMILY_VERSION ||
        le32_load(bytes + 12) != SALT_DPR_MENTOR_FAMILY_FILE_BYTES ||
        le32_load(bytes + 16) != SALT_DPR_MENTOR_FAMILY_VERSION ||
        le32_load(bytes + 20) != 0 ||
        mentor_family_file_sha256(bytes, actual) != 0 ||
        memcmp(actual, bytes + 184, 32) != 0)
        return -1;
    for (size_t i = 216; i < SALT_DPR_MENTOR_FAMILY_FILE_BYTES; i++)
        if (bytes[i] != 0) return -1;
    memset(family, 0, sizeof *family);
    family->schema_version = le32_load(bytes + 16);
    memcpy(family->mindset_sha256, bytes + 24, 32);
    memcpy(family->state_compatibility_sha256, bytes + 56, 32);
    memcpy(family->policy_sha256, bytes + 88, 32);
    memcpy(family->nomogram_root_sha256, bytes + 120, 32);
    memcpy(family->provenance_sha256, bytes + 152, 32);
    memcpy(family->family_sha256, actual, 32);
    return salt_dpr_mentor_family_validate(family);
}

int salt_dpr_mentor_family_publish(
        const char *root, const SaltDprMentorFamilyMaterial *material,
        uint8_t family_sha256[32]) {
    uint8_t bytes[SALT_DPR_MENTOR_FAMILY_FILE_BYTES];
    uint8_t readback[SALT_DPR_MENTOR_FAMILY_FILE_BYTES];
    SaltDprMentorFamily decoded;
    char directory[PATH_MAX], target[PATH_MAX];
    if (!root || !family_sha256 ||
        salt_dpr_store_prepare(root) != 0 ||
        salt_dpr_mentor_family_encode(material, bytes, family_sha256) != 0 ||
        join_path(root, "families", directory, sizeof directory) != 0 ||
        salt_dpr_mentor_family_path(
            root, family_sha256, target, sizeof target) != 0 ||
        publish_atomic(directory, target, bytes, sizeof bytes) != 0 ||
        read_exact_private(target, readback, sizeof readback) != 0 ||
        salt_dpr_mentor_family_decode(readback, &decoded) != 0 ||
        memcmp(decoded.family_sha256, family_sha256, 32) != 0)
        return -1;
    return 0;
}

int salt_dpr_mentor_binding_path(
        const char *root, const uint8_t binding_sha256[32],
        char *out, size_t out_size) {
    char hex[65], suffix[96];
    int count;
    if (!root || !binding_sha256 || digest_is_zero(binding_sha256) ||
        !out || out_size == 0)
        return -1;
    hex_encode(binding_sha256, hex);
    count = snprintf(suffix, sizeof suffix, "bindings/%s.binding", hex);
    if (count < 0 || (size_t)count >= sizeof suffix) return -1;
    return join_path(root, suffix, out, out_size);
}

static int mentor_binding_material_validate(
        const SaltDprMentorBindingMaterial *material) {
    if (!material ||
        material->schema_version != SALT_DPR_MENTOR_BINDING_VERSION ||
        digest_is_zero(material->family_sha256) ||
        digest_is_zero(material->edge_sha256) ||
        digest_is_zero(material->source_node_sha256) ||
        digest_is_zero(material->state_compatibility_sha256) ||
        digest_is_zero(material->policy_sha256))
        return -1;
    return 0;
}

static int mentor_binding_validate(const SaltDprMentorBinding *binding) {
    if (!binding ||
        binding->schema_version != SALT_DPR_MENTOR_BINDING_VERSION ||
        digest_is_zero(binding->family_sha256) ||
        digest_is_zero(binding->edge_sha256) ||
        digest_is_zero(binding->source_node_sha256) ||
        digest_is_zero(binding->state_compatibility_sha256) ||
        digest_is_zero(binding->policy_sha256) ||
        digest_is_zero(binding->binding_sha256))
        return -1;
    return 0;
}

static int mentor_binding_file_sha256(
        const uint8_t bytes[SALT_DPR_MENTOR_BINDING_FILE_BYTES],
        uint8_t out[32]) {
    SaltSha256 hasher;
    if (!bytes || !out) return -1;
    salt_sha256_init(&hasher);
    salt_sha256_update(&hasher, bytes, 184);
    salt_sha256_update(&hasher, bytes + 216,
                       SALT_DPR_MENTOR_BINDING_FILE_BYTES - 216u);
    salt_sha256_final(&hasher, out);
    return 0;
}

int salt_dpr_mentor_binding_encode(
        const SaltDprMentorBindingMaterial *material,
        uint8_t out[SALT_DPR_MENTOR_BINDING_FILE_BYTES],
        uint8_t binding_sha256[32]) {
    if (!out || !binding_sha256 ||
        mentor_binding_material_validate(material) != 0)
        return -1;
    memset(out, 0, SALT_DPR_MENTOR_BINDING_FILE_BYTES);
    memcpy(out, BINDING_MAGIC, sizeof BINDING_MAGIC);
    le32_store(out + 8, SALT_DPR_MENTOR_BINDING_VERSION);
    le32_store(out + 12, SALT_DPR_MENTOR_BINDING_FILE_BYTES);
    le32_store(out + 16, material->schema_version);
    memcpy(out + 24, material->family_sha256, 32);
    memcpy(out + 56, material->edge_sha256, 32);
    memcpy(out + 88, material->source_node_sha256, 32);
    memcpy(out + 120, material->state_compatibility_sha256, 32);
    memcpy(out + 152, material->policy_sha256, 32);
    if (mentor_binding_file_sha256(out, binding_sha256) != 0) return -1;
    memcpy(out + 184, binding_sha256, 32);
    return 0;
}

int salt_dpr_mentor_binding_decode(
        const uint8_t bytes[SALT_DPR_MENTOR_BINDING_FILE_BYTES],
        SaltDprMentorBinding *binding) {
    uint8_t actual[32];
    if (!bytes || !binding ||
        memcmp(bytes, BINDING_MAGIC, sizeof BINDING_MAGIC) != 0 ||
        le32_load(bytes + 8) != SALT_DPR_MENTOR_BINDING_VERSION ||
        le32_load(bytes + 12) != SALT_DPR_MENTOR_BINDING_FILE_BYTES ||
        le32_load(bytes + 16) != SALT_DPR_MENTOR_BINDING_VERSION ||
        le32_load(bytes + 20) != 0 ||
        mentor_binding_file_sha256(bytes, actual) != 0 ||
        memcmp(actual, bytes + 184, 32) != 0)
        return -1;
    for (size_t i = 216; i < SALT_DPR_MENTOR_BINDING_FILE_BYTES; i++)
        if (bytes[i] != 0) return -1;
    memset(binding, 0, sizeof *binding);
    binding->schema_version = le32_load(bytes + 16);
    memcpy(binding->family_sha256, bytes + 24, 32);
    memcpy(binding->edge_sha256, bytes + 56, 32);
    memcpy(binding->source_node_sha256, bytes + 88, 32);
    memcpy(binding->state_compatibility_sha256, bytes + 120, 32);
    memcpy(binding->policy_sha256, bytes + 152, 32);
    memcpy(binding->binding_sha256, actual, 32);
    binding->family_index = SIZE_MAX;
    binding->edge_index = SIZE_MAX;
    return mentor_binding_validate(binding);
}

int salt_dpr_mentor_binding_publish(
        const char *root, const SaltDprMentorBindingMaterial *material,
        uint8_t binding_sha256[32]) {
    uint8_t bytes[SALT_DPR_MENTOR_BINDING_FILE_BYTES];
    uint8_t readback[SALT_DPR_MENTOR_BINDING_FILE_BYTES];
    SaltDprMentorBinding decoded;
    char directory[PATH_MAX], target[PATH_MAX];
    if (!root || !binding_sha256 ||
        salt_dpr_store_prepare(root) != 0 ||
        salt_dpr_mentor_binding_encode(material, bytes, binding_sha256) != 0 ||
        join_path(root, "bindings", directory, sizeof directory) != 0 ||
        salt_dpr_mentor_binding_path(
            root, binding_sha256, target, sizeof target) != 0 ||
        publish_atomic(directory, target, bytes, sizeof bytes) != 0 ||
        read_exact_private(target, readback, sizeof readback) != 0 ||
        salt_dpr_mentor_binding_decode(readback, &decoded) != 0 ||
        memcmp(decoded.binding_sha256, binding_sha256, 32) != 0)
        return -1;
    return 0;
}

int salt_dpr_attention_plan_path(
        const char *root, const uint8_t plan_sha256[32],
        char *out, size_t out_size) {
    char hex[65], suffix[96];
    int count;
    if (!root || !plan_sha256 || digest_is_zero(plan_sha256) ||
        !out || out_size == 0)
        return -1;
    hex_encode(plan_sha256, hex);
    count = snprintf(suffix, sizeof suffix, "attention/%s.attention", hex);
    if (count < 0 || (size_t)count >= sizeof suffix) return -1;
    return join_path(root, suffix, out, out_size);
}

static int attention_item_same(const SaltDprRelevanceItem *left,
                               const SaltDprRelevanceItem *right) {
    return left->kind == right->kind && left->layer == right->layer &&
        left->index == right->index && left->start == right->start &&
        left->length == right->length;
}

static int attention_plan_structural_validate(const SaltDprAttentionPlan *plan) {
    if (!plan || plan->schema_version != SALT_DPR_ATTENTION_PLAN_VERSION ||
        plan->exactness != SALT_DPR_EXACTNESS_A0 ||
        digest_is_zero(plan->dpr_sha256) ||
        digest_is_zero(plan->mindset_sha256) ||
        digest_is_zero(plan->provenance_sha256) ||
        digest_is_zero(plan->intent_sha256) ||
        digest_is_zero(plan->state_compatibility_sha256) ||
        digest_is_zero(plan->policy_sha256) ||
        plan->item_count == 0 ||
        plan->item_count > SALT_DPR_MAX_RELEVANCE_ITEMS ||
        plan->expected_saved_numerator == 0 ||
        plan->expected_saved_denominator == 0 ||
        plan->provenance_authenticated != 1 || plan->policy_allowed != 1)
        return -1;
    for (uint32_t i = 0; i < SALT_DPR_MAX_RELEVANCE_ITEMS; i++) {
        const SaltDprRelevanceItem *item = &plan->items[i];
        if (i >= plan->item_count) {
            if (item->kind != SALT_DPR_RELEVANCE_INVALID || item->layer != 0 ||
                item->index != 0 || item->start != 0 || item->length != 0 ||
                item->priority != 0)
                return -1;
            continue;
        }
        if (item->priority == 0 ||
            (i > 0 && item->priority > plan->items[i - 1].priority))
            return -1;
        if (item->kind == SALT_DPR_RELEVANCE_EXPERT_PREFETCH) {
            if (item->start != 0 || item->length != 0) return -1;
        } else if (item->kind == SALT_DPR_RELEVANCE_KV_PREFETCH) {
            if (item->index != 0 || item->length == 0) return -1;
        } else {
            return -1;
        }
        for (uint32_t prior = 0; prior < i; prior++)
            if (attention_item_same(item, &plan->items[prior])) return -1;
    }
    return 0;
}

static int attention_plan_file_sha256(
        const uint8_t bytes[SALT_DPR_ATTENTION_PLAN_FILE_BYTES],
        uint8_t out[32]) {
    SaltSha256 hasher;
    if (!bytes || !out) return -1;
    salt_sha256_init(&hasher);
    salt_sha256_update(&hasher, bytes, 2288);
    salt_sha256_update(&hasher, bytes + 2320,
                       SALT_DPR_ATTENTION_PLAN_FILE_BYTES - 2320u);
    salt_sha256_final(&hasher, out);
    return 0;
}

int salt_dpr_attention_plan_encode(
        const SaltDprAttentionPlan *plan,
        uint8_t out[SALT_DPR_ATTENTION_PLAN_FILE_BYTES],
        uint8_t plan_sha256[32]) {
    if (!out || !plan_sha256 || attention_plan_structural_validate(plan) != 0)
        return -1;
    memset(out, 0, SALT_DPR_ATTENTION_PLAN_FILE_BYTES);
    memcpy(out, ATTENTION_MAGIC, sizeof ATTENTION_MAGIC);
    le32_store(out + 8, SALT_DPR_ATTENTION_PLAN_VERSION);
    le32_store(out + 12, SALT_DPR_ATTENTION_PLAN_FILE_BYTES);
    le32_store(out + 16, plan->schema_version);
    le32_store(out + 20, (uint32_t)plan->exactness);
    le32_store(out + 24, plan->item_count);
    le32_store(out + 28, 3u);
    memcpy(out + 32, plan->dpr_sha256, 32);
    memcpy(out + 64, plan->mindset_sha256, 32);
    memcpy(out + 96, plan->provenance_sha256, 32);
    memcpy(out + 128, plan->intent_sha256, 32);
    memcpy(out + 160, plan->state_compatibility_sha256, 32);
    memcpy(out + 192, plan->policy_sha256, 32);
    le64_store(out + 224, plan->expected_saved_numerator);
    le64_store(out + 232, plan->expected_saved_denominator);
    for (uint32_t i = 0; i < SALT_DPR_MAX_RELEVANCE_ITEMS; i++) {
        uint8_t *item = out + 240 + (size_t)i * 32u;
        le32_store(item, (uint32_t)plan->items[i].kind);
        le32_store(item + 4, plan->items[i].layer);
        le32_store(item + 8, plan->items[i].index);
        le32_store(item + 12, plan->items[i].priority);
        le64_store(item + 16, plan->items[i].start);
        le64_store(item + 24, plan->items[i].length);
    }
    if (attention_plan_file_sha256(out, plan_sha256) != 0) return -1;
    memcpy(out + 2288, plan_sha256, 32);
    return 0;
}

int salt_dpr_attention_plan_decode(
        const uint8_t bytes[SALT_DPR_ATTENTION_PLAN_FILE_BYTES],
        SaltDprAttentionPlan *plan, uint8_t plan_sha256[32]) {
    uint8_t actual[32];
    if (!bytes || !plan || !plan_sha256 ||
        memcmp(bytes, ATTENTION_MAGIC, sizeof ATTENTION_MAGIC) != 0 ||
        le32_load(bytes + 8) != SALT_DPR_ATTENTION_PLAN_VERSION ||
        le32_load(bytes + 12) != SALT_DPR_ATTENTION_PLAN_FILE_BYTES ||
        le32_load(bytes + 16) != SALT_DPR_ATTENTION_PLAN_VERSION ||
        le32_load(bytes + 28) != 3u ||
        attention_plan_file_sha256(bytes, actual) != 0 ||
        memcmp(actual, bytes + 2288, 32) != 0)
        return -1;
    for (size_t i = 2320; i < SALT_DPR_ATTENTION_PLAN_FILE_BYTES; i++)
        if (bytes[i] != 0) return -1;
    memset(plan, 0, sizeof *plan);
    plan->schema_version = le32_load(bytes + 16);
    plan->exactness = (SaltDprExactnessLevel)le32_load(bytes + 20);
    plan->item_count = le32_load(bytes + 24);
    memcpy(plan->dpr_sha256, bytes + 32, 32);
    memcpy(plan->mindset_sha256, bytes + 64, 32);
    memcpy(plan->provenance_sha256, bytes + 96, 32);
    memcpy(plan->intent_sha256, bytes + 128, 32);
    memcpy(plan->state_compatibility_sha256, bytes + 160, 32);
    memcpy(plan->policy_sha256, bytes + 192, 32);
    plan->expected_saved_numerator = le64_load(bytes + 224);
    plan->expected_saved_denominator = le64_load(bytes + 232);
    for (uint32_t i = 0; i < SALT_DPR_MAX_RELEVANCE_ITEMS; i++) {
        const uint8_t *item = bytes + 240 + (size_t)i * 32u;
        plan->items[i].kind = (SaltDprRelevanceKind)le32_load(item);
        plan->items[i].layer = le32_load(item + 4);
        plan->items[i].index = le32_load(item + 8);
        plan->items[i].priority = le32_load(item + 12);
        plan->items[i].start = le64_load(item + 16);
        plan->items[i].length = le64_load(item + 24);
    }
    plan->provenance_authenticated = 1;
    plan->policy_allowed = 1;
    memcpy(plan_sha256, actual, 32);
    return attention_plan_structural_validate(plan);
}

int salt_dpr_attention_plan_publish(
        const char *root, const SaltDprAttentionPlan *plan,
        uint8_t plan_sha256[32]) {
    uint8_t bytes[SALT_DPR_ATTENTION_PLAN_FILE_BYTES];
    uint8_t readback[SALT_DPR_ATTENTION_PLAN_FILE_BYTES], decoded_sha[32];
    SaltDprAttentionPlan decoded;
    char directory[PATH_MAX], target[PATH_MAX];
    if (!root || !plan_sha256 || salt_dpr_store_prepare(root) != 0 ||
        salt_dpr_attention_plan_encode(plan, bytes, plan_sha256) != 0 ||
        join_path(root, "attention", directory, sizeof directory) != 0 ||
        salt_dpr_attention_plan_path(
            root, plan_sha256, target, sizeof target) != 0 ||
        publish_atomic(directory, target, bytes, sizeof bytes) != 0 ||
        read_exact_private(target, readback, sizeof readback) != 0 ||
        salt_dpr_attention_plan_decode(
            readback, &decoded, decoded_sha) != 0 ||
        memcmp(decoded_sha, plan_sha256, 32) != 0)
        return -1;
    return 0;
}

static int edge_file_sha256(
        const uint8_t bytes[SALT_DPR_EDGE_FILE_BYTES], uint8_t out[32]) {
    SaltSha256 hasher;
    if (!bytes || !out) return -1;
    salt_sha256_init(&hasher);
    salt_sha256_update(&hasher, bytes, 440);
    salt_sha256_update(&hasher, bytes + 472, 40);
    salt_sha256_final(&hasher, out);
    return 0;
}

int salt_dpr_edge_encode(const SaltDprEdge *edge,
                         uint8_t out[SALT_DPR_EDGE_FILE_BYTES],
                         uint8_t edge_sha256[32]) {
    if (!out || !edge_sha256 || edge_structural_validate(edge) != 0)
        return -1;
    memset(out, 0, SALT_DPR_EDGE_FILE_BYTES);
    memcpy(out, EDGE_MAGIC, sizeof EDGE_MAGIC);
    le32_store(out + 8, SALT_DPR_EDGE_VERSION);
    le32_store(out + 12, SALT_DPR_EDGE_FILE_BYTES);
    le32_store(out + 16, (uint32_t)edge->mode);
    le32_store(out + 20, edge->horizon);
    le32_store(out + 24, edge->candidate_count);
    memcpy(out + 32, edge->source_node_sha256, 32);
    memcpy(out + 64, edge->reference_sha256, 32);
    le64_store(out + 96, edge->reference_bytes);
    le64_store(out + 104, edge->reference_position);
    for (uint32_t i = 0; i < SALT_DPR_MAX_HORIZON; i++)
        le32_store(out + 112 + (size_t)i * 4u,
                   (uint32_t)edge->candidate_token_ids[i]);
    le32_store(out + 368, edge->chart.horizon);
    le64_store(out + 376, edge->chart.lookup_ns);
    le64_store(out + 384, edge->chart.recover_ns);
    le64_store(out + 392, edge->chart.draft_ns);
    le64_store(out + 400, edge->chart.verify_ns);
    le64_store(out + 408, edge->chart.commit_ns);
    le64_store(out + 416, edge->chart.accepted_tokens_total);
    le64_store(out + 424, edge->chart.rounds);
    le64_store(out + 432, edge->chart.sample_count);
    le32_store(out + 472, (uint32_t)edge->kind);
    le32_store(out + 476, (uint32_t)edge->bonus_token_id);
    memcpy(out + 480, edge->next_node_sha256, 32);
    if (edge_file_sha256(out, edge_sha256) != 0) return -1;
    memcpy(out + 440, edge_sha256, 32);
    return 0;
}

int salt_dpr_edge_decode(const uint8_t bytes[SALT_DPR_EDGE_FILE_BYTES],
                         SaltDprEdge *edge, uint8_t edge_sha256[32]) {
    uint8_t actual[32];
    if (!bytes || !edge || !edge_sha256 ||
        memcmp(bytes, EDGE_MAGIC, sizeof EDGE_MAGIC) != 0 ||
        le32_load(bytes + 8) != SALT_DPR_EDGE_VERSION ||
        le32_load(bytes + 12) != SALT_DPR_EDGE_FILE_BYTES ||
        le32_load(bytes + 28) != 0 || le32_load(bytes + 372) != 0 ||
        edge_file_sha256(bytes, actual) != 0 ||
        memcmp(actual, bytes + 440, 32) != 0)
        return -1;
    memset(edge, 0, sizeof *edge);
    edge->schema_version = le32_load(bytes + 8);
    edge->mode = (SaltDprMode)le32_load(bytes + 16);
    edge->horizon = le32_load(bytes + 20);
    edge->candidate_count = le32_load(bytes + 24);
    memcpy(edge->source_node_sha256, bytes + 32, 32);
    memcpy(edge->reference_sha256, bytes + 64, 32);
    edge->reference_bytes = le64_load(bytes + 96);
    edge->reference_position = le64_load(bytes + 104);
    for (uint32_t i = 0; i < SALT_DPR_MAX_HORIZON; i++)
        edge->candidate_token_ids[i] =
            (int32_t)le32_load(bytes + 112 + (size_t)i * 4u);
    edge->chart.horizon = le32_load(bytes + 368);
    edge->chart.lookup_ns = le64_load(bytes + 376);
    edge->chart.recover_ns = le64_load(bytes + 384);
    edge->chart.draft_ns = le64_load(bytes + 392);
    edge->chart.verify_ns = le64_load(bytes + 400);
    edge->chart.commit_ns = le64_load(bytes + 408);
    edge->chart.accepted_tokens_total = le64_load(bytes + 416);
    edge->chart.rounds = le64_load(bytes + 424);
    edge->chart.sample_count = le64_load(bytes + 432);
    edge->kind = (SaltDprEdgeKind)le32_load(bytes + 472);
    edge->bonus_token_id = (int32_t)le32_load(bytes + 476);
    memcpy(edge->next_node_sha256, bytes + 480, 32);
    if (edge_structural_validate(edge) != 0) return -1;
    memcpy(edge_sha256, actual, 32);
    return 0;
}

int salt_dpr_edge_publish(const char *root, const SaltDprEdge *edge,
                          uint8_t edge_sha256[32]) {
    uint8_t bytes[SALT_DPR_EDGE_FILE_BYTES], readback[SALT_DPR_EDGE_FILE_BYTES];
    uint8_t readback_sha[32];
    SaltDprEdge readback_edge;
    char directory[PATH_MAX], target[PATH_MAX];
    if (salt_dpr_store_prepare(root) != 0 ||
        salt_dpr_edge_encode(edge, bytes, edge_sha256) != 0 ||
        join_path(root, "edges", directory, sizeof directory) != 0 ||
        salt_dpr_edge_path(root, edge_sha256, target, sizeof target) != 0 ||
        publish_atomic(directory, target, bytes, sizeof bytes) != 0 ||
        read_exact_private(target, readback, sizeof readback) != 0 ||
        salt_dpr_edge_decode(readback, &readback_edge, readback_sha) != 0 ||
        memcmp(readback_sha, edge_sha256, 32) != 0)
        return -1;
    return 0;
}

int salt_dpr_reference_measure(const char *path,
                               SaltDprReferenceMeasurement *measurement) {
    int fd = -1, rc = -1;
    struct stat before, after;
    if (!path || !*path || !measurement) return -1;
    memset(measurement, 0, sizeof *measurement);
    fd = open(path, O_RDONLY | O_NOFOLLOW);
    if (fd < 0 || fstat(fd, &before) != 0 ||
        !stat_private_regular(&before) || before.st_size == 0 ||
        salt_sha256_fd(fd, measurement->file_sha256) != 0 ||
        fstat(fd, &after) != 0 || !same_identity(&before, &after))
        goto done;
    measurement->bytes = (uint64_t)before.st_size;
    measurement->device = (uint64_t)before.st_dev;
    measurement->inode = (uint64_t)before.st_ino;
    measurement->mtime_ns = stat_mtime_ns(&before);
    measurement->ctime_ns = stat_ctime_ns(&before);
    rc = 0;
done:
    if (fd >= 0) close(fd);
    return rc;
}

static int current_measurement(const char *path,
                               SaltDprReferenceMeasurement *measurement) {
    int fd = -1, rc = -1;
    struct stat value;
    if (!path || !measurement) return -1;
    fd = open(path, O_RDONLY | O_NOFOLLOW);
    if (fd < 0 || fstat(fd, &value) != 0 ||
        !stat_private_regular(&value) || value.st_size == 0)
        goto done;
    memset(measurement, 0, sizeof *measurement);
    measurement->bytes = (uint64_t)value.st_size;
    measurement->device = (uint64_t)value.st_dev;
    measurement->inode = (uint64_t)value.st_ino;
    measurement->mtime_ns = stat_mtime_ns(&value);
    measurement->ctime_ns = stat_ctime_ns(&value);
    rc = 0;
done:
    if (fd >= 0) close(fd);
    return rc;
}

static int measurement_identity_equal(
        const SaltDprReferenceMeasurement *left,
        const SaltDprReferenceMeasurement *right) {
    return left && right && left->bytes == right->bytes &&
           left->device == right->device && left->inode == right->inode &&
           left->mtime_ns == right->mtime_ns && left->ctime_ns == right->ctime_ns;
}

static int reference_encode(
        const uint8_t edge_sha256[32], const SaltDprEdge *edge,
        const char *reference_path,
        const SaltDprReferenceMeasurement *measurement,
        uint8_t *out, size_t out_capacity, size_t *out_bytes) {
    SaltSha256 hasher;
    uint8_t digest[32];
    size_t path_bytes, total;
    if (!edge_sha256 || !edge || !reference_path || reference_path[0] != '/' ||
        !measurement || !out || !out_bytes ||
        memcmp(edge->reference_sha256, measurement->file_sha256, 32) != 0 ||
        edge->reference_bytes != measurement->bytes)
        return -1;
    path_bytes = strlen(reference_path);
    if (path_bytes == 0 || path_bytes > SALT_DPR_PATH_MAX ||
        path_bytes > SIZE_MAX - SALT_DPR_REFERENCE_HEADER_BYTES)
        return -1;
    total = SALT_DPR_REFERENCE_HEADER_BYTES + path_bytes;
    if (total > out_capacity) return -1;
    memset(out, 0, total);
    memcpy(out, REFERENCE_MAGIC, sizeof REFERENCE_MAGIC);
    le32_store(out + 8, 1u);
    le32_store(out + 12, SALT_DPR_REFERENCE_HEADER_BYTES);
    le64_store(out + 16, (uint64_t)total);
    memcpy(out + 24, edge_sha256, 32);
    memcpy(out + 56, measurement->file_sha256, 32);
    le64_store(out + 88, measurement->bytes);
    le64_store(out + 96, edge->reference_position);
    le64_store(out + 104, measurement->device);
    le64_store(out + 112, measurement->inode);
    le64_store(out + 120, measurement->mtime_ns);
    le64_store(out + 128, measurement->ctime_ns);
    le32_store(out + 136, (uint32_t)path_bytes);
    memcpy(out + SALT_DPR_REFERENCE_HEADER_BYTES, reference_path, path_bytes);
    salt_sha256_init(&hasher);
    salt_sha256_update(&hasher, out, 144);
    salt_sha256_update(&hasher, out + 176, total - 176);
    salt_sha256_final(&hasher, digest);
    memcpy(out + 144, digest, 32);
    *out_bytes = total;
    return 0;
}

static int reference_decode(const uint8_t *bytes, size_t count,
                            SaltDprReferenceLease *receipt) {
    SaltSha256 hasher;
    uint8_t digest[32];
    uint32_t path_bytes;
    uint64_t encoded_bytes;
    if (!bytes || !receipt || count < SALT_DPR_REFERENCE_HEADER_BYTES ||
        count > DPR_REFERENCE_MAX_BYTES ||
        memcmp(bytes, REFERENCE_MAGIC, sizeof REFERENCE_MAGIC) != 0 ||
        le32_load(bytes + 8) != 1u ||
        le32_load(bytes + 12) != SALT_DPR_REFERENCE_HEADER_BYTES)
        return -1;
    encoded_bytes = le64_load(bytes + 16);
    path_bytes = le32_load(bytes + 136);
    if (encoded_bytes != (uint64_t)count || path_bytes == 0 ||
        path_bytes > SALT_DPR_PATH_MAX ||
        SALT_DPR_REFERENCE_HEADER_BYTES + (size_t)path_bytes != count ||
        bytes[SALT_DPR_REFERENCE_HEADER_BYTES] != '/')
        return -1;
    for (size_t i = 140; i < 144; i++) if (bytes[i] != 0) return -1;
    for (size_t i = 176; i < SALT_DPR_REFERENCE_HEADER_BYTES; i++)
        if (bytes[i] != 0) return -1;
    if (memchr(bytes + SALT_DPR_REFERENCE_HEADER_BYTES, 0, path_bytes))
        return -1;
    salt_sha256_init(&hasher);
    salt_sha256_update(&hasher, bytes, 144);
    salt_sha256_update(&hasher, bytes + 176, count - 176);
    salt_sha256_final(&hasher, digest);
    if (memcmp(digest, bytes + 144, 32) != 0) return -1;
    memset(receipt, 0, sizeof *receipt);
    receipt->fd = -1;
    memcpy(receipt->edge_sha256, bytes + 24, 32);
    memcpy(receipt->reference_sha256, bytes + 56, 32);
    receipt->bytes = le64_load(bytes + 88);
    receipt->position = le64_load(bytes + 96);
    receipt->device = le64_load(bytes + 104);
    receipt->inode = le64_load(bytes + 112);
    receipt->mtime_ns = le64_load(bytes + 120);
    receipt->ctime_ns = le64_load(bytes + 128);
    memcpy(receipt->path, bytes + SALT_DPR_REFERENCE_HEADER_BYTES, path_bytes);
    receipt->path[path_bytes] = 0;
    return 0;
}

int salt_dpr_reference_bind(const char *root,
                            const uint8_t edge_sha256[32],
                            const SaltDprEdge *edge,
                            const char *reference_path,
                            const SaltDprReferenceMeasurement *measurement) {
    SaltDprReferenceMeasurement current;
    uint8_t edge_bytes[SALT_DPR_EDGE_FILE_BYTES], computed_edge_sha[32];
    uint8_t bytes[DPR_REFERENCE_MAX_BYTES];
    size_t count = 0;
    char directory[PATH_MAX], target[PATH_MAX];
    if (!root || !edge_sha256 || !edge || !reference_path || !measurement ||
        edge_structural_validate(edge) != 0 ||
        salt_dpr_edge_encode(edge, edge_bytes, computed_edge_sha) != 0 ||
        memcmp(edge_sha256, computed_edge_sha, 32) != 0 ||
        current_measurement(reference_path, &current) != 0 ||
        !measurement_identity_equal(&current, measurement) ||
        salt_dpr_store_prepare(root) != 0 ||
        reference_encode(edge_sha256, edge, reference_path, measurement,
                         bytes, sizeof bytes, &count) != 0 ||
        join_path(root, "references", directory, sizeof directory) != 0 ||
        salt_dpr_reference_receipt_path(
            root, edge_sha256, target, sizeof target) != 0 ||
        publish_atomic(directory, target, bytes, count) != 0)
        return -1;
    return 0;
}

static int read_receipt(const char *path, SaltDprReferenceLease *receipt) {
    int fd = -1, rc = -1;
    struct stat before, after;
    uint8_t bytes[DPR_REFERENCE_MAX_BYTES];
    size_t count, offset = 0;
    fd = open(path, O_RDONLY | O_NOFOLLOW);
    if (fd < 0 || fstat(fd, &before) != 0 ||
        !stat_private_regular(&before) ||
        before.st_size < (off_t)SALT_DPR_REFERENCE_HEADER_BYTES ||
        (uint64_t)before.st_size > (uint64_t)sizeof bytes)
        goto done;
    count = (size_t)before.st_size;
    while (offset < count) {
        ssize_t got = read(fd, bytes + offset, count - offset);
        if (got <= 0) goto done;
        offset += (size_t)got;
    }
    if (read(fd, bytes, 1) != 0 || fstat(fd, &after) != 0 ||
        !same_identity(&before, &after) ||
        reference_decode(bytes, count, receipt) != 0)
        goto done;
    rc = 0;
done:
    if (fd >= 0) close(fd);
    return rc;
}

int salt_dpr_reference_open(const char *root,
                            const SaltDprStoredEdge *stored_edge,
                            SaltDprReferenceLease *lease) {
    SaltDprReferenceLease receipt;
    struct stat value;
    char receipt_path[PATH_MAX];
    int fd = -1;
    if (!root || !stored_edge || !lease ||
        salt_dpr_reference_receipt_path(
            root, stored_edge->edge_sha256,
            receipt_path, sizeof receipt_path) != 0 ||
        read_receipt(receipt_path, &receipt) != 0 ||
        memcmp(receipt.edge_sha256, stored_edge->edge_sha256, 32) != 0 ||
        memcmp(receipt.reference_sha256,
               stored_edge->edge.reference_sha256, 32) != 0 ||
        receipt.bytes != stored_edge->edge.reference_bytes ||
        receipt.position != stored_edge->edge.reference_position)
        return -1;
    fd = open(receipt.path, O_RDONLY | O_NOFOLLOW);
    if (fd < 0 || fstat(fd, &value) != 0 ||
        !stat_private_regular(&value) ||
        (uint64_t)value.st_size != receipt.bytes ||
        (uint64_t)value.st_dev != receipt.device ||
        (uint64_t)value.st_ino != receipt.inode ||
        stat_mtime_ns(&value) != receipt.mtime_ns ||
        stat_ctime_ns(&value) != receipt.ctime_ns) {
        if (fd >= 0) close(fd);
        return -1;
    }
    *lease = receipt;
    lease->fd = fd;
    return 0;
}

void salt_dpr_reference_close(SaltDprReferenceLease *lease) {
    if (!lease) return;
    if (lease->fd >= 0) close(lease->fd);
    memset(lease, 0, sizeof *lease);
    lease->fd = -1;
}

int salt_dpr_store_init(SaltDprStore *store,
                        SaltDprStoredEdge *edge_slots,
                        size_t edge_capacity,
                        uint32_t *buckets,
                        size_t bucket_count) {
    if (!store || !edge_slots || edge_capacity == 0 || !buckets ||
        bucket_count == 0 || edge_capacity > UINT32_MAX)
        return -1;
    memset(store, 0, sizeof *store);
    memset(edge_slots, 0, edge_capacity * sizeof *edge_slots);
    for (size_t i = 0; i < bucket_count; i++) buckets[i] = SALT_DPR_STORE_NONE;
    store->edges = edge_slots;
    store->edge_capacity = edge_capacity;
    store->buckets = buckets;
    store->bucket_count = bucket_count;
    return 0;
}

static size_t node_bucket(const uint8_t node[32], size_t bucket_count) {
    uint64_t hash = UINT64_C(1469598103934665603);
    for (size_t i = 0; i < 32; i++) {
        hash ^= node[i];
        hash *= UINT64_C(1099511628211);
    }
    return (size_t)(hash % bucket_count);
}

int salt_dpr_attention_plan_store_init(
        SaltDprAttentionPlanStore *store,
        SaltDprStoredAttentionPlan *plan_slots, size_t plan_capacity,
        uint32_t *buckets, size_t bucket_count) {
    if (!store || !plan_slots || plan_capacity == 0 || !buckets ||
        bucket_count == 0 || plan_capacity > UINT32_MAX)
        return -1;
    memset(store, 0, sizeof *store);
    memset(plan_slots, 0, plan_capacity * sizeof *plan_slots);
    for (size_t i = 0; i < plan_capacity; i++)
        plan_slots[i].next_index = SALT_DPR_STORE_NONE;
    for (size_t i = 0; i < bucket_count; i++)
        buckets[i] = SALT_DPR_STORE_NONE;
    store->plans = plan_slots;
    store->plan_capacity = plan_capacity;
    store->buckets = buckets;
    store->bucket_count = bucket_count;
    store->runtime_index = SIZE_MAX;
    return 0;
}

static int attention_filename_sha(const char *name, uint8_t out[32]) {
    char hex[65];
    if (!name || !out || strlen(name) != 74 ||
        strcmp(name + 64, ".attention") != 0)
        return -1;
    memcpy(hex, name, 64);
    hex[64] = 0;
    return hex_parse(hex, out);
}

int salt_dpr_attention_plan_store_load(
        SaltDprAttentionPlanStore *store, const char *root) {
    char directory[PATH_MAX], path[PATH_MAX];
    DIR *dir = NULL;
    struct dirent *entry;
    if (!store || !store->plans || !store->buckets || !root ||
        store->plan_capacity == 0 || store->bucket_count == 0 ||
        salt_dpr_store_prepare(root) != 0 ||
        join_path(root, "attention", directory, sizeof directory) != 0)
        return -1;
    store->plan_count = 0;
    store->rejected_plans = 0;
    store->runtime_index = SIZE_MAX;
    store->runtime_present = 0;
    memset(store->plans, 0, store->plan_capacity * sizeof *store->plans);
    for (size_t i = 0; i < store->plan_capacity; i++)
        store->plans[i].next_index = SALT_DPR_STORE_NONE;
    for (size_t i = 0; i < store->bucket_count; i++)
        store->buckets[i] = SALT_DPR_STORE_NONE;
    dir = opendir(directory);
    if (!dir) return -1;
    while ((entry = readdir(dir)) != NULL) {
        uint8_t expected_sha[32], actual_sha[32];
        uint8_t bytes[SALT_DPR_ATTENTION_PLAN_FILE_BYTES];
        SaltDprAttentionPlan plan;
        size_t bucket;
        uint32_t index;
        int count;
        if (entry->d_name[0] == '.') continue;
        if (attention_filename_sha(entry->d_name, expected_sha) != 0) {
            store->rejected_plans++;
            continue;
        }
        count = snprintf(path, sizeof path, "%s/%s", directory, entry->d_name);
        if (count < 0 || (size_t)count >= sizeof path ||
            read_exact_private(path, bytes, sizeof bytes) != 0 ||
            salt_dpr_attention_plan_decode(bytes, &plan, actual_sha) != 0 ||
            memcmp(expected_sha, actual_sha, 32) != 0) {
            store->rejected_plans++;
            continue;
        }
        if (store->plan_count >= store->plan_capacity) {
            closedir(dir);
            return -1;
        }
        index = (uint32_t)store->plan_count;
        store->plans[index].plan = plan;
        memcpy(store->plans[index].plan_sha256, actual_sha, 32);
        bucket = node_bucket(plan.intent_sha256, store->bucket_count);
        store->plans[index].next_index = store->buckets[bucket];
        store->buckets[bucket] = index;
        store->plan_count++;
    }
    return closedir(dir) == 0 ? 0 : -1;
}

int salt_dpr_attention_plan_store_upsert_runtime(
        SaltDprAttentionPlanStore *store, const SaltDprAttentionPlan *plan,
        uint8_t plan_sha256[32], size_t *stored_index) {
    uint8_t bytes[SALT_DPR_ATTENTION_PLAN_FILE_BYTES], decoded_sha[32];
    SaltDprAttentionPlan decoded;
    size_t index, bucket;
    if (!store || !store->plans || !store->buckets || !plan ||
        !plan_sha256 || !stored_index || store->plan_capacity == 0 ||
        store->bucket_count == 0 ||
        salt_dpr_attention_plan_encode(plan, bytes, plan_sha256) != 0 ||
        salt_dpr_attention_plan_decode(bytes, &decoded, decoded_sha) != 0 ||
        memcmp(plan_sha256, decoded_sha, 32) != 0)
        return -1;
    if (store->runtime_present) {
        uint32_t *link;
        size_t walked = 0;
        if (store->runtime_index >= store->plan_count) return -1;
        index = store->runtime_index;
        bucket = node_bucket(
            store->plans[index].plan.intent_sha256, store->bucket_count);
        link = &store->buckets[bucket];
        while (*link != SALT_DPR_STORE_NONE && *link != (uint32_t)index) {
            if (*link >= store->plan_count || ++walked > store->plan_count)
                return -1;
            link = &store->plans[*link].next_index;
        }
        if (*link != (uint32_t)index) return -1;
        *link = store->plans[index].next_index;
    } else {
        if (store->plan_count >= store->plan_capacity) return -1;
        index = store->plan_count++;
        store->runtime_index = index;
        store->runtime_present = 1;
    }
    store->plans[index].plan = decoded;
    memcpy(store->plans[index].plan_sha256, decoded_sha, 32);
    bucket = node_bucket(decoded.intent_sha256, store->bucket_count);
    store->plans[index].next_index = store->buckets[bucket];
    store->buckets[bucket] = (uint32_t)index;
    *stored_index = index;
    return 0;
}

int salt_dpr_attention_plan_store_clear_runtime(
        SaltDprAttentionPlanStore *store) {
    uint32_t *link;
    size_t index, bucket, walked = 0;
    if (!store || !store->plans || !store->buckets ||
        store->bucket_count == 0)
        return -1;
    if (!store->runtime_present) return 0;
    index = store->runtime_index;
    if (index >= store->plan_count || index + 1u != store->plan_count)
        return -1;
    bucket = node_bucket(
        store->plans[index].plan.intent_sha256, store->bucket_count);
    link = &store->buckets[bucket];
    while (*link != SALT_DPR_STORE_NONE && *link != (uint32_t)index) {
        if (*link >= store->plan_count || ++walked > store->plan_count)
            return -1;
        link = &store->plans[*link].next_index;
    }
    if (*link != (uint32_t)index) return -1;
    *link = store->plans[index].next_index;
    memset(&store->plans[index], 0, sizeof store->plans[index]);
    store->plans[index].next_index = SALT_DPR_STORE_NONE;
    store->plan_count--;
    store->runtime_index = SIZE_MAX;
    store->runtime_present = 0;
    return 0;
}

static int ratio_compare(uint64_t left_num, uint64_t left_den,
                         uint64_t right_num, uint64_t right_den) {
    int reversed = 0;
    for (;;) {
        uint64_t left_q = left_num / left_den;
        uint64_t right_q = right_num / right_den;
        uint64_t left_r, right_r;
        if (left_q != right_q) {
            int result = left_q < right_q ? -1 : 1;
            return reversed ? -result : result;
        }
        left_r = left_num % left_den;
        right_r = right_num % right_den;
        if (left_r == 0 || right_r == 0) {
            int result;
            if (left_r == 0 && right_r == 0) return 0;
            result = left_r == 0 ? -1 : 1;
            return reversed ? -result : result;
        }
        left_num = left_den;
        left_den = left_r;
        right_num = right_den;
        right_den = right_r;
        reversed = !reversed;
    }
}

int salt_dpr_attention_plan_store_select(
        const SaltDprAttentionPlanStore *store,
        const SaltDprComputeIntent *intent,
        uint32_t max_layers, uint32_t max_experts, uint64_t max_context,
        size_t *selected_index) {
    uint8_t intent_sha[32];
    uint32_t index;
    size_t bucket, walked = 0, best = SIZE_MAX;
    if (!store || !store->plans || !store->buckets || !intent ||
        !selected_index || store->bucket_count == 0 ||
        salt_dpr_compute_intent_sha256(intent, intent_sha) != 0)
        return -1;
    *selected_index = SIZE_MAX;
    bucket = node_bucket(intent_sha, store->bucket_count);
    index = store->buckets[bucket];
    while (index != SALT_DPR_STORE_NONE) {
        const SaltDprStoredAttentionPlan *stored;
        int admitted;
        if (index >= store->plan_count || ++walked > store->plan_count)
            return -1;
        stored = &store->plans[index];
        admitted = salt_dpr_attention_plan_validate(
            &stored->plan, intent, max_layers, max_experts, max_context);
        if (admitted < 0) return -1;
        if (admitted > 0 &&
            (best == SIZE_MAX || ratio_compare(
                stored->plan.expected_saved_numerator,
                stored->plan.expected_saved_denominator,
                store->plans[best].plan.expected_saved_numerator,
                store->plans[best].plan.expected_saved_denominator) > 0 ||
             (ratio_compare(
                stored->plan.expected_saved_numerator,
                stored->plan.expected_saved_denominator,
                store->plans[best].plan.expected_saved_numerator,
                store->plans[best].plan.expected_saved_denominator) == 0 &&
              memcmp(stored->plan_sha256,
                     store->plans[best].plan_sha256, 32) < 0)))
            best = index;
        index = stored->next_index;
    }
    if (best == SIZE_MAX) return 0;
    *selected_index = best;
    return 1;
}

int salt_dpr_mentor_family_store_init(
        SaltDprMentorFamilyStore *store,
        SaltDprMentorFamily *family_slots, uint32_t *next_indices,
        size_t family_capacity, uint32_t *buckets, size_t bucket_count) {
    if (!store || !family_slots || !next_indices || family_capacity == 0 ||
        !buckets || bucket_count == 0 || family_capacity > UINT32_MAX)
        return -1;
    memset(store, 0, sizeof *store);
    memset(family_slots, 0, family_capacity * sizeof *family_slots);
    for (size_t i = 0; i < family_capacity; i++)
        next_indices[i] = SALT_DPR_STORE_NONE;
    for (size_t i = 0; i < bucket_count; i++)
        buckets[i] = SALT_DPR_STORE_NONE;
    store->families = family_slots;
    store->next_indices = next_indices;
    store->family_capacity = family_capacity;
    store->buckets = buckets;
    store->bucket_count = bucket_count;
    return 0;
}

int salt_dpr_mentor_family_store_find(
        const SaltDprMentorFamilyStore *store,
        const uint8_t family_sha256[32], size_t *family_index) {
    uint32_t index;
    size_t bucket, walked = 0;
    if (!store || !store->families || !store->next_indices ||
        !store->buckets || store->bucket_count == 0 ||
        !family_sha256 || digest_is_zero(family_sha256) || !family_index)
        return -1;
    *family_index = SIZE_MAX;
    bucket = node_bucket(family_sha256, store->bucket_count);
    index = store->buckets[bucket];
    while (index != SALT_DPR_STORE_NONE) {
        if (index >= store->family_count || ++walked > store->family_count)
            return -1;
        if (memcmp(store->families[index].family_sha256,
                   family_sha256, 32) == 0) {
            *family_index = index;
            return 1;
        }
        index = store->next_indices[index];
    }
    return 0;
}

static int mentor_family_filename_sha(const char *name, uint8_t out[32]) {
    size_t length;
    char hex[65];
    if (!name || !out) return -1;
    length = strlen(name);
    if (length != 71 || strcmp(name + 64, ".family") != 0) return -1;
    memcpy(hex, name, 64);
    hex[64] = 0;
    return hex_parse(hex, out);
}

int salt_dpr_mentor_family_store_load(
        SaltDprMentorFamilyStore *store, const char *root) {
    char directory[PATH_MAX], path[PATH_MAX];
    DIR *dir = NULL;
    struct dirent *entry;
    if (!store || !store->families || !store->next_indices ||
        !store->buckets || !root || store->family_capacity == 0 ||
        store->bucket_count == 0 || salt_dpr_store_prepare(root) != 0 ||
        join_path(root, "families", directory, sizeof directory) != 0)
        return -1;
    store->family_count = 0;
    store->rejected_families = 0;
    memset(store->families, 0,
           store->family_capacity * sizeof *store->families);
    for (size_t i = 0; i < store->family_capacity; i++)
        store->next_indices[i] = SALT_DPR_STORE_NONE;
    for (size_t i = 0; i < store->bucket_count; i++)
        store->buckets[i] = SALT_DPR_STORE_NONE;
    dir = opendir(directory);
    if (!dir) return -1;
    while ((entry = readdir(dir)) != NULL) {
        uint8_t expected_sha[32];
        uint8_t bytes[SALT_DPR_MENTOR_FAMILY_FILE_BYTES];
        SaltDprMentorFamily family;
        size_t bucket, duplicate = SIZE_MAX;
        uint32_t index;
        int count, found;
        if (entry->d_name[0] == '.') continue;
        if (mentor_family_filename_sha(entry->d_name, expected_sha) != 0) {
            store->rejected_families++;
            continue;
        }
        count = snprintf(path, sizeof path, "%s/%s", directory, entry->d_name);
        if (count < 0 || (size_t)count >= sizeof path ||
            read_exact_private(path, bytes, sizeof bytes) != 0 ||
            salt_dpr_mentor_family_decode(bytes, &family) != 0 ||
            memcmp(expected_sha, family.family_sha256, 32) != 0) {
            store->rejected_families++;
            continue;
        }
        found = salt_dpr_mentor_family_store_find(
            store, family.family_sha256, &duplicate);
        if (found < 0) {
            closedir(dir);
            return -1;
        }
        if (found > 0) {
            store->rejected_families++;
            continue;
        }
        if (store->family_count >= store->family_capacity) {
            closedir(dir);
            return -1;
        }
        index = (uint32_t)store->family_count;
        bucket = node_bucket(family.family_sha256, store->bucket_count);
        store->families[index] = family;
        store->next_indices[index] = store->buckets[bucket];
        store->buckets[bucket] = index;
        store->family_count++;
    }
    return closedir(dir) == 0 ? 0 : -1;
}

int salt_dpr_mentor_binding_store_init(
        SaltDprMentorBindingStore *store,
        SaltDprMentorBinding *binding_slots, uint32_t *next_indices,
        size_t binding_capacity, uint32_t *buckets, size_t bucket_count) {
    if (!store || !binding_slots || !next_indices || binding_capacity == 0 ||
        !buckets || bucket_count == 0 || binding_capacity > UINT32_MAX)
        return -1;
    memset(store, 0, sizeof *store);
    memset(binding_slots, 0, binding_capacity * sizeof *binding_slots);
    for (size_t i = 0; i < binding_capacity; i++)
        next_indices[i] = SALT_DPR_STORE_NONE;
    for (size_t i = 0; i < bucket_count; i++)
        buckets[i] = SALT_DPR_STORE_NONE;
    store->bindings = binding_slots;
    store->next_indices = next_indices;
    store->binding_capacity = binding_capacity;
    store->buckets = buckets;
    store->bucket_count = bucket_count;
    return 0;
}

static int edge_store_find_sha(const SaltDprStore *store,
                               const uint8_t edge_sha256[32],
                               size_t *edge_index) {
    if (!store || !store->edges || !edge_sha256 ||
        digest_is_zero(edge_sha256) || !edge_index)
        return -1;
    *edge_index = SIZE_MAX;
    for (size_t i = 0; i < store->edge_count; i++) {
        if (memcmp(store->edges[i].edge_sha256, edge_sha256, 32) == 0) {
            *edge_index = i;
            return 1;
        }
    }
    return 0;
}

static int mentor_binding_filename_sha(const char *name, uint8_t out[32]) {
    size_t length;
    char hex[65];
    if (!name || !out) return -1;
    length = strlen(name);
    if (length != 72 || strcmp(name + 64, ".binding") != 0) return -1;
    memcpy(hex, name, 64);
    hex[64] = 0;
    return hex_parse(hex, out);
}

int salt_dpr_mentor_binding_store_load(
        SaltDprMentorBindingStore *store, const char *root,
        const SaltDprMentorFamilyStore *family_store,
        const SaltDprStore *edge_store) {
    char directory[PATH_MAX], path[PATH_MAX];
    DIR *dir = NULL;
    struct dirent *entry;
    if (!store || !store->bindings || !store->next_indices ||
        !store->buckets || !root || !family_store || !edge_store ||
        store->binding_capacity == 0 || store->bucket_count == 0 ||
        salt_dpr_store_prepare(root) != 0 ||
        join_path(root, "bindings", directory, sizeof directory) != 0)
        return -1;
    store->binding_count = 0;
    store->rejected_bindings = 0;
    memset(store->bindings, 0,
           store->binding_capacity * sizeof *store->bindings);
    for (size_t i = 0; i < store->binding_capacity; i++)
        store->next_indices[i] = SALT_DPR_STORE_NONE;
    for (size_t i = 0; i < store->bucket_count; i++)
        store->buckets[i] = SALT_DPR_STORE_NONE;
    dir = opendir(directory);
    if (!dir) return -1;
    while ((entry = readdir(dir)) != NULL) {
        uint8_t expected_sha[32];
        uint8_t bytes[SALT_DPR_MENTOR_BINDING_FILE_BYTES];
        SaltDprMentorBinding binding;
        const SaltDprMentorFamily *family;
        const SaltDprStoredEdge *edge;
        size_t family_index = SIZE_MAX, edge_index = SIZE_MAX, bucket;
        uint32_t index;
        int count, found_family, found_edge;
        if (entry->d_name[0] == '.') continue;
        if (mentor_binding_filename_sha(entry->d_name, expected_sha) != 0) {
            store->rejected_bindings++;
            continue;
        }
        count = snprintf(path, sizeof path, "%s/%s", directory, entry->d_name);
        if (count < 0 || (size_t)count >= sizeof path ||
            read_exact_private(path, bytes, sizeof bytes) != 0 ||
            salt_dpr_mentor_binding_decode(bytes, &binding) != 0 ||
            memcmp(expected_sha, binding.binding_sha256, 32) != 0) {
            store->rejected_bindings++;
            continue;
        }
        found_family = salt_dpr_mentor_family_store_find(
            family_store, binding.family_sha256, &family_index);
        found_edge = edge_store_find_sha(
            edge_store, binding.edge_sha256, &edge_index);
        if (found_family < 0 || found_edge < 0) {
            closedir(dir);
            return -1;
        }
        if (found_family == 0 || found_edge == 0) {
            store->rejected_bindings++;
            continue;
        }
        family = &family_store->families[family_index];
        edge = &edge_store->edges[edge_index];
        if (edge->edge.kind != SALT_DPR_EDGE_NOMOGRAM_DRAFT ||
            memcmp(binding.source_node_sha256,
                   edge->edge.source_node_sha256, 32) != 0 ||
            memcmp(binding.state_compatibility_sha256,
                   family->state_compatibility_sha256, 32) != 0 ||
            memcmp(binding.policy_sha256, family->policy_sha256, 32) != 0) {
            store->rejected_bindings++;
            continue;
        }
        if (store->binding_count >= store->binding_capacity) {
            closedir(dir);
            return -1;
        }
        binding.family_index = family_index;
        binding.edge_index = edge_index;
        index = (uint32_t)store->binding_count;
        bucket = node_bucket(binding.source_node_sha256, store->bucket_count);
        store->bindings[index] = binding;
        store->next_indices[index] = store->buckets[bucket];
        store->buckets[bucket] = index;
        store->binding_count++;
    }
    return closedir(dir) == 0 ? 0 : -1;
}

int salt_dpr_mentor_candidates_build(
        const SaltDprMentorBindingStore *binding_store,
        const SaltDprMentorFamilyStore *family_store,
        const SaltDprStore *edge_store,
        const uint8_t *retained, const SaltDprChart *effective_charts,
        SaltDprMode mode,
        const uint8_t target_node_sha256[32],
        uint64_t serial_ns_per_token, uint64_t minimum_samples,
        SaltDprSelectorCandidate *candidates, size_t candidate_capacity,
        size_t *candidate_count) {
    uint32_t index;
    size_t bucket, walked = 0, count = 0;
    if (!binding_store || !binding_store->bindings ||
        !binding_store->next_indices || !binding_store->buckets ||
        !family_store || !family_store->families || !edge_store ||
        !edge_store->edges || !target_node_sha256 ||
        digest_is_zero(target_node_sha256) || !serial_ns_per_token ||
        !minimum_samples || !candidates || candidate_capacity == 0 ||
        !candidate_count || !retained || binding_store->bucket_count == 0 ||
        (mode != SALT_DPR_PERSIST && mode != SALT_DPR_DYNAMIC))
        return -1;
    *candidate_count = 0;
    bucket = node_bucket(target_node_sha256, binding_store->bucket_count);
    index = binding_store->buckets[bucket];
    while (index != SALT_DPR_STORE_NONE) {
        const SaltDprMentorBinding *binding;
        const SaltDprMentorFamily *family;
        const SaltDprStoredEdge *edge;
        const SaltDprChart *chart;
        uint64_t saved_numerator, saved_denominator;
        if (index >= binding_store->binding_count ||
            ++walked > binding_store->binding_count)
            return -1;
        binding = &binding_store->bindings[index];
        if (mentor_binding_validate(binding) != 0 ||
            binding->family_index >= family_store->family_count ||
            binding->edge_index >= edge_store->edge_count)
            return -1;
        family = &family_store->families[binding->family_index];
        edge = &edge_store->edges[binding->edge_index];
        chart = effective_charts
            ? &effective_charts[binding->edge_index] : &edge->edge.chart;
        if (memcmp(binding->family_sha256, family->family_sha256, 32) != 0 ||
            memcmp(binding->edge_sha256, edge->edge_sha256, 32) != 0 ||
            memcmp(binding->source_node_sha256,
                   edge->edge.source_node_sha256, 32) != 0 ||
            memcmp(binding->state_compatibility_sha256,
                   family->state_compatibility_sha256, 32) != 0 ||
            memcmp(binding->policy_sha256, family->policy_sha256, 32) != 0)
            return -1;
        if (retained[binding->edge_index] && edge->edge.mode == mode &&
            memcmp(binding->source_node_sha256, target_node_sha256, 32) == 0 &&
            salt_dpr_chart_is_green(
                chart, serial_ns_per_token, minimum_samples)) {
            if (salt_dpr_chart_expected_saved(
                    chart, serial_ns_per_token,
                    &saved_numerator, &saved_denominator) != 0)
                return -1;
            if (saved_numerator != 0) {
                SaltDprSelectorCandidate *candidate;
                if (count >= candidate_capacity) return -1;
                candidate = &candidates[count++];
                memset(candidate, 0, sizeof *candidate);
                candidate->schema_version = SALT_DPR_SELECTOR_CANDIDATE_VERSION;
                candidate->kind = SALT_DPR_SELECTOR_MENTOR_NOMOGRAM;
                candidate->family_index = binding->family_index;
                memcpy(candidate->edge_sha256, edge->edge_sha256, 32);
                memcpy(candidate->binding_sha256,
                       binding->binding_sha256, 32);
                memcpy(candidate->source_node_sha256,
                       binding->source_node_sha256, 32);
                memcpy(candidate->state_compatibility_sha256,
                       family->state_compatibility_sha256, 32);
                memcpy(candidate->policy_sha256, family->policy_sha256, 32);
                candidate->expected_saved_numerator = saved_numerator;
                candidate->expected_saved_denominator = saved_denominator;
                candidate->provenance_authenticated = 1;
                candidate->policy_allowed = 1;
            }
        }
        index = binding_store->next_indices[index];
    }
    *candidate_count = count;
    return 0;
}

int salt_dpr_mentor_walk_nomogram(
        const SaltDprMentorBindingStore *binding_store,
        const SaltDprMentorFamilyStore *family_store,
        const SaltDprStore *edge_store, const uint8_t *retained,
        const SaltDprChart *effective_charts,
        size_t family_index, SaltDprMode mode,
        const uint8_t start_node_sha256[32],
        uint64_t serial_ns_per_token, uint64_t minimum_samples,
        int32_t vocab_size, uint64_t max_context,
        SaltDprStopTokenFn stop_token, void *stop_opaque,
        uint32_t max_tokens, int32_t *candidate_token_ids,
        size_t *edge_indices, size_t *binding_indices,
        uint32_t *candidate_count) {
    uint8_t node[32], visited[SALT_DPR_MAX_WALK_HORIZON][32];
    uint32_t count = 0;
    if (!binding_store || !binding_store->bindings ||
        !binding_store->next_indices || !binding_store->buckets ||
        !family_store || !family_store->families ||
        family_index >= family_store->family_count || !edge_store ||
        !edge_store->edges || !retained || !start_node_sha256 ||
        digest_is_zero(start_node_sha256) || !serial_ns_per_token ||
        !minimum_samples || vocab_size <= 0 || max_context == 0 ||
        max_tokens == 0 || max_tokens > SALT_DPR_MAX_WALK_HORIZON ||
        !candidate_token_ids || !candidate_count ||
        binding_store->bucket_count == 0 ||
        (mode != SALT_DPR_PERSIST && mode != SALT_DPR_DYNAMIC))
        return -1;
    memcpy(node, start_node_sha256, 32);
    while (count < max_tokens) {
        size_t bucket = node_bucket(node, binding_store->bucket_count);
        uint32_t index = binding_store->buckets[bucket];
        size_t walked = 0, best_binding = SIZE_MAX, best_edge = SIZE_MAX;
        for (uint32_t i = 0; i < count; i++)
            if (memcmp(visited[i], node, 32) == 0) return -1;
        memcpy(visited[count], node, 32);
        while (index != SALT_DPR_STORE_NONE) {
            const SaltDprMentorBinding *binding;
            const SaltDprStoredEdge *edge;
            const SaltDprChart *chart;
            int comparison = 0;
            if (index >= binding_store->binding_count ||
                ++walked > binding_store->binding_count)
                return -1;
            binding = &binding_store->bindings[index];
            if (binding->edge_index >= edge_store->edge_count ||
                binding->family_index >= family_store->family_count)
                return -1;
            edge = &edge_store->edges[binding->edge_index];
            chart = effective_charts
                ? &effective_charts[binding->edge_index] : &edge->edge.chart;
            if (binding->family_index == family_index &&
                retained[binding->edge_index] && edge->edge.mode == mode &&
                memcmp(binding->source_node_sha256, node, 32) == 0 &&
                memcmp(binding->family_sha256,
                       family_store->families[family_index].family_sha256,
                       32) == 0 &&
                memcmp(binding->edge_sha256, edge->edge_sha256, 32) == 0 &&
                salt_dpr_edge_validate(
                    &edge->edge, vocab_size, max_context,
                    stop_token, stop_opaque) == 0 &&
                salt_dpr_chart_is_green(
                    chart, serial_ns_per_token,
                    minimum_samples)) {
                if (best_binding == SIZE_MAX) {
                    best_binding = index;
                    best_edge = binding->edge_index;
                } else if (salt_dpr_chart_compare(
                        chart,
                        effective_charts
                            ? &effective_charts[best_edge]
                            : &edge_store->edges[best_edge].edge.chart,
                        &comparison) != 0) {
                    return -1;
                } else if (comparison < 0 ||
                           (comparison == 0 &&
                            memcmp(edge->edge_sha256,
                                   edge_store->edges[best_edge].edge_sha256,
                                   32) < 0)) {
                    best_binding = index;
                    best_edge = binding->edge_index;
                }
            }
            index = binding_store->next_indices[index];
        }
        if (best_binding == SIZE_MAX) break;
        candidate_token_ids[count] =
            edge_store->edges[best_edge].edge.candidate_token_ids[0];
        if (edge_indices) edge_indices[count] = best_edge;
        if (binding_indices) binding_indices[count] = best_binding;
        memcpy(node, edge_store->edges[best_edge].edge.next_node_sha256, 32);
        count++;
    }
    *candidate_count = count;
    return count == 0 ? 0 : 1;
}

static int edge_filename_sha(const char *name, uint8_t out[32]) {
    size_t length;
    char hex[65];
    if (!name || !out) return -1;
    length = strlen(name);
    if (length != 68 || strcmp(name + 64, ".dpr") != 0) return -1;
    memcpy(hex, name, 64);
    hex[64] = 0;
    return hex_parse(hex, out);
}

int salt_dpr_store_load(SaltDprStore *store, const char *root) {
    char directory[PATH_MAX], path[PATH_MAX];
    DIR *dir = NULL;
    struct dirent *entry;
    if (!store || !store->edges || !store->buckets || !root ||
        salt_dpr_store_prepare(root) != 0 ||
        join_path(root, "edges", directory, sizeof directory) != 0)
        return -1;
    store->edge_count = 0;
    store->rejected_edges = 0;
    for (size_t i = 0; i < store->bucket_count; i++)
        store->buckets[i] = SALT_DPR_STORE_NONE;
    dir = opendir(directory);
    if (!dir) return -1;
    while ((entry = readdir(dir)) != NULL) {
        uint8_t expected_sha[32], actual_sha[32];
        uint8_t bytes[SALT_DPR_EDGE_FILE_BYTES];
        SaltDprEdge edge;
        size_t bucket;
        uint32_t index;
        int count;
        if (entry->d_name[0] == '.') continue;
        if (edge_filename_sha(entry->d_name, expected_sha) != 0) {
            store->rejected_edges++;
            continue;
        }
        count = snprintf(path, sizeof path, "%s/%s", directory, entry->d_name);
        if (count < 0 || (size_t)count >= sizeof path ||
            read_exact_private(path, bytes, sizeof bytes) != 0 ||
            salt_dpr_edge_decode(bytes, &edge, actual_sha) != 0 ||
            memcmp(expected_sha, actual_sha, 32) != 0) {
            store->rejected_edges++;
            continue;
        }
        if (store->edge_count >= store->edge_capacity) {
            closedir(dir);
            return -1;
        }
        index = (uint32_t)store->edge_count;
        bucket = node_bucket(edge.source_node_sha256, store->bucket_count);
        store->edges[index].edge = edge;
        memcpy(store->edges[index].edge_sha256, actual_sha, 32);
        store->edges[index].next_index = store->buckets[bucket];
        store->buckets[bucket] = index;
        store->edge_count++;
    }
    return closedir(dir) == 0 ? 0 : -1;
}

int salt_dpr_store_select_kind(const SaltDprStore *store,
                               const uint8_t source_node_sha256[32],
                               SaltDprMode mode, SaltDprEdgeKind kind,
                               uint64_t serial_ns_per_token,
                               uint64_t minimum_samples,
                               int32_t vocab_size,
                               uint64_t max_context,
                               SaltDprStopTokenFn stop_token,
                               void *stop_opaque,
                               size_t *selected_index) {
    uint32_t index;
    size_t bucket, walked = 0, best = SIZE_MAX;
    if (!store || !store->edges || !store->buckets ||
        !source_node_sha256 || !selected_index || store->bucket_count == 0 ||
        (mode != SALT_DPR_OFF && mode != SALT_DPR_PERSIST &&
         mode != SALT_DPR_DYNAMIC) ||
        (kind != SALT_DPR_EDGE_PARENT_EXACT &&
         kind != SALT_DPR_EDGE_NOMOGRAM_DRAFT &&
         kind != SALT_DPR_EDGE_PARENT_PREFILL &&
         kind != SALT_DPR_EDGE_NOMOGRAM_PREFILL))
        return -1;
    *selected_index = SIZE_MAX;
    if (mode == SALT_DPR_OFF) return 0;
    bucket = node_bucket(source_node_sha256, store->bucket_count);
    index = store->buckets[bucket];
    while (index != SALT_DPR_STORE_NONE) {
        int comparison = 0;
        const SaltDprStoredEdge *candidate;
        if (index >= store->edge_count || ++walked > store->edge_count)
            return -1;
        candidate = &store->edges[index];
        if (candidate->edge.mode == mode && candidate->edge.kind == kind &&
            memcmp(candidate->edge.source_node_sha256,
                   source_node_sha256, 32) == 0 &&
            salt_dpr_edge_validate(
                &candidate->edge, vocab_size, max_context,
                stop_token, stop_opaque) == 0 &&
            salt_dpr_chart_is_green(
                &candidate->edge.chart,
                serial_ns_per_token, minimum_samples)) {
            if (best == SIZE_MAX) {
                best = index;
            } else if (salt_dpr_chart_compare(
                    &candidate->edge.chart,
                    &store->edges[best].edge.chart, &comparison) != 0) {
                return -1;
            } else if (comparison < 0) {
                best = index;
            }
        }
        index = candidate->next_index;
    }
    if (best == SIZE_MAX) return 0;
    *selected_index = best;
    return 1;
}

int salt_dpr_store_select(const SaltDprStore *store,
                          const uint8_t source_node_sha256[32],
                          SaltDprMode mode,
                          uint64_t serial_ns_per_token,
                          uint64_t minimum_samples,
                          int32_t vocab_size,
                          uint64_t max_context,
                          SaltDprStopTokenFn stop_token,
                          void *stop_opaque,
                          size_t *selected_index) {
    return salt_dpr_store_select_kind(
        store, source_node_sha256, mode, SALT_DPR_EDGE_PARENT_EXACT,
        serial_ns_per_token, minimum_samples, vocab_size, max_context,
        stop_token, stop_opaque, selected_index);
}

int salt_dpr_store_walk_nomogram(const SaltDprStore *store,
                               const uint8_t start_node_sha256[32],
                               SaltDprMode mode,
                               uint64_t serial_ns_per_token,
                               uint64_t minimum_samples,
                               int32_t vocab_size,
                               uint64_t max_context,
                               SaltDprStopTokenFn stop_token,
                               void *stop_opaque,
                               uint32_t max_tokens,
                               int32_t *candidate_token_ids,
                               size_t *edge_indices,
                               uint32_t *candidate_count) {
    uint8_t node[32], visited[SALT_DPR_MAX_WALK_HORIZON][32];
    uint32_t count = 0;
    if (!store || !start_node_sha256 || !candidate_token_ids ||
        !candidate_count || max_tokens == 0 ||
        max_tokens > SALT_DPR_MAX_WALK_HORIZON)
        return -1;
    memcpy(node, start_node_sha256, 32);
    while (count < max_tokens) {
        size_t selected = SIZE_MAX;
        int found;
        for (uint32_t i = 0; i < count; i++)
            if (memcmp(visited[i], node, 32) == 0) return -1;
        memcpy(visited[count], node, 32);
        found = salt_dpr_store_select_kind(
            store, node, mode, SALT_DPR_EDGE_NOMOGRAM_DRAFT,
            serial_ns_per_token, minimum_samples, vocab_size, max_context,
            stop_token, stop_opaque, &selected);
        if (found < 0) return -1;
        if (found == 0) break;
        candidate_token_ids[count] =
            store->edges[selected].edge.candidate_token_ids[0];
        if (edge_indices) edge_indices[count] = selected;
        memcpy(node, store->edges[selected].edge.next_node_sha256, 32);
        count++;
    }
    *candidate_count = count;
    return count == 0 ? 0 : 1;
}
