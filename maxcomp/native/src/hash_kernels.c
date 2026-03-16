/* hash_kernels.c — hash preimage search kernel */

#include "maxcomp_native.h"
#include "md5.h"
#include "sha1.h"
#include "sha256.h"
#include <string.h>

/* Hash type constants */
#define HASH_MD5    0
#define HASH_SHA1   1
#define HASH_SHA256 2

/* Digest sizes in bytes */
#define MD5_DIGEST_LEN    16
#define SHA1_DIGEST_LEN   20
#define SHA256_DIGEST_LEN 32

/* Maximum digest length across all types */
#define MAX_DIGEST_LEN 32

/* Number of output bits for each hash type */
static const uint32_t hash_output_bits[3] = { 128, 160, 256 };

void hash_compute(int hash_type, const uint8_t *preimage, uint32_t preimage_len, uint8_t *digest) {
    /* Zero entire 32-byte digest buffer so callers can rely on trailing zeros */
    memset(digest, 0, MAX_DIGEST_LEN);

    switch (hash_type) {
    case HASH_MD5:
        md5_hash(preimage, preimage_len, digest);
        break;
    case HASH_SHA1:
        sha1_hash(preimage, preimage_len, digest);
        break;
    case HASH_SHA256:
        sha256_hash(preimage, preimage_len, digest);
        break;
    default:
        break;
    }
}

/*
 * Compare the first target_bit_length bits of digest against target.
 * Returns 1 if they match, 0 otherwise.
 */
static int bits_match(const uint8_t *digest, const uint8_t *target,
                      uint32_t target_bit_length) {
    uint32_t full_bytes = target_bit_length / 8;
    uint32_t remaining_bits = target_bit_length % 8;

    /* Compare full bytes */
    if (full_bytes > 0 && memcmp(digest, target, full_bytes) != 0)
        return 0;

    /* Compare remaining partial byte (high bits) */
    if (remaining_bits > 0) {
        uint8_t mask = (uint8_t)(0xFF << (8 - remaining_bits));
        if ((digest[full_bytes] & mask) != (target[full_bytes] & mask))
            return 0;
    }

    return 1;
}

/*
 * Increment a big-endian byte array of given length.
 * Returns 1 on overflow (all bytes wrapped to 0), 0 otherwise.
 */
static int increment_be(uint8_t *buf, uint32_t len) {
    for (int i = (int)len - 1; i >= 0; i--) {
        if (++buf[i] != 0)
            return 0;
    }
    return 1; /* overflow: all bytes wrapped to 0 */
}

int hash_preimage_search(
    const uint8_t *target, uint32_t target_bit_length,
    uint32_t max_preimage_len,
    int *out_hash_type, uint8_t *out_preimage, uint32_t *out_preimage_len
) {
    if (target_bit_length == 0)
        return -1;

    uint8_t digest[MAX_DIGEST_LEN];
    uint8_t preimage[256]; /* max preimage buffer — well above practical limit */

    for (int hash_type = HASH_MD5; hash_type <= HASH_SHA256; hash_type++) {
        /* Skip if target needs more bits than this hash produces */
        if (target_bit_length > hash_output_bits[hash_type])
            continue;

        for (uint32_t preimage_len = 1; preimage_len <= max_preimage_len; preimage_len++) {
            /* Start with all-zero preimage (i.e. value 0 in big-endian) */
            memset(preimage, 0, preimage_len);

            /* Enumerate all 256^preimage_len values */
            uint64_t total;
            if (preimage_len >= 8) {
                /* More than 2^64 — practically unreachable, but avoid overflow.
                 * In practice max_preimage_len is small (1-3). */
                total = 0; /* sentinel: use overflow detection instead */
            } else {
                total = 1;
                for (uint32_t j = 0; j < preimage_len; j++)
                    total *= 256;
            }

            if (total > 0) {
                for (uint64_t i = 0; i < total; i++) {
                    hash_compute(hash_type, preimage, preimage_len, digest);

                    if (bits_match(digest, target, target_bit_length)) {
                        *out_hash_type = hash_type;
                        memcpy(out_preimage, preimage, preimage_len);
                        *out_preimage_len = preimage_len;
                        return 0;
                    }

                    /* Advance to next preimage (big-endian increment) */
                    if (i + 1 < total)
                        increment_be(preimage, preimage_len);
                }
            } else {
                /* preimage_len >= 8: iterate until overflow */
                for (;;) {
                    hash_compute(hash_type, preimage, preimage_len, digest);

                    if (bits_match(digest, target, target_bit_length)) {
                        *out_hash_type = hash_type;
                        memcpy(out_preimage, preimage, preimage_len);
                        *out_preimage_len = preimage_len;
                        return 0;
                    }

                    if (increment_be(preimage, preimage_len))
                        break; /* wrapped around — exhausted all values */
                }
            }
        }
    }

    return -1;
}
