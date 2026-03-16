/*
 * ifs_kernels.c — IFS block matching with ARM NEON acceleration
 *
 * For each range block, finds the best matching domain block (normal or
 * inverted) using Hamming distance.  Uses NEON vcntq_u8 for fast popcount
 * on 16-byte chunks, with scalar __builtin_popcount fallback for shorter
 * blocks.
 */

#include "maxcomp_native.h"
#include <arm_neon.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* Helpers                                                             */
/* ------------------------------------------------------------------ */

/*
 * Compute Hamming distance between two packed bit arrays of exactly
 * `byte_count` bytes.  Only the first `num_bits` bits matter; any
 * trailing bits in the last byte beyond num_bits are masked out.
 */
static uint32_t hamming_packed(const uint8_t *a, const uint8_t *b,
                               uint32_t byte_count, uint32_t num_bits)
{
    uint32_t total = 0;
    uint32_t i = 0;

    /* NEON path: process 16 bytes at a time */
    while (i + 16 <= byte_count) {
        uint8x16_t va = vld1q_u8(a + i);
        uint8x16_t vb = vld1q_u8(b + i);
        uint8x16_t xored = veorq_u8(va, vb);
        uint8x16_t counts = vcntq_u8(xored);
        total += vaddlvq_u8(counts);
        i += 16;
    }

    /* Scalar remainder */
    for (; i < byte_count; i++) {
        total += (uint32_t)__builtin_popcount((unsigned)(a[i] ^ b[i]));
    }

    /* Mask off unused trailing bits in the last byte */
    uint32_t tail = num_bits % 8;
    if (tail != 0 && byte_count > 0) {
        uint8_t mask = (uint8_t)(0xFF >> tail);  /* bits to IGNORE */
        uint8_t extra = (a[byte_count - 1] ^ b[byte_count - 1]) & mask;
        total -= (uint32_t)__builtin_popcount((unsigned)extra);
    }

    return total;
}

/* ------------------------------------------------------------------ */
/* Public API                                                          */
/* ------------------------------------------------------------------ */

uint32_t ifs_match_blocks(
    const uint8_t *domain_pool, uint32_t num_domains,
    const uint8_t *range_blocks, uint32_t num_range_blocks,
    uint32_t block_size,
    uint32_t max_errors,
    uint32_t *out_domain_indices,
    uint8_t  *out_inverted,
    uint32_t *out_errors)
{
    if (block_size == 0 || num_domains == 0 || num_range_blocks == 0) {
        return 0;
    }

    const uint32_t block_bytes = (block_size + 7) / 8;
    uint32_t matched = 0;

    for (uint32_t r = 0; r < num_range_blocks; r++) {
        const uint8_t *rb = range_blocks + r * block_bytes;

        uint32_t best_err = UINT32_MAX;
        uint32_t best_domain = 0;
        uint8_t  best_inv = 0;

        for (uint32_t d = 0; d < num_domains; d++) {
            const uint8_t *db = domain_pool + d * block_bytes;

            /* Normal (non-inverted) match */
            uint32_t errs = hamming_packed(db, rb, block_bytes, block_size);
            if (errs < best_err) {
                best_err = errs;
                best_domain = d;
                best_inv = 0;
                if (errs == 0) {
                    break;  /* perfect match — early exit */
                }
            }

            /* Inverted match: hamming(~db, rb) == block_size - hamming(db, rb) */
            uint32_t errs_inv = block_size - errs;
            if (errs_inv < best_err) {
                best_err = errs_inv;
                best_domain = d;
                best_inv = 1;
                if (errs_inv == 0) {
                    break;  /* perfect inverted match */
                }
            }
        }

        out_domain_indices[r] = best_domain;
        out_inverted[r] = best_inv;
        out_errors[r] = best_err;

        if (best_err <= max_errors) {
            matched++;
        }
    }

    return matched;
}
