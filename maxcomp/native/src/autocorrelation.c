/*
 * autocorrelation.c — Autocorrelation for packed bit arrays with ARM NEON
 *
 * Computes normalized autocorrelation using packed bit-level operations.
 * The key insight: for binary {0,1} data packed MSB-first, we can compute
 * the dot-product between the signal and its lag-shifted copy by ANDing
 * the two packed arrays and counting the 1s.
 */

#include "maxcomp_native.h"
#include <arm_neon.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

/* ------------------------------------------------------------------ */
/* Helpers                                                             */
/* ------------------------------------------------------------------ */

/*
 * Popcount of the first `num_bits` bits of packed data.
 * Uses NEON for 16-byte chunks, scalar for the remainder.
 */
static uint32_t packed_popcount(const uint8_t *data, uint32_t num_bits)
{
    if (num_bits == 0) return 0;

    uint32_t full_bytes = num_bits / 8;
    uint32_t tail_bits  = num_bits % 8;
    uint32_t total = 0;
    uint32_t i = 0;

    /* NEON: 16 bytes at a time */
    while (i + 16 <= full_bytes) {
        uint8x16_t v = vld1q_u8(data + i);
        uint8x16_t c = vcntq_u8(v);
        total += vaddlvq_u8(c);
        i += 16;
    }

    /* Scalar remainder of full bytes */
    for (; i < full_bytes; i++) {
        total += (uint32_t)__builtin_popcount((unsigned)data[i]);
    }

    /* Partial last byte: only the top `tail_bits` bits count (MSB-first) */
    if (tail_bits > 0) {
        uint8_t mask = (uint8_t)(0xFF << (8 - tail_bits));
        total += (uint32_t)__builtin_popcount((unsigned)(data[full_bytes] & mask));
    }

    return total;
}

/*
 * Popcount of (a AND b) over the first `num_bits` bits.
 * Both arrays are packed MSB-first and must be at least ceil(num_bits/8) bytes.
 */
static uint32_t packed_and_popcount(const uint8_t *a, const uint8_t *b,
                                    uint32_t num_bits)
{
    if (num_bits == 0) return 0;

    uint32_t full_bytes = num_bits / 8;
    uint32_t tail_bits  = num_bits % 8;
    uint32_t total = 0;
    uint32_t i = 0;

    while (i + 16 <= full_bytes) {
        uint8x16_t va = vld1q_u8(a + i);
        uint8x16_t vb = vld1q_u8(b + i);
        uint8x16_t c  = vcntq_u8(vandq_u8(va, vb));
        total += vaddlvq_u8(c);
        i += 16;
    }

    for (; i < full_bytes; i++) {
        total += (uint32_t)__builtin_popcount((unsigned)(a[i] & b[i]));
    }

    if (tail_bits > 0) {
        uint8_t mask = (uint8_t)(0xFF << (8 - tail_bits));
        total += (uint32_t)__builtin_popcount(
            (unsigned)((a[full_bytes] & b[full_bytes]) & mask));
    }

    return total;
}

/*
 * Create a copy of `src` (num_bits long, packed MSB-first) shifted left
 * by `lag` bits.  That is, shifted[i] = src[i + lag] for i in [0, n - lag).
 * The last `lag` bits of the shifted array are zero.
 *
 * Caller must free the returned buffer.  Returns NULL on allocation failure.
 */
static uint8_t *make_shifted(const uint8_t *src, uint32_t num_bits,
                              uint32_t lag)
{
    uint32_t total_bytes = (num_bits + 7) / 8;
    uint8_t *dst = (uint8_t *)calloc(total_bytes, 1);
    if (!dst) return NULL;

    uint32_t byte_off = lag / 8;
    uint32_t bit_off  = lag % 8;

    /* Number of source bytes that are still relevant after the byte offset */
    uint32_t src_remaining = (byte_off < total_bytes)
                             ? total_bytes - byte_off
                             : 0;

    if (bit_off == 0) {
        /* Byte-aligned shift: simple copy */
        if (src_remaining > 0) {
            memcpy(dst, src + byte_off, src_remaining);
        }
    } else {
        /* Non-aligned: merge adjacent source bytes */
        for (uint32_t i = 0; i < src_remaining; i++) {
            uint8_t hi = src[byte_off + i] << bit_off;
            uint8_t lo = 0;
            if (byte_off + i + 1 < total_bytes) {
                lo = src[byte_off + i + 1] >> (8 - bit_off);
            }
            dst[i] = hi | lo;
        }
    }

    return dst;
}

/* ------------------------------------------------------------------ */
/* Public API                                                          */
/* ------------------------------------------------------------------ */

double autocorrelation_packed(
    const uint8_t *bits, uint32_t num_bits, uint32_t lag)
{
    if (num_bits == 0 || lag == 0 || lag >= num_bits) {
        return 0.0;
    }

    /* Number of overlapping positions */
    uint32_t n = num_bits;
    uint32_t overlap = n - lag;

    /* Ones count of the full signal */
    uint32_t ones_total = packed_popcount(bits, n);
    double mean = (double)ones_total / (double)n;

    /* Variance: for binary {0,1}, var = mean * (1 - mean) */
    double var = mean * (1.0 - mean);
    if (var < 1e-15) {
        /* Constant stream — define autocorrelation as 1.0 */
        return 1.0;
    }

    /*
     * Build the lag-shifted copy.  shifted[i] = bits[i + lag].
     * Then:
     *   agreement = popcount(bits AND shifted)  over the first `overlap` bits
     *             = sum of bits[i] * bits[i+lag] for i in [0, overlap)
     *
     * Covariance:
     *   cov = (1/overlap) * sum((bits[i] - mean)(bits[i+lag] - mean))
     *       = (1/overlap) * [ agreement - mean * ones_in_first_overlap
     *                         - mean * ones_in_shifted_overlap + overlap * mean^2 ]
     *
     * Where:
     *   ones_in_first_overlap  = popcount(bits[0..overlap-1])
     *   ones_in_shifted_overlap = popcount(bits[lag..lag+overlap-1])
     *                           = popcount(shifted[0..overlap-1])
     */
    uint8_t *shifted = make_shifted(bits, num_bits, lag);
    if (!shifted) return 0.0;

    uint32_t agreement = packed_and_popcount(bits, shifted, overlap);
    uint32_t ones_first   = packed_popcount(bits, overlap);
    uint32_t ones_shifted = packed_popcount(shifted, overlap);

    double cov = ((double)agreement
                  - mean * (double)ones_first
                  - mean * (double)ones_shifted
                  + (double)overlap * mean * mean) / (double)overlap;

    free(shifted);

    return cov / var;
}


int has_self_similarity_packed(
    const uint8_t *bits, uint32_t num_bits, double threshold)
{
    if (num_bits < 4) {
        return 0;
    }

    uint32_t n = num_bits;

    /*
     * Candidate lags, mirroring the Python logic:
     *   1. Small values 1..min(16, n-1)
     *   2. Powers of 2
     *   3. Fractions of length: n/2, n/3, n/4, n/5, n/6, n/8
     *
     * We collect unique lags into an array, sort, and test each.
     */

    /* Upper bound on number of candidate lags */
    uint32_t lags[128];
    uint32_t num_lags = 0;

    /* Small values 1..16 */
    for (uint32_t k = 1; k <= 16 && k < n; k++) {
        lags[num_lags++] = k;
    }

    /* Powers of 2 */
    uint32_t p = 1;
    while (p < n && num_lags < 128) {
        /* Check if already present */
        int found = 0;
        for (uint32_t j = 0; j < num_lags; j++) {
            if (lags[j] == p) { found = 1; break; }
        }
        if (!found) lags[num_lags++] = p;
        if (p > UINT32_MAX / 2) break;
        p *= 2;
    }

    /* Fractions of length */
    static const uint32_t divs[] = {2, 3, 4, 5, 6, 8};
    for (int d = 0; d < 6 && num_lags < 128; d++) {
        uint32_t frac = n / divs[d];
        if (frac == 0) continue;
        int found = 0;
        for (uint32_t j = 0; j < num_lags; j++) {
            if (lags[j] == frac) { found = 1; break; }
        }
        if (!found) lags[num_lags++] = frac;
    }

    /* Simple insertion sort (small array) */
    for (uint32_t i = 1; i < num_lags; i++) {
        uint32_t key = lags[i];
        int j = (int)i - 1;
        while (j >= 0 && lags[j] > key) {
            lags[j + 1] = lags[j];
            j--;
        }
        lags[j + 1] = key;
    }

    /* Test each lag */
    for (uint32_t i = 0; i < num_lags; i++) {
        uint32_t lag = lags[i];
        if (lag >= n) continue;
        double ac = autocorrelation_packed(bits, num_bits, lag);
        if (ac > threshold || ac < -threshold) {
            return 1;
        }
    }

    return 0;
}
