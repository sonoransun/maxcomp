/* prng_kernels.c — PRNG seed brute-force search with NEON acceleration.
 *
 * Implements MT19937, xorshift128+, and glibc-style LCG.
 * MT19937 seeding matches CPython's random.Random(seed) exactly.
 */

#include "maxcomp_native.h"
#include <arm_neon.h>
#include <string.h>

/* ============================================================
 * MT19937 — Mersenne Twister
 *
 * Matches CPython's random module, which uses init_by_array
 * for integer seeds. For seed S, the key array is the little-
 * endian 32-bit word decomposition of S (with at least [0]
 * for seed == 0).
 * ============================================================ */

#define MT_N       624
#define MT_M       397
#define MT_MATRIX_A    0x9908B0DFU
#define MT_UPPER_MASK  0x80000000U
#define MT_LOWER_MASK  0x7FFFFFFFU

typedef struct {
    uint32_t mt[MT_N];
    int      mti;
} mt_state;

static void mt_init_genrand(mt_state *s, uint32_t seed)
{
    s->mt[0] = seed;
    for (s->mti = 1; s->mti < MT_N; s->mti++) {
        s->mt[s->mti] =
            1812433253U * (s->mt[s->mti - 1] ^ (s->mt[s->mti - 1] >> 30))
            + (uint32_t)s->mti;
    }
}

static void mt_init_by_array(mt_state *s, const uint32_t *key, uint32_t key_length)
{
    uint32_t i, j, k;
    mt_init_genrand(s, 19650218U);
    i = 1; j = 0;
    k = MT_N > key_length ? MT_N : key_length;
    for (; k; k--) {
        s->mt[i] = (s->mt[i] ^ ((s->mt[i - 1] ^ (s->mt[i - 1] >> 30)) * 1664525U))
                    + key[j] + j;
        i++; j++;
        if (i >= MT_N) { s->mt[0] = s->mt[MT_N - 1]; i = 1; }
        if (j >= key_length) j = 0;
    }
    for (k = MT_N - 1; k; k--) {
        s->mt[i] = (s->mt[i] ^ ((s->mt[i - 1] ^ (s->mt[i - 1] >> 30)) * 1566083941U))
                    - i;
        i++;
        if (i >= MT_N) { s->mt[0] = s->mt[MT_N - 1]; i = 1; }
    }
    s->mt[0] = 0x80000000U;  /* MSB is 1; assuring non-zero initial array */
    s->mti = MT_N;  /* force twist on first extraction */
}

/* Seed MT state the same way CPython does for integer seeds.
 * For a non-negative integer seed, CPython decomposes it into
 * 32-bit little-endian words and calls init_by_array. */
static void mt_seed_python(mt_state *s, uint32_t seed)
{
    /* For seeds 0..2^32-1 the key is either [0] (seed==0) or [seed]. */
    uint32_t key[1];
    key[0] = seed;
    mt_init_by_array(s, key, 1);
}

static void mt_twist(mt_state *s)
{
    uint32_t y;
    int kk;
    static const uint32_t mag01[2] = {0, MT_MATRIX_A};

    for (kk = 0; kk < MT_N - MT_M; kk++) {
        y = (s->mt[kk] & MT_UPPER_MASK) | (s->mt[kk + 1] & MT_LOWER_MASK);
        s->mt[kk] = s->mt[kk + MT_M] ^ (y >> 1) ^ mag01[y & 1];
    }
    for (; kk < MT_N - 1; kk++) {
        y = (s->mt[kk] & MT_UPPER_MASK) | (s->mt[kk + 1] & MT_LOWER_MASK);
        s->mt[kk] = s->mt[kk + (MT_M - MT_N)] ^ (y >> 1) ^ mag01[y & 1];
    }
    y = (s->mt[MT_N - 1] & MT_UPPER_MASK) | (s->mt[0] & MT_LOWER_MASK);
    s->mt[MT_N - 1] = s->mt[MT_M - 1] ^ (y >> 1) ^ mag01[y & 1];

    s->mti = 0;
}

static uint32_t mt_genrand_uint32(mt_state *s)
{
    uint32_t y;

    if (s->mti >= MT_N)
        mt_twist(s);

    y = s->mt[s->mti++];

    /* Tempering */
    y ^= (y >> 11);
    y ^= (y <<  7) & 0x9D2C5680U;
    y ^= (y << 15) & 0xEFC60000U;
    y ^= (y >> 18);

    return y;
}

/* Generate bits matching Python's getrandbits(n).
 *
 * Python getrandbits(n):
 *   k = ceil(n / 32)
 *   Generate k uint32 values.
 *   Build integer: word[0] is lowest 32 bits, word[k-1] is highest.
 *   Take the low n bits of that integer.
 *   Return as big-endian bytes.
 *
 * So the FIRST word generated goes into the LOWEST position.
 */
static void mt_getrandbits(mt_state *s, uint32_t n, uint8_t *out)
{
    uint32_t num_bytes = (n + 7) / 8;
    uint32_t k = (n + 31) / 32;
    /* We need at most ceil(n/32) words = up to ~8K for 256K bits, but for
     * our use cases n is at most a few thousand bits. Use stack for small,
     * otherwise we just stream directly. */

    /* Strategy: generate k words. The integer value is:
     *   val = word[0] + word[1]*2^32 + ... + word[k-1]*2^(32*(k-1))
     * Then take low n bits, convert to big-endian bytes.
     *
     * We build the output byte array directly:
     * The words are little-endian in significance. So word[k-1] provides
     * the most significant bits. We place it first in big-endian output.
     */

    memset(out, 0, num_bytes);

    /* Generate words into a temporary array. For efficiency, generate
     * directly into a reversed layout. */
    if (k == 0) return;

    /* We'll fill from the end of the output buffer. Each word contributes
     * 4 bytes in big-endian order, starting from the least significant (last). */

    /* Generate all k words. */
    uint32_t words[k];
    for (uint32_t i = 0; i < k; i++) {
        words[i] = mt_genrand_uint32(s);
    }

    /* Mask the top word if n is not a multiple of 32. */
    uint32_t top_bits = n % 32;
    if (top_bits != 0) {
        words[k - 1] &= (1U << top_bits) - 1;
    }

    /* Now the integer value is sum(words[i] * 2^(32*i)).
     * Convert to big-endian bytes of length num_bytes.
     *
     * Byte layout (big-endian):
     *   out[0] is the most significant byte.
     *   out[num_bytes-1] is the least significant byte.
     *
     * Word layout:
     *   words[0] contains bits 0..31  (least significant)
     *   words[k-1] contains the most significant bits
     *
     * For each word[i], its 4 bytes go at positions:
     *   byte offset from the END: i*4 .. i*4+3
     *   i.e., out[num_bytes - 1 - i*4 - 0] gets lowest byte of word[i]
     *         out[num_bytes - 1 - i*4 - 1] gets next byte, etc.
     *
     * But we must be careful about the top word which may have fewer than 4 bytes.
     */

    /* Simplest correct approach: fill byte-by-byte from the integer. */
    /* Byte j (0-indexed from the end, i.e., out[num_bytes-1-j]) gets
     * bit range [8*j .. 8*j+7] from the integer.
     * That bit range is in word[j/4], bits [(j%4)*8 .. (j%4)*8+7]. */
    for (uint32_t j = 0; j < num_bytes; j++) {
        uint32_t word_idx = j / 4;
        uint32_t byte_in_word = j % 4;
        if (word_idx < k) {
            out[num_bytes - 1 - j] = (uint8_t)(words[word_idx] >> (byte_in_word * 8));
        }
    }
}

/* Compare generated bytes against target. Returns 1 if match, 0 if not. */
static int compare_bits(const uint8_t *generated, const uint8_t *target,
                        uint32_t bit_length)
{
    uint32_t full_bytes = bit_length / 8;
    uint32_t remaining_bits = bit_length % 8;

    if (full_bytes > 0 && memcmp(generated, target, full_bytes) != 0)
        return 0;

    if (remaining_bits > 0) {
        uint8_t mask = (uint8_t)(0xFF << (8 - remaining_bits));
        if ((generated[full_bytes] & mask) != (target[full_bytes] & mask))
            return 0;
    }

    return 1;
}

/* ============================================================
 * MT19937 brute-force search (scalar — MT is hard to vectorize)
 * ============================================================ */

int64_t prng_mt19937_search(const uint8_t *target, uint32_t target_bit_length,
                            uint32_t max_seed)
{
    uint32_t num_bytes = (target_bit_length + 7) / 8;
    if (num_bytes == 0 || target_bit_length == 0)
        return -1;

    /* Stack buffer for generated output — sufficient for typical use */
    uint8_t gen[num_bytes];
    mt_state st;

    for (uint32_t seed = 0; seed < max_seed; seed++) {
        mt_seed_python(&st, seed);
        mt_getrandbits(&st, target_bit_length, gen);
        if (compare_bits(gen, target, target_bit_length))
            return (int64_t)seed;
    }

    return -1;
}

/* ============================================================
 * xorshift128+ — NEON-accelerated (2 seeds in parallel)
 * ============================================================ */

/* Scalar xorshift128+ for a single seed. Writes output to out. */
static void xorshift128_generate(uint32_t seed, uint32_t bit_length, uint8_t *out)
{
    uint32_t num_bytes = (bit_length + 7) / 8;
    memset(out, 0, num_bytes);

    uint64_t s0 = ((uint64_t)seed) | 1;
    uint64_t s1 = ((uint64_t)seed * 6364136223846793005ULL + 1) | 1;

    uint32_t bits_generated = 0;

    while (bits_generated < bit_length) {
        /* xorshift128+ step */
        uint64_t x = s0;
        uint64_t y = s1;
        s0 = y;
        x ^= (x << 23);
        s1 = x ^ y ^ (x >> 17) ^ (y >> 26);
        uint64_t val = s1 + y;

        uint32_t bits_needed = bit_length - bits_generated;
        if (bits_needed > 64) bits_needed = 64;

        /* Take top bits_needed bits of val */
        uint64_t top_bits = val >> (64 - bits_needed);

        /* Pack into output buffer big-endian, MSB first */
        /* We're writing bits_needed bits starting at bit position bits_generated */
        for (uint32_t i = 0; i < bits_needed; i++) {
            uint32_t bit_val = (top_bits >> (bits_needed - 1 - i)) & 1;
            uint32_t pos = bits_generated + i;
            if (bit_val)
                out[pos / 8] |= (uint8_t)(0x80 >> (pos % 8));
        }

        bits_generated += bits_needed;
    }
}

/* NEON-accelerated xorshift128+ search, processing 2 seeds in parallel. */
int64_t prng_xorshift128_search(const uint8_t *target, uint32_t target_bit_length,
                                uint32_t max_seed)
{
    uint32_t num_bytes = (target_bit_length + 7) / 8;
    if (num_bytes == 0 || target_bit_length == 0)
        return -1;

    /* For seeds that can be processed 2 at a time via NEON */
    uint32_t seed = 0;

    /* NEON path: process 2 seeds at a time using uint64x2_t.
     * We run the xorshift128+ iterations for 2 seeds simultaneously,
     * then compare each output against target. */
    for (; seed + 2 <= max_seed; seed += 2) {
        uint64_t seed0 = seed;
        uint64_t seed1 = seed + 1;

        /* Init s0 for both lanes */
        uint64x2_t vs0 = {(seed0 | 1), (seed1 | 1)};
        /* Init s1 for both lanes */
        uint64x2_t vs1 = {
            (seed0 * 6364136223846793005ULL + 1) | 1,
            (seed1 * 6364136223846793005ULL + 1) | 1
        };

        /* Generate bits for both seeds and compare against target */
        uint8_t gen0[num_bytes];
        uint8_t gen1[num_bytes];
        memset(gen0, 0, num_bytes);
        memset(gen1, 0, num_bytes);

        uint32_t bits_generated = 0;
        int match0 = 1, match1 = 1;

        while (bits_generated < target_bit_length && (match0 || match1)) {
            /* xorshift128+ step — NEON parallel */
            uint64x2_t vx = vs0;
            uint64x2_t vy = vs1;
            vs0 = vy;
            vx = veorq_u64(vx, vshlq_n_u64(vx, 23));
            uint64x2_t vxr17 = vshrq_n_u64(vx, 17);
            uint64x2_t vyr26 = vshrq_n_u64(vy, 26);
            vs1 = veorq_u64(veorq_u64(vx, vy), veorq_u64(vxr17, vyr26));
            uint64x2_t vout = vaddq_u64(vs1, vy);

            uint64_t out0 = vgetq_lane_u64(vout, 0);
            uint64_t out1 = vgetq_lane_u64(vout, 1);

            uint32_t bits_needed = target_bit_length - bits_generated;
            if (bits_needed > 64) bits_needed = 64;

            uint64_t top0 = out0 >> (64 - bits_needed);
            uint64_t top1 = out1 >> (64 - bits_needed);

            /* Pack and compare incrementally */
            for (uint32_t i = 0; i < bits_needed; i++) {
                uint32_t pos = bits_generated + i;
                uint32_t byte_idx = pos / 8;
                uint8_t bit_mask = (uint8_t)(0x80 >> (pos % 8));

                if (match0) {
                    uint32_t bv = (top0 >> (bits_needed - 1 - i)) & 1;
                    if (bv)
                        gen0[byte_idx] |= bit_mask;
                    /* Early exit: compare this byte when complete or at end */
                    if ((pos % 8 == 7) || (pos == target_bit_length - 1)) {
                        uint8_t tmask = 0xFF;
                        if (pos == target_bit_length - 1 && target_bit_length % 8 != 0) {
                            tmask = (uint8_t)(0xFF << (8 - (target_bit_length % 8)));
                        }
                        if ((gen0[byte_idx] & tmask) != (target[byte_idx] & tmask))
                            match0 = 0;
                    }
                }
                if (match1) {
                    uint32_t bv = (top1 >> (bits_needed - 1 - i)) & 1;
                    if (bv)
                        gen1[byte_idx] |= bit_mask;
                    if ((pos % 8 == 7) || (pos == target_bit_length - 1)) {
                        uint8_t tmask = 0xFF;
                        if (pos == target_bit_length - 1 && target_bit_length % 8 != 0) {
                            tmask = (uint8_t)(0xFF << (8 - (target_bit_length % 8)));
                        }
                        if ((gen1[byte_idx] & tmask) != (target[byte_idx] & tmask))
                            match1 = 0;
                    }
                }
            }

            bits_generated += bits_needed;
        }

        if (match0 && compare_bits(gen0, target, target_bit_length))
            return (int64_t)seed;
        if (match1 && compare_bits(gen1, target, target_bit_length))
            return (int64_t)(seed + 1);
    }

    /* Handle remaining seed (if max_seed is odd) */
    for (; seed < max_seed; seed++) {
        uint8_t gen[num_bytes];
        xorshift128_generate(seed, target_bit_length, gen);
        if (compare_bits(gen, target, target_bit_length))
            return (int64_t)seed;
    }

    return -1;
}

/* ============================================================
 * LCG — NEON-accelerated (4 seeds in parallel)
 * ============================================================ */

/* Scalar LCG generate for a single seed. */
static void lcg_generate(uint32_t seed, uint32_t bit_length, uint8_t *out)
{
    uint32_t num_bytes = (bit_length + 7) / 8;
    memset(out, 0, num_bytes);

    uint32_t state = seed & 0x7FFFFFFFU;
    uint32_t bits_generated = 0;

    while (bits_generated < bit_length) {
        state = (1103515245U * state + 12345U) & 0x7FFFFFFFU;
        uint32_t bits_needed = bit_length - bits_generated;
        if (bits_needed > 16) bits_needed = 16;

        uint32_t extracted = (state >> (31 - bits_needed)) & ((1U << bits_needed) - 1);

        /* Pack big-endian */
        for (uint32_t i = 0; i < bits_needed; i++) {
            uint32_t bit_val = (extracted >> (bits_needed - 1 - i)) & 1;
            uint32_t pos = bits_generated + i;
            if (bit_val)
                out[pos / 8] |= (uint8_t)(0x80 >> (pos % 8));
        }

        bits_generated += bits_needed;
    }
}

/* NEON-accelerated LCG search, processing 4 seeds in parallel. */
int64_t prng_lcg_search(const uint8_t *target, uint32_t target_bit_length,
                        uint32_t max_seed)
{
    uint32_t num_bytes = (target_bit_length + 7) / 8;
    if (num_bytes == 0 || target_bit_length == 0)
        return -1;

    uint32_t seed = 0;

    /* NEON path: process 4 seeds at a time using uint32x4_t. */
    for (; seed + 4 <= max_seed; seed += 4) {
        /* Initialize 4 LCG states */
        uint32x4_t vstate = {
            (seed)     & 0x7FFFFFFFU,
            (seed + 1) & 0x7FFFFFFFU,
            (seed + 2) & 0x7FFFFFFFU,
            (seed + 3) & 0x7FFFFFFFU
        };

        uint32x4_t va = vdupq_n_u32(1103515245U);
        uint32x4_t vc = vdupq_n_u32(12345U);
        uint32x4_t vmask31 = vdupq_n_u32(0x7FFFFFFFU);

        uint8_t gen[4][num_bytes];
        memset(gen, 0, 4 * num_bytes);

        uint32_t bits_generated = 0;
        int active[4] = {1, 1, 1, 1};

        while (bits_generated < target_bit_length) {
            /* LCG step for all 4 lanes */
            /* state = (1103515245 * state + 12345) & 0x7FFFFFFF */
            vstate = vandq_u32(vmlaq_u32(vc, va, vstate), vmask31);

            uint32_t bits_needed = target_bit_length - bits_generated;
            if (bits_needed > 16) bits_needed = 16;

            uint32_t shift = 31 - bits_needed;
            uint32_t extract_mask = (1U << bits_needed) - 1;

            /* Extract the 4 states */
            uint32_t states[4];
            vst1q_u32(states, vstate);

            for (int lane = 0; lane < 4; lane++) {
                if (!active[lane]) continue;

                uint32_t extracted = (states[lane] >> shift) & extract_mask;

                for (uint32_t i = 0; i < bits_needed; i++) {
                    uint32_t bit_val = (extracted >> (bits_needed - 1 - i)) & 1;
                    uint32_t pos = bits_generated + i;
                    if (bit_val)
                        gen[lane][pos / 8] |= (uint8_t)(0x80 >> (pos % 8));
                }

                /* Early byte-level comparison */
                uint32_t end_pos = bits_generated + bits_needed - 1;
                uint32_t start_byte = bits_generated / 8;
                uint32_t end_byte = end_pos / 8;

                for (uint32_t b = start_byte; b <= end_byte; b++) {
                    int is_last_byte = (b == (target_bit_length - 1) / 8);
                    int byte_complete = ((b + 1) * 8 <= bits_generated + bits_needed);
                    if (byte_complete || (end_pos == target_bit_length - 1 && is_last_byte)) {
                        uint8_t tmask = 0xFF;
                        if (is_last_byte && target_bit_length % 8 != 0) {
                            tmask = (uint8_t)(0xFF << (8 - (target_bit_length % 8)));
                        }
                        if ((gen[lane][b] & tmask) != (target[b] & tmask)) {
                            active[lane] = 0;
                            break;
                        }
                    }
                }
            }

            bits_generated += bits_needed;

            /* If no lanes active, break early */
            if (!active[0] && !active[1] && !active[2] && !active[3])
                break;
        }

        for (int lane = 0; lane < 4; lane++) {
            if (active[lane] && compare_bits(gen[lane], target, target_bit_length))
                return (int64_t)(seed + (uint32_t)lane);
        }
    }

    /* Handle remaining seeds */
    for (; seed < max_seed; seed++) {
        uint8_t gen[num_bytes];
        lcg_generate(seed, target_bit_length, gen);
        if (compare_bits(gen, target, target_bit_length))
            return (int64_t)seed;
    }

    return -1;
}

/* ============================================================
 * prng_search_all — try MT19937, xorshift128, LCG in order
 * ============================================================ */

int prng_search_all(const uint8_t *target, uint32_t target_bit_length,
                    uint32_t max_seed, int *out_type, uint64_t *out_seed)
{
    int64_t result;

    /* Type 0: MT19937 */
    result = prng_mt19937_search(target, target_bit_length, max_seed);
    if (result >= 0) {
        *out_type = 0;
        *out_seed = (uint64_t)result;
        return 0;
    }

    /* Type 1: xorshift128+ */
    result = prng_xorshift128_search(target, target_bit_length, max_seed);
    if (result >= 0) {
        *out_type = 1;
        *out_seed = (uint64_t)result;
        return 0;
    }

    /* Type 2: LCG */
    result = prng_lcg_search(target, target_bit_length, max_seed);
    if (result >= 0) {
        *out_type = 2;
        *out_seed = (uint64_t)result;
        return 0;
    }

    return -1;
}

/* ============================================================
 * prng_generate — generate bits for decompression
 * ============================================================ */

void prng_generate(int prng_type, uint64_t seed, uint32_t bit_length,
                   uint8_t *output)
{
    uint32_t num_bytes = (bit_length + 7) / 8;
    if (num_bytes == 0 || bit_length == 0)
        return;

    switch (prng_type) {
    case 0: {
        /* MT19937 */
        mt_state st;
        mt_seed_python(&st, (uint32_t)seed);
        mt_getrandbits(&st, bit_length, output);
        break;
    }
    case 1:
        /* xorshift128+ */
        xorshift128_generate((uint32_t)seed, bit_length, output);
        break;
    case 2:
        /* LCG */
        lcg_generate((uint32_t)seed, bit_length, output);
        break;
    default:
        memset(output, 0, num_bytes);
        break;
    }
}
