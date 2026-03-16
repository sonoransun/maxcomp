#ifndef MAXCOMP_NATIVE_H
#define MAXCOMP_NATIVE_H

#include <stdint.h>
#include <stddef.h>

/* Version check */
uint32_t maxcomp_native_version(void);

/* ============================================================
 * PRNG seed brute-force
 * ============================================================ */

/* Search all 3 PRNG types (0=MT19937, 1=xorshift128, 2=LCG) for seeds
 * [0, max_seed) that produce target bits.
 * On success: writes prng_type to *out_type, seed to *out_seed, returns 0.
 * On failure: returns -1. */
int prng_search_all(
    const uint8_t *target, uint32_t target_bit_length,
    uint32_t max_seed,
    int *out_type, uint64_t *out_seed
);

/* Generate bits from a given PRNG type and seed.
 * output must hold at least (bit_length + 7) / 8 bytes. */
void prng_generate(
    int prng_type, uint64_t seed, uint32_t bit_length,
    uint8_t *output
);

/* Individual PRNG search functions. Return matching seed or -1. */
int64_t prng_mt19937_search(const uint8_t *target, uint32_t target_bit_length, uint32_t max_seed);
int64_t prng_xorshift128_search(const uint8_t *target, uint32_t target_bit_length, uint32_t max_seed);
int64_t prng_lcg_search(const uint8_t *target, uint32_t target_bit_length, uint32_t max_seed);

/* ============================================================
 * Hash preimage search
 * ============================================================ */

/* Search for a preimage (1..max_preimage_len bytes) whose hash prefix matches target.
 * hash types: 0=MD5, 1=SHA1, 2=SHA256.
 * On success: writes to out_hash_type, out_preimage, out_preimage_len. Returns 0.
 * On failure: returns -1.
 * out_preimage must have room for max_preimage_len bytes. */
int hash_preimage_search(
    const uint8_t *target, uint32_t target_bit_length,
    uint32_t max_preimage_len,
    int *out_hash_type, uint8_t *out_preimage, uint32_t *out_preimage_len
);

/* Compute hash of preimage. digest must hold at least 32 bytes. */
void hash_compute(int hash_type, const uint8_t *preimage, uint32_t preimage_len, uint8_t *digest);

/* ============================================================
 * IFS block matching
 * ============================================================ */

/* For each range block, find best matching domain block.
 * domain_pool: contiguous array of num_domains blocks, each (block_size+7)/8 bytes.
 * range_blocks: contiguous array of num_range_blocks blocks.
 * max_errors: maximum bit errors for a valid match.
 * Returns number of range blocks that found a valid match (<= max_errors). */
uint32_t ifs_match_blocks(
    const uint8_t *domain_pool, uint32_t num_domains,
    const uint8_t *range_blocks, uint32_t num_range_blocks,
    uint32_t block_size,
    uint32_t max_errors,
    uint32_t *out_domain_indices,
    uint8_t *out_inverted,
    uint32_t *out_errors
);

/* ============================================================
 * Cellular automata
 * ============================================================ */

/* Run elementary CA. output must hold (width * generations + 7) / 8 bytes. */
void automata_run(
    uint8_t rule, uint32_t width, uint32_t generations,
    const uint8_t *initial_state,
    uint8_t *output
);

/* Search all 256 rules x widths for target match.
 * widths: array of width values; num_widths: count.
 * On success: writes results, returns 0. On failure: returns -1. */
int automata_search(
    const uint8_t *target, uint32_t target_bit_length,
    const uint32_t *widths, uint32_t num_widths,
    uint8_t *out_rule, uint32_t *out_width, uint32_t *out_generations,
    uint8_t *out_init_state, uint32_t *out_init_state_len
);

/* ============================================================
 * Autocorrelation
 * ============================================================ */

/* Compute normalized autocorrelation of packed bit array at given lag. */
double autocorrelation_packed(
    const uint8_t *bits, uint32_t num_bits, uint32_t lag
);

/* Check self-similarity: returns 1 if any lag exceeds threshold, 0 otherwise. */
int has_self_similarity_packed(
    const uint8_t *bits, uint32_t num_bits, double threshold
);

/* ============================================================
 * BitStream operations
 * ============================================================ */

/* Extract sub-bitstream. dst must hold (length + 7) / 8 bytes. */
void bitstream_slice(
    const uint8_t *src, uint32_t src_bit_length,
    uint32_t start, uint32_t length,
    uint8_t *dst
);

/* Concatenate two bit arrays. dst must hold (a_bits + b_bits + 7) / 8 bytes. */
void bitstream_concat(
    const uint8_t *a, uint32_t a_bit_length,
    const uint8_t *b, uint32_t b_bit_length,
    uint8_t *dst
);

/* Hamming distance between two packed bit arrays. */
uint32_t bitstream_hamming(
    const uint8_t *a, const uint8_t *b, uint32_t num_bits
);

/* Popcount of packed bit array. */
uint32_t bitstream_popcount(
    const uint8_t *data, uint32_t num_bits
);

#endif /* MAXCOMP_NATIVE_H */
