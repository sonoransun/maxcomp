/* automata_kernels.c — Elementary cellular automata C kernel with NEON. */

#include "maxcomp_native.h"
#include <arm_neon.h>
#include <string.h>
#include <stdlib.h>

/* ------------------------------------------------------------------ */
/*  Helpers                                                            */
/* ------------------------------------------------------------------ */

/* Get bit `pos` from packed MSB-first array. */
static inline uint8_t get_bit(const uint8_t *packed, uint32_t pos) {
    return (packed[pos >> 3] >> (7 - (pos & 7))) & 1;
}

/* Set bit `pos` in packed MSB-first array. */
static inline void set_bit(uint8_t *packed, uint32_t pos, uint8_t val) {
    uint32_t byte_idx = pos >> 3;
    uint32_t bit_off  = 7 - (pos & 7);
    if (val)
        packed[byte_idx] |=  (1u << bit_off);
    else
        packed[byte_idx] &= ~(1u << bit_off);
}

/* Number of bytes needed for `nbits` packed bits. */
static inline uint32_t packed_bytes(uint32_t nbits) {
    return (nbits + 7) >> 3;
}

/* ------------------------------------------------------------------ */
/*  automata_run                                                       */
/* ------------------------------------------------------------------ */

void automata_run(
    uint8_t rule, uint32_t width, uint32_t generations,
    const uint8_t *initial_state, uint8_t *output)
{
    if (width == 0 || generations == 0) return;

    /* Allocate two unpacked cell rows (cur / nxt). */
    uint8_t stack_a[64], stack_b[64];
    uint8_t *cur, *nxt, *heap_buf = NULL;

    if (width <= 64) {
        cur = stack_a;
        nxt = stack_b;
    } else {
        heap_buf = (uint8_t *)malloc(width * 2);
        cur = heap_buf;
        nxt = heap_buf + width;
    }

    /* Unpack initial_state (packed MSB-first) into cur[]. */
    memset(cur, 0, width);
    for (uint32_t i = 0; i < width; i++) {
        cur[i] = get_bit(initial_state, i);
    }

    /* Zero the output buffer. */
    uint32_t total_bits = width * generations;
    memset(output, 0, packed_bytes(total_bits));

    /* Write generation 0 into output. */
    uint32_t out_pos = 0;
    for (uint32_t i = 0; i < width; i++) {
        set_bit(output, out_pos++, cur[i]);
    }

    /* Remaining generations. */
    for (uint32_t g = 1; g < generations; g++) {

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
        if (width >= 8) {
            /* Build NEON lookup table: table[i] = (rule >> i) & 1 for i=0..7. */
            uint8_t tbl_data[8];
            for (int k = 0; k < 8; k++)
                tbl_data[k] = (rule >> k) & 1;
            uint8x8_t tbl = vld1_u8(tbl_data);

            uint32_t i = 0;
            for (; i + 8 <= width; i += 8) {
                /* Compute neighborhood index for 8 consecutive cells. */
                uint8_t idx_arr[8];
                for (uint32_t j = 0; j < 8; j++) {
                    uint32_t ci = i + j;
                    uint8_t left   = cur[(ci == 0) ? width - 1 : ci - 1];
                    uint8_t center = cur[ci];
                    uint8_t right  = cur[(ci + 1 == width) ? 0 : ci + 1];
                    idx_arr[j] = (uint8_t)((left << 2) | (center << 1) | right);
                }

                uint8x8_t indices = vld1_u8(idx_arr);
                uint8x8_t result  = vtbl1_u8(tbl, indices);

                uint8_t res_arr[8];
                vst1_u8(res_arr, result);

                for (uint32_t j = 0; j < 8; j++)
                    nxt[i + j] = res_arr[j];
            }
            /* Scalar tail for remaining cells. */
            for (; i < width; i++) {
                uint8_t left   = cur[(i == 0) ? width - 1 : i - 1];
                uint8_t center = cur[i];
                uint8_t right  = cur[(i + 1 == width) ? 0 : i + 1];
                uint8_t neighborhood = (uint8_t)((left << 2) | (center << 1) | right);
                nxt[i] = (rule >> neighborhood) & 1;
            }
        } else
#endif
        {
            /* Scalar path for narrow widths (< 8 or non-NEON). */
            for (uint32_t i = 0; i < width; i++) {
                uint8_t left   = cur[(i == 0) ? width - 1 : i - 1];
                uint8_t center = cur[i];
                uint8_t right  = cur[(i + 1 == width) ? 0 : i + 1];
                uint8_t neighborhood = (uint8_t)((left << 2) | (center << 1) | right);
                nxt[i] = (rule >> neighborhood) & 1;
            }
        }

        /* Write this generation into output. */
        for (uint32_t i = 0; i < width; i++) {
            set_bit(output, out_pos++, nxt[i]);
        }

        /* Swap cur / nxt. */
        uint8_t *tmp = cur;
        cur = nxt;
        nxt = tmp;
    }

    if (heap_buf) free(heap_buf);
}

/* ------------------------------------------------------------------ */
/*  automata_search                                                    */
/* ------------------------------------------------------------------ */

/* Compare first `bit_length` bits of `a` and `b` (packed MSB-first).
 * Returns 1 if equal, 0 otherwise. */
static int bits_equal(const uint8_t *a, const uint8_t *b, uint32_t bit_length) {
    uint32_t full_bytes = bit_length >> 3;
    uint32_t remaining  = bit_length & 7;

    /* Compare full bytes. */
    if (full_bytes > 0 && memcmp(a, b, full_bytes) != 0)
        return 0;

    /* Compare remaining bits in the last partial byte. */
    if (remaining > 0) {
        uint8_t mask = (uint8_t)(0xFF << (8 - remaining));
        if ((a[full_bytes] & mask) != (b[full_bytes] & mask))
            return 0;
    }

    return 1;
}

/* Transpose grid (width x generations) from row-major packed into column-major
 * packed buffer.  Both are MSB-first packed bit arrays. */
static void transpose_grid(
    const uint8_t *row_major, uint32_t width, uint32_t generations,
    uint8_t *col_major)
{
    uint32_t total = width * generations;
    memset(col_major, 0, packed_bytes(total));

    uint32_t dst = 0;
    for (uint32_t col = 0; col < width; col++) {
        for (uint32_t row = 0; row < generations; row++) {
            uint32_t src_pos = row * width + col;
            set_bit(col_major, dst, get_bit(row_major, src_pos));
            dst++;
        }
    }
}

int automata_search(
    const uint8_t *target, uint32_t target_bit_length,
    const uint32_t *widths, uint32_t num_widths,
    uint8_t *out_rule, uint32_t *out_width, uint32_t *out_generations,
    uint8_t *out_init_state, uint32_t *out_init_state_len)
{
    if (target_bit_length == 0) return -1;

    /* Scratch buffers — reuse across iterations.  Track max size needed. */
    uint8_t *output_buf   = NULL;
    uint8_t *col_buf      = NULL;
    uint8_t *init_buf     = NULL;
    uint32_t alloc_output = 0;
    uint32_t alloc_init   = 0;

    for (uint32_t r = 0; r < 256; r++) {
        uint8_t rule = (uint8_t)r;

        for (uint32_t wi = 0; wi < num_widths; wi++) {
            uint32_t width = widths[wi];
            if (width == 0) continue;

            /* Compute generations = ceil(target_bit_length / width). */
            uint32_t generations = (target_bit_length + width - 1) / width;
            if (generations < 1) generations = 1;

            uint32_t total_bits  = width * generations;
            if (total_bits < target_bit_length) continue;

            uint32_t out_bytes  = packed_bytes(total_bits);
            uint32_t init_bytes = packed_bytes(width);

            /* Ensure output buffer is large enough. */
            if (out_bytes > alloc_output) {
                free(output_buf);
                free(col_buf);
                alloc_output = out_bytes;
                output_buf = (uint8_t *)malloc(alloc_output);
                col_buf    = (uint8_t *)malloc(alloc_output);
            }

            /* Ensure init buffer is large enough. */
            if (init_bytes > alloc_init) {
                free(init_buf);
                alloc_init = init_bytes;
                init_buf = (uint8_t *)malloc(alloc_init);
            }

            /* Build initial state: all zeros except middle cell = 1. */
            memset(init_buf, 0, init_bytes);
            uint32_t center = width / 2;
            set_bit(init_buf, center, 1);

            /* Run the automaton. */
            automata_run(rule, width, generations, init_buf, output_buf);

            /* --- Row-major comparison (read_order == 0) --- */
            if (bits_equal(output_buf, target, target_bit_length)) {
                *out_rule          = rule;
                *out_width         = width;
                *out_generations   = generations;
                memcpy(out_init_state, init_buf, init_bytes);
                *out_init_state_len = init_bytes;

                free(output_buf);
                free(col_buf);
                free(init_buf);
                return 0;
            }

            /* --- Column-major comparison (read_order == 1) --- */
            transpose_grid(output_buf, width, generations, col_buf);

            if (bits_equal(col_buf, target, target_bit_length)) {
                *out_rule          = rule;
                *out_width         = width;
                *out_generations   = generations;
                memcpy(out_init_state, init_buf, init_bytes);
                *out_init_state_len = init_bytes;

                free(output_buf);
                free(col_buf);
                free(init_buf);
                return 0;
            }
        }
    }

    free(output_buf);
    free(col_buf);
    free(init_buf);
    return -1;
}
