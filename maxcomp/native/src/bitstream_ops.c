/*
 * bitstream_ops.c — Bit-level operations with ARM NEON acceleration
 *
 * Packed MSB-first format: bit index 0 is the MSB of byte 0.
 * All functions handle edge cases: 0 bits, non-byte-aligned lengths,
 * and inputs shorter than 16 bytes.
 */

#include "maxcomp_native.h"
#include <arm_neon.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* bitstream_slice                                                     */
/* ------------------------------------------------------------------ */

void bitstream_slice(
    const uint8_t *src, uint32_t src_bit_length,
    uint32_t start, uint32_t length,
    uint8_t *dst)
{
    if (length == 0) return;

    /* Clamp to available bits */
    if (start + length > src_bit_length) {
        if (start >= src_bit_length) {
            memset(dst, 0, (length + 7) / 8);
            return;
        }
        length = src_bit_length - start;
    }

    uint32_t dst_bytes = (length + 7) / 8;
    uint32_t byte_off = start / 8;
    uint32_t bit_off  = start % 8;

    if (bit_off == 0) {
        /* Byte-aligned fast path */
        uint32_t full_bytes = length / 8;
        if (full_bytes > 0) {
            memcpy(dst, src + byte_off, full_bytes);
        }
        /* Partial last byte */
        uint32_t tail = length % 8;
        if (tail > 0) {
            uint8_t mask = (uint8_t)(0xFF << (8 - tail));
            dst[full_bytes] = src[byte_off + full_bytes] & mask;
        }
    } else {
        /* Non-aligned: shift+OR adjacent bytes */
        uint32_t src_total_bytes = (src_bit_length + 7) / 8;
        for (uint32_t i = 0; i < dst_bytes; i++) {
            uint32_t si = byte_off + i;
            uint8_t hi = 0, lo = 0;
            if (si < src_total_bytes) {
                hi = src[si] << bit_off;
            }
            if (si + 1 < src_total_bytes) {
                lo = src[si + 1] >> (8 - bit_off);
            }
            dst[i] = hi | lo;
        }
        /* Zero out unused trailing bits in last byte */
        uint32_t tail = length % 8;
        if (tail > 0) {
            uint8_t mask = (uint8_t)(0xFF << (8 - tail));
            dst[dst_bytes - 1] &= mask;
        }
    }
}

/* ------------------------------------------------------------------ */
/* bitstream_concat                                                    */
/* ------------------------------------------------------------------ */

void bitstream_concat(
    const uint8_t *a, uint32_t a_bit_length,
    const uint8_t *b, uint32_t b_bit_length,
    uint8_t *dst)
{
    uint32_t total_bits = a_bit_length + b_bit_length;
    if (total_bits == 0) return;

    uint32_t dst_bytes = (total_bits + 7) / 8;
    memset(dst, 0, dst_bytes);

    /* Copy all of a into dst */
    uint32_t a_full_bytes = a_bit_length / 8;
    uint32_t a_tail = a_bit_length % 8;

    if (a_full_bytes > 0) {
        memcpy(dst, a, a_full_bytes);
    }

    /* If a has a partial last byte, copy its used bits */
    if (a_tail > 0) {
        uint8_t mask = (uint8_t)(0xFF << (8 - a_tail));
        dst[a_full_bytes] = a[a_full_bytes] & mask;
    }

    /* Append b starting at bit position a_bit_length */
    if (b_bit_length == 0) return;

    uint32_t b_bytes = (b_bit_length + 7) / 8;
    uint32_t write_bit_off = a_bit_length % 8;

    if (write_bit_off == 0) {
        /* b starts on a byte boundary — simple copy */
        uint32_t write_byte = a_bit_length / 8;
        uint32_t b_full_bytes = b_bit_length / 8;
        if (b_full_bytes > 0) {
            memcpy(dst + write_byte, b, b_full_bytes);
        }
        uint32_t b_tail = b_bit_length % 8;
        if (b_tail > 0) {
            uint8_t mask = (uint8_t)(0xFF << (8 - b_tail));
            dst[write_byte + b_full_bytes] = b[b_full_bytes] & mask;
        }
    } else {
        /* b starts mid-byte — need to merge */
        uint32_t write_byte = a_bit_length / 8;
        uint32_t right_shift = write_bit_off;
        uint32_t left_shift = 8 - right_shift;

        for (uint32_t i = 0; i < b_bytes; i++) {
            /* Top part goes into current byte */
            dst[write_byte + i] |= b[i] >> right_shift;
            /* Bottom part goes into next byte */
            if (write_byte + i + 1 < dst_bytes) {
                dst[write_byte + i + 1] |= b[i] << left_shift;
            }
        }

        /* Zero out unused trailing bits in the last dst byte */
        uint32_t total_tail = total_bits % 8;
        if (total_tail > 0) {
            uint8_t mask = (uint8_t)(0xFF << (8 - total_tail));
            dst[dst_bytes - 1] &= mask;
        }
    }
}

/* ------------------------------------------------------------------ */
/* bitstream_hamming                                                   */
/* ------------------------------------------------------------------ */

uint32_t bitstream_hamming(
    const uint8_t *a, const uint8_t *b, uint32_t num_bits)
{
    if (num_bits == 0) return 0;

    uint32_t full_bytes = num_bits / 8;
    uint32_t tail_bits  = num_bits % 8;
    uint32_t total = 0;
    uint32_t i = 0;

    /* NEON: 16 bytes at a time */
    while (i + 16 <= full_bytes) {
        uint8x16_t va = vld1q_u8(a + i);
        uint8x16_t vb = vld1q_u8(b + i);
        uint8x16_t xored = veorq_u8(va, vb);
        uint8x16_t counts = vcntq_u8(xored);
        total += vaddlvq_u8(counts);
        i += 16;
    }

    /* Scalar remainder of full bytes */
    for (; i < full_bytes; i++) {
        total += (uint32_t)__builtin_popcount((unsigned)(a[i] ^ b[i]));
    }

    /* Partial last byte: only top `tail_bits` bits matter (MSB-first) */
    if (tail_bits > 0) {
        uint8_t mask = (uint8_t)(0xFF << (8 - tail_bits));
        total += (uint32_t)__builtin_popcount(
            (unsigned)((a[full_bytes] ^ b[full_bytes]) & mask));
    }

    return total;
}

/* ------------------------------------------------------------------ */
/* bitstream_popcount                                                  */
/* ------------------------------------------------------------------ */

uint32_t bitstream_popcount(
    const uint8_t *data, uint32_t num_bits)
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

    /* Partial last byte: only top `tail_bits` bits matter */
    if (tail_bits > 0) {
        uint8_t mask = (uint8_t)(0xFF << (8 - tail_bits));
        total += (uint32_t)__builtin_popcount((unsigned)(data[full_bytes] & mask));
    }

    return total;
}
