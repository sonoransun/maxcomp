/* sha1.c — standalone SHA-1 implementation */

#include "sha1.h"
#include <string.h>

static inline uint32_t rotate_left(uint32_t x, uint32_t n) {
    return (x << n) | (x >> (32 - n));
}

static inline uint32_t be32(const uint8_t *p) {
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) |
           ((uint32_t)p[2] << 8) | (uint32_t)p[3];
}

static inline void put_be32(uint8_t *p, uint32_t v) {
    p[0] = (uint8_t)(v >> 24);
    p[1] = (uint8_t)(v >> 16);
    p[2] = (uint8_t)(v >> 8);
    p[3] = (uint8_t)(v);
}

static void sha1_compress(uint32_t state[5], const uint8_t block[64]) {
    uint32_t W[80];
    for (int i = 0; i < 16; i++)
        W[i] = be32(block + 4 * i);
    for (int i = 16; i < 80; i++)
        W[i] = rotate_left(W[i-3] ^ W[i-8] ^ W[i-14] ^ W[i-16], 1);

    uint32_t a = state[0], b = state[1], c = state[2],
             d = state[3], e = state[4];

    for (int i = 0; i < 80; i++) {
        uint32_t f, k;
        if (i < 20) {
            f = (b & c) | (~b & d);
            k = 0x5A827999;
        } else if (i < 40) {
            f = b ^ c ^ d;
            k = 0x6ED9EBA1;
        } else if (i < 60) {
            f = (b & c) | (b & d) | (c & d);
            k = 0x8F1BBCDC;
        } else {
            f = b ^ c ^ d;
            k = 0xCA62C1D6;
        }
        uint32_t temp = rotate_left(a, 5) + f + e + k + W[i];
        e = d;
        d = c;
        c = rotate_left(b, 30);
        b = a;
        a = temp;
    }

    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
}

void sha1_hash(const uint8_t *data, size_t len, uint8_t digest[20]) {
    uint32_t state[5] = {
        0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0
    };

    /* Process full 64-byte blocks */
    size_t offset = 0;
    while (offset + 64 <= len) {
        sha1_compress(state, data + offset);
        offset += 64;
    }

    /* Padding */
    uint8_t buf[128];
    size_t remaining = len - offset;
    memcpy(buf, data + offset, remaining);
    buf[remaining] = 0x80;
    memset(buf + remaining + 1, 0, sizeof(buf) - remaining - 1);

    size_t pad_blocks;
    if (remaining + 1 + 8 > 64)
        pad_blocks = 128;
    else
        pad_blocks = 64;

    /* Append bit length as 64-bit BE at end of last block */
    uint64_t bit_len = (uint64_t)len * 8;
    put_be32(buf + pad_blocks - 8, (uint32_t)(bit_len >> 32));
    put_be32(buf + pad_blocks - 4, (uint32_t)(bit_len));

    for (size_t i = 0; i < pad_blocks; i += 64)
        sha1_compress(state, buf + i);

    /* Write digest in BE */
    for (int i = 0; i < 5; i++)
        put_be32(digest + 4 * i, state[i]);
}
