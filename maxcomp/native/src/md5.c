/* md5.c — standalone MD5 implementation (RFC 1321) */

#include "md5.h"
#include <string.h>

/* Per-round shift amounts */
static const uint32_t S[64] = {
    7,12,17,22, 7,12,17,22, 7,12,17,22, 7,12,17,22,
    5, 9,14,20, 5, 9,14,20, 5, 9,14,20, 5, 9,14,20,
    4,11,16,23, 4,11,16,23, 4,11,16,23, 4,11,16,23,
    6,10,15,21, 6,10,15,21, 6,10,15,21, 6,10,15,21,
};

/* Pre-computed T[i] = floor(2^32 * |sin(i+1)|) */
static const uint32_t K[64] = {
    0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee,
    0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501,
    0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be,
    0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821,
    0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa,
    0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
    0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed,
    0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a,
    0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c,
    0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70,
    0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05,
    0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
    0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039,
    0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
    0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1,
    0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391,
};

static inline uint32_t rotate_left(uint32_t x, uint32_t n) {
    return (x << n) | (x >> (32 - n));
}

static inline uint32_t le32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static inline void put_le32(uint8_t *p, uint32_t v) {
    p[0] = (uint8_t)(v);
    p[1] = (uint8_t)(v >> 8);
    p[2] = (uint8_t)(v >> 16);
    p[3] = (uint8_t)(v >> 24);
}

static void md5_compress(uint32_t state[4], const uint8_t block[64]) {
    uint32_t M[16];
    for (int i = 0; i < 16; i++)
        M[i] = le32(block + 4 * i);

    uint32_t a = state[0], b = state[1], c = state[2], d = state[3];

    for (int i = 0; i < 64; i++) {
        uint32_t f, g;
        if (i < 16) {
            f = (b & c) | (~b & d);
            g = (uint32_t)i;
        } else if (i < 32) {
            f = (d & b) | (~d & c);
            g = (5 * (uint32_t)i + 1) % 16;
        } else if (i < 48) {
            f = b ^ c ^ d;
            g = (3 * (uint32_t)i + 5) % 16;
        } else {
            f = c ^ (b | ~d);
            g = (7 * (uint32_t)i) % 16;
        }
        f = f + a + K[i] + M[g];
        a = d;
        d = c;
        c = b;
        b = b + rotate_left(f, S[i]);
    }

    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
}

void md5_hash(const uint8_t *data, size_t len, uint8_t digest[16]) {
    uint32_t state[4] = {
        0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476
    };

    /* Process full 64-byte blocks */
    size_t offset = 0;
    while (offset + 64 <= len) {
        md5_compress(state, data + offset);
        offset += 64;
    }

    /* Padding: build final block(s) */
    uint8_t buf[128]; /* up to 2 blocks for padding */
    size_t remaining = len - offset;
    memcpy(buf, data + offset, remaining);
    buf[remaining] = 0x80;
    memset(buf + remaining + 1, 0, sizeof(buf) - remaining - 1);

    /* If remaining + 1 + 8 > 64 we need two blocks */
    size_t pad_blocks;
    if (remaining + 1 + 8 > 64) {
        pad_blocks = 128;
    } else {
        pad_blocks = 64;
    }

    /* Append bit length as 64-bit LE at end of last block */
    uint64_t bit_len = (uint64_t)len * 8;
    put_le32(buf + pad_blocks - 8, (uint32_t)(bit_len));
    put_le32(buf + pad_blocks - 4, (uint32_t)(bit_len >> 32));

    for (size_t i = 0; i < pad_blocks; i += 64)
        md5_compress(state, buf + i);

    /* Write digest in LE */
    for (int i = 0; i < 4; i++)
        put_le32(digest + 4 * i, state[i]);
}
