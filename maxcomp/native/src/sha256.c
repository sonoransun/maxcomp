/* sha256.c — standalone SHA-256 with optional ARM crypto extensions */

#include "sha256.h"
#include <string.h>

static const uint32_t SHA256_K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
};

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

/* ------------------------------------------------------------------ */
#if defined(__ARM_FEATURE_SHA2) && defined(__aarch64__)
/* ------------------------------------------------------------------ */

#include <arm_neon.h>

static void sha256_compress(uint32_t state[8], const uint8_t block[64]) {
    /* Load state into NEON registers.
     * ARM SHA-256 intrinsics expect ABCD in one register and EFGH in another. */
    uint32x4_t STATE0 = vld1q_u32(&state[0]); /* A B C D */
    uint32x4_t STATE1 = vld1q_u32(&state[4]); /* E F G H */

    /* Save original state for final addition */
    uint32x4_t ABCD_SAVE = STATE0;
    uint32x4_t EFGH_SAVE = STATE1;

    /* Load message into 4 NEON registers (big-endian) */
    uint32x4_t MSG0 = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(block +  0)));
    uint32x4_t MSG1 = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(block + 16)));
    uint32x4_t MSG2 = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(block + 32)));
    uint32x4_t MSG3 = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(block + 48)));

    uint32x4_t TMP0, TMP1, TMP2;

    /* Rounds 0-3 */
    TMP0 = vaddq_u32(MSG0, vld1q_u32(&SHA256_K[0]));
    TMP1 = STATE0;
    TMP2 = STATE1;
    STATE0 = vsha256hq_u32(TMP1, TMP2, TMP0);
    STATE1 = vsha256h2q_u32(TMP2, TMP1, TMP0);
    MSG0 = vsha256su0q_u32(MSG0, MSG1);

    /* Rounds 4-7 */
    TMP0 = vaddq_u32(MSG1, vld1q_u32(&SHA256_K[4]));
    TMP1 = STATE0;
    TMP2 = STATE1;
    STATE0 = vsha256hq_u32(TMP1, TMP2, TMP0);
    STATE1 = vsha256h2q_u32(TMP2, TMP1, TMP0);
    MSG1 = vsha256su0q_u32(MSG1, MSG2);
    MSG0 = vsha256su1q_u32(MSG0, MSG2, MSG3);

    /* Rounds 8-11 */
    TMP0 = vaddq_u32(MSG2, vld1q_u32(&SHA256_K[8]));
    TMP1 = STATE0;
    TMP2 = STATE1;
    STATE0 = vsha256hq_u32(TMP1, TMP2, TMP0);
    STATE1 = vsha256h2q_u32(TMP2, TMP1, TMP0);
    MSG2 = vsha256su0q_u32(MSG2, MSG3);
    MSG1 = vsha256su1q_u32(MSG1, MSG3, MSG0);

    /* Rounds 12-15 */
    TMP0 = vaddq_u32(MSG3, vld1q_u32(&SHA256_K[12]));
    TMP1 = STATE0;
    TMP2 = STATE1;
    STATE0 = vsha256hq_u32(TMP1, TMP2, TMP0);
    STATE1 = vsha256h2q_u32(TMP2, TMP1, TMP0);
    MSG3 = vsha256su0q_u32(MSG3, MSG0);
    MSG2 = vsha256su1q_u32(MSG2, MSG0, MSG1);

    /* Rounds 16-19 */
    TMP0 = vaddq_u32(MSG0, vld1q_u32(&SHA256_K[16]));
    TMP1 = STATE0;
    TMP2 = STATE1;
    STATE0 = vsha256hq_u32(TMP1, TMP2, TMP0);
    STATE1 = vsha256h2q_u32(TMP2, TMP1, TMP0);
    MSG0 = vsha256su0q_u32(MSG0, MSG1);
    MSG3 = vsha256su1q_u32(MSG3, MSG1, MSG2);

    /* Rounds 20-23 */
    TMP0 = vaddq_u32(MSG1, vld1q_u32(&SHA256_K[20]));
    TMP1 = STATE0;
    TMP2 = STATE1;
    STATE0 = vsha256hq_u32(TMP1, TMP2, TMP0);
    STATE1 = vsha256h2q_u32(TMP2, TMP1, TMP0);
    MSG1 = vsha256su0q_u32(MSG1, MSG2);
    MSG0 = vsha256su1q_u32(MSG0, MSG2, MSG3);

    /* Rounds 24-27 */
    TMP0 = vaddq_u32(MSG2, vld1q_u32(&SHA256_K[24]));
    TMP1 = STATE0;
    TMP2 = STATE1;
    STATE0 = vsha256hq_u32(TMP1, TMP2, TMP0);
    STATE1 = vsha256h2q_u32(TMP2, TMP1, TMP0);
    MSG2 = vsha256su0q_u32(MSG2, MSG3);
    MSG1 = vsha256su1q_u32(MSG1, MSG3, MSG0);

    /* Rounds 28-31 */
    TMP0 = vaddq_u32(MSG3, vld1q_u32(&SHA256_K[28]));
    TMP1 = STATE0;
    TMP2 = STATE1;
    STATE0 = vsha256hq_u32(TMP1, TMP2, TMP0);
    STATE1 = vsha256h2q_u32(TMP2, TMP1, TMP0);
    MSG3 = vsha256su0q_u32(MSG3, MSG0);
    MSG2 = vsha256su1q_u32(MSG2, MSG0, MSG1);

    /* Rounds 32-35 */
    TMP0 = vaddq_u32(MSG0, vld1q_u32(&SHA256_K[32]));
    TMP1 = STATE0;
    TMP2 = STATE1;
    STATE0 = vsha256hq_u32(TMP1, TMP2, TMP0);
    STATE1 = vsha256h2q_u32(TMP2, TMP1, TMP0);
    MSG0 = vsha256su0q_u32(MSG0, MSG1);
    MSG3 = vsha256su1q_u32(MSG3, MSG1, MSG2);

    /* Rounds 36-39 */
    TMP0 = vaddq_u32(MSG1, vld1q_u32(&SHA256_K[36]));
    TMP1 = STATE0;
    TMP2 = STATE1;
    STATE0 = vsha256hq_u32(TMP1, TMP2, TMP0);
    STATE1 = vsha256h2q_u32(TMP2, TMP1, TMP0);
    MSG1 = vsha256su0q_u32(MSG1, MSG2);
    MSG0 = vsha256su1q_u32(MSG0, MSG2, MSG3);

    /* Rounds 40-43 */
    TMP0 = vaddq_u32(MSG2, vld1q_u32(&SHA256_K[40]));
    TMP1 = STATE0;
    TMP2 = STATE1;
    STATE0 = vsha256hq_u32(TMP1, TMP2, TMP0);
    STATE1 = vsha256h2q_u32(TMP2, TMP1, TMP0);
    MSG2 = vsha256su0q_u32(MSG2, MSG3);
    MSG1 = vsha256su1q_u32(MSG1, MSG3, MSG0);

    /* Rounds 44-47 */
    TMP0 = vaddq_u32(MSG3, vld1q_u32(&SHA256_K[44]));
    TMP1 = STATE0;
    TMP2 = STATE1;
    STATE0 = vsha256hq_u32(TMP1, TMP2, TMP0);
    STATE1 = vsha256h2q_u32(TMP2, TMP1, TMP0);
    MSG3 = vsha256su0q_u32(MSG3, MSG0);
    MSG2 = vsha256su1q_u32(MSG2, MSG0, MSG1);

    /* Rounds 48-51 */
    TMP0 = vaddq_u32(MSG0, vld1q_u32(&SHA256_K[48]));
    TMP1 = STATE0;
    TMP2 = STATE1;
    STATE0 = vsha256hq_u32(TMP1, TMP2, TMP0);
    STATE1 = vsha256h2q_u32(TMP2, TMP1, TMP0);
    MSG3 = vsha256su1q_u32(MSG3, MSG1, MSG2);

    /* Rounds 52-55 */
    TMP0 = vaddq_u32(MSG1, vld1q_u32(&SHA256_K[52]));
    TMP1 = STATE0;
    TMP2 = STATE1;
    STATE0 = vsha256hq_u32(TMP1, TMP2, TMP0);
    STATE1 = vsha256h2q_u32(TMP2, TMP1, TMP0);

    /* Rounds 56-59 */
    TMP0 = vaddq_u32(MSG2, vld1q_u32(&SHA256_K[56]));
    TMP1 = STATE0;
    TMP2 = STATE1;
    STATE0 = vsha256hq_u32(TMP1, TMP2, TMP0);
    STATE1 = vsha256h2q_u32(TMP2, TMP1, TMP0);

    /* Rounds 60-63 */
    TMP0 = vaddq_u32(MSG3, vld1q_u32(&SHA256_K[60]));
    TMP1 = STATE0;
    TMP2 = STATE1;
    STATE0 = vsha256hq_u32(TMP1, TMP2, TMP0);
    STATE1 = vsha256h2q_u32(TMP2, TMP1, TMP0);

    /* Add saved state */
    STATE0 = vaddq_u32(STATE0, ABCD_SAVE);
    STATE1 = vaddq_u32(STATE1, EFGH_SAVE);

    /* Store result */
    vst1q_u32(&state[0], STATE0);
    vst1q_u32(&state[4], STATE1);
}

#else
/* ------------------------------------------------------------------ */
/* Software fallback                                                   */
/* ------------------------------------------------------------------ */

static inline uint32_t rotr32(uint32_t x, uint32_t n) {
    return (x >> n) | (x << (32 - n));
}

#define CH(x,y,z)  (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x,y,z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define EP0(x)  (rotr32(x, 2) ^ rotr32(x,13) ^ rotr32(x,22))
#define EP1(x)  (rotr32(x, 6) ^ rotr32(x,11) ^ rotr32(x,25))
#define SIG0(x) (rotr32(x, 7) ^ rotr32(x,18) ^ ((x) >> 3))
#define SIG1(x) (rotr32(x,17) ^ rotr32(x,19) ^ ((x) >> 10))

static void sha256_compress(uint32_t state[8], const uint8_t block[64]) {
    uint32_t W[64];
    for (int i = 0; i < 16; i++)
        W[i] = be32(block + 4 * i);
    for (int i = 16; i < 64; i++)
        W[i] = SIG1(W[i-2]) + W[i-7] + SIG0(W[i-15]) + W[i-16];

    uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
    uint32_t e = state[4], f = state[5], g = state[6], h = state[7];

    for (int i = 0; i < 64; i++) {
        uint32_t t1 = h + EP1(e) + CH(e,f,g) + SHA256_K[i] + W[i];
        uint32_t t2 = EP0(a) + MAJ(a,b,c);
        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }

    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

#endif /* __ARM_FEATURE_SHA2 */

/* ------------------------------------------------------------------ */
/* Public API                                                          */
/* ------------------------------------------------------------------ */

void sha256_hash(const uint8_t *data, size_t len, uint8_t digest[32]) {
    uint32_t state[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    };

    /* Process full 64-byte blocks */
    size_t offset = 0;
    while (offset + 64 <= len) {
        sha256_compress(state, data + offset);
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
        sha256_compress(state, buf + i);

    /* Write digest in BE */
    for (int i = 0; i < 8; i++)
        put_be32(digest + 4 * i, state[i]);
}
