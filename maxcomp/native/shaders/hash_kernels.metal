#include <metal_stdlib>
using namespace metal;

/* SHA-256 constants */
constant uint K[64] = {
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
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

uint rotr(uint x, uint n) { return (x >> n) | (x << (32 - n)); }
uint ch(uint x, uint y, uint z) { return (x & y) ^ (~x & z); }
uint maj(uint x, uint y, uint z) { return (x & y) ^ (x & z) ^ (y & z); }
uint sigma0(uint x) { return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22); }
uint sigma1(uint x) { return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25); }
uint gamma0(uint x) { return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3); }
uint gamma1(uint x) { return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10); }

/* Single-block SHA-256 for messages up to 55 bytes.
 * Writes 32-byte digest. */
void sha256_single_block(const thread uint8_t *msg, uint msg_len, thread uint8_t *digest) {
    /* Pad message into a single 64-byte block */
    uint W[64];
    for (uint i = 0; i < 16; i++) W[i] = 0;

    for (uint i = 0; i < msg_len; i++) {
        W[i / 4] |= uint(msg[i]) << (24 - (i % 4) * 8);
    }
    /* Append 0x80 */
    W[msg_len / 4] |= 0x80u << (24 - (msg_len % 4) * 8);
    /* Append length in bits at end of block */
    W[15] = msg_len * 8;

    /* Message schedule */
    for (uint i = 16; i < 64; i++) {
        W[i] = gamma1(W[i-2]) + W[i-7] + gamma0(W[i-15]) + W[i-16];
    }

    /* Compression */
    uint h0 = 0x6a09e667, h1 = 0xbb67ae85, h2 = 0x3c6ef372, h3 = 0xa54ff53a;
    uint h4 = 0x510e527f, h5 = 0x9b05688c, h6 = 0x1f83d9ab, h7 = 0x5be0cd19;
    uint a = h0, b = h1, c = h2, d = h3, e = h4, f = h5, g = h6, h = h7;

    for (uint i = 0; i < 64; i++) {
        uint t1 = h + sigma1(e) + ch(e, f, g) + K[i] + W[i];
        uint t2 = sigma0(a) + maj(a, b, c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    h0 += a; h1 += b; h2 += c; h3 += d;
    h4 += e; h5 += f; h6 += g; h7 += h;

    /* Write digest big-endian */
    uint hvals[8] = {h0, h1, h2, h3, h4, h5, h6, h7};
    for (uint i = 0; i < 8; i++) {
        digest[i*4+0] = uint8_t(hvals[i] >> 24);
        digest[i*4+1] = uint8_t(hvals[i] >> 16);
        digest[i*4+2] = uint8_t(hvals[i] >> 8);
        digest[i*4+3] = uint8_t(hvals[i]);
    }
}

/* SHA-256 preimage search kernel — one thread per preimage index */
kernel void sha256_preimage_search(
    device const uint8_t *target       [[buffer(0)]],
    device atomic_int *result_index    [[buffer(1)]],
    constant uint &target_bit_length   [[buffer(2)]],
    constant uint &preimage_len        [[buffer(3)]],
    constant uint &batch_offset        [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (atomic_load_explicit(result_index, memory_order_relaxed) >= 0)
        return;

    uint idx = tid + batch_offset;

    /* Construct preimage from index */
    uint8_t preimage[3];
    if (preimage_len == 1) {
        if (idx >= 256) return;
        preimage[0] = uint8_t(idx);
    } else if (preimage_len == 2) {
        if (idx >= 65536) return;
        preimage[0] = uint8_t(idx >> 8);
        preimage[1] = uint8_t(idx & 0xFF);
    } else {
        if (idx >= 16777216) return;
        preimage[0] = uint8_t(idx >> 16);
        preimage[1] = uint8_t((idx >> 8) & 0xFF);
        preimage[2] = uint8_t(idx & 0xFF);
    }

    /* Compute SHA-256 */
    uint8_t digest[32];
    sha256_single_block(preimage, preimage_len, digest);

    /* Compare first target_bit_length bits */
    uint full_bytes = target_bit_length / 8;
    bool match = true;

    for (uint i = 0; i < full_bytes && match; i++) {
        if (digest[i] != target[i]) match = false;
    }

    uint remaining_bits = target_bit_length % 8;
    if (match && remaining_bits > 0 && full_bytes < 32) {
        uint8_t mask = uint8_t(0xFF << (8 - remaining_bits));
        if ((digest[full_bytes] & mask) != (target[full_bytes] & mask))
            match = false;
    }

    if (match) {
        atomic_store_explicit(result_index, int(idx), memory_order_relaxed);
    }
}
