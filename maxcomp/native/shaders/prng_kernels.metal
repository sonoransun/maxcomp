#include <metal_stdlib>
using namespace metal;

/* xorshift128+ seed search — one thread per seed */
kernel void xorshift128_search(
    device const uint8_t *target      [[buffer(0)]],
    device atomic_int *result_seed    [[buffer(1)]],
    constant uint &target_bit_length  [[buffer(2)]],
    constant uint &seed_offset        [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    /* Early exit if another thread already found a match */
    if (atomic_load_explicit(result_seed, memory_order_relaxed) >= 0)
        return;

    uint seed = tid + seed_offset;
    uint target_bytes = (target_bit_length + 7) / 8;

    /* Initialize xorshift128+ state from seed */
    ulong s0 = (ulong(seed) | 1uL);
    ulong s1 = ((ulong(seed) * 6364136223846793005uL + 1uL) | 1uL);

    /* Generate and compare */
    uint bits_generated = 0;
    uint byte_idx = 0;
    bool match = true;

    while (bits_generated < target_bit_length && match) {
        /* xorshift128+ step */
        ulong x = s0;
        ulong y = s1;
        s0 = y;
        x ^= (x << 23);
        s1 = (x ^ y ^ (x >> 17) ^ (y >> 26));
        ulong out = s1 + y;

        /* Compare up to 64 bits from this output */
        uint bits_remaining = target_bit_length - bits_generated;
        uint bits_to_check = min(64u, bits_remaining);

        for (uint b = 0; b < bits_to_check && match; b += 8) {
            uint bits_in_byte = min(8u, bits_to_check - b);
            uint8_t generated_byte = uint8_t((out >> (56 - b)) & 0xFF);

            if (byte_idx < target_bytes) {
                if (bits_in_byte == 8) {
                    if (generated_byte != target[byte_idx]) {
                        match = false;
                    }
                } else {
                    /* Partial last byte — mask */
                    uint8_t mask = uint8_t(0xFF << (8 - bits_in_byte));
                    if ((generated_byte & mask) != (target[byte_idx] & mask)) {
                        match = false;
                    }
                }
                byte_idx++;
            }
        }
        bits_generated += bits_to_check;
    }

    if (match) {
        atomic_store_explicit(result_seed, int(seed), memory_order_relaxed);
    }
}

/* LCG seed search — one thread per seed */
kernel void lcg_search(
    device const uint8_t *target      [[buffer(0)]],
    device atomic_int *result_seed    [[buffer(1)]],
    constant uint &target_bit_length  [[buffer(2)]],
    constant uint &seed_offset        [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (atomic_load_explicit(result_seed, memory_order_relaxed) >= 0)
        return;

    uint seed = tid + seed_offset;
    uint state = seed & 0x7FFFFFFF;

    uint bits_generated = 0;
    uint byte_idx = 0;
    bool match = true;
    uint accum = 0;
    uint accum_bits = 0;

    while (bits_generated < target_bit_length && match) {
        /* LCG step */
        state = (1103515245u * state + 12345u) & 0x7FFFFFFFu;

        uint bits_remaining = target_bit_length - bits_generated;
        uint bits_needed = min(16u, bits_remaining);
        uint extracted = (state >> (31 - bits_needed)) & ((1u << bits_needed) - 1u);

        /* Accumulate bits and compare byte-by-byte */
        accum = (accum << bits_needed) | extracted;
        accum_bits += bits_needed;
        bits_generated += bits_needed;

        while (accum_bits >= 8 && match) {
            accum_bits -= 8;
            uint8_t gen_byte = uint8_t((accum >> accum_bits) & 0xFF);
            if (byte_idx < (target_bit_length + 7) / 8) {
                if (bits_generated >= target_bit_length && accum_bits == 0) {
                    /* Last byte might be partial */
                    uint tail_bits = target_bit_length % 8;
                    if (tail_bits == 0) tail_bits = 8;
                    uint8_t mask = uint8_t(0xFF << (8 - tail_bits));
                    if ((gen_byte & mask) != (target[byte_idx] & mask))
                        match = false;
                } else {
                    if (gen_byte != target[byte_idx])
                        match = false;
                }
                byte_idx++;
            }
        }
    }

    /* Handle remaining accumulated bits */
    if (match && accum_bits > 0 && byte_idx < (target_bit_length + 7) / 8) {
        uint8_t gen_byte = uint8_t((accum << (8 - accum_bits)) & 0xFF);
        uint8_t mask = uint8_t(0xFF << (8 - accum_bits));
        if ((gen_byte & mask) != (target[byte_idx] & mask))
            match = false;
    }

    if (match) {
        atomic_store_explicit(result_seed, int(seed), memory_order_relaxed);
    }
}
