#include <metal_stdlib>
using namespace metal;

/* IFS block matching — compute Hamming distance for each (range, domain) pair.
 * 2D dispatch: x = domain_idx, y = range_idx.
 * Writes hamming distance to results[range_idx * num_domains + domain_idx]. */
kernel void ifs_block_match(
    device const uint8_t *domain_pool   [[buffer(0)]],
    device const uint8_t *range_blocks  [[buffer(1)]],
    device uint *results                [[buffer(2)]],
    constant uint &num_domains          [[buffer(3)]],
    constant uint &bytes_per_block      [[buffer(4)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint d_idx = tid.x;
    uint r_idx = tid.y;

    if (d_idx >= num_domains) return;

    uint hamming = 0;
    for (uint i = 0; i < bytes_per_block; i++) {
        uint8_t xored = domain_pool[d_idx * bytes_per_block + i]
                       ^ range_blocks[r_idx * bytes_per_block + i];
        hamming += popcount(uint(xored));
    }
    results[r_idx * num_domains + d_idx] = hamming;
}
