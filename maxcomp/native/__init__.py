"""Native acceleration loader for maxcomp.

Tries to load the C shared library and Metal GPU support.
Sets NATIVE_AVAILABLE and METAL_AVAILABLE flags.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

NATIVE_AVAILABLE = False
METAL_AVAILABLE = False

_lib = None
_ffi = None
_metal = None

# cffi C declarations matching maxcomp_native.h
_CDEFS = """
uint32_t maxcomp_native_version(void);

int prng_search_all(
    const uint8_t *target, uint32_t target_bit_length,
    uint32_t max_seed,
    int *out_type, uint64_t *out_seed
);
void prng_generate(
    int prng_type, uint64_t seed, uint32_t bit_length,
    uint8_t *output
);
int64_t prng_mt19937_search(const uint8_t *target, uint32_t target_bit_length, uint32_t max_seed);
int64_t prng_xorshift128_search(const uint8_t *target, uint32_t target_bit_length, uint32_t max_seed);
int64_t prng_lcg_search(const uint8_t *target, uint32_t target_bit_length, uint32_t max_seed);

int hash_preimage_search(
    const uint8_t *target, uint32_t target_bit_length,
    uint32_t max_preimage_len,
    int *out_hash_type, uint8_t *out_preimage, uint32_t *out_preimage_len
);
void hash_compute(int hash_type, const uint8_t *preimage, uint32_t preimage_len, uint8_t *digest);

uint32_t ifs_match_blocks(
    const uint8_t *domain_pool, uint32_t num_domains,
    const uint8_t *range_blocks, uint32_t num_range_blocks,
    uint32_t block_size, uint32_t max_errors,
    uint32_t *out_domain_indices, uint8_t *out_inverted, uint32_t *out_errors
);

void automata_run(
    uint8_t rule, uint32_t width, uint32_t generations,
    const uint8_t *initial_state, uint8_t *output
);
int automata_search(
    const uint8_t *target, uint32_t target_bit_length,
    const uint32_t *widths, uint32_t num_widths,
    uint8_t *out_rule, uint32_t *out_width, uint32_t *out_generations,
    uint8_t *out_init_state, uint32_t *out_init_state_len
);

double autocorrelation_packed(const uint8_t *bits, uint32_t num_bits, uint32_t lag);
int has_self_similarity_packed(const uint8_t *bits, uint32_t num_bits, double threshold);

void bitstream_slice(
    const uint8_t *src, uint32_t src_bit_length,
    uint32_t start, uint32_t length, uint8_t *dst
);
void bitstream_concat(
    const uint8_t *a, uint32_t a_bit_length,
    const uint8_t *b, uint32_t b_bit_length, uint8_t *dst
);
uint32_t bitstream_hamming(const uint8_t *a, const uint8_t *b, uint32_t num_bits);
uint32_t bitstream_popcount(const uint8_t *data, uint32_t num_bits);
"""


def _find_native_lib() -> str | None:
    """Locate _maxcomp_native.dylib relative to this package."""
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    for name in ("_maxcomp_native.dylib", "_maxcomp_native.so"):
        path = os.path.join(pkg_dir, name)
        if os.path.isfile(path):
            return path
    return None


def _load_native() -> None:
    """Attempt to load the native C library via cffi."""
    global NATIVE_AVAILABLE, _lib, _ffi

    lib_path = _find_native_lib()
    if lib_path is None:
        logger.debug("Native library not found; using pure Python fallback")
        return

    try:
        import cffi

        ffi = cffi.FFI()
        ffi.cdef(_CDEFS)
        lib = ffi.dlopen(lib_path)
        # Verify it loaded correctly
        ver = lib.maxcomp_native_version()
        _lib = lib
        _ffi = ffi
        NATIVE_AVAILABLE = True
        logger.info("Native acceleration v%d loaded from %s", ver, lib_path)
    except Exception as e:
        logger.debug("Failed to load native library: %s", e)


def _load_metal() -> None:
    """Attempt to initialize Metal GPU support."""
    global METAL_AVAILABLE, _metal
    try:
        from maxcomp.native.metal_bridge import MetalBridge

        bridge = MetalBridge()
        if bridge.available:
            _metal = bridge
            METAL_AVAILABLE = True
            logger.info("Metal GPU acceleration available")
    except Exception as e:
        logger.debug("Metal not available: %s", e)


def get_lib():
    """Get the cffi library handle (or None)."""
    return _lib


def get_ffi():
    """Get the cffi FFI instance (or None)."""
    return _ffi


def get_metal():
    """Get the Metal bridge instance (or None)."""
    return _metal


# Load on import
_load_native()
_load_metal()
