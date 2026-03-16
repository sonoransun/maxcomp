"""PRNG seed brute-force search for generative compression."""

from __future__ import annotations

import random
import time
from typing import Optional

try:
    from maxcomp.native import NATIVE_AVAILABLE, METAL_AVAILABLE, get_lib, get_ffi, get_metal
except ImportError:
    NATIVE_AVAILABLE = METAL_AVAILABLE = False


def _xorshift128(seed: int, num_bits: int) -> int:
    """Generate bits using xorshift128 algorithm.

    Uses 128-bit state derived from a single seed.
    """
    MASK64 = (1 << 64) - 1
    # Derive 128-bit state from seed (avoid zero state)
    s0 = (seed | 1) & MASK64
    s1 = ((seed * 6364136223846793005 + 1) | 1) & MASK64

    result = 0
    bits_generated = 0
    while bits_generated < num_bits:
        # xorshift128+ step
        x = s0
        y = s1
        s0 = y
        x ^= (x << 23) & MASK64
        s1 = (x ^ y ^ (x >> 17) ^ (y >> 26)) & MASK64
        out = (s1 + y) & MASK64

        # Take 64 bits (or fewer if near the end)
        bits_needed = min(64, num_bits - bits_generated)
        # Take the top bits_needed bits of out
        top_bits = out >> (64 - bits_needed)
        result = (result << bits_needed) | top_bits
        bits_generated += bits_needed

    return result


def _lcg(seed: int, num_bits: int) -> int:
    """Generate bits using glibc LCG parameters."""
    a = 1103515245
    c = 12345
    m = 1 << 31

    state = seed & (m - 1)
    result = 0
    bits_generated = 0
    while bits_generated < num_bits:
        state = (a * state + c) % m
        # Take 16 bits from the middle (bits 16..30) as glibc does
        bits_needed = min(16, num_bits - bits_generated)
        extracted = (state >> (31 - bits_needed)) & ((1 << bits_needed) - 1)
        result = (result << bits_needed) | extracted
        bits_generated += bits_needed

    return result


def _mt19937(seed: int, num_bits: int) -> int:
    """Generate bits using Python's Mersenne Twister."""
    rng = random.Random(seed)
    return rng.getrandbits(num_bits)


def _generate_bits(prng_type: int, seed: int, num_bits: int) -> int:
    """Generate bits from the given PRNG type and seed."""
    if prng_type == 0:
        return _mt19937(seed, num_bits)
    elif prng_type == 1:
        return _xorshift128(seed, num_bits)
    elif prng_type == 2:
        return _lcg(seed, num_bits)
    else:
        raise ValueError(f"Unknown PRNG type: {prng_type}")


def generate_bytes(prng_type: int, seed: int, bit_length: int) -> bytes:
    """Generate bytes from the given PRNG type, seed, and bit length.

    Used by the decompressor to reconstruct data.
    """
    if NATIVE_AVAILABLE and prng_type in (1, 2):
        try:
            ffi = get_ffi()
            lib = get_lib()
            num_bytes = (bit_length + 7) // 8
            out = ffi.new(f"uint8_t[{num_bytes}]")
            lib.prng_generate(prng_type, seed, bit_length, out)
            return bytes(ffi.buffer(out, num_bytes))
        except Exception:
            pass
    val = _generate_bits(prng_type, seed, bit_length)
    num_bytes = (bit_length + 7) // 8
    return val.to_bytes(num_bytes, "big")


def _search_prng_native(
    target_bytes: bytes, target_bit_length: int, max_seed: int
) -> Optional[tuple[int, int]]:
    """Search using native C acceleration (xorshift + LCG only, not MT19937)."""
    ffi = get_ffi()
    lib = get_lib()
    out_type = ffi.new("int *")
    out_seed = ffi.new("uint64_t *")
    result = lib.prng_search_all(target_bytes, target_bit_length, max_seed, out_type, out_seed)
    if result == 0:
        return (out_type[0], int(out_seed[0]))
    return None


def _search_prng_python(
    target_bytes: bytes,
    target_bit_length: int,
    max_seed: int,
    timeout: float,
) -> Optional[tuple[int, int]]:
    """Pure Python PRNG seed search (fallback)."""
    target_int = int.from_bytes(target_bytes, "big")
    padding = len(target_bytes) * 8 - target_bit_length
    if padding > 0:
        target_int >>= padding

    start_time = time.monotonic()

    for prng_type in (0, 1, 2):
        for seed in range(max_seed):
            if seed % 1000 == 0 and (time.monotonic() - start_time) > timeout:
                return None
            try:
                generated = _generate_bits(prng_type, seed, target_bit_length)
                if generated == target_int:
                    return (prng_type, seed)
            except Exception:
                continue
    return None


def search_prng(
    target_bytes: bytes,
    target_bit_length: int,
    max_seed: int = 65536,
    timeout: float = 30.0,
) -> Optional[tuple[int, int]]:
    """Search for a PRNG seed that generates the target bits.

    Dispatches to GPU, native C, or pure Python depending on availability.
    """
    if target_bit_length == 0:
        return None

    # GPU path (for large seed spaces)
    if METAL_AVAILABLE and max_seed >= 4096:
        try:
            metal = get_metal()
            for prng_type in (1, 2):  # xorshift and LCG only on GPU
                seed = metal.dispatch_prng_search(
                    target_bytes, target_bit_length, prng_type, max_seed
                )
                if seed >= 0:
                    return (prng_type, seed)
        except Exception:
            pass

    # Native C path
    if NATIVE_AVAILABLE:
        try:
            result = _search_prng_native(target_bytes, target_bit_length, max_seed)
            if result is not None:
                return result
        except Exception:
            pass

    # Pure Python fallback
    return _search_prng_python(target_bytes, target_bit_length, max_seed, timeout)
