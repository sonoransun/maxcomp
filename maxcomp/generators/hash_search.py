"""Truncated hash output matching for generative compression."""

from __future__ import annotations

import hashlib
import time
from typing import Optional

try:
    from maxcomp.native import NATIVE_AVAILABLE, get_lib, get_ffi
except ImportError:
    NATIVE_AVAILABLE = False


_HASH_FUNCTIONS = {
    0: ("md5", hashlib.md5),
    1: ("sha1", hashlib.sha1),
    2: ("sha256", hashlib.sha256),
}


def compute_hash(hash_type: int, preimage: bytes) -> bytes:
    """Compute hash digest for the given type and preimage.

    Used by the decompressor to reconstruct data.
    """
    if NATIVE_AVAILABLE:
        try:
            ffi = get_ffi()
            lib = get_lib()
            digest = ffi.new("uint8_t[32]")
            lib.hash_compute(hash_type, preimage, len(preimage), digest)
            sizes = {0: 16, 1: 20, 2: 32}
            return bytes(ffi.buffer(digest, sizes[hash_type]))
        except Exception:
            pass
    _, hash_fn = _HASH_FUNCTIONS[hash_type]
    return hash_fn(preimage).digest()


def _search_hash_native(
    target_bytes: bytes,
    target_bit_length: int,
    max_preimage_len: int,
) -> Optional[tuple[int, bytes]]:
    """Search using native C acceleration."""
    ffi = get_ffi()
    lib = get_lib()
    out_hash_type = ffi.new("int *")
    out_preimage = ffi.new(f"uint8_t[{max_preimage_len}]")
    out_preimage_len = ffi.new("uint32_t *")

    result = lib.hash_preimage_search(
        target_bytes, target_bit_length, max_preimage_len,
        out_hash_type, out_preimage, out_preimage_len,
    )
    if result == 0:
        plen = out_preimage_len[0]
        return (out_hash_type[0], bytes(ffi.buffer(out_preimage, plen)))
    return None


def _search_hash_python(
    target_bytes: bytes,
    target_bit_length: int,
    max_preimage_len: int,
    timeout: float,
) -> Optional[tuple[int, bytes]]:
    """Pure Python hash preimage search (fallback)."""
    target_byte_len = (target_bit_length + 7) // 8
    target_int = int.from_bytes(target_bytes[:target_byte_len], "big")
    padding = target_byte_len * 8 - target_bit_length
    if padding > 0:
        target_int >>= padding

    start_time = time.monotonic()

    for hash_type, (_, hash_fn) in _HASH_FUNCTIONS.items():
        test_digest = hash_fn(b"x").digest()
        hash_bit_length = len(test_digest) * 8
        if target_bit_length > hash_bit_length:
            continue

        for preimage_len in range(1, max_preimage_len + 1):
            total = 256 ** preimage_len
            for i in range(total):
                if i % 1000 == 0 and (time.monotonic() - start_time) > timeout:
                    return None
                preimage = i.to_bytes(preimage_len, "big")
                digest = hash_fn(preimage).digest()
                digest_int = int.from_bytes(digest[:target_byte_len], "big")
                if padding > 0:
                    digest_int >>= padding
                if digest_int == target_int:
                    return (hash_type, preimage)
    return None


def search_hash(
    target_bytes: bytes,
    target_bit_length: int,
    max_preimage_len: int = 3,
    timeout: float = 30.0,
) -> Optional[tuple[int, bytes]]:
    """Search for a hash preimage whose output prefix matches the target.

    Dispatches to native C or pure Python depending on availability.
    """
    if target_bit_length == 0:
        return None

    # Native C path
    if NATIVE_AVAILABLE:
        try:
            result = _search_hash_native(target_bytes, target_bit_length, max_preimage_len)
            if result is not None:
                return result
        except Exception:
            pass

    # Pure Python fallback
    return _search_hash_python(target_bytes, target_bit_length, max_preimage_len, timeout)
