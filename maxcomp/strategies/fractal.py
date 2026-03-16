"""Fractal / self-similarity compression strategy.

Combines IFS-based and L-system-based approaches.
"""

from __future__ import annotations

import struct
from typing import Optional

from maxcomp.bitstream import BitStream
from maxcomp.fractal.analysis import has_self_similarity
from maxcomp.fractal.ifs import ifs_compress, ifs_decompress
from maxcomp.fractal.lsystem import (
    LSystemRule,
    generate_lsystem,
    search_lsystem,
)
from maxcomp.strategies import register
from maxcomp.strategies.base import (
    CompressionResult,
    CompressionStrategy,
    MethodID,
)


def _bits_to_packed_bytes(bits: list[int]) -> bytes:
    """Pack a list of 0/1 ints into bytes (MSB first, zero-padded)."""
    if not bits:
        return b""
    n = len(bits)
    padded = bits + [0] * ((8 - n % 8) % 8)
    out = bytearray()
    for i in range(0, len(padded), 8):
        byte = 0
        for j in range(8):
            byte = (byte << 1) | padded[i + j]
        out.append(byte)
    return bytes(out)


def _packed_bytes_to_bits(data: bytes, num_bits: int) -> list[int]:
    """Unpack bytes into a list of 0/1 ints, returning *num_bits* bits."""
    bits: list[int] = []
    for byte in data:
        for j in range(7, -1, -1):
            bits.append((byte >> j) & 1)
    return bits[:num_bits]


# ---------------------------------------------------------------------------
# IFS payload helpers
# ---------------------------------------------------------------------------

def _encode_ifs_payload(
    transforms: list[tuple[int, bool]],
    seed_bits: list[int],
    block_size: int,
) -> bytes:
    """Encode IFS parameters into a binary payload.

    Layout:
        block_size     : 1 byte
        num_transforms : 2 bytes (uint16 BE)
        seed_length    : 2 bytes (uint16 BE)  — number of bits
        seed_bits      : packed bytes
        transforms     : for each — domain_index (uint16 BE) + flags (1 byte)
    """
    buf = bytearray()
    buf.append(block_size & 0xFF)
    buf += struct.pack("!H", len(transforms))
    buf += struct.pack("!H", len(seed_bits))
    buf += _bits_to_packed_bytes(seed_bits)
    for domain_idx, inverted in transforms:
        buf += struct.pack("!H", domain_idx)
        flags = 1 if inverted else 0
        buf.append(flags)
    return bytes(buf)


def _decode_ifs_payload(
    payload: bytes,
) -> tuple[list[tuple[int, bool]], list[int], int]:
    """Decode an IFS payload back into (transforms, seed_bits, block_size)."""
    off = 0
    block_size = payload[off]
    off += 1
    num_transforms = struct.unpack_from("!H", payload, off)[0]
    off += 2
    seed_length = struct.unpack_from("!H", payload, off)[0]
    off += 2
    seed_byte_count = (seed_length + 7) // 8
    seed_bytes = payload[off: off + seed_byte_count]
    off += seed_byte_count
    seed_bits = _packed_bytes_to_bits(seed_bytes, seed_length)

    transforms: list[tuple[int, bool]] = []
    for _ in range(num_transforms):
        domain_idx = struct.unpack_from("!H", payload, off)[0]
        off += 2
        flags = payload[off]
        off += 1
        transforms.append((domain_idx, bool(flags & 1)))

    return transforms, seed_bits, block_size


# ---------------------------------------------------------------------------
# L-system payload helpers
# ---------------------------------------------------------------------------

def _encode_lsystem_payload(
    axiom: str,
    rules: list[LSystemRule],
    iterations: int,
    bit_mapping: dict[str, str],
) -> bytes:
    """Encode L-system parameters into a binary payload.

    Layout:
        num_rules   : 1 byte
        for each rule:
            pred_len : 1 byte
            pred     : N bytes (UTF-8)
            succ_len : 1 byte
            succ     : N bytes (UTF-8)
        axiom_len   : 1 byte
        axiom       : N bytes (UTF-8)
        iterations  : 4 bytes (uint32 BE)
        num_mappings: 1 byte
        for each mapping:
            char     : 1 byte (UTF-8)
            bits_len : 1 byte
            bits     : N bytes (UTF-8 of '0'/'1' string)
    """
    buf = bytearray()
    buf.append(len(rules) & 0xFF)
    for rule in rules:
        pred_bytes = rule.predecessor.encode("utf-8")
        succ_bytes = rule.successor.encode("utf-8")
        buf.append(len(pred_bytes) & 0xFF)
        buf += pred_bytes
        buf.append(len(succ_bytes) & 0xFF)
        buf += succ_bytes
    axiom_bytes = axiom.encode("utf-8")
    buf.append(len(axiom_bytes) & 0xFF)
    buf += axiom_bytes
    buf += struct.pack("!I", iterations)
    buf.append(len(bit_mapping) & 0xFF)
    for char, bits_str in sorted(bit_mapping.items()):
        char_byte = char.encode("utf-8")
        bits_bytes = bits_str.encode("utf-8")
        buf.append(char_byte[0])
        buf.append(len(bits_bytes) & 0xFF)
        buf += bits_bytes
    return bytes(buf)


def _decode_lsystem_payload(
    payload: bytes,
) -> tuple[str, list[LSystemRule], int, dict[str, str]]:
    """Decode an L-system payload."""
    off = 0
    num_rules = payload[off]
    off += 1
    rules: list[LSystemRule] = []
    for _ in range(num_rules):
        pred_len = payload[off]
        off += 1
        pred = payload[off: off + pred_len].decode("utf-8")
        off += pred_len
        succ_len = payload[off]
        off += 1
        succ = payload[off: off + succ_len].decode("utf-8")
        off += succ_len
        rules.append(LSystemRule(pred, succ))

    axiom_len = payload[off]
    off += 1
    axiom = payload[off: off + axiom_len].decode("utf-8")
    off += axiom_len

    iterations = struct.unpack_from("!I", payload, off)[0]
    off += 4

    num_mappings = payload[off]
    off += 1
    bit_mapping: dict[str, str] = {}
    for _ in range(num_mappings):
        char = chr(payload[off])
        off += 1
        bits_len = payload[off]
        off += 1
        bits_str = payload[off: off + bits_len].decode("utf-8")
        off += bits_len
        bit_mapping[char] = bits_str

    return axiom, rules, iterations, bit_mapping


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

class FractalStrategy(CompressionStrategy):
    """Fractal / self-similarity compression using IFS and L-systems."""

    def name(self) -> str:
        return "fractal"

    def method_ids(self) -> list[MethodID]:
        return [MethodID.FRACTAL_IFS, MethodID.FRACTAL_LSYSTEM]

    def compress(
        self, data: BitStream, timeout_seconds: float = 30.0
    ) -> Optional[CompressionResult]:
        bits = list(data)
        n = len(bits)
        if n == 0:
            return None

        # Quick pre-check: does the data look self-similar?
        if not has_self_similarity(bits, threshold=0.7):
            return None

        best: Optional[CompressionResult] = None

        # --- IFS attempt ---
        ifs_result = ifs_compress(bits, block_size=8)
        if ifs_result is not None:
            transforms, seed_bits, block_size = ifs_result
            payload = _encode_ifs_payload(transforms, seed_bits, block_size)
            compressed_bits = len(payload) * 8
            if compressed_bits < n:
                result = CompressionResult(
                    method=MethodID.FRACTAL_IFS,
                    payload=payload,
                    original_bit_length=n,
                    compressed_bit_length=compressed_bits,
                    metadata={"strategy": "fractal_ifs", "block_size": block_size,
                              "num_transforms": len(transforms)},
                )
                if best is None or result.ratio < best.ratio:
                    best = result

        # --- L-system attempt ---
        if n < 256:
            ls_timeout = min(timeout_seconds / 2, 10.0)
            ls_result = search_lsystem(bits, max_rules=2, max_iterations=8, timeout=ls_timeout)
            if ls_result is not None:
                axiom, rules, iterations, bit_mapping = ls_result
                payload = _encode_lsystem_payload(axiom, rules, iterations, bit_mapping)
                compressed_bits = len(payload) * 8
                if compressed_bits < n:
                    result = CompressionResult(
                        method=MethodID.FRACTAL_LSYSTEM,
                        payload=payload,
                        original_bit_length=n,
                        compressed_bit_length=compressed_bits,
                        metadata={"strategy": "fractal_lsystem", "axiom": axiom,
                                  "iterations": iterations},
                    )
                    if best is None or result.ratio < best.ratio:
                        best = result

        return best

    def decompress(
        self, method: MethodID, payload: bytes, original_bit_length: int
    ) -> BitStream:
        if method == MethodID.FRACTAL_IFS:
            transforms, seed_bits, block_size = _decode_ifs_payload(payload)
            bits = ifs_decompress(
                transforms, seed_bits, block_size, original_bit_length, iterations=20,
            )
            bin_str = "".join(str(b) for b in bits[:original_bit_length])
            return BitStream.from_bin_str(bin_str)

        if method == MethodID.FRACTAL_LSYSTEM:
            axiom, rules, iterations, bit_mapping = _decode_lsystem_payload(payload)
            expanded = generate_lsystem(axiom, rules, iterations)
            bits: list[int] = []
            for ch in expanded:
                if ch in bit_mapping:
                    bits.extend(int(c) for c in bit_mapping[ch])
            bin_str = "".join(str(b) for b in bits[:original_bit_length])
            return BitStream.from_bin_str(bin_str)

        raise ValueError(f"FractalStrategy cannot decompress method {method!r}")


register(FractalStrategy())
