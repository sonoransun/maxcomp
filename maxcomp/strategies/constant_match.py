"""Constant-match compression strategy.

Searches for input data as a substring within binary expansions of
mathematical constants and chaotic sequences. Supports configurable
search windows and tiered progressive search.
"""

from __future__ import annotations

import struct
import time
from typing import Optional

from maxcomp.bitstream import BitStream
from maxcomp.constants.catalog import (
    CLASSIC_CONSTANTS,
    CONSTANT_NAMES,
    ConstantID,
)
from maxcomp.constants.provider import get_binary_expansion
from maxcomp.strategies import register
from maxcomp.strategies.base import (
    CompressionResult,
    CompressionStrategy,
    MethodID,
)

# Payload: constant_id (1 byte) + offset (8 bytes, uint64 BE) + length (8 bytes, uint64 BE)
_PAYLOAD_SIZE = 17
_DEFAULT_PRECISION_BITS = 100_000

# Tiers for progressive search (in bits)
SEARCH_TIERS = [8_192, 65_536, 100_000, 524_288, 2_097_152, 8_388_608]


class ConstantMatchStrategy(CompressionStrategy):
    """Compress by matching data against binary expansions of constants/sequences."""

    def __init__(
        self,
        precision_bits: int = _DEFAULT_PRECISION_BITS,
        tiered: bool = False,
        include_chaotic: bool = False,
    ) -> None:
        self._precision_bits = precision_bits
        self._tiered = tiered
        self._include_chaotic = include_chaotic

    def name(self) -> str:
        return "constant_match"

    def method_ids(self) -> list[MethodID]:
        return [MethodID.CONSTANT_MATCH]

    def _get_search_ids(self) -> list[ConstantID]:
        """Return the list of ConstantIDs to search."""
        if self._include_chaotic:
            return list(ConstantID)
        return [cid for cid in ConstantID if cid in CLASSIC_CONSTANTS]

    def _build_result(
        self, constant_id: ConstantID, offset: int, bit_length: int
    ) -> CompressionResult:
        payload = struct.pack(">B", constant_id.value)
        payload += struct.pack(">Q", offset)
        payload += struct.pack(">Q", bit_length)
        return CompressionResult(
            method=MethodID.CONSTANT_MATCH,
            payload=payload,
            original_bit_length=bit_length,
            compressed_bit_length=_PAYLOAD_SIZE * 8,
            metadata={
                "strategy": "constant_match",
                "constant": CONSTANT_NAMES[constant_id],
                "offset": offset,
                "length": bit_length,
            },
        )

    def compress(
        self, data: BitStream, timeout_seconds: float = 30.0
    ) -> Optional[CompressionResult]:
        original_bytes = (len(data) + 7) // 8
        if _PAYLOAD_SIZE >= original_bytes:
            return None

        if self._tiered:
            return self._compress_tiered(data, timeout_seconds)
        return self._compress_fixed(data, timeout_seconds)

    def _compress_fixed(
        self, data: BitStream, timeout_seconds: float
    ) -> Optional[CompressionResult]:
        """Search all constants at the configured precision."""
        needle = data.to_bin_str()
        start_time = time.monotonic()
        search_ids = self._get_search_ids()

        for constant_id in search_ids:
            elapsed = time.monotonic() - start_time
            if elapsed >= timeout_seconds:
                break

            try:
                expansion = get_binary_expansion(constant_id, self._precision_bits)
            except Exception:
                continue
            offset = expansion.find(needle)
            if offset >= 0:
                return self._build_result(constant_id, offset, len(data))

        return None

    def _compress_tiered(
        self, data: BitStream, timeout_seconds: float
    ) -> Optional[CompressionResult]:
        """Progressive search: try small windows first, expand if time allows."""
        needle = data.to_bin_str()
        start_time = time.monotonic()
        search_ids = self._get_search_ids()

        # Filter tiers up to configured max
        tiers = [t for t in SEARCH_TIERS if t <= self._precision_bits]
        if self._precision_bits not in tiers:
            tiers.append(self._precision_bits)

        for tier_bits in tiers:
            elapsed = time.monotonic() - start_time
            remaining = timeout_seconds - elapsed
            if remaining <= 0:
                break

            for constant_id in search_ids:
                if (time.monotonic() - start_time) >= timeout_seconds:
                    return None
                try:
                    expansion = get_binary_expansion(constant_id, tier_bits)
                except Exception:
                    continue
                offset = expansion.find(needle)
                if offset >= 0:
                    return self._build_result(constant_id, offset, len(data))

        return None

    def decompress(
        self, method: MethodID, payload: bytes, original_bit_length: int
    ) -> BitStream:
        if len(payload) != _PAYLOAD_SIZE:
            raise ValueError(
                f"Expected {_PAYLOAD_SIZE}-byte payload, got {len(payload)}"
            )

        constant_id_val = struct.unpack(">B", payload[0:1])[0]
        offset = struct.unpack(">Q", payload[1:9])[0]
        length = struct.unpack(">Q", payload[9:17])[0]

        constant_id = ConstantID(constant_id_val)

        needed = offset + length
        expansion = get_binary_expansion(constant_id, needed)
        bits_str = expansion[offset : offset + length]

        return BitStream.from_bin_str(bits_str)


register(ConstantMatchStrategy())
