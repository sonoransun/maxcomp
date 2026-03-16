"""Identity strategy — store raw bits (fallback)."""

from __future__ import annotations

from typing import Optional

from maxcomp.bitstream import BitStream
from maxcomp.strategies import register
from maxcomp.strategies.base import (
    CompressionResult,
    CompressionStrategy,
    MethodID,
)


class IdentityStrategy(CompressionStrategy):
    """Passthrough strategy that stores bits as-is. Always succeeds."""

    def name(self) -> str:
        return "identity"

    def method_ids(self) -> list[MethodID]:
        return [MethodID.IDENTITY]

    def compress(
        self, data: BitStream, timeout_seconds: float = 30.0
    ) -> Optional[CompressionResult]:
        payload = data.to_bytes()
        # Compressed size = payload size in bits (no extra overhead beyond format header)
        compressed_bits = len(payload) * 8
        return CompressionResult(
            method=MethodID.IDENTITY,
            payload=payload,
            original_bit_length=len(data),
            compressed_bit_length=compressed_bits,
            metadata={"strategy": "identity"},
        )

    def decompress(
        self, method: MethodID, payload: bytes, original_bit_length: int
    ) -> BitStream:
        return BitStream(payload, original_bit_length)


register(IdentityStrategy())
