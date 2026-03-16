"""Compressed output format (.mxc) — serialize and deserialize."""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

from maxcomp.bitstream import BitStream
from maxcomp.strategies.base import CompressionResult, MethodID

MAGIC = b"MXC\x00"
FORMAT_VERSION = 1
HEADER_SIZE = 16
# Header: magic(4) + version(1) + flags(1) + method_id(1) + reserved(1) + original_bits(8)
HEADER_FMT = ">4sBBBBQ"

FLAG_CHUNKED = 0x01


@dataclass
class CompressedOutput:
    """The final compressed blob, ready for serialization."""

    original_bit_length: int
    method: MethodID
    payload: bytes
    chunk_results: list[CompressionResult] | None = None
    all_results: list[CompressionResult] = field(default_factory=list)

    @property
    def is_chunked(self) -> bool:
        return self.method == MethodID.CHUNKED_COMPOSITE

    @property
    def total_compressed_size(self) -> int:
        """Total compressed size in bytes, including header."""
        return HEADER_SIZE + len(self.payload)

    def serialize(self) -> bytes:
        """Serialize to the .mxc binary format."""
        flags = FLAG_CHUNKED if self.is_chunked else 0
        header = struct.pack(
            HEADER_FMT,
            MAGIC,
            FORMAT_VERSION,
            flags,
            int(self.method),
            0,  # reserved
            self.original_bit_length,
        )
        return header + self.payload

    @classmethod
    def deserialize(cls, data: bytes) -> CompressedOutput:
        """Deserialize from .mxc binary format."""
        if len(data) < HEADER_SIZE:
            raise ValueError(f"Data too short: need at least {HEADER_SIZE} bytes")

        magic, version, flags, method_id, _, original_bits = struct.unpack(
            HEADER_FMT, data[:HEADER_SIZE]
        )

        if magic != MAGIC:
            raise ValueError(f"Invalid magic: {magic!r}")
        if version != FORMAT_VERSION:
            raise ValueError(f"Unsupported format version: {version}")

        method = MethodID(method_id)
        payload = data[HEADER_SIZE:]

        chunk_results = None
        if flags & FLAG_CHUNKED:
            chunk_results = _deserialize_chunks(payload)

        return cls(
            original_bit_length=original_bits,
            method=method,
            payload=payload,
            chunk_results=chunk_results,
        )

    def decompress(self) -> BitStream:
        """Decompress back to original BitStream."""
        from maxcomp.strategies import get_decompressor

        if self.is_chunked and self.chunk_results is not None:
            parts: list[BitStream] = []
            for chunk in self.chunk_results:
                strategy = get_decompressor(chunk.method)
                part = strategy.decompress(
                    chunk.method, chunk.payload, chunk.original_bit_length
                )
                parts.append(part)
            # Concatenate all chunks
            result = BitStream(b"", 0)
            for part in parts:
                result = result.concat(part)
            return result
        else:
            strategy = get_decompressor(self.method)
            return strategy.decompress(
                self.method, self.payload, self.original_bit_length
            )


def serialize_chunks(chunk_results: list[CompressionResult]) -> bytes:
    """Serialize chunk results into the chunked payload format."""
    parts = [struct.pack(">I", len(chunk_results))]
    for chunk in chunk_results:
        parts.append(
            struct.pack(
                ">IBI",
                chunk.original_bit_length,
                int(chunk.method),
                len(chunk.payload),
            )
        )
        parts.append(chunk.payload)
    return b"".join(parts)


def _deserialize_chunks(payload: bytes) -> list[CompressionResult]:
    """Deserialize chunked payload into CompressionResult list."""
    offset = 0
    (num_chunks,) = struct.unpack(">I", payload[offset : offset + 4])
    offset += 4

    results = []
    for _ in range(num_chunks):
        bit_length, method_id, payload_length = struct.unpack(
            ">IBI", payload[offset : offset + 9]
        )
        offset += 9
        chunk_payload = payload[offset : offset + payload_length]
        offset += payload_length

        results.append(
            CompressionResult(
                method=MethodID(method_id),
                payload=chunk_payload,
                original_bit_length=bit_length,
                compressed_bit_length=9 * 8 + payload_length * 8,  # chunk header + payload
            )
        )
    return results


def total_size_with_overhead(result: CompressionResult) -> int:
    """Total size in bits including the .mxc header overhead."""
    return HEADER_SIZE * 8 + len(result.payload) * 8
