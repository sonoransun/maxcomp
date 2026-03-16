"""Chunker — split and reassemble BitStreams for per-chunk compression."""

from __future__ import annotations

from maxcomp.bitstream import BitStream


DEFAULT_CHUNK_SIZES = [64, 128, 256, 512, 1024, 2048, 4096, 8192]

EXTENDED_CHUNK_SIZES = [
    64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536,
]


def chunk_bitstream(
    data: BitStream, chunk_size: int
) -> list[BitStream]:
    """Split a BitStream into chunks of the given size (last may be smaller)."""
    return data.chunks(chunk_size)


def reassemble(chunks: list[BitStream]) -> BitStream:
    """Reassemble chunks into a single BitStream."""
    if not chunks:
        return BitStream(b"", 0)
    result = chunks[0]
    for chunk in chunks[1:]:
        result = result.concat(chunk)
    return result
