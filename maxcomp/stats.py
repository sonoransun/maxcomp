"""Statistics collection and reporting."""

from __future__ import annotations

import json
from dataclasses import dataclass

from maxcomp.format import HEADER_SIZE, CompressedOutput, total_size_with_overhead
from maxcomp.strategies.base import CompressionResult


@dataclass
class StrategyStats:
    """Stats for a single strategy result."""

    name: str
    method: str
    original_bits: int
    compressed_bits: int
    total_with_overhead_bits: int
    ratio: float
    saved_bits: int

    @classmethod
    def from_result(cls, result: CompressionResult) -> StrategyStats:
        total = total_size_with_overhead(result)
        return cls(
            name=result.metadata.get("strategy", result.method.name),
            method=result.method.name,
            original_bits=result.original_bit_length,
            compressed_bits=result.compressed_bit_length,
            total_with_overhead_bits=total,
            ratio=total / result.original_bit_length if result.original_bit_length > 0 else float("inf"),
            saved_bits=result.original_bit_length - total,
        )


def format_stats(output: CompressedOutput, as_json: bool = False) -> str:
    """Format compression statistics for display."""
    results = output.all_results
    stats = [StrategyStats.from_result(r) for r in results]
    stats.sort(key=lambda s: s.total_with_overhead_bits)

    if as_json:
        return json.dumps(
            {
                "original_bits": output.original_bit_length,
                "original_bytes": (output.original_bit_length + 7) // 8,
                "best_method": output.method.name,
                "best_total_bytes": output.total_compressed_size,
                "best_ratio": output.total_compressed_size * 8 / output.original_bit_length
                if output.original_bit_length > 0
                else None,
                "strategies": [
                    {
                        "name": s.name,
                        "method": s.method,
                        "ratio": round(s.ratio, 6),
                        "total_bits": s.total_with_overhead_bits,
                        "saved_bits": s.saved_bits,
                    }
                    for s in stats
                ],
            },
            indent=2,
        )

    lines = [
        f"Original: {output.original_bit_length} bits ({(output.original_bit_length + 7) // 8} bytes)",
        f"Best method: {output.method.name}",
        f"Compressed: {output.total_compressed_size} bytes (ratio: {output.total_compressed_size * 8 / output.original_bit_length:.4f})"
        if output.original_bit_length > 0
        else "Compressed: 0 bytes",
        "",
        "Strategy results (sorted by size):",
        f"{'Strategy':<30} {'Total bits':>12} {'Ratio':>10} {'Saved bits':>12}",
        "-" * 66,
    ]
    for s in stats:
        lines.append(
            f"{s.name:<30} {s.total_with_overhead_bits:>12} {s.ratio:>10.4f} {s.saved_bits:>12}"
        )
    return "\n".join(lines)


def format_info(output: CompressedOutput) -> str:
    """Format .mxc file metadata for display."""
    lines = [
        f"Format: MXC v1",
        f"Method: {output.method.name} ({int(output.method)})",
        f"Original size: {output.original_bit_length} bits ({(output.original_bit_length + 7) // 8} bytes)",
        f"Compressed size: {output.total_compressed_size} bytes",
        f"Chunked: {output.is_chunked}",
    ]
    if output.is_chunked and output.chunk_results:
        lines.append(f"Chunks: {len(output.chunk_results)}")
        for i, chunk in enumerate(output.chunk_results):
            lines.append(
                f"  Chunk {i}: {chunk.method.name}, {chunk.original_bit_length} bits, "
                f"payload {len(chunk.payload)} bytes"
            )
    return "\n".join(lines)
