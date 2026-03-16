"""Compression engine — orchestrates strategies and selects the best result."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from maxcomp.bitstream import BitStream
from maxcomp.chunker import DEFAULT_CHUNK_SIZES, EXTENDED_CHUNK_SIZES, chunk_bitstream
from maxcomp.format import (
    HEADER_SIZE,
    CompressedOutput,
    serialize_chunks,
    total_size_with_overhead,
)
from maxcomp.strategies import _ensure_registered, get_all_strategies
from maxcomp.strategies.base import (
    CompressionResult,
    CompressionStrategy,
    MethodID,
)

logger = logging.getLogger(__name__)


class CompressionEngine:
    """Try all strategies (whole and chunked) and pick the best result."""

    def __init__(
        self,
        strategies: list[CompressionStrategy] | None = None,
        chunk_sizes: list[int] | None = None,
        max_workers: int | None = None,
        timeout_per_strategy: float = 30.0,
        enable_chunking: bool = True,
        precision_bits: int | None = None,
        tiered_search: bool = False,
        include_chaotic: bool = False,
    ) -> None:
        _ensure_registered()
        self.strategies = strategies or get_all_strategies()
        self.max_workers = max_workers
        self.timeout = timeout_per_strategy
        self.enable_chunking = enable_chunking

        # Configure constant_match strategy with precision/chaotic settings
        if precision_bits is not None or tiered_search or include_chaotic:
            from maxcomp.strategies.constant_match import ConstantMatchStrategy

            cm_kwargs: dict = {}
            if precision_bits is not None:
                cm_kwargs["precision_bits"] = precision_bits
            cm_kwargs["tiered"] = tiered_search
            cm_kwargs["include_chaotic"] = include_chaotic

            self.strategies = [
                ConstantMatchStrategy(**cm_kwargs)
                if isinstance(s, ConstantMatchStrategy)
                else s
                for s in self.strategies
            ]

        # Select chunk sizes based on window size
        if chunk_sizes is not None:
            self.chunk_sizes = chunk_sizes
        elif precision_bits is not None and precision_bits >= 524_288:
            self.chunk_sizes = EXTENDED_CHUNK_SIZES
        else:
            self.chunk_sizes = DEFAULT_CHUNK_SIZES

    def compress(self, data: BitStream, verbose: bool = False) -> CompressedOutput:
        """Try all strategies and return the best compressed output."""
        all_results: list[CompressionResult] = []

        # Run all strategies in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {}

            # Whole-stream strategies
            for strategy in self.strategies:
                fut = pool.submit(self._try_strategy, strategy, data)
                futures[fut] = f"whole:{strategy.name()}"

            # Chunked strategies
            if self.enable_chunking and len(data) > 64:
                for chunk_size in self.chunk_sizes:
                    if chunk_size >= len(data):
                        continue
                    fut = pool.submit(self._try_chunked, data, chunk_size)
                    futures[fut] = f"chunked:{chunk_size}"

            for fut in as_completed(futures):
                label = futures[fut]
                try:
                    result = fut.result()
                    if result is not None:
                        all_results.append(result)
                        if verbose:
                            logger.info(
                                "%s: ratio=%.4f (%d -> %d bits)",
                                label,
                                result.ratio,
                                result.original_bit_length,
                                result.compressed_bit_length,
                            )
                except Exception:
                    logger.debug("Strategy %s failed", label, exc_info=True)

        if not all_results:
            # Shouldn't happen — identity always succeeds
            raise RuntimeError("No compression strategy produced a result")

        # Pick best: smallest total size including format overhead
        best = min(all_results, key=lambda r: total_size_with_overhead(r))

        # Build CompressedOutput
        chunk_results = None
        if best.method == MethodID.CHUNKED_COMPOSITE:
            chunk_results = best.metadata.get("_chunk_results")

        return CompressedOutput(
            original_bit_length=len(data),
            method=best.method,
            payload=best.payload,
            chunk_results=chunk_results,
            all_results=all_results,
        )

    def _try_strategy(
        self, strategy: CompressionStrategy, data: BitStream
    ) -> Optional[CompressionResult]:
        """Run a single strategy on the whole BitStream."""
        try:
            return strategy.compress(data, timeout_seconds=self.timeout)
        except Exception:
            logger.debug("Strategy %s raised", strategy.name(), exc_info=True)
            return None

    def _try_chunked(
        self, data: BitStream, chunk_size: int
    ) -> Optional[CompressionResult]:
        """Try per-chunk compression at a given chunk size."""
        chunks = chunk_bitstream(data, chunk_size)
        chunk_results: list[CompressionResult] = []

        for chunk in chunks:
            best_chunk: Optional[CompressionResult] = None
            for strategy in self.strategies:
                try:
                    result = strategy.compress(chunk, timeout_seconds=self.timeout)
                    if result is not None:
                        if best_chunk is None or len(result.payload) < len(
                            best_chunk.payload
                        ):
                            best_chunk = result
                except Exception:
                    continue

            if best_chunk is None:
                return None  # Can't compress this chunk at all
            chunk_results.append(best_chunk)

        # Serialize chunked payload
        payload = serialize_chunks(chunk_results)

        # Calculate total compressed size
        # Header (16 bytes) + payload (includes per-chunk overhead)
        total_payload_bits = len(payload) * 8

        return CompressionResult(
            method=MethodID.CHUNKED_COMPOSITE,
            payload=payload,
            original_bit_length=len(data),
            compressed_bit_length=total_payload_bits,
            metadata={
                "strategy": f"chunked_{chunk_size}",
                "chunk_size": chunk_size,
                "num_chunks": len(chunks),
                "_chunk_results": chunk_results,
            },
        )
