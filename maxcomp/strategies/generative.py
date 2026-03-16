"""Generative compression strategy — uses PRNG, hash, expression, and automata generators."""

from __future__ import annotations

import struct
from typing import Optional

from maxcomp.bitstream import BitStream
from maxcomp.generators.automata import run_automaton_bytes, search_automata
from maxcomp.generators.expressions import evaluate_expression, search_expression
from maxcomp.generators.hash_search import compute_hash, search_hash
from maxcomp.generators.prng import generate_bytes, search_prng
from maxcomp.strategies import register
from maxcomp.strategies.base import (
    CompressionResult,
    CompressionStrategy,
    MethodID,
)


class GenerativeStrategy(CompressionStrategy):
    """Compression strategy that tries to find a short generative program
    (PRNG seed, hash preimage, math expression, or cellular automaton)
    that reproduces the input data."""

    def name(self) -> str:
        return "generative"

    def method_ids(self) -> list[MethodID]:
        return [
            MethodID.GENERATIVE_PRNG,
            MethodID.GENERATIVE_HASH,
            MethodID.GENERATIVE_EXPR,
            MethodID.GENERATIVE_AUTOMATA,
        ]

    def compress(
        self, data: BitStream, timeout_seconds: float = 30.0
    ) -> Optional[CompressionResult]:
        """Try all four generators and return the best compression result."""
        target_bytes = data.to_bytes()
        target_bit_length = len(data)

        if target_bit_length == 0:
            return None

        # Allocate time budget across generators
        per_generator_timeout = timeout_seconds / 4.0

        results: list[CompressionResult] = []

        # --- PRNG ---
        prng_result = search_prng(
            target_bytes, target_bit_length, timeout=per_generator_timeout
        )
        if prng_result is not None:
            prng_type, seed = prng_result
            # Payload: prng_type(1) + seed(8 BE) + bit_length(8 BE) = 17 bytes
            payload = struct.pack(">B Q Q", prng_type, seed, target_bit_length)
            results.append(
                CompressionResult(
                    method=MethodID.GENERATIVE_PRNG,
                    payload=payload,
                    original_bit_length=target_bit_length,
                    compressed_bit_length=len(payload) * 8,
                    metadata={"strategy": "generative_prng", "prng_type": prng_type, "seed": seed},
                )
            )

        # --- HASH ---
        hash_result = search_hash(
            target_bytes, target_bit_length, timeout=per_generator_timeout
        )
        if hash_result is not None:
            hash_type, preimage = hash_result
            # Payload: hash_type(1) + preimage_length(1) + preimage(N) + truncate_bits(4 BE)
            payload = struct.pack(
                ">B B", hash_type, len(preimage)
            ) + preimage + struct.pack(">I", target_bit_length)
            results.append(
                CompressionResult(
                    method=MethodID.GENERATIVE_HASH,
                    payload=payload,
                    original_bit_length=target_bit_length,
                    compressed_bit_length=len(payload) * 8,
                    metadata={"strategy": "generative_hash", "hash_type": hash_type, "preimage": preimage},
                )
            )

        # --- EXPRESSION ---
        target_int = data.to_int()
        expr_result = search_expression(
            target_int, target_bit_length, timeout=per_generator_timeout
        )
        if expr_result is not None:
            expr_str, bit_offset, bit_length = expr_result
            expr_bytes = expr_str.encode("utf-8")
            # Payload: expr_length(2 BE) + expression(N UTF-8) + bit_offset(4 BE) + bit_length(4 BE)
            payload = struct.pack(">H", len(expr_bytes)) + expr_bytes + struct.pack(">I I", bit_offset, bit_length)
            results.append(
                CompressionResult(
                    method=MethodID.GENERATIVE_EXPR,
                    payload=payload,
                    original_bit_length=target_bit_length,
                    compressed_bit_length=len(payload) * 8,
                    metadata={"strategy": "generative_expr", "expression": expr_str, "bit_offset": bit_offset},
                )
            )

        # --- AUTOMATA ---
        automata_result = search_automata(
            target_bytes, target_bit_length, timeout=per_generator_timeout
        )
        if automata_result is not None:
            rule, width, generations, init_state, read_order = automata_result
            # Payload: rule(1) + width(2 BE) + generations(4 BE) + init_state_length(2 BE) + init_state(N) + read_order(1)
            payload = struct.pack(
                ">B H I H", rule, width, generations, len(init_state)
            ) + init_state + struct.pack(">B", read_order)
            results.append(
                CompressionResult(
                    method=MethodID.GENERATIVE_AUTOMATA,
                    payload=payload,
                    original_bit_length=target_bit_length,
                    compressed_bit_length=len(payload) * 8,
                    metadata={
                        "strategy": "generative_automata",
                        "rule": rule,
                        "width": width,
                        "generations": generations,
                    },
                )
            )

        if not results:
            return None

        # Return the best result (smallest compressed size that actually compresses)
        best = min(results, key=lambda r: r.compressed_bit_length)
        if best.compressed_bit_length >= best.original_bit_length:
            return None
        return best

    def decompress(
        self, method: MethodID, payload: bytes, original_bit_length: int
    ) -> BitStream:
        """Reconstruct original bits from payload by re-running the generator."""
        if method == MethodID.GENERATIVE_PRNG:
            return self._decompress_prng(payload, original_bit_length)
        elif method == MethodID.GENERATIVE_HASH:
            return self._decompress_hash(payload, original_bit_length)
        elif method == MethodID.GENERATIVE_EXPR:
            return self._decompress_expr(payload, original_bit_length)
        elif method == MethodID.GENERATIVE_AUTOMATA:
            return self._decompress_automata(payload, original_bit_length)
        else:
            raise ValueError(f"GenerativeStrategy cannot decompress method {method!r}")

    def _decompress_prng(self, payload: bytes, original_bit_length: int) -> BitStream:
        """Decompress PRNG payload."""
        prng_type, seed, bit_length = struct.unpack(">B Q Q", payload)
        data = generate_bytes(prng_type, seed, bit_length)
        return BitStream(data, bit_length)

    def _decompress_hash(self, payload: bytes, original_bit_length: int) -> BitStream:
        """Decompress hash payload."""
        hash_type = payload[0]
        preimage_length = payload[1]
        preimage = payload[2 : 2 + preimage_length]
        truncate_bits = struct.unpack(">I", payload[2 + preimage_length : 6 + preimage_length])[0]

        digest = compute_hash(hash_type, preimage)
        return BitStream(digest, truncate_bits)

    def _decompress_expr(self, payload: bytes, original_bit_length: int) -> BitStream:
        """Decompress expression payload."""
        expr_length = struct.unpack(">H", payload[:2])[0]
        expr_str = payload[2 : 2 + expr_length].decode("utf-8")
        bit_offset, bit_length = struct.unpack(
            ">I I", payload[2 + expr_length : 10 + expr_length]
        )

        value = evaluate_expression(expr_str)
        value_binstr = bin(value)[2:]

        # Extract the substring at bit_offset with bit_length
        extracted = value_binstr[bit_offset : bit_offset + bit_length]
        extracted_int = int(extracted, 2)

        return BitStream.from_int(extracted_int, bit_length)

    def _decompress_automata(self, payload: bytes, original_bit_length: int) -> BitStream:
        """Decompress automata payload."""
        rule, width, generations, init_state_length = struct.unpack(
            ">B H I H", payload[:9]
        )
        init_state = payload[9 : 9 + init_state_length]
        read_order = payload[9 + init_state_length]

        data = run_automaton_bytes(
            rule, width, generations, init_state, read_order, original_bit_length
        )
        return BitStream(data, original_bit_length)


register(GenerativeStrategy())
