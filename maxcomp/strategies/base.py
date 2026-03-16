"""Base classes for compression strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

from maxcomp.bitstream import BitStream


class MethodID(IntEnum):
    """Identifies the compression method in the binary format."""

    IDENTITY = 0
    CONSTANT_MATCH = 1
    FRACTAL_IFS = 2
    FRACTAL_LSYSTEM = 3
    GENERATIVE_PRNG = 4
    GENERATIVE_HASH = 5
    GENERATIVE_EXPR = 6
    GENERATIVE_AUTOMATA = 7
    CHUNKED_COMPOSITE = 255


@dataclass(frozen=True)
class CompressionResult:
    """The output of a single strategy attempt on a BitStream."""

    method: MethodID
    payload: bytes
    original_bit_length: int
    compressed_bit_length: int
    metadata: dict = field(default_factory=dict)

    @property
    def ratio(self) -> float:
        """Compression ratio. < 1.0 means compression was achieved."""
        if self.original_bit_length == 0:
            return float("inf")
        return self.compressed_bit_length / self.original_bit_length

    @property
    def saved_bits(self) -> int:
        return self.original_bit_length - self.compressed_bit_length


class CompressionStrategy(ABC):
    """Abstract base class for all compression strategies."""

    @abstractmethod
    def name(self) -> str:
        """Human-readable strategy name."""
        ...

    @abstractmethod
    def method_ids(self) -> list[MethodID]:
        """Method IDs this strategy handles."""
        ...

    @abstractmethod
    def compress(
        self, data: BitStream, timeout_seconds: float = 30.0
    ) -> Optional[CompressionResult]:
        """Attempt to compress. Returns None if no compression achieved."""
        ...

    @abstractmethod
    def decompress(
        self, method: MethodID, payload: bytes, original_bit_length: int
    ) -> BitStream:
        """Reconstruct original bits from payload."""
        ...
