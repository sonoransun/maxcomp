"""BitStream — immutable sequence of bits with efficient operations."""

from __future__ import annotations

import sys
from typing import Iterator, overload


class BitStream:
    """Immutable sequence of bits backed by bytes.

    Tracks exact bit length — the last byte may be partial.
    """

    __slots__ = ("_data", "_bit_length", "_hash")

    def __init__(self, data: bytes, bit_length: int | None = None) -> None:
        if bit_length is None:
            bit_length = len(data) * 8
        if bit_length < 0:
            raise ValueError("bit_length must be non-negative")
        needed_bytes = (bit_length + 7) // 8
        if len(data) < needed_bytes:
            raise ValueError(
                f"Need at least {needed_bytes} bytes for {bit_length} bits, got {len(data)}"
            )
        # Mask off unused trailing bits in last byte
        if bit_length % 8 != 0 and needed_bytes > 0:
            mask = 0xFF << (8 - bit_length % 8) & 0xFF
            data = data[: needed_bytes - 1] + bytes([data[needed_bytes - 1] & mask])
        else:
            data = data[:needed_bytes]
        object.__setattr__(self, "_data", bytes(data))
        object.__setattr__(self, "_bit_length", bit_length)
        object.__setattr__(self, "_hash", None)

    # -- Constructors --

    @classmethod
    def from_bytes(cls, data: bytes) -> BitStream:
        """Create from raw bytes (all bits used)."""
        return cls(data)

    @classmethod
    def from_file(cls, path: str) -> BitStream:
        """Create from file contents."""
        with open(path, "rb") as f:
            return cls(f.read())

    @classmethod
    def from_hex(cls, hex_str: str) -> BitStream:
        """Create from hex string (e.g. 'deadbeef')."""
        hex_str = hex_str.strip()
        if hex_str.startswith(("0x", "0X")):
            hex_str = hex_str[2:]
        return cls(bytes.fromhex(hex_str))

    @classmethod
    def from_bin_str(cls, bin_str: str) -> BitStream:
        """Create from binary string (e.g. '11010010')."""
        bin_str = bin_str.strip()
        if bin_str.startswith(("0b", "0B")):
            bin_str = bin_str[2:]
        bit_length = len(bin_str)
        if bit_length == 0:
            return cls(b"", 0)
        # Pack into bytes
        padded = bin_str + "0" * ((8 - bit_length % 8) % 8)
        data = bytes(int(padded[i : i + 8], 2) for i in range(0, len(padded), 8))
        return cls(data, bit_length)

    @classmethod
    def from_int(cls, value: int, bit_length: int) -> BitStream:
        """Create from integer with explicit bit length."""
        if value < 0:
            raise ValueError("value must be non-negative")
        if bit_length == 0:
            return cls(b"", 0)
        needed_bytes = (bit_length + 7) // 8
        # Left-shift value so bits occupy MSB positions in the packed bytes
        padding = needed_bytes * 8 - bit_length
        shifted = value << padding
        data = shifted.to_bytes(needed_bytes, "big")
        return cls(data, bit_length)

    @classmethod
    def from_stdin(cls) -> BitStream:
        """Create from stdin (binary mode)."""
        return cls(sys.stdin.buffer.read())

    # -- Accessors --

    @property
    def data(self) -> bytes:
        """Underlying packed bytes."""
        return self._data

    def bit_at(self, index: int) -> int:
        """Get bit value (0 or 1) at the given index."""
        if index < 0:
            index += self._bit_length
        if not 0 <= index < self._bit_length:
            raise IndexError(f"Bit index {index} out of range for length {self._bit_length}")
        byte_idx = index // 8
        bit_idx = 7 - (index % 8)
        return (self._data[byte_idx] >> bit_idx) & 1

    def to_bytes(self) -> bytes:
        """Return packed bytes (last byte zero-padded if needed)."""
        return self._data

    def to_int(self) -> int:
        """Convert to integer."""
        if not self._data:
            return 0
        val = int.from_bytes(self._data, "big")
        # Shift right to remove padding bits
        padding = (8 - self._bit_length % 8) % 8
        return val >> padding if padding else val

    def to_bin_str(self) -> str:
        """Convert to binary string representation."""
        if self._bit_length == 0:
            return ""
        full_bin = bin(int.from_bytes(self._data, "big"))[2:].zfill(len(self._data) * 8)
        return full_bin[: self._bit_length]

    def slice(self, start: int, length: int) -> BitStream:
        """Extract a sub-BitStream starting at bit position `start` with given `length`."""
        if start < 0:
            start += self._bit_length
        if start < 0 or start + length > self._bit_length:
            raise IndexError(
                f"Slice [{start}:{start + length}] out of range for length {self._bit_length}"
            )
        if length == 0:
            return BitStream(b"", 0)
        # Extract bits via integer arithmetic
        val = self.to_int()
        # Shift and mask
        shift = self._bit_length - start - length
        mask = (1 << length) - 1
        sub_val = (val >> shift) & mask
        return BitStream.from_int(sub_val, length)

    def chunks(self, chunk_size: int) -> list[BitStream]:
        """Split into chunks of `chunk_size` bits. Last chunk may be smaller."""
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        result = []
        for i in range(0, self._bit_length, chunk_size):
            length = min(chunk_size, self._bit_length - i)
            result.append(self.slice(i, length))
        return result

    def concat(self, other: BitStream) -> BitStream:
        """Concatenate two BitStreams."""
        if self._bit_length == 0:
            return other
        if other._bit_length == 0:
            return self
        total = self._bit_length + other._bit_length
        val = (self.to_int() << other._bit_length) | other.to_int()
        return BitStream.from_int(val, total)

    # -- Dunder methods --

    def __len__(self) -> int:
        return self._bit_length

    @overload
    def __getitem__(self, key: int) -> int: ...

    @overload
    def __getitem__(self, key: slice) -> BitStream: ...

    def __getitem__(self, key: int | slice) -> int | BitStream:
        if isinstance(key, int):
            return self.bit_at(key)
        start, stop, step = key.indices(self._bit_length)
        if step != 1:
            # Fallback for non-unit step
            bits = [self.bit_at(i) for i in range(start, stop, step)]
            return BitStream.from_bin_str("".join(str(b) for b in bits))
        return self.slice(start, stop - start)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BitStream):
            return NotImplemented
        return self._bit_length == other._bit_length and self._data == other._data

    def __hash__(self) -> int:
        h = object.__getattribute__(self, "_hash")
        if h is None:
            h = hash((self._data, self._bit_length))
            object.__setattr__(self, "_hash", h)
        return h

    def __repr__(self) -> str:
        if self._bit_length <= 64:
            return f"BitStream('{self.to_bin_str()}', len={self._bit_length})"
        preview = self.slice(0, 32).to_bin_str()
        return f"BitStream('{preview}...', len={self._bit_length})"

    def __iter__(self) -> Iterator[int]:
        for i in range(self._bit_length):
            yield self.bit_at(i)

    def __bool__(self) -> bool:
        return self._bit_length > 0
