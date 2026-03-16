"""Shared test fixtures."""

import random

import pytest

from maxcomp.bitstream import BitStream


@pytest.fixture
def random_bits_64():
    """64 bytes of pseudo-random data (seeded for reproducibility)."""
    rng = random.Random(12345)
    return BitStream.from_bytes(bytes(rng.getrandbits(8) for _ in range(64)))


@pytest.fixture
def all_zeros_32():
    """32 bytes of zeros."""
    return BitStream.from_bytes(b"\x00" * 32)


@pytest.fixture
def all_ones_32():
    """32 bytes of ones."""
    return BitStream.from_bytes(b"\xff" * 32)


@pytest.fixture
def sequential_bytes():
    """Bytes 0-255 in order."""
    return BitStream.from_bytes(bytes(range(256)))


@pytest.fixture
def mt19937_seed42():
    """256 bits from MT19937 seeded with 42."""
    rng = random.Random(42)
    data = rng.getrandbits(256).to_bytes(32, "big")
    return BitStream.from_bytes(data)


@pytest.fixture
def self_similar_pattern():
    """A bit stream with self-similar structure (repeated pattern with variations)."""
    # Pattern: 10110100 repeated 8 times
    pattern = "10110100" * 8
    return BitStream.from_bin_str(pattern)


@pytest.fixture
def short_bits():
    """Short 16-bit stream."""
    return BitStream.from_bin_str("1101001011110001")
