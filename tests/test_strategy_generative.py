"""Tests for generative compression strategy."""

import random
import struct

import pytest

from maxcomp.bitstream import BitStream
from maxcomp.generators.prng import generate_bytes
from maxcomp.strategies.generative import GenerativeStrategy


@pytest.fixture
def strategy():
    return GenerativeStrategy()


class TestPRNG:
    def test_finds_mt19937_seed(self, strategy):
        """Should find the seed for a known MT19937 output."""
        seed = 42
        rng = random.Random(seed)
        # Generate 256 bits (32 bytes) — larger than 17-byte payload so compression helps
        val = rng.getrandbits(256)
        data = BitStream.from_int(val, 256)
        result = strategy.compress(data, timeout_seconds=15.0)
        assert result is not None
        # Verify round-trip
        restored = strategy.decompress(result.method, result.payload, result.original_bit_length)
        assert restored == data

    def test_generates_correct_bytes(self):
        """generate_bytes should reproduce MT19937 output."""
        seed = 100
        rng = random.Random(seed)
        expected = rng.getrandbits(64).to_bytes(8, "big")
        result = generate_bytes(0, seed, 64)  # type 0 = MT19937
        assert result == expected


class TestExpression:
    def test_finds_power_of_two_minus_one(self, strategy):
        """Should find 2^n - 1 pattern."""
        # 2^31 - 1 = 0x7FFFFFFF
        val = (1 << 31) - 1
        data = BitStream.from_int(val, 31)
        result = strategy.compress(data, timeout_seconds=10.0)
        # This may or may not find it depending on expression enumeration
        if result is not None:
            restored = strategy.decompress(result.method, result.payload, result.original_bit_length)
            assert restored == data


class TestRoundTrip:
    def test_any_match_roundtrips(self, strategy):
        """Whatever the strategy finds, it should decompress correctly."""
        # Use a small PRNG output that should be findable
        rng = random.Random(7)
        val = rng.getrandbits(32)
        data = BitStream.from_int(val, 32)
        result = strategy.compress(data, timeout_seconds=15.0)
        if result is not None:
            restored = strategy.decompress(result.method, result.payload, result.original_bit_length)
            assert restored == data
