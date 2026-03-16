"""Tests for constant-match compression strategy."""

import pytest

from maxcomp.bitstream import BitStream
from maxcomp.constants.provider import get_binary_expansion
from maxcomp.constants.catalog import ConstantID
from maxcomp.strategies.constant_match import ConstantMatchStrategy


@pytest.fixture
def strategy():
    return ConstantMatchStrategy(precision_bits=10_000)


class TestConstantMatch:
    def test_finds_known_pi_bits(self, strategy):
        """The first N bits of pi's fractional expansion should match at offset 0."""
        pi_bits = get_binary_expansion(ConstantID.PI, 200)
        # Take bits 0..200 — should be found at offset 0
        data = BitStream.from_bin_str(pi_bits[:200])
        result = strategy.compress(data, timeout_seconds=10.0)
        assert result is not None
        assert result.metadata["constant"] == "pi"
        assert result.metadata["offset"] == 0

    def test_finds_pi_bits_at_offset(self, strategy):
        """Bits from an offset within pi should be found."""
        pi_bits = get_binary_expansion(ConstantID.PI, 1000)
        # Take 200 bits starting at offset 500
        data = BitStream.from_bin_str(pi_bits[500:700])
        result = strategy.compress(data, timeout_seconds=10.0)
        assert result is not None
        assert result.metadata["constant"] == "pi"
        assert result.metadata["offset"] == 500

    def test_finds_e_bits(self, strategy):
        """Should find bits in e's expansion."""
        e_bits = get_binary_expansion(ConstantID.E, 200)
        data = BitStream.from_bin_str(e_bits[:200])
        result = strategy.compress(data, timeout_seconds=10.0)
        assert result is not None
        assert result.metadata["offset"] == 0

    def test_roundtrip(self, strategy):
        """Compress then decompress should return original data."""
        pi_bits = get_binary_expansion(ConstantID.PI, 500)
        original = BitStream.from_bin_str(pi_bits[100:300])
        result = strategy.compress(original, timeout_seconds=10.0)
        assert result is not None
        restored = strategy.decompress(result.method, result.payload, result.original_bit_length)
        assert restored == original

    def test_too_short_returns_none(self, strategy):
        """Input shorter than payload size should return None."""
        data = BitStream.from_bytes(b"\xab")  # 8 bits, 1 byte — smaller than 17-byte payload
        result = strategy.compress(data, timeout_seconds=5.0)
        assert result is None

    def test_random_data_likely_none(self, strategy):
        """Random data is unlikely to appear in first 10k bits of any constant."""
        import os
        data = BitStream.from_bytes(os.urandom(64))
        result = strategy.compress(data, timeout_seconds=5.0)
        # Can't guarantee None, but very likely with only 10k precision
        # Just verify it doesn't crash
        if result is not None:
            restored = strategy.decompress(result.method, result.payload, result.original_bit_length)
            assert restored == data
