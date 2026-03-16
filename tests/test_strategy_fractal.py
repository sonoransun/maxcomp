"""Tests for fractal compression strategy."""

import pytest

from maxcomp.bitstream import BitStream
from maxcomp.fractal.analysis import autocorrelation, has_self_similarity
from maxcomp.fractal.lsystem import LSystemRule, generate_lsystem
from maxcomp.strategies.fractal import FractalStrategy


@pytest.fixture
def strategy():
    return FractalStrategy()


class TestAnalysis:
    def test_autocorrelation_perfect_period(self):
        """Perfect period-4 signal should have high autocorrelation at lag 4."""
        bits = [1, 0, 1, 1] * 20
        r = autocorrelation(bits, 4)
        assert r == pytest.approx(1.0)

    def test_autocorrelation_no_period(self):
        """Random-ish signal should have low autocorrelation."""
        import random
        rng = random.Random(999)
        bits = [rng.randint(0, 1) for _ in range(100)]
        r = autocorrelation(bits, 7)
        assert abs(r) < 0.5  # unlikely to be high for random data

    def test_self_similarity_periodic(self):
        """Periodic data should be detected as self-similar."""
        bits = [1, 0, 1, 0] * 30
        assert has_self_similarity(bits, threshold=0.7)

    def test_self_similarity_random(self):
        """Random data should generally not be self-similar."""
        import random
        rng = random.Random(42)
        bits = [rng.randint(0, 1) for _ in range(200)]
        # Might or might not be, but shouldn't crash
        has_self_similarity(bits, threshold=0.9)


class TestLSystem:
    def test_generate_basic(self):
        """Basic L-system generation."""
        rules = [LSystemRule("A", "AB"), LSystemRule("B", "A")]
        result = generate_lsystem("A", rules, 3)
        # A -> AB -> ABA -> ABAAB
        assert result == "ABAAB"

    def test_generate_fibonacci(self):
        """Fibonacci L-system produces expected lengths."""
        rules = [LSystemRule("A", "AB"), LSystemRule("B", "A")]
        # Lengths: 1, 2, 3, 5, 8, 13, ...
        for n, expected_len in [(0, 1), (1, 2), (2, 3), (3, 5), (4, 8)]:
            result = generate_lsystem("A", rules, n)
            assert len(result) == expected_len


class TestFractalStrategy:
    def test_periodic_data(self, strategy):
        """Highly periodic data might be compressible."""
        pattern = "10110100" * 10  # 80 bits, clearly periodic
        data = BitStream.from_bin_str(pattern)
        result = strategy.compress(data, timeout_seconds=10.0)
        # If it finds something, verify roundtrip
        if result is not None:
            restored = strategy.decompress(
                result.method, result.payload, result.original_bit_length
            )
            assert restored == data

    def test_no_crash_on_random(self, strategy):
        """Random data should not crash the strategy."""
        import os
        data = BitStream.from_bytes(os.urandom(32))
        result = strategy.compress(data, timeout_seconds=5.0)
        # Most likely None for random data
        if result is not None:
            restored = strategy.decompress(
                result.method, result.payload, result.original_bit_length
            )
            assert restored == data
