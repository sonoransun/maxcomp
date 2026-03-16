"""Tests for chaotic sequence generation and integration."""

import pytest

from maxcomp.bitstream import BitStream
from maxcomp.constants.catalog import CHAOTIC_SPECS, ConstantID
from maxcomp.constants.chaotic import generate_chaotic_bits
from maxcomp.constants.provider import get_binary_expansion
from maxcomp.strategies.constant_match import ConstantMatchStrategy


class TestChaoticGeneration:
    def test_logistic_deterministic(self):
        """Same parameters produce same bits every time."""
        spec = CHAOTIC_SPECS[ConstantID.LOGISTIC_R4_0_X0_0_1]
        bits1 = generate_chaotic_bits(spec, 200)
        bits2 = generate_chaotic_bits(spec, 200)
        assert bits1 == bits2
        assert len(bits1) == 200
        assert all(c in "01" for c in bits1)

    def test_tent_deterministic(self):
        spec = CHAOTIC_SPECS[ConstantID.TENT_MU2_X0_0_1]
        bits1 = generate_chaotic_bits(spec, 200)
        bits2 = generate_chaotic_bits(spec, 200)
        assert bits1 == bits2

    def test_sine_deterministic(self):
        spec = CHAOTIC_SPECS[ConstantID.SINE_A1_0_X0_0_1]
        bits1 = generate_chaotic_bits(spec, 100)
        bits2 = generate_chaotic_bits(spec, 100)
        assert bits1 == bits2

    def test_gauss_deterministic(self):
        spec = CHAOTIC_SPECS[ConstantID.GAUSS_A4_9_BN0_1_X0_0_5]
        bits1 = generate_chaotic_bits(spec, 100)
        bits2 = generate_chaotic_bits(spec, 100)
        assert bits1 == bits2

    def test_lorenz_deterministic(self):
        spec = CHAOTIC_SPECS[ConstantID.LORENZ_STD_X]
        bits1 = generate_chaotic_bits(spec, 100)
        bits2 = generate_chaotic_bits(spec, 100)
        assert bits1 == bits2

    def test_henon_deterministic(self):
        spec = CHAOTIC_SPECS[ConstantID.HENON_STD]
        bits1 = generate_chaotic_bits(spec, 100)
        bits2 = generate_chaotic_bits(spec, 100)
        assert bits1 == bits2

    def test_logistic_has_both_bits(self):
        """Chaotic logistic map at r=4 should produce both 0s and 1s."""
        spec = CHAOTIC_SPECS[ConstantID.LOGISTIC_R4_0_X0_0_1]
        bits = generate_chaotic_bits(spec, 200)
        assert "0" in bits
        assert "1" in bits

    def test_different_params_different_bits(self):
        """Different parameters should produce different sequences."""
        bits_a = generate_chaotic_bits(CHAOTIC_SPECS[ConstantID.LOGISTIC_R4_0_X0_0_1], 200)
        bits_b = generate_chaotic_bits(CHAOTIC_SPECS[ConstantID.LOGISTIC_R4_0_X0_0_6], 200)
        assert bits_a != bits_b


class TestBernoulliEquivalence:
    def test_bernoulli_pi_equals_pi(self):
        """Bernoulli shift with x0=pi-3 should equal pi's binary expansion."""
        pi_bits = get_binary_expansion(ConstantID.PI, 500)
        bernoulli_bits = get_binary_expansion(ConstantID.BERNOULLI_PI_FRAC, 500)
        assert pi_bits == bernoulli_bits

    def test_bernoulli_e_equals_e(self):
        e_bits = get_binary_expansion(ConstantID.E, 500)
        bernoulli_bits = get_binary_expansion(ConstantID.BERNOULLI_E_FRAC, 500)
        assert e_bits == bernoulli_bits


class TestProviderCaching:
    def test_chaotic_caching(self):
        """Second call should return cached result."""
        bits1 = get_binary_expansion(ConstantID.TENT_MU2_X0_0_3, 200)
        bits2 = get_binary_expansion(ConstantID.TENT_MU2_X0_0_3, 200)
        assert bits1 == bits2

    def test_chaotic_prefix(self):
        """Requesting fewer bits should return a prefix of the cached expansion."""
        bits_long = get_binary_expansion(ConstantID.LOGISTIC_R3_9_X0_0_7, 200)
        bits_short = get_binary_expansion(ConstantID.LOGISTIC_R3_9_X0_0_7, 100)
        assert bits_long[:100] == bits_short


class TestConstantMatchWithChaotic:
    def test_finds_logistic_match(self):
        """Data from a logistic sequence should be found by constant_match."""
        bits = get_binary_expansion(ConstantID.LOGISTIC_R4_0_X0_0_1, 500)
        data = BitStream.from_bin_str(bits[:200])
        strategy = ConstantMatchStrategy(precision_bits=500, include_chaotic=True)
        result = strategy.compress(data, timeout_seconds=30.0)
        assert result is not None
        assert result.metadata["constant"] == "logistic_r4.0_x0.1"
        assert result.metadata["offset"] == 0

    def test_finds_tent_match(self):
        bits = get_binary_expansion(ConstantID.TENT_MU2_X0_0_7, 500)
        data = BitStream.from_bin_str(bits[100:300])
        strategy = ConstantMatchStrategy(precision_bits=500, include_chaotic=True)
        result = strategy.compress(data, timeout_seconds=30.0)
        assert result is not None
        # Verify round-trip regardless of which sequence matched
        restored = strategy.decompress(result.method, result.payload, result.original_bit_length)
        assert restored == data

    def test_roundtrip_chaotic(self):
        """Compress and decompress data from a chaotic sequence."""
        bits = get_binary_expansion(ConstantID.SINE_A1_0_X0_0_5, 500)
        original = BitStream.from_bin_str(bits[:200])
        strategy = ConstantMatchStrategy(precision_bits=500, include_chaotic=True)
        result = strategy.compress(original, timeout_seconds=30.0)
        assert result is not None
        restored = strategy.decompress(result.method, result.payload, result.original_bit_length)
        assert restored == original

    def test_classic_only_skips_chaotic(self):
        """Without include_chaotic, chaotic sequences should not be searched."""
        bits = get_binary_expansion(ConstantID.LOGISTIC_R4_0_X0_0_1, 500)
        data = BitStream.from_bin_str(bits[:200])
        strategy = ConstantMatchStrategy(precision_bits=500, include_chaotic=False)
        result = strategy.compress(data, timeout_seconds=10.0)
        # Should NOT find it in classic constants (very unlikely to match pi/e/etc)
        # (might accidentally match, so we just verify it doesn't crash)
        if result is not None:
            assert result.metadata["constant"] not in [
                "logistic_r4.0_x0.1",
            ]
