"""Tests for long window scopes and tiered search."""

import subprocess
import sys

import pytest

from maxcomp.bitstream import BitStream
from maxcomp.cli import WINDOW_PRESETS
from maxcomp.constants.catalog import ConstantID
from maxcomp.constants.provider import get_binary_expansion
from maxcomp.engine import CompressionEngine
from maxcomp.strategies.constant_match import ConstantMatchStrategy


class TestWindowPresets:
    def test_preset_values(self):
        assert WINDOW_PRESETS["small"] == 8_192
        assert WINDOW_PRESETS["medium"] == 65_536
        assert WINDOW_PRESETS["default"] == 100_000
        assert WINDOW_PRESETS["large"] == 524_288
        assert WINDOW_PRESETS["xlarge"] == 2_097_152
        assert WINDOW_PRESETS["huge"] == 8_388_608

    def test_all_presets_present(self):
        expected = {"small", "medium", "default", "large", "xlarge", "huge"}
        assert set(WINDOW_PRESETS.keys()) == expected


class TestLargeWindowMatch:
    @pytest.mark.timeout(60)
    def test_pi_deep_offset(self):
        """Data embedded deep in pi's expansion should be found with large window."""
        # Get bits from an offset that exceeds the default 100k window
        pi_bits = get_binary_expansion(ConstantID.PI, 150_000)
        # Take 200 bits starting at offset 110k (beyond default window)
        data = BitStream.from_bin_str(pi_bits[110_000:110_200])
        strategy = ConstantMatchStrategy(precision_bits=150_000)
        result = strategy.compress(data, timeout_seconds=30.0)
        assert result is not None
        assert result.metadata["offset"] == 110_000


class TestTieredSearch:
    def test_tiered_finds_early_match(self):
        """Tiered search should find data in the first tier quickly."""
        pi_bits = get_binary_expansion(ConstantID.PI, 8_192)
        data = BitStream.from_bin_str(pi_bits[100:300])
        strategy = ConstantMatchStrategy(
            precision_bits=524_288, tiered=True
        )
        result = strategy.compress(data, timeout_seconds=30.0)
        assert result is not None
        assert result.metadata["offset"] == 100

    def test_tiered_no_false_positives(self):
        """Tiered search should not return wrong results."""
        import os
        data = BitStream.from_bytes(os.urandom(32))
        strategy = ConstantMatchStrategy(
            precision_bits=8_192, tiered=True
        )
        result = strategy.compress(data, timeout_seconds=5.0)
        if result is not None:
            # Verify the match is real
            restored = strategy.decompress(
                result.method, result.payload, result.original_bit_length
            )
            assert restored == data


class TestEngineWithWindow:
    def test_engine_precision_bits(self):
        """Engine should configure ConstantMatchStrategy with precision_bits."""
        engine = CompressionEngine(
            precision_bits=50_000,
            timeout_per_strategy=5.0,
            enable_chunking=False,
        )
        # Verify the strategy was configured
        from maxcomp.strategies.constant_match import ConstantMatchStrategy
        cm = [s for s in engine.strategies if isinstance(s, ConstantMatchStrategy)]
        assert len(cm) == 1
        assert cm[0]._precision_bits == 50_000

    def test_engine_chaotic_flag(self):
        engine = CompressionEngine(
            include_chaotic=True,
            timeout_per_strategy=5.0,
            enable_chunking=False,
        )
        from maxcomp.strategies.constant_match import ConstantMatchStrategy
        cm = [s for s in engine.strategies if isinstance(s, ConstantMatchStrategy)]
        assert len(cm) == 1
        assert cm[0]._include_chaotic is True


class TestCLIWindowFlags:
    def _run(self, args: list[str], input_data: bytes = b"", **kwargs):
        return subprocess.run(
            [sys.executable, "-m", "maxcomp"] + args,
            capture_output=True, input=input_data, timeout=30, **kwargs,
        )

    def test_window_flag(self):
        """CLI should accept --window flag."""
        result = self._run([
            "analyze", "-f", "hex", "--window", "small",
            "--no-chunking", "--timeout", "5",
        ], input_data=b"deadbeefcafebabe" * 4)
        assert result.returncode == 0

    def test_enable_chaotic_flag(self):
        """CLI should accept --enable-chaotic flag."""
        result = self._run([
            "analyze", "-f", "hex", "--enable-chaotic",
            "--window", "small", "--no-chunking", "--timeout", "5",
        ], input_data=b"deadbeefcafebabe" * 4)
        assert result.returncode == 0
