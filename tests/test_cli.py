"""Tests for the CLI interface."""

import os
import subprocess
import sys

import pytest


@pytest.fixture
def sample_file(tmp_path):
    """Create a small sample binary file."""
    f = tmp_path / "sample.bin"
    f.write_bytes(b"\xde\xad\xbe\xef" * 8)  # 32 bytes
    return f


class TestCLI:
    def _run(self, args: list[str], **kwargs):
        return subprocess.run(
            [sys.executable, "-m", "maxcomp"] + args,
            capture_output=True,
            timeout=60,
            **kwargs,
        )

    def test_compress_decompress_roundtrip(self, tmp_path, sample_file):
        """Compress then decompress should restore original."""
        mxc = tmp_path / "output.mxc"
        restored = tmp_path / "restored.bin"

        # Compress
        result = self._run([
            "compress", "-i", str(sample_file), "-o", str(mxc), "--timeout", "5",
        ])
        assert result.returncode == 0, result.stderr.decode()
        assert mxc.exists()

        # Decompress
        result = self._run([
            "decompress", "-i", str(mxc), "-o", str(restored),
        ])
        assert result.returncode == 0, result.stderr.decode()
        assert restored.read_bytes() == sample_file.read_bytes()

    def test_analyze_json(self, sample_file):
        """Analyze should produce valid JSON."""
        import json
        result = self._run([
            "analyze", "-i", str(sample_file), "--json", "--timeout", "5",
        ])
        assert result.returncode == 0, result.stderr.decode()
        data = json.loads(result.stdout.decode())
        assert "original_bits" in data
        assert "strategies" in data

    def test_analyze_text(self, sample_file):
        """Analyze should produce text output."""
        result = self._run([
            "analyze", "-i", str(sample_file), "--timeout", "5",
        ])
        assert result.returncode == 0, result.stderr.decode()
        output = result.stdout.decode()
        assert "Original:" in output
        assert "Best method:" in output

    def test_info(self, tmp_path, sample_file):
        """Info should display metadata."""
        mxc = tmp_path / "output.mxc"
        self._run([
            "compress", "-i", str(sample_file), "-o", str(mxc), "--timeout", "5",
        ])
        result = self._run(["info", str(mxc)])
        assert result.returncode == 0, result.stderr.decode()
        output = result.stdout.decode()
        assert "Format: MXC v1" in output

    def test_compress_hex_input(self, tmp_path):
        """Compress from hex input format."""
        hex_file = tmp_path / "input.hex"
        hex_file.write_text("deadbeefcafebabe" * 4)
        mxc = tmp_path / "output.mxc"
        result = self._run([
            "compress", "-i", str(hex_file), "-f", "hex",
            "-o", str(mxc), "--timeout", "5",
        ])
        assert result.returncode == 0, result.stderr.decode()

    def test_compress_no_chunking(self, tmp_path, sample_file):
        """Compress with chunking disabled."""
        mxc = tmp_path / "output.mxc"
        restored = tmp_path / "restored.bin"
        self._run([
            "compress", "-i", str(sample_file), "-o", str(mxc),
            "--no-chunking", "--timeout", "5",
        ])
        self._run(["decompress", "-i", str(mxc), "-o", str(restored)])
        assert restored.read_bytes() == sample_file.read_bytes()
