"""End-to-end compress/decompress round-trip tests."""

import os
import random

import pytest

from maxcomp.bitstream import BitStream
from maxcomp.engine import CompressionEngine
from maxcomp.format import CompressedOutput


@pytest.fixture
def engine():
    return CompressionEngine(timeout_per_strategy=5.0, enable_chunking=False)


@pytest.fixture
def engine_with_chunking():
    return CompressionEngine(timeout_per_strategy=5.0, enable_chunking=True)


class TestRoundTrip:
    """Verify that compress -> serialize -> deserialize -> decompress == original."""

    def _roundtrip(self, engine: CompressionEngine, data: BitStream):
        compressed = engine.compress(data)
        serialized = compressed.serialize()
        deserialized = CompressedOutput.deserialize(serialized)
        restored = deserialized.decompress()
        assert restored == data, (
            f"Round-trip failed: original={data.to_bin_str()[:64]}... "
            f"restored={restored.to_bin_str()[:64]}..."
        )

    def test_minimal_byte(self, engine):
        self._roundtrip(engine, BitStream.from_bytes(b"\x42"))

    def test_all_zeros(self, engine):
        self._roundtrip(engine, BitStream.from_bytes(b"\x00" * 32))

    def test_all_ones(self, engine):
        self._roundtrip(engine, BitStream.from_bytes(b"\xff" * 32))

    def test_sequential(self, engine):
        self._roundtrip(engine, BitStream.from_bytes(bytes(range(256))))

    def test_random_64_bytes(self, engine):
        data = BitStream.from_bytes(os.urandom(64))
        self._roundtrip(engine, data)

    def test_random_small(self, engine):
        data = BitStream.from_bytes(os.urandom(4))
        self._roundtrip(engine, data)

    def test_repeated_pattern(self, engine):
        pattern = b"\xaa\x55" * 32
        self._roundtrip(engine, BitStream.from_bytes(pattern))

    def test_mt19937_seed42(self, engine):
        rng = random.Random(42)
        data = BitStream.from_bytes(rng.getrandbits(256).to_bytes(32, "big"))
        self._roundtrip(engine, data)


class TestRoundTripChunked:
    @pytest.mark.timeout(300)
    def test_chunked_roundtrip(self, engine_with_chunking):
        data = BitStream.from_bytes(os.urandom(128))
        compressed = engine_with_chunking.compress(data)
        serialized = compressed.serialize()
        deserialized = CompressedOutput.deserialize(serialized)
        restored = deserialized.decompress()
        assert restored == data
