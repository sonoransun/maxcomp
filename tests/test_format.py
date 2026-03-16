"""Tests for the .mxc format serialization."""

import pytest

from maxcomp.bitstream import BitStream
from maxcomp.format import (
    HEADER_SIZE,
    MAGIC,
    CompressedOutput,
    serialize_chunks,
)
from maxcomp.strategies.base import CompressionResult, MethodID


class TestSerializeDeserialize:
    def test_identity_roundtrip(self):
        original = CompressedOutput(
            original_bit_length=64,
            method=MethodID.IDENTITY,
            payload=b"\xde\xad\xbe\xef\xca\xfe\xba\xbe",
        )
        data = original.serialize()
        restored = CompressedOutput.deserialize(data)
        assert restored.original_bit_length == 64
        assert restored.method == MethodID.IDENTITY
        assert restored.payload == original.payload

    def test_header_magic(self):
        output = CompressedOutput(
            original_bit_length=8,
            method=MethodID.IDENTITY,
            payload=b"\xff",
        )
        data = output.serialize()
        assert data[:4] == MAGIC

    def test_header_size(self):
        output = CompressedOutput(
            original_bit_length=8,
            method=MethodID.IDENTITY,
            payload=b"\xff",
        )
        data = output.serialize()
        assert len(data) == HEADER_SIZE + 1

    def test_invalid_magic(self):
        data = b"XXXX" + b"\x00" * 12
        with pytest.raises(ValueError, match="Invalid magic"):
            CompressedOutput.deserialize(data)

    def test_data_too_short(self):
        with pytest.raises(ValueError, match="too short"):
            CompressedOutput.deserialize(b"\x00" * 4)

    def test_total_compressed_size(self):
        output = CompressedOutput(
            original_bit_length=64,
            method=MethodID.IDENTITY,
            payload=b"\x00" * 8,
        )
        assert output.total_compressed_size == HEADER_SIZE + 8


class TestChunkedFormat:
    def test_chunked_serialize_deserialize(self):
        chunks = [
            CompressionResult(
                method=MethodID.IDENTITY,
                payload=b"\xaa",
                original_bit_length=8,
                compressed_bit_length=8,
            ),
            CompressionResult(
                method=MethodID.IDENTITY,
                payload=b"\xbb",
                original_bit_length=8,
                compressed_bit_length=8,
            ),
        ]
        payload = serialize_chunks(chunks)
        output = CompressedOutput(
            original_bit_length=16,
            method=MethodID.CHUNKED_COMPOSITE,
            payload=payload,
            chunk_results=chunks,
        )
        data = output.serialize()
        restored = CompressedOutput.deserialize(data)
        assert restored.is_chunked
        assert len(restored.chunk_results) == 2
        assert restored.chunk_results[0].payload == b"\xaa"
        assert restored.chunk_results[1].payload == b"\xbb"


class TestDecompress:
    def test_decompress_identity(self):
        from maxcomp.strategies import _ensure_registered
        _ensure_registered()

        bs = BitStream.from_hex("deadbeef")
        output = CompressedOutput(
            original_bit_length=len(bs),
            method=MethodID.IDENTITY,
            payload=bs.to_bytes(),
        )
        restored = output.decompress()
        assert restored == bs

    def test_full_roundtrip_via_serialize(self):
        from maxcomp.strategies import _ensure_registered
        _ensure_registered()

        bs = BitStream.from_hex("cafebabe")
        output = CompressedOutput(
            original_bit_length=len(bs),
            method=MethodID.IDENTITY,
            payload=bs.to_bytes(),
        )
        serialized = output.serialize()
        deserialized = CompressedOutput.deserialize(serialized)
        restored = deserialized.decompress()
        assert restored == bs
