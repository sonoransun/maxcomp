"""Tests for native C acceleration layer."""

import random
import hashlib
import time

import pytest

from maxcomp.native import NATIVE_AVAILABLE, get_lib, get_ffi

pytestmark = pytest.mark.skipif(not NATIVE_AVAILABLE, reason="Native library not available")


@pytest.fixture
def lib():
    return get_lib()


@pytest.fixture
def ffi():
    return get_ffi()


class TestVersion:
    def test_version(self, lib):
        assert lib.maxcomp_native_version() == 1


class TestPRNGNative:
    def test_xorshift_search_finds_seed(self, lib, ffi):
        """Generate xorshift128 output for seed=7, verify C finds it."""
        from maxcomp.generators.prng import _xorshift128

        seed = 7
        val = _xorshift128(seed, 256)
        target = val.to_bytes(32, "big")
        result = lib.prng_xorshift128_search(target, 256, 100)
        assert result == seed

    def test_lcg_search_finds_seed(self, lib, ffi):
        """Generate LCG output for seed=13, verify C finds it."""
        from maxcomp.generators.prng import _lcg

        seed = 13
        val = _lcg(seed, 64)
        target = val.to_bytes(8, "big")
        result = lib.prng_lcg_search(target, 64, 100)
        assert result == seed

    def test_search_all(self, lib, ffi):
        """Test prng_search_all finds xorshift match."""
        from maxcomp.generators.prng import _xorshift128

        seed = 42
        val = _xorshift128(seed, 256)
        target = val.to_bytes(32, "big")

        out_type = ffi.new("int *")
        out_seed = ffi.new("uint64_t *")
        result = lib.prng_search_all(target, 256, 100, out_type, out_seed)
        assert result == 0
        assert out_seed[0] == seed

    def test_generate_xorshift(self, lib, ffi):
        """Verify prng_generate matches Python for xorshift."""
        from maxcomp.generators.prng import _xorshift128

        seed = 99
        bit_length = 128
        expected_val = _xorshift128(seed, bit_length)
        expected = expected_val.to_bytes(16, "big")

        out = ffi.new(f"uint8_t[16]")
        lib.prng_generate(1, seed, bit_length, out)
        actual = bytes(ffi.buffer(out, 16))
        assert actual == expected

    def test_generate_lcg(self, lib, ffi):
        """Verify prng_generate matches Python for LCG."""
        from maxcomp.generators.prng import _lcg

        seed = 50
        bit_length = 64
        expected_val = _lcg(seed, bit_length)
        expected = expected_val.to_bytes(8, "big")

        out = ffi.new(f"uint8_t[8]")
        lib.prng_generate(2, seed, bit_length, out)
        actual = bytes(ffi.buffer(out, 8))
        assert actual == expected


class TestHashNative:
    def test_compute_md5(self, lib, ffi):
        """Verify hash_compute matches hashlib for MD5."""
        preimage = b"\x42"
        expected = hashlib.md5(preimage).digest()

        digest = ffi.new("uint8_t[32]")
        lib.hash_compute(0, preimage, len(preimage), digest)
        actual = bytes(ffi.buffer(digest, 16))
        assert actual == expected

    def test_compute_sha256(self, lib, ffi):
        """Verify hash_compute matches hashlib for SHA-256."""
        preimage = b"\x00\x01\x02"
        expected = hashlib.sha256(preimage).digest()

        digest = ffi.new("uint8_t[32]")
        lib.hash_compute(2, preimage, len(preimage), digest)
        actual = bytes(ffi.buffer(digest, 32))
        assert actual == expected

    def test_preimage_search(self, lib, ffi):
        """Search for a known 1-byte preimage."""
        # Compute SHA-256 of b'\x42', take first 32 bits
        preimage = b"\x42"
        digest = hashlib.sha256(preimage).digest()
        target = digest[:4]

        out_hash_type = ffi.new("int *")
        out_preimage = ffi.new("uint8_t[3]")
        out_preimage_len = ffi.new("uint32_t *")

        result = lib.hash_preimage_search(target, 32, 1, out_hash_type, out_preimage, out_preimage_len)
        assert result == 0
        # Should find the preimage (hash_type 2 = SHA-256, preimage = 0x42)
        found_preimage = bytes(ffi.buffer(out_preimage, out_preimage_len[0]))
        # Verify the found preimage produces the correct hash prefix
        found_digest = hashlib.new(
            ["md5", "sha1", "sha256"][out_hash_type[0]], found_preimage
        ).digest()
        assert found_digest[:4] == target


class TestBitstreamNative:
    def test_hamming_distance(self, lib, ffi):
        """Test bitstream_hamming on known inputs."""
        a = b"\xFF"  # 11111111
        b = b"\x00"  # 00000000
        result = lib.bitstream_hamming(a, b, 8)
        assert result == 8

    def test_hamming_partial(self, lib, ffi):
        """Test hamming distance with partial byte."""
        a = b"\xF0"  # 11110000
        b = b"\x00"  # 00000000
        result = lib.bitstream_hamming(a, b, 4)
        assert result == 4

    def test_popcount(self, lib, ffi):
        """Test bitstream_popcount."""
        data = b"\xFF\x00"
        result = lib.bitstream_popcount(data, 16)
        assert result == 8

    def test_slice(self, lib, ffi):
        """Test bitstream_slice."""
        src = b"\xAB\xCD"  # 10101011 11001101
        dst = ffi.new("uint8_t[1]")
        lib.bitstream_slice(src, 16, 4, 8, dst)
        actual = bytes(ffi.buffer(dst, 1))
        # Bits 4-11 of 10101011 11001101 = 10111100 = 0xBC
        assert actual == b"\xbc"


class TestAutocorrelationNative:
    def test_perfect_period(self, lib, ffi):
        """Perfect period-4 signal should have high autocorrelation at lag 4."""
        pattern = [1, 0, 1, 1] * 20  # 80 bits
        # Pack into bytes
        packed = bytearray()
        for i in range(0, len(pattern), 8):
            byte = 0
            for j in range(8):
                if i + j < len(pattern):
                    byte = (byte << 1) | pattern[i + j]
                else:
                    byte <<= 1
            packed.append(byte)
        packed = bytes(packed)

        result = lib.autocorrelation_packed(packed, 80, 4)
        assert result == pytest.approx(1.0, abs=0.01)

    def test_self_similarity(self, lib, ffi):
        """Periodic data should be detected as self-similar."""
        pattern = [1, 0, 1, 0] * 30  # 120 bits
        packed = bytearray()
        for i in range(0, len(pattern), 8):
            byte = 0
            for j in range(8):
                if i + j < len(pattern):
                    byte = (byte << 1) | pattern[i + j]
                else:
                    byte <<= 1
            packed.append(byte)
        packed = bytes(packed)

        result = lib.has_self_similarity_packed(packed, 120, 0.7)
        assert result == 1


class TestBenchmark:
    """Simple benchmarks comparing native vs Python speed."""

    def test_prng_speedup(self, lib, ffi):
        """Measure PRNG search speedup."""
        from maxcomp.generators.prng import _xorshift128

        seed = 500
        val = _xorshift128(seed, 128)
        target = val.to_bytes(16, "big")

        # Native C timing
        t0 = time.perf_counter()
        result = lib.prng_xorshift128_search(target, 128, 65536)
        t_native = time.perf_counter() - t0
        assert result == seed

        # Python timing
        t0 = time.perf_counter()
        from maxcomp.generators.prng import _search_prng_python
        result_py = _search_prng_python(target, 128, 65536, 30.0)
        t_python = time.perf_counter() - t0

        speedup = t_python / t_native if t_native > 0 else float("inf")
        print(f"\nPRNG xorshift search 65k seeds: Python={t_python:.3f}s, "
              f"Native={t_native:.3f}s, Speedup={speedup:.0f}x")
        assert speedup > 5  # Should be at least 5x faster
