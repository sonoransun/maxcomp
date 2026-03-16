"""Tests for BitStream."""

import pytest

from maxcomp.bitstream import BitStream


class TestConstruction:
    def test_from_bytes(self):
        bs = BitStream.from_bytes(b"\xab")
        assert len(bs) == 8
        assert bs.to_bytes() == b"\xab"

    def test_from_bytes_empty(self):
        bs = BitStream.from_bytes(b"")
        assert len(bs) == 0

    def test_from_hex(self):
        bs = BitStream.from_hex("deadbeef")
        assert len(bs) == 32
        assert bs.to_bytes() == b"\xde\xad\xbe\xef"

    def test_from_hex_0x_prefix(self):
        bs = BitStream.from_hex("0xCAFE")
        assert bs.to_bytes() == b"\xca\xfe"

    def test_from_bin_str(self):
        bs = BitStream.from_bin_str("11010010")
        assert len(bs) == 8
        assert bs.bit_at(0) == 1
        assert bs.bit_at(1) == 1
        assert bs.bit_at(2) == 0
        assert bs.bit_at(3) == 1

    def test_from_bin_str_non_aligned(self):
        bs = BitStream.from_bin_str("101")
        assert len(bs) == 3
        assert bs.bit_at(0) == 1
        assert bs.bit_at(1) == 0
        assert bs.bit_at(2) == 1

    def test_from_bin_str_0b_prefix(self):
        bs = BitStream.from_bin_str("0b1100")
        assert len(bs) == 4

    def test_from_int(self):
        bs = BitStream.from_int(0b1101, 4)
        assert bs.to_bin_str() == "1101"

    def test_from_int_zero(self):
        bs = BitStream.from_int(0, 8)
        assert bs.to_bin_str() == "00000000"

    def test_from_file(self, tmp_path):
        f = tmp_path / "test.bin"
        f.write_bytes(b"\x42\x43")
        bs = BitStream.from_file(str(f))
        assert bs.to_bytes() == b"\x42\x43"


class TestAccessors:
    def test_bit_at(self):
        bs = BitStream.from_bytes(b"\x80")  # 10000000
        assert bs.bit_at(0) == 1
        assert bs.bit_at(1) == 0

    def test_bit_at_negative_index(self):
        bs = BitStream.from_bytes(b"\x01")  # 00000001
        assert bs.bit_at(-1) == 1

    def test_bit_at_out_of_range(self):
        bs = BitStream.from_bytes(b"\x00")
        with pytest.raises(IndexError):
            bs.bit_at(8)

    def test_to_int(self):
        bs = BitStream.from_bin_str("1101")
        assert bs.to_int() == 0b1101

    def test_to_int_byte_aligned(self):
        bs = BitStream.from_bytes(b"\xff")
        assert bs.to_int() == 255

    def test_to_bin_str(self):
        bs = BitStream.from_bytes(b"\xab")
        assert bs.to_bin_str() == "10101011"


class TestSlicing:
    def test_slice(self):
        bs = BitStream.from_bin_str("11010010")
        sub = bs.slice(2, 4)
        assert sub.to_bin_str() == "0100"

    def test_slice_start(self):
        bs = BitStream.from_bin_str("11010010")
        sub = bs.slice(0, 4)
        assert sub.to_bin_str() == "1101"

    def test_slice_end(self):
        bs = BitStream.from_bin_str("11010010")
        sub = bs.slice(4, 4)
        assert sub.to_bin_str() == "0010"

    def test_slice_empty(self):
        bs = BitStream.from_bin_str("11010010")
        sub = bs.slice(0, 0)
        assert len(sub) == 0

    def test_slice_out_of_range(self):
        bs = BitStream.from_bin_str("1101")
        with pytest.raises(IndexError):
            bs.slice(2, 4)

    def test_getitem_int(self):
        bs = BitStream.from_bin_str("1101")
        assert bs[0] == 1
        assert bs[2] == 0

    def test_getitem_slice(self):
        bs = BitStream.from_bin_str("11010010")
        sub = bs[2:6]
        assert sub.to_bin_str() == "0100"


class TestChunking:
    def test_chunks_even(self):
        bs = BitStream.from_bin_str("11001100")
        chunks = bs.chunks(4)
        assert len(chunks) == 2
        assert chunks[0].to_bin_str() == "1100"
        assert chunks[1].to_bin_str() == "1100"

    def test_chunks_uneven(self):
        bs = BitStream.from_bin_str("1101001")
        chunks = bs.chunks(4)
        assert len(chunks) == 2
        assert chunks[0].to_bin_str() == "1101"
        assert chunks[1].to_bin_str() == "001"

    def test_chunks_larger_than_data(self):
        bs = BitStream.from_bin_str("1101")
        chunks = bs.chunks(8)
        assert len(chunks) == 1
        assert chunks[0].to_bin_str() == "1101"


class TestConcat:
    def test_concat(self):
        a = BitStream.from_bin_str("1100")
        b = BitStream.from_bin_str("0011")
        c = a.concat(b)
        assert c.to_bin_str() == "11000011"

    def test_concat_non_aligned(self):
        a = BitStream.from_bin_str("110")
        b = BitStream.from_bin_str("01")
        c = a.concat(b)
        assert c.to_bin_str() == "11001"

    def test_concat_empty(self):
        a = BitStream.from_bin_str("1100")
        b = BitStream(b"", 0)
        assert a.concat(b) == a
        assert b.concat(a) == a


class TestDunder:
    def test_eq(self):
        a = BitStream.from_bin_str("1101")
        b = BitStream.from_bin_str("1101")
        assert a == b

    def test_ne(self):
        a = BitStream.from_bin_str("1101")
        b = BitStream.from_bin_str("1100")
        assert a != b

    def test_ne_different_length(self):
        a = BitStream.from_bin_str("1101")
        b = BitStream.from_bin_str("11010")
        assert a != b

    def test_hash_equal(self):
        a = BitStream.from_bin_str("1101")
        b = BitStream.from_bin_str("1101")
        assert hash(a) == hash(b)

    def test_hashable(self):
        a = BitStream.from_bin_str("1101")
        s = {a}
        assert a in s

    def test_len(self):
        assert len(BitStream.from_bytes(b"\x00" * 10)) == 80

    def test_repr_short(self):
        bs = BitStream.from_bin_str("1101")
        assert "1101" in repr(bs)

    def test_repr_long(self):
        bs = BitStream.from_bytes(b"\x00" * 100)
        assert "..." in repr(bs)

    def test_iter(self):
        bs = BitStream.from_bin_str("1101")
        assert list(bs) == [1, 1, 0, 1]

    def test_bool_true(self):
        assert bool(BitStream.from_bytes(b"\x00"))

    def test_bool_false(self):
        assert not bool(BitStream(b"", 0))
