"""Elementary cellular automata for generative compression."""

from __future__ import annotations

import time
from typing import Optional

try:
    from maxcomp.native import NATIVE_AVAILABLE, get_lib, get_ffi
except ImportError:
    NATIVE_AVAILABLE = False


def _step_rule(rule: int, cells: list[int], width: int) -> list[int]:
    """Apply one step of an elementary cellular automaton.

    Uses periodic (wrap-around) boundary conditions.
    """
    new_cells = [0] * width
    for i in range(width):
        left = cells[(i - 1) % width]
        center = cells[i]
        right = cells[(i + 1) % width]
        neighborhood = (left << 2) | (center << 1) | right
        new_cells[i] = (rule >> neighborhood) & 1
    return new_cells


def run_automaton(
    rule: int,
    width: int,
    generations: int,
    initial_state: bytes,
    read_order: int = 0,
) -> int:
    """Run an elementary cellular automaton and return the output bits as an integer.

    Args:
        rule: Rule number (0-255).
        width: Width of the automaton (number of cells).
        generations: Number of generations to run.
        initial_state: Initial cell states packed as bytes (bit per cell, MSB first).
        read_order: 0 = row-major (default).

    Returns:
        Integer representing the output bits.
    """
    # Unpack initial state from bytes
    cells = []
    for byte_val in initial_state:
        for bit_pos in range(7, -1, -1):
            cells.append((byte_val >> bit_pos) & 1)
            if len(cells) >= width:
                break
        if len(cells) >= width:
            break
    # Pad with zeros if needed
    while len(cells) < width:
        cells.append(0)
    cells = cells[:width]

    # Collect rows
    rows: list[list[int]] = [list(cells)]
    for _ in range(generations - 1):
        cells = _step_rule(rule, cells, width)
        rows.append(list(cells))

    # Read output bits based on read_order
    bits: list[int] = []
    if read_order == 0:  # row-major
        for row in rows:
            bits.extend(row)
    else:
        # Column-major (fallback)
        for col in range(width):
            for row in rows:
                bits.append(row[col])

    # Convert to integer
    result = 0
    for b in bits:
        result = (result << 1) | b
    return result


def run_automaton_bytes(
    rule: int,
    width: int,
    generations: int,
    initial_state: bytes,
    read_order: int,
    bit_length: int,
) -> bytes:
    """Run automaton and return result as bytes, truncated to bit_length.

    Used by the decompressor.
    """
    val = run_automaton(rule, width, generations, initial_state, read_order)
    total_bits = width * generations
    # Truncate to bit_length from the MSB side
    if total_bits > bit_length:
        val >>= (total_bits - bit_length)
    num_bytes = (bit_length + 7) // 8
    return val.to_bytes(num_bytes, "big")


def _search_automata_native(
    target_bytes: bytes, target_bit_length: int
) -> Optional[tuple[int, int, int, bytes, int]]:
    """Search using native C acceleration."""
    ffi = get_ffi()
    lib = get_lib()
    widths_arr = ffi.new("uint32_t[3]", [8, 16, 32])
    out_rule = ffi.new("uint8_t *")
    out_width = ffi.new("uint32_t *")
    out_gens = ffi.new("uint32_t *")
    out_init = ffi.new("uint8_t[8]")  # max width/8 = 4 bytes
    out_init_len = ffi.new("uint32_t *")

    result = lib.automata_search(
        target_bytes, target_bit_length, widths_arr, 3,
        out_rule, out_width, out_gens, out_init, out_init_len,
    )
    if result == 0:
        init_len = out_init_len[0]
        init_state = bytes(ffi.buffer(out_init, init_len))
        return (out_rule[0], out_width[0], out_gens[0], init_state, 0)
    return None


def search_automata(
    target_bytes: bytes,
    target_bit_length: int,
    timeout: float = 30.0,
) -> Optional[tuple[int, int, int, bytes, int]]:
    """Search for a cellular automaton configuration that generates the target.

    Dispatches to native C or pure Python depending on availability.
    """
    if target_bit_length == 0:
        return None

    # Native C path
    if NATIVE_AVAILABLE:
        try:
            result = _search_automata_native(target_bytes, target_bit_length)
            if result is not None:
                return result
            return None  # Native searched exhaustively, no match
        except Exception:
            pass

    # Pure Python fallback
    return _search_automata_python(target_bytes, target_bit_length, timeout)


def _search_automata_python(
    target_bytes: bytes,
    target_bit_length: int,
    timeout: float,
) -> Optional[tuple[int, int, int, bytes, int]]:
    """Pure Python automata search (fallback)."""
    target_int = int.from_bytes(target_bytes, "big")
    padding = len(target_bytes) * 8 - target_bit_length
    if padding > 0:
        target_int >>= padding

    start_time = time.monotonic()
    widths = [8, 16, 32]

    for rule in range(256):
        if (time.monotonic() - start_time) > timeout:
            return None

        for width in widths:
            if (time.monotonic() - start_time) > timeout:
                return None

            init_cells = [0] * width
            init_cells[width // 2] = 1

            init_bytes_list = []
            for i in range(0, width, 8):
                byte_val = 0
                for j in range(8):
                    if i + j < width:
                        byte_val = (byte_val << 1) | init_cells[i + j]
                    else:
                        byte_val <<= 1
                init_bytes_list.append(byte_val)
            init_state = bytes(init_bytes_list)

            generations = (target_bit_length + width - 1) // width
            if generations < 1:
                generations = 1

            total_bits = width * generations
            if total_bits < target_bit_length:
                continue

            try:
                output = run_automaton(rule, width, generations, init_state, 0)
                if total_bits > target_bit_length:
                    output >>= (total_bits - target_bit_length)
                if output == target_int:
                    return (rule, width, generations, init_state, 0)
            except Exception:
                continue

    return None
