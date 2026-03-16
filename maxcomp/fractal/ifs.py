"""1-D IFS (Iterated Function System) fractal compression for bit streams.

Inspired by fractal image compression: divide the stream into small "range
blocks" and express each one as a transformed copy of a larger "domain block"
found elsewhere in the stream.
"""

from __future__ import annotations

from typing import Optional


def _downsample(block: list[int]) -> list[int]:
    """Downsample a block 2:1 by taking every other bit."""
    return block[::2]


def _invert(block: list[int]) -> list[int]:
    return [1 - b for b in block]


def _hamming(a: list[int], b: list[int]) -> int:
    return sum(x != y for x, y in zip(a, b))


def ifs_compress(
    bits: list[int],
    block_size: int = 8,
) -> Optional[tuple[list[tuple[int, bool]], list[int], int]]:
    """Attempt IFS-style compression on *bits*.

    Parameters
    ----------
    bits : list[int]
        The bit stream to compress.
    block_size : int
        Size of each range block.

    Returns
    -------
    (transforms, seed_bits, block_size) or None
        *transforms* is a list of ``(domain_index, is_inverted)`` — one per
        range block.  *seed_bits* is the initial seed (same length as *bits*,
        padded if necessary).  Returns ``None`` if not all range blocks could
        be matched to a domain block.
    """
    n = len(bits)
    if n < block_size * 2:
        return None

    domain_size = block_size * 2
    num_range_blocks = n // block_size

    # Pre-compute all domain blocks (step 1 for overlapping search).
    max_domain_start = n - domain_size
    if max_domain_start < 0:
        return None

    # Build domain pool — use a stride of 1 for best matching, but limit total
    # domains to keep the search tractable.
    stride = max(1, (max_domain_start + 1) // 512)
    domain_starts: list[int] = list(range(0, max_domain_start + 1, stride))

    # Pre-downsample domains.
    domain_ds: list[list[int]] = []
    for ds in domain_starts:
        domain_ds.append(_downsample(bits[ds: ds + domain_size]))

    transforms: list[tuple[int, bool]] = []
    max_errors = max(1, block_size // 8)  # allow small tolerance

    for rb_idx in range(num_range_blocks):
        rb_start = rb_idx * block_size
        rb = bits[rb_start: rb_start + block_size]

        best: Optional[tuple[int, bool, int]] = None  # (domain_idx, inverted, errors)

        for d_idx, ds_block in enumerate(domain_ds):
            # Normal match
            errs = _hamming(ds_block, rb)
            if errs <= max_errors:
                if best is None or errs < best[2]:
                    best = (d_idx, False, errs)
                    if errs == 0:
                        break

            # Inverted match
            errs_inv = _hamming(_invert(ds_block), rb)
            if errs_inv <= max_errors:
                if best is None or errs_inv < best[2]:
                    best = (d_idx, True, errs_inv)
                    if errs_inv == 0:
                        break

        if best is None:
            return None
        transforms.append((best[0], best[1]))

    # Seed is just the original bits (used as starting point for iterative
    # reconstruction).
    seed_bits = list(bits)
    return transforms, seed_bits, block_size


def ifs_decompress(
    transforms: list[tuple[int, bool]],
    seed_bits: list[int],
    block_size: int,
    total_bits: int,
    iterations: int = 20,
) -> list[int]:
    """Reconstruct the bit stream by iteratively applying transforms.

    Parameters
    ----------
    transforms : list[(domain_index, is_inverted)]
        One entry per range block.
    seed_bits : list[int]
        Starting bit values (length >= total_bits ideally).
    block_size : int
        Range block size.
    total_bits : int
        Desired output length.
    iterations : int
        Number of fixed-point iterations.

    Returns
    -------
    list[int]
        Reconstructed bit stream of length *total_bits*.
    """
    domain_size = block_size * 2

    # Pad or trim seed to total_bits.
    current = list(seed_bits[:total_bits])
    while len(current) < total_bits:
        current.append(0)

    # Derive the domain stride that was used during compression.
    max_domain_start = total_bits - domain_size
    if max_domain_start < 0:
        return current[:total_bits]
    stride = max(1, (max_domain_start + 1) // 512)
    domain_starts: list[int] = list(range(0, max_domain_start + 1, stride))

    for _ in range(iterations):
        new = list(current)
        for rb_idx, (d_idx, inverted) in enumerate(transforms):
            if d_idx >= len(domain_starts):
                continue
            ds = domain_starts[d_idx]
            domain_block = current[ds: ds + domain_size]
            if len(domain_block) < domain_size:
                continue
            downsampled = _downsample(domain_block)
            if inverted:
                downsampled = _invert(downsampled)
            rb_start = rb_idx * block_size
            for k in range(block_size):
                pos = rb_start + k
                if pos < total_bits and k < len(downsampled):
                    new[pos] = downsampled[k]
        current = new

    return current[:total_bits]
