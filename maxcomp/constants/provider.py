"""Provider for binary expansions of mathematical constants and chaotic sequences.

Computes binary expansions via mpmath and caches results to disk for reuse.
Dispatches to the appropriate generator based on ConstantID type.
"""

from __future__ import annotations

from pathlib import Path

import mpmath

from maxcomp.constants.catalog import (
    BERNOULLI_DELEGATES,
    CHAOTIC_SPECS,
    CONSTANT_FUNCTIONS,
    CONSTANT_NAMES,
    ConstantID,
)
from maxcomp.constants.chaotic import generate_chaotic_bits

_CACHE_DIR = Path.home() / ".cache" / "maxcomp"

# In-memory cache: constant_id -> binary string
_mem_cache: dict[ConstantID, str] = {}


def _cache_path(constant_id: ConstantID, num_bits: int) -> Path:
    """Return the file path for a cached binary expansion."""
    name = CONSTANT_NAMES[constant_id]
    return _CACHE_DIR / f"{name}_{num_bits}.bits"


def _compute_binary_expansion(constant_id: ConstantID, num_bits: int) -> str:
    """Compute the binary expansion of the fractional part of a classic constant.

    Uses repeated multiply-and-floor on the fractional part.
    """
    dps = int(num_bits * 0.4) + 50
    with mpmath.workdps(dps):
        value = CONSTANT_FUNCTIONS[constant_id]()
        frac = value - int(value)

        bits: list[str] = []
        for _ in range(num_bits):
            frac *= 2
            bit = int(frac)
            frac -= bit
            bits.append(str(bit))

    return "".join(bits)


def _compute_expansion(constant_id: ConstantID, num_bits: int) -> str:
    """Compute binary expansion for any ConstantID, dispatching by type."""
    # Bernoulli shift entries delegate to their underlying classic constant
    if constant_id in BERNOULLI_DELEGATES:
        delegate = BERNOULLI_DELEGATES[constant_id]
        return _compute_binary_expansion(delegate, num_bits)

    # Classic mathematical constants
    if constant_id in CONSTANT_FUNCTIONS:
        return _compute_binary_expansion(constant_id, num_bits)

    # Chaotic sequences
    if constant_id in CHAOTIC_SPECS:
        spec = CHAOTIC_SPECS[constant_id]
        return generate_chaotic_bits(spec, num_bits)

    raise ValueError(f"Unknown ConstantID: {constant_id!r}")


def get_binary_expansion(constant_id: ConstantID, num_bits: int) -> str:
    """Get the binary expansion of a constant or chaotic sequence.

    Returns a string of '0' and '1' characters of length num_bits.
    Results are cached in memory and on disk at ~/.cache/maxcomp/.
    """
    # Check in-memory cache first
    if constant_id in _mem_cache and len(_mem_cache[constant_id]) >= num_bits:
        return _mem_cache[constant_id][:num_bits]

    # Check disk cache
    cache_file = _cache_path(constant_id, num_bits)
    if cache_file.exists():
        expansion = cache_file.read_text()
        if len(expansion) >= num_bits:
            _mem_cache[constant_id] = expansion
            return expansion[:num_bits]

    # Compute from scratch
    expansion = _compute_expansion(constant_id, num_bits)

    # Store to disk cache
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(expansion)

    # Store in memory
    _mem_cache[constant_id] = expansion

    return expansion
