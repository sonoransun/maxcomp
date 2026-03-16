"""Autocorrelation and fractal dimension analysis for bit streams."""

from __future__ import annotations

try:
    from maxcomp.native import NATIVE_AVAILABLE, get_lib, get_ffi
except ImportError:
    NATIVE_AVAILABLE = False


def _pack_bits(bits: list[int]) -> bytes:
    """Pack a list of 0/1 ints into MSB-first bytes."""
    n = len(bits)
    padded = bits + [0] * ((8 - n % 8) % 8)
    out = bytearray()
    for i in range(0, len(padded), 8):
        byte = 0
        for j in range(8):
            byte = (byte << 1) | padded[i + j]
        out.append(byte)
    return bytes(out)


def autocorrelation(bits: list[int], lag: int) -> float:
    """Compute normalised autocorrelation of *bits* at the given *lag*.

    Returns a value in [-1, 1].  A value near 1 means the stream is highly
    correlated with itself shifted by *lag* positions.
    """
    n = len(bits)
    if n == 0 or lag <= 0 or lag >= n:
        return 0.0

    # Mean
    mean = sum(bits) / n

    # Variance
    var = sum((b - mean) ** 2 for b in bits) / n
    if var == 0.0:
        # Constant stream — perfectly self-similar at every lag.
        return 1.0

    cov = sum((bits[i] - mean) * (bits[i + lag] - mean) for i in range(n - lag)) / (n - lag)
    return cov / var


def has_self_similarity(bits: list[int], threshold: float = 0.7) -> bool:
    """Return True if the stream shows self-similarity at any tested lag.

    Tests a range of lags based on small multiples and powers-of-two to catch
    periodic as well as fractal-like repetition.
    """
    n = len(bits)
    if n < 4:
        return False

    # Try native C path
    if NATIVE_AVAILABLE:
        try:
            ffi = get_ffi()
            lib = get_lib()
            packed = _pack_bits(bits)
            result = lib.has_self_similarity_packed(packed, n, threshold)
            return bool(result)
        except Exception:
            pass

    # Pure Python fallback
    lags: set[int] = set()
    for k in range(1, min(17, n)):
        lags.add(k)
    p = 1
    while p < n:
        lags.add(p)
        p *= 2
    for div in (2, 3, 4, 5, 6, 8):
        frac = n // div
        if frac > 0:
            lags.add(frac)

    for lag in sorted(lags):
        if lag >= n:
            continue
        if abs(autocorrelation(bits, lag)) >= threshold:
            return True
    return False


def box_counting_dimension(bits: list[int]) -> float:
    """Estimate the fractal (box-counting) dimension of a 1-D bit stream.

    The bit stream is treated as a 1-D signal (indices on X, values on Y).
    We overlay boxes of varying size *s* and count the number of occupied boxes.
    The dimension is the negative slope of log(count) vs log(s).

    For a truly random stream the dimension will be close to 1.0; for very
    regular or constant streams it will be lower.
    """
    n = len(bits)
    if n < 4:
        return 0.0

    import math

    # Generate a set of box sizes (powers of 2 up to n//2)
    sizes: list[int] = []
    s = 1
    while s <= n // 2:
        sizes.append(s)
        s *= 2
    if not sizes:
        return 0.0

    log_inv_sizes: list[float] = []
    log_counts: list[float] = []

    for s in sizes:
        # Number of boxes along the x-axis
        nx = (n + s - 1) // s
        # Y is binary so value range is [0, 1].  With box size s the y-grid has
        # at most 2 cells (one for 0, one for 1) when s >= 1.
        # We use a simple occupied-cell counting: for each x-strip we mark the
        # y-cell that contains each bit value.
        occupied: set[tuple[int, int]] = set()
        for i, b in enumerate(bits):
            x_cell = i // s
            # Map bit value to y-cell (0 -> 0, 1 -> 0 or 1 based on scale).
            # At finest resolution each bit is its own y-cell.
            y_cell = b  # either 0 or 1
            occupied.add((x_cell, y_cell))
        count = len(occupied)
        if count > 0:
            log_inv_sizes.append(math.log(1.0 / s))
            log_counts.append(math.log(count))

    if len(log_inv_sizes) < 2:
        return 0.0

    # Simple linear regression: slope of log(count) vs log(1/s)
    n_pts = len(log_inv_sizes)
    mean_x = sum(log_inv_sizes) / n_pts
    mean_y = sum(log_counts) / n_pts
    num = sum((log_inv_sizes[i] - mean_x) * (log_counts[i] - mean_y) for i in range(n_pts))
    den = sum((log_inv_sizes[i] - mean_x) ** 2 for i in range(n_pts))
    if den == 0.0:
        return 0.0
    slope = num / den
    return max(0.0, slope)
