"""Math expression enumeration for generative compression."""

from __future__ import annotations

import math
import time
from typing import Optional


def _fibonacci(n: int) -> int:
    """Compute the nth Fibonacci number."""
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def _sieve_primes(limit: int) -> list[int]:
    """Simple sieve of Eratosthenes."""
    if limit < 2:
        return []
    sieve = bytearray(b"\x01") * (limit + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = 0
    return [i for i in range(2, limit + 1) if sieve[i]]


def _int_to_binstr(value: int) -> str:
    """Convert a non-negative integer to binary string (no '0b' prefix)."""
    if value == 0:
        return "0"
    return bin(value)[2:]


def evaluate_expression(expr: str) -> int:
    """Evaluate a math expression string and return its integer value.

    Supported forms:
        2^N-1, N!, fib(N), prime(N), B^E
    """
    expr = expr.strip()

    if expr.startswith("fib(") and expr.endswith(")"):
        n = int(expr[4:-1])
        return _fibonacci(n)

    if expr.startswith("prime(") and expr.endswith(")"):
        n = int(expr[6:-1])
        primes = _sieve_primes(n * 20)  # generous upper bound
        if n - 1 < len(primes):
            return primes[n - 1]  # 1-indexed
        raise ValueError(f"prime({n}) out of range")

    if "^" in expr and "-" in expr:
        # 2^N-1 form
        parts = expr.split("-")
        if len(parts) == 2:
            base_exp = parts[0].strip()
            sub = int(parts[1].strip())
            base, exp = base_exp.split("^")
            return int(base.strip()) ** int(exp.strip()) - sub

    if expr.endswith("!"):
        n = int(expr[:-1])
        return math.factorial(n)

    if "^" in expr:
        base, exp = expr.split("^")
        return int(base.strip()) ** int(exp.strip())

    # Fallback: try direct integer parse
    return int(expr)


def search_expression(
    target_int: int,
    target_bit_length: int,
    timeout: float = 30.0,
) -> Optional[tuple[str, int, int]]:
    """Search for a math expression whose binary representation contains the target.

    Args:
        target_int: The target value as an integer.
        target_bit_length: Bit length of the target.
        timeout: Maximum time in seconds.

    Returns:
        Tuple of (expression_string, bit_offset, bit_length) or None.
        The shortest expression that works is returned.
    """
    if target_bit_length == 0:
        return None

    target_binstr = bin(target_int)[2:].zfill(target_bit_length)
    start_time = time.monotonic()

    candidates: list[tuple[str, int]] = []  # (expr, value)

    # 2^n - 1 for n=1..1024
    for n in range(1, 1025):
        if (time.monotonic() - start_time) > timeout:
            break
        val = (1 << n) - 1
        candidates.append((f"2^{n}-1", val))

    # n! for n=1..100
    for n in range(1, 101):
        if (time.monotonic() - start_time) > timeout:
            break
        candidates.append((f"{n}!", math.factorial(n)))

    # fib(n) for n=1..1000
    a, b = 0, 1
    for n in range(1, 1001):
        if (time.monotonic() - start_time) > timeout:
            break
        a, b = b, a + b
        if a > 0:
            candidates.append((f"fib({n})", a))

    # Powers of small bases: b^e for b in 2..10, e in 2..200
    for base in range(2, 11):
        for exp in range(2, 201):
            if (time.monotonic() - start_time) > timeout:
                break
            candidates.append((f"{base}^{exp}", base**exp))

    # Primes concatenated (individual primes as values)
    primes = _sieve_primes(10000)
    for i, p in enumerate(primes[:1000], 1):
        if (time.monotonic() - start_time) > timeout:
            break
        candidates.append((f"prime({i})", p))

    # Search for the shortest expression matching the target
    best: Optional[tuple[str, int, int]] = None

    for expr, val in candidates:
        if (time.monotonic() - start_time) > timeout:
            break
        if val <= 0:
            continue

        val_binstr = _int_to_binstr(val)
        if len(val_binstr) < target_bit_length:
            continue

        idx = val_binstr.find(target_binstr)
        if idx >= 0:
            # Compute expression string length as a heuristic for "shortness"
            expr_cost = len(expr.encode("utf-8")) + 8  # expr + offset + length fields
            if best is None or expr_cost < len(best[0].encode("utf-8")) + 8:
                best = (expr, idx, target_bit_length)

    return best
