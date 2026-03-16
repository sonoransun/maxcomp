"""L-system generation and matching for bit streams.

Searches for a simple L-system (small alphabet, short productions) whose
expansion, when mapped to bits, reproduces a target bit sequence.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class LSystemRule:
    """A single production rule: predecessor -> successor."""

    predecessor: str
    successor: str


def generate_lsystem(
    axiom: str,
    rules: list[LSystemRule],
    iterations: int,
) -> str:
    """Expand an L-system starting from *axiom* for *iterations* steps.

    Characters not covered by any rule are passed through unchanged.
    """
    rule_map: dict[str, str] = {r.predecessor: r.successor for r in rules}
    current = axiom
    for _ in range(iterations):
        current = "".join(rule_map.get(ch, ch) for ch in current)
    return current


def _string_to_bits(s: str, mapping: dict[str, str]) -> Optional[list[int]]:
    """Convert an L-system string to bits using *mapping*.

    Returns ``None`` if any character is unmapped.
    """
    parts: list[str] = []
    for ch in s:
        if ch not in mapping:
            return None
        parts.append(mapping[ch])
    bit_str = "".join(parts)
    return [int(c) for c in bit_str]


def _bits_match(a: list[int], b: list[int]) -> bool:
    """Return True if *a* is a prefix of *b* (or exact match) and covers *b*."""
    if len(a) < len(b):
        return False
    return a[: len(b)] == b


def search_lsystem(
    target_bits: list[int],
    max_rules: int = 2,
    max_iterations: int = 8,
    timeout: float = 10.0,
) -> Optional[tuple[str, list[LSystemRule], int, dict[str, str]]]:
    """Search for an L-system that generates *target_bits*.

    Limited to inputs of length < 256 to keep the search tractable.

    Parameters
    ----------
    target_bits : list[int]
        The target bit sequence.
    max_rules : int
        Maximum number of production rules (1 or 2).
    max_iterations : int
        Maximum expansion iterations to try.
    timeout : float
        Wall-clock seconds before giving up.

    Returns
    -------
    (axiom, rules, iterations, bit_mapping) or None
    """
    n = len(target_bits)
    if n == 0 or n >= 256:
        return None

    deadline = time.monotonic() + timeout
    symbols = ["A", "B"]

    # Possible bit mappings: each symbol maps to a short bit string (1-3 bits).
    bit_options = ["0", "1", "00", "01", "10", "11", "000", "001", "010", "011",
                   "100", "101", "110", "111"]

    # Enumerate mappings.
    all_mappings: list[dict[str, str]] = []
    for a_bits in bit_options:
        for b_bits in bit_options:
            all_mappings.append({"A": a_bits, "B": b_bits})

    # Enumerate successor strings of length 1-4 over the alphabet.
    def _successors(max_len: int = 4) -> list[str]:
        result: list[str] = []
        for length in range(1, max_len + 1):
            for combo in itertools.product(symbols, repeat=length):
                result.append("".join(combo))
        return result

    succ_list = _successors()

    # Axiom candidates: short strings of symbols (length 1-3).
    axiom_candidates: list[str] = []
    for length in range(1, 4):
        for combo in itertools.product(symbols, repeat=length):
            axiom_candidates.append("".join(combo))

    # Try 1-rule systems first (just A -> something, B is identity or absent).
    for axiom in axiom_candidates:
        if time.monotonic() > deadline:
            return None
        for succ_a in succ_list:
            if time.monotonic() > deadline:
                return None
            rules_1: list[LSystemRule] = [LSystemRule("A", succ_a)]
            # B stays as B (identity).
            for iters in range(1, max_iterations + 1):
                expanded = generate_lsystem(axiom, rules_1, iters)
                # Bail early if expansion is way too long.
                if len(expanded) > n * 4:
                    break
                for mapping in all_mappings:
                    bits = _string_to_bits(expanded, mapping)
                    if bits is not None and _bits_match(bits, target_bits):
                        return axiom, rules_1, iters, mapping

    if max_rules < 2:
        return None

    # Try 2-rule systems (A -> ..., B -> ...).
    for axiom in axiom_candidates:
        if time.monotonic() > deadline:
            return None
        for succ_a in succ_list:
            if time.monotonic() > deadline:
                return None
            for succ_b in succ_list:
                if time.monotonic() > deadline:
                    return None
                rules_2: list[LSystemRule] = [
                    LSystemRule("A", succ_a),
                    LSystemRule("B", succ_b),
                ]
                for iters in range(1, max_iterations + 1):
                    expanded = generate_lsystem(axiom, rules_2, iters)
                    if len(expanded) > n * 4:
                        break
                    for mapping in all_mappings:
                        bits = _string_to_bits(expanded, mapping)
                        if bits is not None and _bits_match(bits, target_bits):
                            return axiom, rules_2, iters, mapping

    return None
