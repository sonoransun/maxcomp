"""Catalog of mathematical constants and chaotic sequences."""

from __future__ import annotations

from enum import IntEnum
from typing import Callable

import mpmath

from maxcomp.constants.chaotic import ChaoticMapSpec


class ConstantID(IntEnum):
    """Identifies a mathematical constant or chaotic sequence."""

    # Classic mathematical constants (0-4)
    PI = 0
    E = 1
    SQRT2 = 2
    PHI = 3
    LN2 = 4

    # Logistic map family (16-20): x_{n+1} = r * x * (1-x)
    LOGISTIC_R3_57_X0_0_1 = 16
    LOGISTIC_R3_8_X0_0_3 = 17
    LOGISTIC_R3_9_X0_0_7 = 18
    LOGISTIC_R4_0_X0_0_1 = 19
    LOGISTIC_R4_0_X0_0_6 = 20

    # Tent map family (24-26): x_{n+1} = mu * min(x, 1-x)
    TENT_MU2_X0_0_1 = 24
    TENT_MU2_X0_0_3 = 25
    TENT_MU2_X0_0_7 = 26

    # Bernoulli shift / doubling map (32-34): x_{n+1} = 2x mod 1
    BERNOULLI_PI_FRAC = 32
    BERNOULLI_E_FRAC = 33
    BERNOULLI_SQRT2_FRAC = 34

    # Sine map family (40-42): x_{n+1} = a * sin(pi * x)
    SINE_A1_0_X0_0_1 = 40
    SINE_A1_0_X0_0_5 = 41
    SINE_A0_9_X0_0_3 = 42

    # Gauss iterated map (48-49): x_{n+1} = exp(-alpha * x^2) + beta
    GAUSS_A4_9_BN0_1_X0_0_5 = 48
    GAUSS_A6_2_BN0_1_X0_0_5 = 49

    # Lorenz system (56-58): discretized ODE, sign-bit quantize
    LORENZ_STD_X = 56
    LORENZ_STD_Y = 57
    LORENZ_STD_Z = 58

    # Henon map (64): x_{n+1} = 1 - a*x^2 + y, y_{n+1} = b*x
    HENON_STD = 64


def _phi() -> mpmath.mpf:
    return (1 + mpmath.sqrt(5)) / 2


# Pure mathematical constants (IDs 0-4) — computed via mpmath real-number expansion
CONSTANT_FUNCTIONS: dict[ConstantID, Callable[[], mpmath.mpf]] = {
    ConstantID.PI: lambda: mpmath.pi,
    ConstantID.E: lambda: mpmath.e,
    ConstantID.SQRT2: lambda: mpmath.sqrt(2),
    ConstantID.PHI: _phi,
    ConstantID.LN2: lambda: mpmath.ln(2),
}

# Human-readable names (used for cache file paths)
CONSTANT_NAMES: dict[ConstantID, str] = {
    # Classic constants
    ConstantID.PI: "pi",
    ConstantID.E: "e",
    ConstantID.SQRT2: "sqrt2",
    ConstantID.PHI: "phi",
    ConstantID.LN2: "ln2",
    # Logistic
    ConstantID.LOGISTIC_R3_57_X0_0_1: "logistic_r3.57_x0.1",
    ConstantID.LOGISTIC_R3_8_X0_0_3: "logistic_r3.8_x0.3",
    ConstantID.LOGISTIC_R3_9_X0_0_7: "logistic_r3.9_x0.7",
    ConstantID.LOGISTIC_R4_0_X0_0_1: "logistic_r4.0_x0.1",
    ConstantID.LOGISTIC_R4_0_X0_0_6: "logistic_r4.0_x0.6",
    # Tent
    ConstantID.TENT_MU2_X0_0_1: "tent_mu2_x0.1",
    ConstantID.TENT_MU2_X0_0_3: "tent_mu2_x0.3",
    ConstantID.TENT_MU2_X0_0_7: "tent_mu2_x0.7",
    # Bernoulli
    ConstantID.BERNOULLI_PI_FRAC: "bernoulli_pi_frac",
    ConstantID.BERNOULLI_E_FRAC: "bernoulli_e_frac",
    ConstantID.BERNOULLI_SQRT2_FRAC: "bernoulli_sqrt2_frac",
    # Sine
    ConstantID.SINE_A1_0_X0_0_1: "sine_a1.0_x0.1",
    ConstantID.SINE_A1_0_X0_0_5: "sine_a1.0_x0.5",
    ConstantID.SINE_A0_9_X0_0_3: "sine_a0.9_x0.3",
    # Gauss
    ConstantID.GAUSS_A4_9_BN0_1_X0_0_5: "gauss_a4.9_b-0.1_x0.5",
    ConstantID.GAUSS_A6_2_BN0_1_X0_0_5: "gauss_a6.2_b-0.1_x0.5",
    # Lorenz
    ConstantID.LORENZ_STD_X: "lorenz_std_x",
    ConstantID.LORENZ_STD_Y: "lorenz_std_y",
    ConstantID.LORENZ_STD_Z: "lorenz_std_z",
    # Henon
    ConstantID.HENON_STD: "henon_std",
}

# Chaotic sequence specifications (IDs 16+)
CHAOTIC_SPECS: dict[ConstantID, ChaoticMapSpec] = {
    # Logistic map
    ConstantID.LOGISTIC_R3_57_X0_0_1: ChaoticMapSpec("logistic", {"r": 3.57, "x0": 0.1}, "logistic_r3.57_x0.1"),
    ConstantID.LOGISTIC_R3_8_X0_0_3: ChaoticMapSpec("logistic", {"r": 3.8, "x0": 0.3}, "logistic_r3.8_x0.3"),
    ConstantID.LOGISTIC_R3_9_X0_0_7: ChaoticMapSpec("logistic", {"r": 3.9, "x0": 0.7}, "logistic_r3.9_x0.7"),
    ConstantID.LOGISTIC_R4_0_X0_0_1: ChaoticMapSpec("logistic", {"r": 4.0, "x0": 0.1}, "logistic_r4.0_x0.1"),
    ConstantID.LOGISTIC_R4_0_X0_0_6: ChaoticMapSpec("logistic", {"r": 4.0, "x0": 0.6}, "logistic_r4.0_x0.6"),
    # Tent map
    ConstantID.TENT_MU2_X0_0_1: ChaoticMapSpec("tent", {"mu": 2.0, "x0": 0.1}, "tent_mu2_x0.1"),
    ConstantID.TENT_MU2_X0_0_3: ChaoticMapSpec("tent", {"mu": 2.0, "x0": 0.3}, "tent_mu2_x0.3"),
    ConstantID.TENT_MU2_X0_0_7: ChaoticMapSpec("tent", {"mu": 2.0, "x0": 0.7}, "tent_mu2_x0.7"),
    # Bernoulli shift — these delegate to the corresponding classic constant in provider.py
    # (included here for completeness; provider handles them specially)
    # Sine map
    ConstantID.SINE_A1_0_X0_0_1: ChaoticMapSpec("sine", {"a": 1.0, "x0": 0.1}, "sine_a1.0_x0.1"),
    ConstantID.SINE_A1_0_X0_0_5: ChaoticMapSpec("sine", {"a": 1.0, "x0": 0.5}, "sine_a1.0_x0.5"),
    ConstantID.SINE_A0_9_X0_0_3: ChaoticMapSpec("sine", {"a": 0.9, "x0": 0.3}, "sine_a0.9_x0.3"),
    # Gauss map
    ConstantID.GAUSS_A4_9_BN0_1_X0_0_5: ChaoticMapSpec("gauss", {"alpha": 4.9, "beta": -0.1, "x0": 0.5}, "gauss_a4.9_b-0.1_x0.5"),
    ConstantID.GAUSS_A6_2_BN0_1_X0_0_5: ChaoticMapSpec("gauss", {"alpha": 6.2, "beta": -0.1, "x0": 0.5}, "gauss_a6.2_b-0.1_x0.5"),
    # Lorenz system
    ConstantID.LORENZ_STD_X: ChaoticMapSpec("lorenz", {"sigma": 10.0, "rho": 28.0, "beta": 8.0 / 3.0, "dt": 0.01, "component": 0, "x0": 1.0, "y0": 1.0, "z0": 1.0}, "lorenz_std_x", "sign"),
    ConstantID.LORENZ_STD_Y: ChaoticMapSpec("lorenz", {"sigma": 10.0, "rho": 28.0, "beta": 8.0 / 3.0, "dt": 0.01, "component": 1, "x0": 1.0, "y0": 1.0, "z0": 1.0}, "lorenz_std_y", "sign"),
    ConstantID.LORENZ_STD_Z: ChaoticMapSpec("lorenz", {"sigma": 10.0, "rho": 28.0, "beta": 8.0 / 3.0, "dt": 0.01, "component": 2, "x0": 1.0, "y0": 1.0, "z0": 1.0}, "lorenz_std_z", "sign"),
    # Henon map
    ConstantID.HENON_STD: ChaoticMapSpec("henon", {"a": 1.4, "b": 0.3, "x0": 0.1, "y0": 0.1}, "henon_std", "sign"),
}

# Groupings for filtering
CLASSIC_CONSTANTS: frozenset[ConstantID] = frozenset({
    ConstantID.PI, ConstantID.E, ConstantID.SQRT2, ConstantID.PHI, ConstantID.LN2,
})

# Bernoulli entries map to their underlying classic constant
BERNOULLI_DELEGATES: dict[ConstantID, ConstantID] = {
    ConstantID.BERNOULLI_PI_FRAC: ConstantID.PI,
    ConstantID.BERNOULLI_E_FRAC: ConstantID.E,
    ConstantID.BERNOULLI_SQRT2_FRAC: ConstantID.SQRT2,
}
