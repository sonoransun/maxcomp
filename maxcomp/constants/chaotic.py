"""Chaotic sequence generators for exotic compression matching.

Each chaotic map is iterated at arbitrary precision using mpmath, and the
iterates are quantized to bits (threshold or sign-bit method).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mpmath


@dataclass(frozen=True)
class ChaoticMapSpec:
    """Specification for a chaotic map sequence."""

    map_type: str  # "logistic", "tent", "bernoulli", "sine", "gauss", "lorenz", "henon"
    params: dict[str, float] = field(default_factory=dict)
    name: str = ""
    quantize_method: str = "threshold"  # "threshold" or "sign"


def generate_chaotic_bits(spec: ChaoticMapSpec, num_bits: int) -> str:
    """Generate a binary string from a chaotic map specification."""
    generators = {
        "logistic": _generate_logistic,
        "tent": _generate_tent,
        "sine": _generate_sine,
        "gauss": _generate_gauss,
        "lorenz": _generate_lorenz,
        "henon": _generate_henon,
    }
    gen = generators.get(spec.map_type)
    if gen is None:
        raise ValueError(f"Unknown chaotic map type: {spec.map_type}")
    return gen(spec.params, num_bits)


# --------------------------------------------------------------------------
# Logistic map: x_{n+1} = r * x_n * (1 - x_n)
# --------------------------------------------------------------------------

def _generate_logistic(params: dict[str, float], num_bits: int) -> str:
    r_val = params["r"]
    x0_val = params["x0"]
    with mpmath.workdps(num_bits + 50):
        r = mpmath.mpf(str(r_val))
        x = mpmath.mpf(str(x0_val))
        half = mpmath.mpf("0.5")
        bits: list[str] = []
        for _ in range(num_bits):
            x = r * x * (1 - x)
            bits.append("1" if x >= half else "0")
    return "".join(bits)


# --------------------------------------------------------------------------
# Tent map: x_{n+1} = mu * min(x_n, 1 - x_n)
# --------------------------------------------------------------------------

def _generate_tent(params: dict[str, float], num_bits: int) -> str:
    mu_val = params["mu"]
    x0_val = params["x0"]
    with mpmath.workdps(num_bits + 50):
        mu = mpmath.mpf(str(mu_val))
        x = mpmath.mpf(str(x0_val))
        one = mpmath.mpf("1")
        half = mpmath.mpf("0.5")
        bits: list[str] = []
        for _ in range(num_bits):
            x = mu * min(x, one - x)
            bits.append("1" if x >= half else "0")
    return "".join(bits)


# --------------------------------------------------------------------------
# Sine map: x_{n+1} = a * sin(pi * x_n)
# --------------------------------------------------------------------------

def _generate_sine(params: dict[str, float], num_bits: int) -> str:
    a_val = params["a"]
    x0_val = params["x0"]
    with mpmath.workdps(num_bits + 50):
        a = mpmath.mpf(str(a_val))
        x = mpmath.mpf(str(x0_val))
        half = mpmath.mpf("0.5")
        bits: list[str] = []
        for _ in range(num_bits):
            x = a * mpmath.sin(mpmath.pi * x)
            bits.append("1" if x >= half else "0")
    return "".join(bits)


# --------------------------------------------------------------------------
# Gauss iterated map: x_{n+1} = exp(-alpha * x_n^2) + beta
# --------------------------------------------------------------------------

def _generate_gauss(params: dict[str, float], num_bits: int) -> str:
    alpha_val = params["alpha"]
    beta_val = params["beta"]
    x0_val = params["x0"]
    with mpmath.workdps(num_bits + 50):
        alpha = mpmath.mpf(str(alpha_val))
        beta = mpmath.mpf(str(beta_val))
        x = mpmath.mpf(str(x0_val))
        half = mpmath.mpf("0.5")
        bits: list[str] = []
        for _ in range(num_bits):
            x = mpmath.exp(-alpha * x * x) + beta
            bits.append("1" if x >= half else "0")
    return "".join(bits)


# --------------------------------------------------------------------------
# Lorenz system (discretized via RK4)
# dx/dt = sigma*(y-x), dy/dt = x*(rho-z)-y, dz/dt = x*y - beta*z
# --------------------------------------------------------------------------

def _lorenz_deriv(state, sigma, rho, beta_param):
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta_param * z
    return [dx, dy, dz]


def _generate_lorenz(params: dict[str, float], num_bits: int) -> str:
    sigma_val = params.get("sigma", 10.0)
    rho_val = params.get("rho", 28.0)
    beta_val = params.get("beta", 8.0 / 3.0)
    dt_val = params.get("dt", 0.01)
    component = int(params.get("component", 0))  # 0=x, 1=y, 2=z
    x0 = params.get("x0", 1.0)
    y0 = params.get("y0", 1.0)
    z0 = params.get("z0", 1.0)

    # Lorenz doesn't need extreme mpmath precision since we only
    # extract sign bits. Use moderate precision to keep it tractable.
    with mpmath.workdps(50):
        sigma = mpmath.mpf(str(sigma_val))
        rho = mpmath.mpf(str(rho_val))
        beta_p = mpmath.mpf(str(beta_val))
        dt = mpmath.mpf(str(dt_val))
        state = [mpmath.mpf(str(x0)), mpmath.mpf(str(y0)), mpmath.mpf(str(z0))]
        zero = mpmath.mpf("0")

        bits: list[str] = []
        for _ in range(num_bits):
            # RK4 integration step
            k1 = _lorenz_deriv(state, sigma, rho, beta_p)
            s2 = [state[j] + dt / 2 * k1[j] for j in range(3)]
            k2 = _lorenz_deriv(s2, sigma, rho, beta_p)
            s3 = [state[j] + dt / 2 * k2[j] for j in range(3)]
            k3 = _lorenz_deriv(s3, sigma, rho, beta_p)
            s4 = [state[j] + dt * k3[j] for j in range(3)]
            k4 = _lorenz_deriv(s4, sigma, rho, beta_p)
            state = [
                state[j] + dt / 6 * (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j])
                for j in range(3)
            ]
            bits.append("1" if state[component] > zero else "0")
    return "".join(bits)


# --------------------------------------------------------------------------
# Henon map: x_{n+1} = 1 - a*x_n^2 + y_n, y_{n+1} = b*x_n
# --------------------------------------------------------------------------

def _generate_henon(params: dict[str, float], num_bits: int) -> str:
    a_val = params["a"]
    b_val = params["b"]
    x0_val = params.get("x0", 0.1)
    y0_val = params.get("y0", 0.1)
    with mpmath.workdps(num_bits + 50):
        a = mpmath.mpf(str(a_val))
        b = mpmath.mpf(str(b_val))
        x = mpmath.mpf(str(x0_val))
        y = mpmath.mpf(str(y0_val))
        zero = mpmath.mpf("0")
        bits: list[str] = []
        for _ in range(num_bits):
            x_new = 1 - a * x * x + y
            y_new = b * x
            x, y = x_new, y_new
            bits.append("1" if x >= zero else "0")
    return "".join(bits)
