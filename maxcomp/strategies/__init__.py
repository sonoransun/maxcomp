"""Strategy registry for compression methods."""

from __future__ import annotations

from maxcomp.strategies.base import CompressionStrategy, MethodID

_REGISTRY: dict[MethodID, CompressionStrategy] = {}
_ALL_STRATEGIES: list[CompressionStrategy] = []
_INITIALIZED = False


def register(strategy: CompressionStrategy) -> CompressionStrategy:
    """Register a strategy instance for its method IDs."""
    for method_id in strategy.method_ids():
        _REGISTRY[method_id] = strategy
    _ALL_STRATEGIES.append(strategy)
    return strategy


def get_all_strategies() -> list[CompressionStrategy]:
    """Return all registered strategies."""
    _ensure_registered()
    return list(_ALL_STRATEGIES)


def get_decompressor(method: MethodID) -> CompressionStrategy:
    """Get the strategy that can decompress the given method ID."""
    _ensure_registered()
    if method not in _REGISTRY:
        raise ValueError(f"No decompressor registered for method {method!r}")
    return _REGISTRY[method]


def _ensure_registered() -> None:
    """Import all strategy modules to trigger registration."""
    global _INITIALIZED
    if _INITIALIZED:
        return
    _INITIALIZED = True
    import maxcomp.strategies.identity  # noqa: F401
    import maxcomp.strategies.constant_match  # noqa: F401
    import maxcomp.strategies.fractal  # noqa: F401
    import maxcomp.strategies.generative  # noqa: F401
