"""CLI interface for maxcomp."""

from __future__ import annotations

import argparse
import logging
import sys

from maxcomp.bitstream import BitStream
from maxcomp.engine import CompressionEngine
from maxcomp.format import CompressedOutput
from maxcomp.stats import format_info, format_stats

WINDOW_PRESETS: dict[str, int] = {
    "small": 8_192,
    "medium": 65_536,
    "default": 100_000,
    "large": 524_288,
    "xlarge": 2_097_152,
    "huge": 8_388_608,
}


def _resolve_window(args: argparse.Namespace) -> int | None:
    """Resolve search window size from CLI args."""
    if hasattr(args, "window_bits") and args.window_bits is not None:
        return args.window_bits
    if hasattr(args, "window") and args.window != "default":
        return WINDOW_PRESETS[args.window]
    return None


def _auto_timeout(precision_bits: int | None, user_timeout: float) -> float:
    """Auto-scale timeout based on window size if user didn't override."""
    if user_timeout != 30.0:
        return user_timeout
    if precision_bits is None:
        return 30.0
    if precision_bits >= 8_388_608:
        return 300.0
    if precision_bits >= 2_097_152:
        return 120.0
    if precision_bits >= 524_288:
        return 60.0
    return 30.0


def _parse_input(args: argparse.Namespace) -> BitStream:
    """Parse input based on CLI arguments."""
    fmt = getattr(args, "format", "auto")

    if hasattr(args, "input") and args.input:
        if fmt == "hex":
            with open(args.input) as f:
                return BitStream.from_hex(f.read())
        elif fmt == "bin":
            with open(args.input) as f:
                return BitStream.from_bin_str(f.read())
        else:
            return BitStream.from_file(args.input)
    else:
        raw = sys.stdin.buffer.read()
        if fmt == "hex":
            return BitStream.from_hex(raw.decode())
        elif fmt == "bin":
            return BitStream.from_bin_str(raw.decode())
        else:
            return BitStream.from_bytes(raw)


def cmd_compress(args: argparse.Namespace) -> None:
    """Handle the compress subcommand."""
    data = _parse_input(args)

    if len(data) == 0:
        print("Error: empty input", file=sys.stderr)
        sys.exit(1)

    precision_bits = _resolve_window(args)
    effective_timeout = _auto_timeout(precision_bits, args.timeout)
    include_chaotic = getattr(args, "enable_chaotic", False)
    tiered_search = getattr(args, "tiered_search", False)

    engine = CompressionEngine(
        timeout_per_strategy=effective_timeout,
        enable_chunking=not args.no_chunking,
        precision_bits=precision_bits,
        tiered_search=tiered_search,
        include_chaotic=include_chaotic,
    )
    output = engine.compress(data, verbose=args.verbose)

    serialized = output.serialize()

    if args.output:
        with open(args.output, "wb") as f:
            f.write(serialized)
        if args.verbose:
            print(format_stats(output), file=sys.stderr)
        if not args.quiet:
            print(
                f"Compressed {len(data)} bits -> {len(serialized)} bytes "
                f"({output.method.name})",
                file=sys.stderr,
            )
    else:
        sys.stdout.buffer.write(serialized)


def cmd_decompress(args: argparse.Namespace) -> None:
    """Handle the decompress subcommand."""
    from maxcomp.strategies import _ensure_registered

    _ensure_registered()

    if hasattr(args, "input") and args.input:
        with open(args.input, "rb") as f:
            raw = f.read()
    else:
        raw = sys.stdin.buffer.read()

    output = CompressedOutput.deserialize(raw)
    restored = output.decompress()

    out_fmt = getattr(args, "out_format", "raw")

    if args.output:
        if out_fmt == "hex":
            with open(args.output, "w") as f:
                f.write(restored.to_bytes().hex())
        elif out_fmt == "bin":
            with open(args.output, "w") as f:
                f.write(restored.to_bin_str())
        else:
            with open(args.output, "wb") as f:
                f.write(restored.to_bytes())
    else:
        if out_fmt == "hex":
            sys.stdout.write(restored.to_bytes().hex())
        elif out_fmt == "bin":
            sys.stdout.write(restored.to_bin_str())
        else:
            sys.stdout.buffer.write(restored.to_bytes())


def cmd_analyze(args: argparse.Namespace) -> None:
    """Handle the analyze subcommand."""
    data = _parse_input(args)

    if len(data) == 0:
        print("Error: empty input", file=sys.stderr)
        sys.exit(1)

    precision_bits = _resolve_window(args)
    effective_timeout = _auto_timeout(precision_bits, args.timeout)
    include_chaotic = getattr(args, "enable_chaotic", False)
    tiered_search = getattr(args, "tiered_search", False)

    engine = CompressionEngine(
        timeout_per_strategy=effective_timeout,
        enable_chunking=not args.no_chunking,
        precision_bits=precision_bits,
        tiered_search=tiered_search,
        include_chaotic=include_chaotic,
    )
    output = engine.compress(data, verbose=args.verbose)

    use_json = getattr(args, "json", False)
    print(format_stats(output, as_json=use_json))


def cmd_info(args: argparse.Namespace) -> None:
    """Handle the info subcommand."""
    from maxcomp.strategies import _ensure_registered

    _ensure_registered()

    with open(args.file, "rb") as f:
        raw = f.read()

    output = CompressedOutput.deserialize(raw)
    print(format_info(output))


def main(argv: list[str] | None = None) -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="maxcomp",
        description="Universal re-compression of arbitrary bit streams",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show detailed progress"
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="Suppress non-error output"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # compress
    p_comp = subparsers.add_parser("compress", help="Compress input")
    p_comp.add_argument("-i", "--input", help="Input file (default: stdin)")
    p_comp.add_argument("-o", "--output", help="Output file (default: stdout)")
    p_comp.add_argument(
        "-f", "--format", choices=["auto", "raw", "hex", "bin"], default="auto",
        help="Input format",
    )
    p_comp.add_argument(
        "--timeout", type=float, default=30.0,
        help="Per-strategy timeout in seconds",
    )
    p_comp.add_argument(
        "--no-chunking", action="store_true",
        help="Disable chunked compression",
    )
    p_comp.add_argument(
        "--window",
        choices=["small", "medium", "default", "large", "xlarge", "huge"],
        default="default",
        help="Search window: small(1KB) medium(8KB) default(~12KB) large(64KB) xlarge(256KB) huge(1MB)",
    )
    p_comp.add_argument(
        "--window-bits", type=int, default=None,
        help="Exact search window in bits (overrides --window)",
    )
    p_comp.add_argument(
        "--tiered-search", action="store_true",
        help="Progressive search: start small, expand if time allows",
    )
    p_comp.add_argument(
        "--enable-chaotic", action="store_true",
        help="Include chaotic sequences (logistic, tent, Lorenz, etc.) in search",
    )
    p_comp.set_defaults(func=cmd_compress)

    # decompress
    p_dec = subparsers.add_parser("decompress", help="Decompress .mxc file")
    p_dec.add_argument("-i", "--input", help="Input .mxc file (default: stdin)")
    p_dec.add_argument("-o", "--output", help="Output file (default: stdout)")
    p_dec.add_argument(
        "-f", "--out-format", choices=["raw", "hex", "bin"], default="raw",
        help="Output format",
    )
    p_dec.set_defaults(func=cmd_decompress)

    # analyze
    p_analyze = subparsers.add_parser("analyze", help="Analyze compression strategies")
    p_analyze.add_argument("-i", "--input", help="Input file (default: stdin)")
    p_analyze.add_argument(
        "-f", "--format", choices=["auto", "raw", "hex", "bin"], default="auto",
        help="Input format",
    )
    p_analyze.add_argument("--json", action="store_true", help="Output as JSON")
    p_analyze.add_argument(
        "--timeout", type=float, default=30.0,
        help="Per-strategy timeout in seconds",
    )
    p_analyze.add_argument(
        "--no-chunking", action="store_true",
        help="Disable chunked compression",
    )
    p_analyze.add_argument(
        "--window",
        choices=["small", "medium", "default", "large", "xlarge", "huge"],
        default="default",
        help="Search window: small(1KB) medium(8KB) default(~12KB) large(64KB) xlarge(256KB) huge(1MB)",
    )
    p_analyze.add_argument(
        "--window-bits", type=int, default=None,
        help="Exact search window in bits (overrides --window)",
    )
    p_analyze.add_argument(
        "--tiered-search", action="store_true",
        help="Progressive search: start small, expand if time allows",
    )
    p_analyze.add_argument(
        "--enable-chaotic", action="store_true",
        help="Include chaotic sequences in search",
    )
    p_analyze.set_defaults(func=cmd_analyze)

    # info
    p_info = subparsers.add_parser("info", help="Show .mxc file metadata")
    p_info.add_argument("file", help="The .mxc file to inspect")
    p_info.set_defaults(func=cmd_info)

    args = parser.parse_args(argv)

    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    args.func(args)
