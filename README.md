# maxcomp

Universal re-compression of arbitrary bit streams. Explores whether data can be represented more compactly as references into mathematical constants, chaotic sequences, fractal self-similarity, or short generative programs.

## How It Works

maxcomp tries 8 compression strategies in parallel against your input data, picks whichever produces the smallest output, and wraps the result in a `.mxc` file that can be losslessly decompressed back to the original.

```mermaid
graph LR
    Input["Input\n(file, stdin, hex)"] --> Engine

    subgraph Engine["Compression Engine"]
        direction TB
        S1["Constant Match\npi, e, sqrt2, chaotic..."]
        S2["Fractal IFS\nself-similar blocks"]
        S3["L-System\nproduction rules"]
        S4["PRNG Seed\nMT19937, xorshift, LCG"]
        S5["Hash Preimage\nMD5, SHA1, SHA256"]
        S6["Expression\n2^n-1, n!, fib(n)"]
        S7["Cellular Automata\n256 rules x 3 widths"]
        S8["Identity\nraw fallback"]
    end

    Engine --> Select["Pick Smallest"]
    Select --> Output[".mxc File"]
```

The core idea: if your data happens to appear as a substring in pi starting at bit 47,293 — store `(pi, 47293, length)` in 17 bytes instead of the original data. The same principle applies to chaotic sequences, PRNG outputs, hash digests, and more.

## Installation

```bash
git clone https://github.com/martinpeck/maxcomp.git
cd maxcomp
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

**Optional — native C acceleration (macOS arm64):**

```bash
pip install cffi
cd maxcomp/native && make && cd ../..
```

This builds `_maxcomp_native.dylib` with ARM NEON SIMD and crypto intrinsics, providing up to 18,000x speedup on brute-force searches.

## Quick Start

```bash
# Compress a file
maxcomp compress -i data.bin -o data.mxc

# Decompress back
maxcomp decompress -i data.mxc -o restored.bin

# Analyze all strategies without writing output
maxcomp analyze -i data.bin --json

# Inspect a compressed file
maxcomp info data.mxc
```

## Compression Strategies

```mermaid
graph TD
    subgraph Strategies["8 Compression Strategies"]
        direction LR

        subgraph Sequence["Sequence Matching"]
            CM["Constant Match\n5 constants + 20 chaotic\nsequences"]
        end

        subgraph Fractal["Fractal Analysis"]
            IFS["IFS\nblock matching\nwith inversion"]
            LS["L-System\nrule search\n< 256 bits"]
        end

        subgraph Generative["Generative Search"]
            PRNG["PRNG Seeds\n3 generators\n65k seeds"]
            Hash["Hash Preimage\n3 algorithms\n16.7M candidates"]
            Expr["Expressions\n2^n, n!, fib\n~4k candidates"]
            CA["Cell. Automata\n256 rules\n3 widths"]
        end

        ID["Identity\nraw fallback"]
    end

    CM --> |"17 bytes"|R["Result"]
    IFS --> |"transforms"|R
    LS --> |"rules + axiom"|R
    PRNG --> |"type + seed"|R
    Hash --> |"type + preimage"|R
    Expr --> |"expression"|R
    CA --> |"rule + width"|R
    ID --> |"raw bytes"|R
```

### Strategy Details

| # | Strategy | What it does | Payload |
|---|----------|-------------|---------|
| 0 | **Identity** | Store raw bits (always succeeds) | Original bytes |
| 1 | **Constant Match** | Find input as substring in pi, e, logistic map, Lorenz, etc. | 17 bytes |
| 2 | **Fractal IFS** | Detect self-similar block mappings with optional inversion | Variable |
| 3 | **L-System** | Find production rules that generate the bit pattern | Variable |
| 4 | **PRNG Seed** | Brute-force MT19937/xorshift128/LCG seeds | 17 bytes |
| 5 | **Hash Preimage** | Find short input whose MD5/SHA1/SHA256 prefix matches | Variable |
| 6 | **Expression** | Match against 2^n-1, n!, fib(n), primes, powers | Variable |
| 7 | **Cellular Automata** | Search 256 elementary CA rules across 3 widths | Variable |

## Searchable Sequences

The constant match strategy searches for input data as a substring within 25 binary sequences:

```mermaid
graph TD
    subgraph Classic["Classic Constants (5)"]
        PI["π"] --- E["e"] --- SQ["√2"] --- PHI["φ"] --- LN["ln2"]
    end

    subgraph Chaotic["Chaotic Sequences (20) — opt-in via --enable-chaotic"]
        subgraph LogMap["Logistic Map (5)\nx → r·x·(1-x)"]
            L1["r=3.57"] --- L2["r=3.8"] --- L3["r=3.9"] --- L4["r=4.0 x₀=0.1"] --- L5["r=4.0 x₀=0.6"]
        end
        subgraph TentMap["Tent Map (3)\nx → μ·min(x,1-x)"]
            T1["x₀=0.1"] --- T2["x₀=0.3"] --- T3["x₀=0.7"]
        end
        subgraph Bern["Bernoulli Shift (3)\nx → 2x mod 1"]
            B1["π frac"] --- B2["e frac"] --- B3["√2 frac"]
        end
        subgraph SineMap["Sine Map (3)\nx → a·sin(πx)"]
            S1["a=1.0"] --- S2["a=0.9"]
        end
        subgraph GaussMap["Gauss Map (2)\nx → e^(-αx²)+β"]
            G1["α=4.9"] --- G2["α=6.2"]
        end
        subgraph LorenzSys["Lorenz System (3)\nRK4, σ=10 ρ=28"]
            LX["x comp"] --- LY["y comp"] --- LZ["z comp"]
        end
        subgraph HenonMap["Henon Map (1)\nx→1-ax²+y"]
            H1["a=1.4 b=0.3"]
        end
    end
```

All sequences are computed at arbitrary precision using mpmath and cached to disk (`~/.cache/maxcomp/`) for reuse.

## Search Windows

Control how deep into each sequence to search:

| Preset | Size | Bits | CLI Flag |
|--------|------|------|----------|
| small | 1 KB | 8,192 | `--window small` |
| medium | 8 KB | 65,536 | `--window medium` |
| default | ~12 KB | 100,000 | *(default)* |
| large | 64 KB | 524,288 | `--window large` |
| xlarge | 256 KB | 2,097,152 | `--window xlarge` |
| huge | 1 MB | 8,388,608 | `--window huge` |

Timeouts auto-scale with window size (30s for small/default, up to 5 minutes for huge). Use `--tiered-search` for progressive expansion — tries the smallest window first and only expands if time remains.

```bash
# Search pi/e/etc up to 64KB deep with chaotic sequences enabled
maxcomp compress -i data.bin -o data.mxc --window large --enable-chaotic

# Progressive search up to 1MB
maxcomp compress -i data.bin -o data.mxc --window huge --tiered-search

# Exact window size
maxcomp compress -i data.bin -o data.mxc --window-bits 500000
```

## Chunked Compression

For larger inputs, maxcomp splits the data into chunks and compresses each independently, picking the best strategy per chunk:

```mermaid
graph LR
    Input["Input Stream"] --> Chunker

    subgraph Chunker["Chunker"]
        C1["Chunk 1\n(64-8192 bits)"]
        C2["Chunk 2"]
        C3["Chunk 3"]
        CN["Chunk N"]
    end

    C1 --> |"Best: PRNG"|R1["seed=42"]
    C2 --> |"Best: Constant"|R2["pi@offset 891"]
    C3 --> |"Best: Identity"|R3["raw bits"]
    CN --> |"Best: IFS"|RN["transforms"]

    R1 --> Composite[".mxc\nCHUNKED_COMPOSITE"]
    R2 --> Composite
    R3 --> Composite
    RN --> Composite
```

**Default chunk sizes:** 64, 128, 256, 512, 1024, 2048, 4096, 8192 bits.
**Extended** (for `--window large` and above): adds 16384, 32768, 65536 bits.

## Native Acceleration

On macOS arm64, the native C library provides massive speedups via ARM NEON SIMD, ARM crypto extensions, and optional Metal GPU compute:

```mermaid
graph TD
    subgraph Dispatch["Python Dispatcher"]
        Check{"Native\navailable?"}
    end

    Check -->|Yes| GPU{"GPU\navailable?"}
    Check -->|No| Python["Pure Python\n(fallback)"]

    GPU -->|Yes| Metal["Metal GPU\n(65k+ threads)"]
    GPU -->|No| CNEON["C + NEON SIMD"]

    Metal --> Result["Result"]
    CNEON --> Result
    Python --> Result

    subgraph Accelerated["Native C Kernels"]
        P["PRNG Search\nNEON 2/4-wide"]
        H["Hash Search\nARM SHA2 crypto"]
        I["IFS Matching\nvcntq popcount"]
        A["Automata\nvtbl1 rule lookup"]
        AC["Autocorrelation\npacked bit ops"]
    end
```

| Component | Technique | Speedup |
|-----------|-----------|---------|
| PRNG xorshift search | NEON `uint64x2_t` (2 seeds/cycle) | ~18,000x |
| PRNG LCG search | NEON `uint32x4_t` (4 seeds/cycle) | ~10,000x |
| SHA-256 hashing | ARM crypto (`vsha256hq_u32`) | ~5-10x |
| IFS block matching | NEON `vcntq_u8` popcount | ~100x |
| Automata rules | NEON `vtbl1_u8` lookup | ~50x |

Build with `cd maxcomp/native && make`. GPU shaders: `make metal`.

## .mxc File Format

```
┌─────────────────────────────────────────┐
│  Header (16 bytes)                      │
│  ┌──────────┬───────────────────────┐   │
│  │ Magic    │ MXC\0  (4 bytes)      │   │
│  │ Version  │ 1      (1 byte)       │   │
│  │ Flags    │ 0x01 = chunked        │   │
│  │ Method   │ MethodID (1 byte)     │   │
│  │ Reserved │ 0x00                  │   │
│  │ Orig Len │ bit count (uint64 BE) │   │
│  └──────────┴───────────────────────┘   │
├─────────────────────────────────────────┤
│  Payload (variable)                     │
│  Method-specific compressed data        │
└─────────────────────────────────────────┘
```

## CLI Reference

```
maxcomp compress  [-i FILE] [-o FILE] [-f auto|raw|hex|bin]
                  [--timeout N] [--no-chunking]
                  [--window small|medium|default|large|xlarge|huge]
                  [--window-bits N] [--tiered-search]
                  [--enable-chaotic]

maxcomp decompress [-i FILE] [-o FILE] [-f raw|hex|bin]

maxcomp analyze   [-i FILE] [-f auto|raw|hex|bin] [--json]
                  [--timeout N] [--no-chunking]
                  [--window ...] [--enable-chaotic]

maxcomp info FILE
```

## Examples

```bash
# Compress from hex string
echo "deadbeefcafebabe0123456789abcdef" | maxcomp compress -f hex -o out.mxc

# Analyze with all strategies including chaotic sequences
maxcomp analyze -i data.bin --enable-chaotic --window medium --json

# Compress with large search window and progressive tiers
maxcomp compress -i data.bin -o data.mxc --window xlarge --tiered-search -v

# Round-trip verification
maxcomp compress -i original.bin -o compressed.mxc
maxcomp decompress -i compressed.mxc -o restored.bin
diff original.bin restored.bin  # should be identical
```

## Project Structure

```
maxcomp/
├── maxcomp/
│   ├── cli.py                    # CLI (compress, decompress, analyze, info)
│   ├── engine.py                 # Parallel strategy orchestration
│   ├── bitstream.py              # Immutable bit sequence type
│   ├── format.py                 # .mxc binary format
│   ├── chunker.py                # Chunk splitting/reassembly
│   ├── stats.py                  # Statistics reporting
│   ├── strategies/               # 4 strategy modules (8 methods)
│   ├── constants/                # 5 constants + 20 chaotic sequences
│   ├── generators/               # PRNG, hash, expression, automata search
│   ├── fractal/                  # IFS, L-system, autocorrelation
│   └── native/                   # C/NEON/Metal acceleration layer
├── tests/                        # 123 tests
├── pyproject.toml
└── LICENSE
```

## Running Tests

```bash
pytest tests/ -v
```

## License

MIT
