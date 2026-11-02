<p align="center">
  <img src="assets/neura-logo.png" alt="Neura" width="420">
</p>

# Neura

A compiled, statically typed programming language for numerical and machine-learning
work, implemented in Rust.

Neura has its own lexer, parser, type system, ownership checker, four IR layers,
optimising compiler, reverse-mode autodiff engine, CPU tensor engine, neural-network
standard library and toolchain. It is not a wrapper around Python or PyTorch, and no
part of the runtime requires a Python installation.

```
source (.neura)
  → lexer → parser → type checker → ownership checker
  → NIR → TIR → ADIR → EIR
  → NAOE optimisation (graph, fusion, memory, precision, device, autotuning)
  → CPU execution (register-machine VM + native tensor kernels)
```

Version `0.1.0`.

## Quick start

```sh
cargo build --release
./target/release/neura run examples/autograd.neura
# Tensor[1](7.0)

./target/release/neura run examples/tensors.neura
./target/release/neura run examples/train_classifier.neura
```

Or run through Cargo:

```sh
cargo run -q -p neura-cli -- run examples/autograd.neura
```

## The language

Indentation-based blocks, static types with inference, tensors with symbolic shapes,
and Rust-style ownership for tensor values.

```neura
import neura.nn as nn
import neura.optim as optim
import neura.data as data

fn main():
    x = Tensor([2.0], requires_grad=true)
    y = x * x + 3.0 * x
    y.backward()
    print(x.grad)              # Tensor[1](7.0)

    train = data.load_csv("examples/datasets/blobs.csv")
    model = nn.Sequential([nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 3)])
    optimizer = optim.Adam(model.parameters(), learning_rate=0.05)
    model.train(train, epochs=20, batch_size=8, optimizer=optimizer)

    print(model.predict(Tensor([[0.0, 0.0], [2.0, 2.0]])))
    model.save("examples/blobs.neuramodel")
```

Supported: functions, `let [mut]` bindings, `if`/`elif`/`else`, `while`, `for`,
`return`/`break`/`continue`, structs, imports, keyword arguments, slices, `&`/`&mut`
borrows, tensor literals, `Tensor[2, 3, f32]` types with `dyn`/symbolic dimensions,
and automatic shape inference through layers.

Full grammar and semantics: [LANGUAGE_SPEC.md](LANGUAGE_SPEC.md).

## Toolchain

| Command | Purpose |
| --- | --- |
| `neura check FILE` | parse, type-check, ownership-check |
| `neura build FILE` | write a hash-verified `.neurabc` artifact (default `FILE.neurabc`) |
| `neura run FILE` | compile, optimise and execute `main` |
| `neura inspect FILE` | inferred types, locals and EIR disassembly |
| `neura graph FILE` | TIR as textual MLIR (`neura_tir` dialect) |
| `neura profile FILE` | per-operation counts, time and memory accounting |
| `neura optimize-report FILE` | NAOE decisions and before/after instruction counts |
| `neura fmt FILE` | canonical formatting (`--check` to verify only) |

Flags: `--opt=auto|aggressive|debug|off`, `--device=auto|cpu|cuda|rocm`,
`--precision=strict|mixed_bf16|mixed_fp16`, `--output=PATH`, `--json`, `--trace`,
`--quiet`, `--training`.

`--json` makes output machine-readable for agents and CI. `neura graph` emits real
MLIR textual form so an installed `mlir-opt`/`llc` toolchain can consume it.

## Packages: `npack`

```sh
npack init my_project          # npack.toml, src/main.neura, tests/
npack install my_project       # resolve dependencies, write npack.lock
npack build my_project         # target/<name>.neurabc
npack run my_project
npack test my_project          # compile and run every tests/*.neura
```

`npack.toml`:

```toml
[package]
name = "my_project"
version = "0.1.0"
entry = "src/main.neura"

[dependencies]
local = { path = "../local" }
pinned = "^0.1.0"
```

Dependencies are resolved recursively (cycles and version conflicts are errors),
locked deterministically, and the build refuses to run against a stale lockfile.
Version-only dependencies resolve from the offline cache (`$NPACK_HOME`, default
`~/.npack`); there is no network access, and a missing cached package is an
explicit error rather than a silent skip.

## Standard library

- `neura.tensor` — creation, arithmetic, matmul, transpose, reshape, slicing,
  broadcasting, reductions, softmax, convolution, gather.
- `neura.nn` — Linear, ReLU, GELU, Sigmoid, Tanh, Softmax, LayerNorm, BatchNorm,
  Dropout, Conv1D, Conv2D, Embedding, Sequential.
- `neura.optim` — SGD (momentum), Adam, AdamW, RMSProp.
- `neura.data` — CSV datasets, deterministic batching, shuffling, splits.
- `neura.autograd` — `backward`, gradient inspection, finite-difference checks.

## Tests and benchmarks

```sh
cargo test --workspace
cargo run --release -p neura-runtime --example benchmarks
```

75 automated tests cover the lexer, parser, type checker, ownership checker, IR,
optimiser, tensor numerics, autograd gradients, layers, optimisers, formatter, CLI,
`npack`, plus subprocess end-to-end tests that run the real `neura` binary
(`cli/tests/e2e.rs`): the required autograd example, tensor ops, MLIR output,
build→run artifacts, diagnostics, formatting, and training with save/load parity.

Benchmarks are measured at run time; no numbers are hard-coded. On the development
machine (aarch64, release): matmul 256³ f32 ≈ 1.7 GFLOP/s, GELU 1M f32
≈ 86 Melem/s, softmax 128×1000 ≈ 98 Melem/s, conv2d 4×8×32×32 ≈ 0.46 GFLOP/s, and
a 64-sample training step at ≈ 6.3k samples/s. These are unoptimised reference
kernels, not vendor-BLAS figures.

## Repository layout

```
compiler/   lexer, parser, AST, diagnostics, type checker, ownership, lowering
ir/         NIR, TIR, ADIR, EIR, shape algebra, MLIR text emission
optimizer/  NAOE passes: graph, fusion, memory, precision, scheduling, autotuning
backends/   CPU tensor engine, CUDA/ROCm interfaces, device discovery
runtime/    EIR VM, autograd, layers, optimizers, datasets, caches, profiler
cli/        the `neura` driver (+ end-to-end tests)
tools/      npack package manager, canonical formatter
examples/   runnable programs and a CSV dataset
```

More detail: [ARCHITECTURE.md](ARCHITECTURE.md), [ROADMAP.md](ROADMAP.md).

## Requirements

- Rust (edition 2021). `cargo build` needs nothing else — there are no third-party
  crates and no Python, PyTorch, LLVM, CUDA or ROCm dependency.
- To consume the emitted MLIR text, install an MLIR/LLVM toolchain separately;
  Neura does not re-implement MLIR.

## Limitations (accurate, not simulated)

- No linked LLVM/MLIR code generation on this host: programs execute on the
  EIR interpreter with native Rust CPU kernels, and MLIR text is emitted for
  external toolchains.
- CUDA and ROCm backends are interfaces: probing reports the missing runtime and
  every allocation/launch returns an explicit error instead of pretending to run.
- The compiler's module system covers `neura.*` only, so symbols from third-party
  `npack` packages cannot be imported yet (dependencies are still resolved, locked
  and build-verified).
- No registry downloads.
- No user-defined generics/traits, struct methods, or closures.
- 
**Editor support:** [Neura for VS Code](https://github.com/Azozrav4/NeuraVSCodePLugin) — syntax highlighting, live diagnostics, build/run, and formatting.
  
## License

MIT
