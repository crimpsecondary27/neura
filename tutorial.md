# Neura Tutorial

A hands-on introduction to Neura 0.1.0: a compiled, statically typed language for
numerical and machine-learning work. This tutorial covers the language basics,
tensors, automatic differentiation, training a model, the CLI, and packages.

All examples are runnable from the repository root.

## 1. Build the toolchain

Neura is written in Rust and has no third-party crate, Python, PyTorch, LLVM, CUDA
or ROCm dependency.

```sh
cargo build --release
```

This produces two binaries:

- `./target/release/neura` — compiler, runner and inspector
- `./target/release/npack` — package manager

Verify the build:

```sh
./target/release/neura version
./target/release/neura run examples/autograd.neura
# Tensor[1](7.0)
```

You can also run through Cargo without building a release binary:

```sh
cargo run -q -p neura-cli -- run examples/autograd.neura
```

## 2. Your first program

Create `hello.neura`:

```neura
fn main():
    print("Hello, Neura!")
```

Run it:

```sh
./target/release/neura run hello.neura
```

Execution always starts at `fn main()` with no parameters. Blocks are defined by
indentation: a `:` opens a block, and the block must be indented. Tabs expand to
four spaces. Comments start with `#`.

## 3. Variables and types

Bindings are inferred, or annotated explicitly:

```neura
fn main():
    let name = "Neura"        # immutable
    let mut count = 0         # mutable
    count = count + 1
    let scale: Float = 2.5
    print(count)
    print(scale)
```

Rules:

- `let name = expr` declares an immutable binding.
- `let mut name = expr` declares a mutable binding.
- A bare `name = expr` for an unbound name declares a mutable binding
  (Python-style), as used throughout the examples.
- Assigning to an immutable binding is diagnostic `E0203`.

Built-in scalar types are `Int`, `Float`, `Bool`, `Str` and `Unit`. Arithmetic
promotes `Int` to `Float`. `List[T]` is a homogeneous list:

Arithmetic operators are `+`, `-`, `*`, `/`, `//` (floor division), `%` and
`**` (exponentiation, right-associative). `x += 1`, `-=`, `*=`, `/=`, `//=`,
`%=` and `**=` update a variable in place. Integer `**` rejects negative
exponents; use a `Float` base for those.

```neura
fn main():
    let mut total = 2 ** 10      # 1024
    total //= 3                  # floor division
    print(total)
    print(-7 // 2)               # -4
```

```neura
fn main():
    values = [1, 2, 3]
    total = 0
    for v in values:
        total = total + v
    print(total)              # 6
```

## 4. Functions

Functions declare parameter types and an optional return type:

```neura
fn square(x: Float) -> Float:
    return x * x

fn main():
    print(square(3.0))        # 9.0
```

A function without `->` returns `Unit`.

Neura uses Rust-style ownership for non-`Copy` values such as tensors, lists,
structs, layers and optimizers:

- Passing an owned value to a user-defined function moves it.
- Scalars are copied.
- Standard-library calls borrow their arguments, so `model.forward(x)` and
  `optimizer.step()` do not consume the model.
- `&x` is a shared borrow; `&mut x` is an exclusive borrow and requires a `mut`
  binding.

## 5. Control flow

`if` / `elif` / `else`, `while`, `for`, `break` and `continue` work as expected:

```neura
fn classify(n: Int) -> Str:
    if n < 0:
        return "negative"
    elif n == 0:
        return "zero"
    else:
        return "positive"

fn main():
    for i in range(5):
        print(classify(i - 2))

    n = 0
    while n < 10:
        n = n + 1
        if n == 3:
            continue
        if n == 6:
            break
        print(n)
```

`range(n)` produces integer values for `for` loops, and `len(value)` returns the
length of a list, string, tensor or dataset.

## 6. Structs

Structs group named, typed fields:

```neura
struct Point:
    x: Float
    y: Float

fn norm(p: Point) -> Float:
    return sqrt(p.x * p.x + p.y * p.y)

fn main():
    p = Point(3.0, 4.0)
    print(norm(p))            # 5.0
```

Fields are initialized positionally. Struct methods, generics and traits are not
implemented in 0.1.0.

## 7. Tensors

Tensors are created from nested lists. Arithmetic supports scalars and
broadcasting, and `@` is matrix multiplication:

```neura
fn main():
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = a @ a
    print(b)
    print(b.shape)            # [2, 2]
    print(a.transpose(0, 1))
    print(a + 10.0)
    print(a.reshape(4))
    print(a.sum())
    print(a.mean())
```

Tensor methods include `matmul`, `transpose`, `reshape`, `flatten`, `slice`,
`sum`, `mean`, `max`, `min`, `relu`, `gelu`, `sigmoid`, `tanh`, `exp`, `log`,
`sqrt`, `softmax`, `conv1d`, `conv2d`, `gather` and `dropout`. The same
operations are available as functions such as `matmul(a, b)`.

Slicing uses `start:end`, with either bound optional:

```neura
fn main():
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    c = a.reshape(4)
    print(c[:2])              # first two elements
```

Types can describe tensor rank, dimensions and dtype:

```neura
fn scale(x: Tensor[2, 3; f32]) -> Tensor[2, 3; f32]:
    return x * 2.0
```

Symbolic dimensions such as `Tensor[N, D; f32]`, unknown shapes (`Tensor`) and
dtype inference are supported. A shape mismatch is reported as `E0341` or
`E0342`, with both shapes printed.

## 8. Automatic differentiation

Set `requires_grad=true`, build an expression, then call `backward()` on the
result. Gradients accumulate into `.grad`:

```neura
fn main():
    x = Tensor([2.0], requires_grad=true)
    y = x * x + 3.0 * x
    y.backward()
    print(x.grad)             # Tensor[1](7.0)
```

`backward()` seeds the output gradient with 1, walks the graph in reverse
topological order, and stores each variable's gradient. For `y = x² + 3x`, the
derivative is `2x + 3`, which is `7` at `x = 2`.

Autodiff works through tensor operations and neural-network layers, so gradients
flow from a loss back into layer parameters.

## 9. Training a model

The standard library provides layers (`neura.nn`), optimizers (`neura.optim`) and
datasets (`neura.data`). A complete training run:

```neura
import neura.nn as nn
import neura.optim as optim
import neura.data as data

fn main():
    train = data.load_csv("examples/datasets/blobs.csv")
    model = nn.Sequential([
        nn.Linear(2, 8),
        nn.ReLU(),
        nn.Linear(8, 3),
    ])
    optimizer = optim.Adam(model.parameters(), learning_rate=0.05)
    model.train(train, epochs=20, batch_size=8, optimizer=optimizer)

    sample = Tensor([[0.0, 0.0], [2.0, 2.0], [-2.0, 2.0]])
    print(model.predict(sample))

    model.save("examples/blobs.neuramodel")
    restored = nn.Sequential([
        nn.Linear(2, 8),
        nn.ReLU(),
        nn.Linear(8, 3),
    ])
    restored.load("examples/blobs.neuramodel")
    print(restored.predict(sample))
```

The CSV file has two features and three classes. `nn.Linear` checks its input
width against the previous layer, so an incompatible architecture fails at
compile time (`E0342`) rather than at run time.

Available layers include `Linear`, `ReLU`, `GELU`, `Sigmoid`, `Tanh`, `Softmax`,
`LayerNorm`, `BatchNorm`, `Dropout`, `Conv1D`, `Conv2D`, `Embedding` and
`Sequential`. Optimizers are `SGD` (with momentum), `Adam`, `AdamW` and
`RMSProp`. `neura.data` provides CSV loading, deterministic batching, shuffling
and splits.

Dataset methods:

```neura
train = data.load_csv("examples/datasets/blobs.csv")
print(len(train))
batches = train.batch(8)
shuffled = train.shuffle()
parts = train.split(0.8)
```

## 10. The `neura` command line

| Command | Purpose |
| --- | --- |
| `neura check FILE` | parse, type-check and ownership-check only |
| `neura build FILE` | write a hash-verified `.neurabc` artifact |
| `neura run FILE` | compile, optimize and execute `main` |
| `neura inspect FILE` | inferred types, locals and EIR disassembly |
| `neura graph FILE` | emit TIR as textual MLIR (`neura_tir` dialect) |
| `neura profile FILE` | per-operation counts, time and memory |
| `neura optimize-report FILE` | optimizer decisions, before/after instruction counts |
| `neura fmt FILE` | canonical formatting (`--check` verifies only) |
| `neura version` | compiler version |

Typical session:

```sh
./target/release/neura check examples/train_classifier.neura
./target/release/neura run examples/tensors.neura
./target/release/neura inspect examples/autograd.neura
./target/release/neura graph examples/tensors.neura
./target/release/neura fmt examples/tensors.neura --check
./target/release/neura profile examples/tensors.neura
```

Policies:

```sh
./target/release/neura run examples/train_classifier.neura \
    --opt=aggressive \
    --precision=mixed_bf16 \
    --device=cpu
```

Flags: `--opt=auto|aggressive|debug|off`, `--device=auto|cpu|cuda|rocm`,
`--precision=strict|mixed_bf16|mixed_fp16`, `--output=PATH`, `--json`,
`--trace`, `--quiet` and `--training`.

`--json` produces machine-readable output for CI and editor integrations.
`--trace` prints each executed EIR instruction.

A `.neurabc` artifact contains a header with the source hash, program name and
policies, plus canonical source. Loading re-verifies the hash, so builds cannot
silently diverge from the source they were produced from.

## 11. Packages with `npack`

Create a project, build it, run it and test it:

```sh
./target/release/npack init my_project
./target/release/npack install my_project
./target/release/npack build my_project
./target/release/npack run my_project
./target/release/npack test my_project
```

`npack init` writes `npack.toml`, `src/main.neura` and a `tests/` directory.
`npack test` compiles and runs every `tests/*.neura` file.

`npack.toml` describes the package and its dependencies:

```toml
[package]
name = "my_project"
version = "0.1.0"
entry = "src/main.neura"

[dependencies]
local = { path = "../local" }
pinned = "^0.1.0"
```

Dependencies are resolved recursively and locked deterministically in
`npack.lock`. Version-only dependencies come from the offline cache
(`$NPACK_HOME`, default `~/.npack`). There is no network access: a missing
cached package is an explicit error. The build refuses to run against a stale
lockfile.

Note: dependency packages are resolved, locked and build-verified, but their
symbols cannot be imported yet. The compiler's module system currently covers
the `neura.*` standard library only.

## 12. Editor support

The VS Code extension in `editors/vscode-neura` provides syntax highlighting,
snippets, diagnostics from `neura check --json`, formatting through `neura fmt`,
and run/build commands for `.neura` files.

## 13. Common diagnostics

| Code | Meaning |
| --- | --- |
| `E0101` | inconsistent indentation |
| `E0102` | invalid token |
| `E0201` | duplicate declaration |
| `E0203` | assignment to immutable binding |
| `E0300` | unknown variable |
| `E0301` | unknown type |
| `E0311` | unsupported assignment target |
| `E0331` | value is not indexable |
| `E0335` | can only borrow a named variable |
| `E0340` | unknown method |
| `E0341` | invalid tensor operation or shape |
| `E0342` | layer input-width mismatch |
| `E0343` | wrong library argument type |
| `E0401` | use of moved value |
| `E0402` | move while borrowed |
| `E0406` | mutable borrow of immutable binding |
| `E0501` | unsupported callee |
| `E0601` | unknown function or library call |

Two frequent cases:

```neura
fn main():
    let x = 1
    x = 2                     # E0203: x is immutable
```

```neura
fn consume(t: Tensor) -> Tensor:
    return t * 2.0

fn main():
    a = Tensor([1.0, 2.0])
    b = consume(a)            # a is moved here
    print(a)                  # E0401: use of moved value
```

Use `let mut` for mutation, and pass `&a` when a function should borrow rather
than consume a value.

## 14. Current limitations

- Programs execute on the EIR interpreter with native Rust CPU kernels. There is
  no linked LLVM/MLIR code generation; MLIR text is emitted for external tools.
- CUDA and ROCm are interfaces: missing runtimes are reported explicitly and
  device operations return errors instead of pretending to run.
- No user-defined generics, traits, struct methods or closures.
- No mutable field/element assignment; assign whole values instead.
- No importing symbols from third-party `npack` packages.
- No registry downloads or network access.

## Where to go next

- `LANGUAGE_SPEC.md` — normative grammar, types, ownership and diagnostics.
- `ARCHITECTURE.md` — compiler, IR layers, optimizer and runtime internals.
- `examples/` — runnable programs: `autograd.neura`, `tensors.neura`,
  `train_classifier.neura`, `vscode_extension.neura`.
- `README.md` — toolchain summary, testing and benchmarks.

Run the test suite and benchmarks:

```sh
cargo test --workspace
cargo run --release -p neura-runtime --example benchmarks
```
