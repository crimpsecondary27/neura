# Neura Language Specification (v0.1.0)

This document is normative for the implementation in `compiler/`. It describes
only features the compiler currently parses, checks and lowers. Constructs listed
under "Not yet implemented" are rejected with diagnostics, never silently ignored.

## 1. Lexical structure

* Source is UTF-8. Line endings `\n`; tabs expand to 4 spaces.
* Indentation defines blocks. A block starts after `:` and must be indented.
  Dedenting closes blocks; mismatched indentation is `E0101`.
* Comments start with `#` and run to end of line.
* Newlines inside `()`, `[]`, `{}` do not terminate statements.
* Identifiers: `[A-Za-z_][A-Za-z0-9_]*`.
* Literals: decimal `Int`, `Float` (`1.0`, `1e-3`), `"strings"` with escapes,
  `true`, `false`.
* Operators: `+ - * / % @ == != < <= > >= && || ! &`, assignment `=`,
  compound-free. `@` is matrix multiplication.
* Delimiters: `( ) [ ] { } , : . ; -> =`.

## 2. Grammar (informal EBNF)

```
program     := item*
item        := import | struct | fn
import      := "import" path ("as" ident)? NEWLINE
path        := ident ("." ident)*
struct      := "struct" ident ":" block(field)
field       := ident ":" type NEWLINE
fn          := "fn" ident "(" params? ")" ("->" type)? ":" block(stmt)
params      := param ("," param)* ","?
param       := ident ":" type | "&" ("mut")? ident ":" type
type        := "Int" | "Float" | "Bool" | "Str" | "Unit"
             | "Tensor" ("[" dims (";" dtype)? "]")?
             | "List" "[" type "]"
             | ident
stmt        := "let" "mut"? ident (":" type)? "=" expr NEWLINE
             | ident "=" expr NEWLINE
             | "if" expr ":" block ("elif" expr ":" block)* ("else" ":" block)?
             | "while" expr ":" block
             | "for" ident "in" expr ":" block
             | "return" expr? NEWLINE
             | "break" | "continue" | "pass"
             | expr NEWLINE
expr        := or
or          := and ("||" and)*
and         := equality ("&&" equality)*
equality    := comparison (("==" | "!=") comparison)*
comparison  := additive (("<" | "<=" | ">" | ">=") additive)*
additive    := multiplicative (("+" | "-") multiplicative)*
multiplicative := unary (("*" | "/" | "%" | "@") unary)*
unary       := ("-" | "!" | "&" "mut"?) unary | postfix
postfix     := primary (("." ident ("(" args? ")")?) | "[" index "]")*
primary     := ident | literal | "(" expr ")" | list | call | "[" expr "]"
list        := "[" (expr ("," expr)* ","?)? "]"
args        := arg ("," arg)* ","?
arg         := (ident "=")? expr
index       := expr | expr? ":" expr?
```

A bare `name = expr` where `name` is not yet bound declares a mutable binding
(Python-style); `let name = expr` declares an immutable binding; `let mut` opts in
to mutation. Assignment to an immutable binding is `E0203`.

## 3. Types

```
Int  Float  Bool  Str  Unit
Tensor[2, 3; f16]      static dimensions
Tensor[N, D; f32]      symbolic dimensions
Tensor                 unknown rank/dtype (f32)
List[T]
struct names, library opaque types (Linear<784,512>, Sequential<10>, Dataset, Adam)
```

* Inference: literals infer `Int`/`Float`/`Bool`/`Str`; arithmetic promotes
  `Int`→`Float`; tensor ops infer and check shapes.
* Tensors support static (`Static(n)`), symbolic (`Symbol("N")`) and unknown
  dimensions. A mismatch is `E0341`/`E0342` with both shapes printed.
* Library calls are typed by *name* (`nn.Linear`, `tensor.matmul`, ...). There is no
  dynamic dispatch.

## 4. Ownership and borrowing

* Scalars are `Copy`; tensors, lists, structs, layers, datasets and optimizers are
  owned values.
* `let y = x` and `return x` move non-Copy values; reuse afterwards is `E0401`.
* `&x` shared, `&mut x` exclusive; exclusive borrow requires a `mut` binding
  (`E0406`), and moving a value while borrowed is `E0402`.
* Passing an owned value to a user-defined function transfers ownership. Calls into
  the standard library borrow their arguments; `model.forward(x)`,
  `optimizer.step()` never consume the model.
* Borrows are call-scoped. Last-use information is computed per local, exposed on
  EIR and used for buffer reuse and `Drop` insertion.
* Tensors needed for backward are marked `needs_autograd`, extending their lifetime
  until the final dependent gradient is produced.

## 5. Tensors and autodiff

Operations (both `x.op(...)` methods and `tensor.op(...)` functions): `+ - * / @`,
`matmul`, `transpose`, `reshape`, `flatten`, `slice`, `sum`, `mean`, `max`, `min`,
`relu`, `gelu`, `sigmoid`, `tanh`, `exp`, `log`, `sqrt`, `softmax`, `conv1d`,
`conv2d`, `gather`, `dropout`, and shape broadcasting.

```neura
x = Tensor([2.0], requires_grad=true)
y = x * x + 3.0 * x
y.backward()
print(x.grad)
```

`y.backward()` seeds `dy/dy = 1`, walks the graph in reverse topological order and
accumulates VJPs into `.grad`. The example prints `Tensor[1](7.0)`. Supported dtypes: `f32`,
`f64`, `f16`, `bf16`, `i8`, `i32`, `i64`, `bool`.

## 6. Standard library surface

```
core builtins     print, input, len, range, int, float, abs, sqrt, exp, log, tanh
neura.tensor      Tensor, scalar, zeros, ones, randn, arange, matmul, add, sub,
                  mul, div, relu, gelu, sigmoid, tanh, exp, log, sqrt, softmax,
                  transpose, reshape, flatten, sum, mean, conv1d, conv2d
neura.nn          Linear, ReLU, GELU, Sigmoid, Softmax, Tanh, Dropout, LayerNorm,
                  BatchNorm, Conv1D, Conv2D, Embedding, Sequential,
                  load_dataset, load_csv, cross_entropy, save, load,
                  parameters, train, forward, predict
neura.optim       SGD, Adam, AdamW, RMSProp  (step, zero_grad)
neura.math        sqrt, exp, log, tanh (scalar or tensor)
neura.data        load_csv, Dataset.batch/shuffle/split/len
neura.autograd    backward, no_grad
neura.device      available, count, default
neura.serialization save, load
```

`input()` reads one line from stdin and returns `Str`. `input(TYPE)` and
`input(prompt, TYPE)` parse that line as `Int`, `Float`, `Bool` or `Str`:

```neura
fn main():
    age: Int = input("age: ", Int)
    ratio = input(Float)
    print(age + 1)
```

## 7. Diagnostics

Structured diagnostics carry a severity, stable code, message, byte span, notes and
an optional help line, and convert to JSON.

| Code | Meaning |
| --- | --- |
| E0101 | inconsistent indentation |
| E0102 | invalid token |
| E0201 | duplicate declaration |
| E0203 | assignment to immutable binding |
| E0300 | unknown variable |
| E0301 | unknown type |
| E0311 | unsupported assignment target |
| E0331 | value is not indexable |
| E0335 | can only borrow a named variable |
| E0340 | unknown method |
| E0341 | invalid tensor operation / shape |
| E0342 | layer input-width mismatch |
| E0343 | wrong library argument type |
| E0401 | use of moved value |
| E0402 | move while borrowed |
| E0406 | mutable borrow of immutable binding |
| E0501 | unsupported callee |
| E0601 | unknown function / library call |

## 8. Program entry and CLI

Execution starts at `fn main()` (no parameters). The toolchain provides:

```
neura check FILE            parse + typecheck + ownership check
neura build FILE            write FILE.neurabc (checked source + header) and cache it
neura run FILE [--opt=...]  compile, optimise and execute main
neura inspect FILE          types and locals per function, plus EIR disassembly
neura graph FILE            TIR as textual MLIR (neura_tir dialect)
neura profile FILE          per-operation execution counts and time
neura optimize-report FILE  NAOE decisions as JSON or text
neura fmt FILE              canonical formatting (--check to verify only)
neura version               compiler version
```

Policies: `--opt=auto|aggressive|debug|off` (default `auto`), `--device=cpu|cuda|rocm|auto`,
`--precision=strict|mixed_bf16|mixed_fp16` (default `strict`).

Other flags: `--output=PATH` (`build`), `--json` (machine-readable output),
`--trace` (print each executed instruction), `--quiet`, `--training` (run layers
in training mode).

A `.neurabc` artifact is self-contained: a header records the source hash,
program name and policies, and the body is canonical source. Loading an artifact
re-verifies the hash, so `build` and `run` cannot silently diverge.

Packages are managed by `npack`:

```
npack init DIR              create npack.toml, src/main.neura, tests/
npack install [DIR]         resolve dependencies and write npack.lock
npack build [DIR]           compile to target/<name>.neurabc
npack run [DIR]             build and execute the entry point
npack test [DIR]            compile and run every tests/*.neura file
```

`npack.toml` uses `[package]` (`name`, `version`, `entry`) and `[dependencies]`
with `name = { path = "..." }` or a semver requirement (`"1.2.3"`, `"^1.2.3"`,
`"~1.2.3"`, `"*"`). Version-only dependencies resolve from the offline cache
(`$NPACK_HOME`, default `~/.npack`); there is no network access in this build, and
a missing cached package is reported as such.

## 9. Not yet implemented

* User-defined generic types and traits; `struct` methods.
* Closures and first-class function values (`E0501`).
* Mutable fields/elements directly (`E0311`); use whole-value assignment.
* GPU kernel generation (interfaces exist; see ARCHITECTURE.md).
* Linked LLVM native object emission; `.neurabc` plus MLIR text is produced instead.
* Importing symbols from third-party `npack` packages: the module system currently
  resolves the `neura.*` standard library only. `npack` still resolves, locks and
  build-verifies dependencies.
* Registry downloads (no network access in this build).
