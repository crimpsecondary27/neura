//! The Neura execution VM.
//!
//! The VM executes the register-based EIR bytecode produced by the compiler.
//! Every tensor instruction goes through the autograd-aware `apply` entry
//! point, so gradients are recorded while the program runs and `backward()`
//! can later walk the graph.
//!
//! The VM is also the place where the standard library is realised: `nn.*`,
//! `optim.*`, `tensor.*`, `math.*` and `device.*` calls are resolved by their
//! canonical (statically checked) names. There is no dynamic dispatch table
//! shared with Python.

use crate::autograd::{self, apply};
use crate::data::Dataset;
use crate::layers::{self, Activation, Layer};
use crate::optim::{self, OptKind};
use crate::support::{MemoryManager, MemoryStats, Profiler};
use crate::value::{tensor as tensor_value, tensor_from_value, trainable, TRef, TNode, Value};
use neura_backends::device;
use neura_backends::tensor::Tensor;
use neura_ir::eir::{EirFunction, EirProgram, Inst, Reg};
use neura_ir::ops::{BinOp, FusedExpr, FusedSpec, TensorOp, UnOp};
use neura_ir::types::{Dim, DType, TensorTy, Ty};
use std::collections::HashMap;
use std::rc::Rc;
use std::time::Instant;

#[derive(Clone, Debug)]
pub struct VmOptions {
    /// Print every instruction before executing it.
    pub trace: bool,
    /// Suppress `print` output on stdout (the output is still captured).
    pub quiet: bool,
    /// Run layers in training mode (dropout, batch statistics).
    pub training: bool,
    /// Collect per-instruction timings.
    pub profile: bool,
}

impl Default for VmOptions {
    fn default() -> Self {
        VmOptions { trace: false, quiet: false, training: false, profile: false }
    }
}

pub struct RunOutcome {
    pub value: Value,
    /// Everything the program printed, in order.
    pub output: Vec<String>,
    /// Human-readable profiler report (empty when profiling is disabled).
    pub report: String,
    pub memory: MemoryStats,
    pub steps: u64,
}

/// Run the program's `main` function.
pub fn run(program: &EirProgram, options: VmOptions) -> Result<RunOutcome, String> {
    let mut m = Machine {
        program,
        options,
        output: Vec::new(),
        profiler: Profiler::new(false),
        memory: MemoryManager::default(),
        rng: program.source_hash | 1,
        steps: 0,
        depth: 0,
    };
    m.profiler = Profiler::new(m.options.profile);
    let main = program.main;
    let value = m.call_function(main, &[])?;
    Ok(RunOutcome {
        value,
        output: m.output,
        report: m.profiler.report(),
        memory: m.memory.stats(),
        steps: m.steps,
    })
}

// ---------------------------------------------------------------------------
// Frames
// ---------------------------------------------------------------------------

enum IterState {
    Range(i64, i64),
    Items(Vec<Value>, usize),
}

struct Frame {
    regs: Vec<Value>,
    iters: HashMap<Reg, IterState>,
}

impl Frame {
    fn new(n: usize) -> Frame {
        Frame { regs: vec![Value::Unit; n], iters: HashMap::new() }
    }
}

enum Flow {
    Next,
    Jump(usize),
    Return(Value),
}

// ---------------------------------------------------------------------------
// Machine
// ---------------------------------------------------------------------------

struct Machine<'p> {
    program: &'p EirProgram,
    options: VmOptions,
    output: Vec<String>,
    profiler: Profiler,
    memory: MemoryManager,
    rng: u64,
    steps: u64,
    depth: usize,
}

const STEP_LIMIT: u64 = 50_000_000;
const DEPTH_LIMIT: usize = 512;

impl<'p> Machine<'p> {
    fn next_seed(&mut self) -> u64 {
        self.rng = self.rng.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut x = self.rng;
        x ^= x >> 30;
        x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        x ^= x >> 27;
        x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
        x ^ (x >> 31)
    }

    fn read(&self, frame: &Frame, r: Reg) -> Value {
        frame.regs.get(r as usize).cloned().unwrap_or(Value::Unit)
    }

    fn write(&mut self, frame: &mut Frame, r: Reg, v: Value) {
        if let Value::Tensor(t) = &v {
            let data = t.borrow().data.clone();
            self.memory.assign(r, &data);
        }
        if (r as usize) < frame.regs.len() {
            frame.regs[r as usize] = v;
        } else {
            frame.regs.resize(r as usize + 1, Value::Unit);
            frame.regs[r as usize] = v;
        }
    }

    fn call_function(&mut self, idx: usize, args: &[Value]) -> Result<Value, String> {
        let program = self.program;
        let f = program
            .functions
            .get(idx)
            .ok_or_else(|| format!("no function with index {}", idx))?;
        if args.len() != f.params.len() {
            return Err(format!(
                "function `{}` expects {} arguments, found {}",
                f.name,
                f.params.len(),
                args.len()
            ));
        }
        if self.depth >= DEPTH_LIMIT {
            return Err(format!("call depth limit reached while calling `{}`", f.name));
        }
        let mut frame = Frame::new(f.types.len());
        for (r, v) in f.params.iter().zip(args.iter()) {
            self.write(&mut frame, *r, v.clone());
        }
        self.depth += 1;
        let out = self.exec(f, &mut frame);
        self.depth -= 1;
        out
    }

    fn exec(&mut self, f: &'p EirFunction, frame: &mut Frame) -> Result<Value, String> {
        let mut pc = 0usize;
        while pc < f.code.len() {
            self.steps += 1;
            if self.steps > STEP_LIMIT {
                return Err(format!(
                    "execution step limit ({}) exceeded in `{}`; the program may not terminate",
                    STEP_LIMIT, f.name
                ));
            }
            if self.options.trace {
                let line = format!("[{} .L{}] {}", f.name, pc, disasm(&f.code[pc]));
                if !self.options.quiet {
                    println!("{}", line);
                }
                self.output.push(line);
            }
            let started = if self.options.profile { Some(Instant::now()) } else { None };
            let flow = self
                .step_inst(&f.code[pc], frame)
                .map_err(|e| format!("runtime error in `{}` at .L{}: {}", f.name, pc, e))?;
            if let Some(t0) = started {
                self.profiler.record(inst_key(&f.code[pc]), t0.elapsed().as_nanos());
            }
            match flow {
                Flow::Next => pc += 1,
                Flow::Jump(t) => {
                    if t > f.code.len() {
                        return Err(format!("jump target .L{} is outside the function", t));
                    }
                    pc = t;
                }
                Flow::Return(v) => return Ok(v),
            }
        }
        Ok(Value::Unit)
    }

    /// Execute a single instruction. The instruction is passed explicitly so
    /// the loop in `exec` keeps ownership of `pc`.
    fn step_inst(&mut self, inst: &Inst, frame: &mut Frame) -> Result<Flow, String> {
        match inst {
            Inst::ConstF64 { dst, v } => {
                self.write(frame, *dst, Value::Float(*v));
                Ok(Flow::Next)
            }
            Inst::ConstInt { dst, v } => {
                self.write(frame, *dst, Value::Int(*v));
                Ok(Flow::Next)
            }
            Inst::ConstBool { dst, v } => {
                self.write(frame, *dst, Value::Bool(*v));
                Ok(Flow::Next)
            }
            Inst::ConstStr { dst, s } => {
                self.write(frame, *dst, Value::Str(s.clone()));
                Ok(Flow::Next)
            }
            Inst::ConstUnit { dst } => {
                self.write(frame, *dst, Value::Unit);
                Ok(Flow::Next)
            }
            Inst::Copy { dst, src } | Inst::Move { dst, src } => {
                let v = self.read(frame, *src);
                self.write(frame, *dst, v);
                if matches!(inst, Inst::Move { .. }) && src != dst {
                    self.write(frame, *src, Value::Unit);
                    self.memory.release(*src);
                }
                Ok(Flow::Next)
            }
            Inst::MakeList { dst, elems } => {
                let vals: Vec<Value> = elems.iter().map(|r| self.read(frame, *r)).collect();
                self.write(frame, *dst, Value::list(vals));
                Ok(Flow::Next)
            }
            Inst::MakeTensor { dst, values, requires_grad, ty } => {
                let v = self.read(frame, *values);
                let data = build_tensor(&v, ty)?;
                let node = if *requires_grad { trainable(data) } else { tensor_value(data) };
                self.write(frame, *dst, Value::Tensor(node));
                Ok(Flow::Next)
            }
            Inst::MakeStruct { dst, struct_id, fields } => {
                let names = self
                    .program
                    .structs
                    .get(*struct_id)
                    .map(|s| s.fields.clone())
                    .unwrap_or_default();
                let mut out = Vec::with_capacity(fields.len());
                for (i, r) in fields.iter().enumerate() {
                    let name = names.get(i).cloned().unwrap_or_else(|| format!("{}", i));
                    out.push((name, self.read(frame, *r)));
                }
                let sname = self
                    .program
                    .structs
                    .get(*struct_id)
                    .map(|s| s.name.clone())
                    .unwrap_or_else(|| "struct".into());
                self.write(frame, *dst, Value::Struct { name: sname, fields: out });
                Ok(Flow::Next)
            }
            Inst::TensorOp { dst, op, args, dyn_dims, ty } => {
                let mut trefs = Vec::with_capacity(args.len());
                for r in args {
                    trefs.push(self.tensor_at(frame, *r)?);
                }
                let op = resolve_dynamic(op, dyn_dims, frame)?;
                let out = self.apply_op(&op, &trefs)?;
                // The compiler's declared element type is authoritative: this
                // is where an opt-in mixed-precision plan takes effect.
                let out = if out.borrow().data.dtype != ty.dtype {
                    crate::autograd::cast(&out, ty.dtype)
                } else {
                    out
                };
                self.write(frame, *dst, Value::Tensor(out));
                Ok(Flow::Next)
            }
            Inst::FusedOp { dst, spec, args, ty } => {
                let mut trefs = Vec::with_capacity(args.len());
                for r in args {
                    trefs.push(self.tensor_at(frame, *r)?);
                }
                let out = self.fused(spec, &trefs, ty)?;
                self.write(frame, *dst, Value::Tensor(out));
                Ok(Flow::Next)
            }
            Inst::BinOp { dst, op, lhs, rhs, ty } => {
                let a = self.read(frame, *lhs);
                let b = self.read(frame, *rhs);
                let v = binary(*op, &a, &b, ty)?;
                self.write(frame, *dst, v);
                Ok(Flow::Next)
            }
            Inst::UnOp { dst, op, src, ty } => {
                let a = self.read(frame, *src);
                let v = unary(*op, &a, ty)?;
                self.write(frame, *dst, v);
                Ok(Flow::Next)
            }
            Inst::Call { dst, func, args, .. } => {
                let vals: Vec<Value> = args.iter().map(|r| self.read(frame, *r)).collect();
                let out = self.call_function(*func, &vals)?;
                self.write(frame, *dst, out);
                Ok(Flow::Next)
            }
            Inst::CallBuiltin { dst, name, args, kwargs, .. } => {
                let vals: Vec<Value> = args.iter().map(|r| self.read(frame, *r)).collect();
                let kws: Vec<(String, Value)> = kwargs
                    .iter()
                    .map(|(k, r)| (k.clone(), self.read(frame, *r)))
                    .collect();
                let out = self.call_builtin(name, vals, kws)?;
                self.write(frame, *dst, out);
                Ok(Flow::Next)
            }
            Inst::MethodCall { dst, recv, method, args, kwargs, .. } => {
                let r = self.read(frame, *recv);
                let vals: Vec<Value> = args.iter().map(|x| self.read(frame, *x)).collect();
                let kws: Vec<(String, Value)> = kwargs
                    .iter()
                    .map(|(k, x)| (k.clone(), self.read(frame, *x)))
                    .collect();
                let out = self.call_method(r, method, vals, kws)?;
                self.write(frame, *dst, out);
                Ok(Flow::Next)
            }
            Inst::GetAttr { dst, obj, name, .. } => {
                let o = self.read(frame, *obj);
                let v = get_attr(&o, name)?;
                self.write(frame, *dst, v);
                Ok(Flow::Next)
            }
            Inst::Index { dst, obj, index, .. } => {
                let o = self.read(frame, *obj);
                let i = self.read(frame, *index);
                let v = index_value(&o, &i)?;
                self.write(frame, *dst, v);
                Ok(Flow::Next)
            }
            Inst::Len { dst, obj } => {
                let o = self.read(frame, *obj);
                let n = length_of(&o)?;
                self.write(frame, *dst, Value::Int(n));
                Ok(Flow::Next)
            }
            Inst::IterInit { dst, iter } => {
                let v = self.read(frame, *iter);
                let state = match v {
                    Value::Range(a, b) => IterState::Range(a, b),
                    Value::List(l) => IterState::Items(l.borrow().clone(), 0),
                    Value::Tensor(t) => {
                        let data = t.borrow().data.clone();
                        let mut items = Vec::with_capacity(data.shape.first().copied().unwrap_or(0));
                        if data.rank() == 0 {
                            items.push(Value::Tensor(t.clone()));
                        } else {
                            for i in 0..data.shape[0] {
                                let row = data.slice(0, i, i + 1)?.reshape(
                                    &data.shape[1..]
                                        .iter()
                                        .map(|d| Dim::Static(*d))
                                        .collect::<Vec<_>>(),
                                )?;
                                items.push(Value::Tensor(tensor_value(row)));
                            }
                        }
                        IterState::Items(items, 0)
                    }
                    Value::Str(s) => {
                        IterState::Items(s.chars().map(|c| Value::Str(c.to_string())).collect(), 0)
                    }
                    other => {
                        return Err(format!("cannot iterate over {}", other.type_name()));
                    }
                };
                frame.iters.insert(*dst, state);
                Ok(Flow::Next)
            }
            Inst::IterNext { dst, state, end } => match frame.iters.get_mut(state) {
                Some(IterState::Range(cur, limit)) => {
                    if cur >= limit {
                        return Ok(Flow::Jump(*end));
                    }
                    let v = *cur;
                    *cur += 1;
                    self.write(frame, *dst, Value::Int(v));
                    Ok(Flow::Next)
                }
                Some(IterState::Items(items, pos)) => {
                    if *pos >= items.len() {
                        return Ok(Flow::Jump(*end));
                    }
                    let v = items[*pos].clone();
                    *pos += 1;
                    self.write(frame, *dst, v);
                    Ok(Flow::Next)
                }
                None => Err("iteration state is missing (internal lowering error)".into()),
            },
            Inst::Jump(t) => Ok(Flow::Jump(*t)),
            Inst::JumpIfFalse { cond, target } => {
                let c = self.read(frame, *cond);
                if c.as_bool()? {
                    Ok(Flow::Next)
                } else {
                    Ok(Flow::Jump(*target))
                }
            }
            Inst::JumpIfTrue { cond, target } => {
                let c = self.read(frame, *cond);
                if c.as_bool()? {
                    Ok(Flow::Jump(*target))
                } else {
                    Ok(Flow::Next)
                }
            }
            Inst::Return(r) => {
                let v = match r {
                    Some(r) => self.read(frame, *r),
                    None => Value::Unit,
                };
                Ok(Flow::Return(v))
            }
            Inst::Drop(r) => {
                self.memory.release(*r);
                self.write(frame, *r, Value::Unit);
                Ok(Flow::Next)
            }
            Inst::Print(r) => {
                let v = self.read(frame, *r);
                let text = v.display();
                if !self.options.quiet {
                    println!("{}", text);
                }
                self.output.push(text);
                Ok(Flow::Next)
            }
            Inst::Input { dst, prompt, ty } => {
                if let Some(r) = prompt {
                    let text = self.read(frame, *r).as_str()?;
                    if !self.options.quiet {
                        print!("{}", text);
                        let _ = std::io::Write::flush(&mut std::io::stdout());
                    }
                }
                let line = read_line()?;
                let v = parse_input(&line, ty)?;
                self.write(frame, *dst, v);
                Ok(Flow::Next)
            }
        }
    }

    fn tensor_at(&self, frame: &Frame, r: Reg) -> Result<TRef, String> {
        match self.read(frame, r) {
            Value::Tensor(t) => Ok(t),
            Value::Int(i) => Ok(tensor_value(Tensor::from_i64(DType::I64, vec![], vec![i])?)),
            Value::Float(f) => Ok(tensor_value(Tensor::from_f64(DType::F64, vec![], vec![f])?)),
            Value::Bool(b) => Ok(tensor_value(Tensor::from_i64(
                DType::Bool,
                vec![],
                vec![b as i64],
            )?)),
            other => Err(format!("expected a tensor, found {}", other.type_name())),
        }
    }

    fn apply_op(&mut self, op: &TensorOp, args: &[TRef]) -> Result<TRef, String> {
        let seed = self.next_seed();
        apply(op, args, self.options.training, seed)
    }

    fn fused(&mut self, spec: &FusedSpec, args: &[TRef], ty: &TensorTy) -> Result<TRef, String> {
        if args.iter().any(|a| a.borrow().requires_grad) {
            // Gradients need a graph, so rebuild the equivalent elementary
            // operations. Numerically this is exactly the same pixel path.
            return rebuild_fused(&spec.expr, args);
        }
        let mut shape: Vec<usize> = Vec::new();
        for a in args {
            let s = a.borrow().data.shape.clone();
            shape = Tensor::broadcast_pair(&shape, &s)?;
        }
        let n: usize = shape.iter().product();
        let mut vals = Vec::with_capacity(n);
        let mut idx = vec![0usize; shape.len()];
        for flat in 0..n {
            let mut rem = flat;
            for d in (0..shape.len()).rev() {
                idx[d] = rem % shape[d].max(1);
                rem /= shape[d].max(1);
            }
            let inputs: Vec<f64> = args.iter().map(|a| take_broadcast(&a.borrow().data, &idx)).collect();
            vals.push(spec.expr.eval(&inputs));
        }
        let data = Tensor::from_f64(ty.dtype, shape, vals)?;
        Ok(tensor_value(data))
    }

    // -- standard library --------------------------------------------------

    fn call_builtin(
        &mut self,
        name: &str,
        args: Vec<Value>,
        kwargs: Vec<(String, Value)>,
    ) -> Result<Value, String> {
        match name {
            "len" => Ok(Value::Int(length_of(args.first().ok_or("`len` expects 1 argument")?)?)),
            "range" => {
                let a = args.first().ok_or("`range` expects 1 or 2 arguments")?.as_i64()?;
                let b = match args.get(1) {
                    Some(v) => v.as_i64()?,
                    None => 0,
                };
                Ok(if args.len() > 1 { Value::Range(a, b) } else { Value::Range(0, a) })
            }
            "abs" => scalar_or_tensor(&args, |x| x.abs(), TensorOp::ReLU, true),
            "sqrt" => scalar_or_tensor(&args, f64::sqrt, TensorOp::Sqrt, false),
            "exp" => scalar_or_tensor(&args, f64::exp, TensorOp::Exp, false),
            "log" => scalar_or_tensor(&args, f64::ln, TensorOp::Log, false),
            "tanh" => scalar_or_tensor(&args, f64::tanh, TensorOp::Tanh, false),
            "int" => Ok(Value::Int(args.first().ok_or("`int` expects 1 argument")?.as_i64()?)),
            "float" => Ok(Value::Float(args.first().ok_or("`float` expects 1 argument")?.as_f64()?)),
            "tensor.scalar" => {
                let v = args.first().ok_or("`scalar` expects 1 argument")?;
                let (dtype, x) = match v {
                    Value::Int(i) => (DType::I64, *i as f64),
                    Value::Bool(b) => (DType::Bool, *b as i64 as f64),
                    other => (DType::F64, other.as_f64()?),
                };
                Ok(Value::Tensor(tensor_value(Tensor::from_f64(dtype, vec![], vec![x])?)))
            }
            "tensor.zeros" | "tensor.ones" | "tensor.randn" => {
                let mut shape = Vec::new();
                for a in &args {
                    shape.push(usize_of(a)?);
                }
                let dtype = dtype_kw(&kwargs).unwrap_or(DType::F32);
                let t = match name {
                    "tensor.zeros" => Tensor::zeros(dtype, shape),
                    "tensor.ones" => Tensor::ones(dtype, shape),
                    _ => {
                        let seed = self.next_seed();
                        let r = Tensor::randn(shape, seed);
                        if dtype == DType::F32 { r } else { r.cast(dtype) }
                    }
                };
                Ok(Value::Tensor(tensor_value(t)))
            }
            "tensor.arange" => {
                let a = args.first().map(|v| v.as_f64()).transpose()?;
                let b = args.get(1).map(|v| v.as_f64()).transpose()?;
                let step = args.get(2).map(|v| v.as_f64()).transpose()?.unwrap_or(1.0);
                let (start, end) = match (a, b) {
                    (Some(s), Some(e)) => (s, e),
                    (Some(e), None) => (0.0, e),
                    _ => return Err("`arange` expects 1 or 2 numeric arguments".into()),
                };
                Ok(Value::Tensor(tensor_value(Tensor::arange(start, end, step)?)))
            }
            "tensor.matmul" | "tensor.add" | "tensor.sub" | "tensor.mul" | "tensor.div" => {
                let op = match name {
                    "tensor.matmul" => TensorOp::MatMul,
                    "tensor.add" => TensorOp::Add,
                    "tensor.sub" => TensorOp::Sub,
                    "tensor.mul" => TensorOp::Mul,
                    _ => TensorOp::Div,
                };
                let a = self.value_to_tref(args.first().ok_or("missing first operand")?)?;
                let b = self.value_to_tref(args.get(1).ok_or("missing second operand")?)?;
                let out = self.apply_op(&op, &[a, b])?;
                Ok(Value::Tensor(out))
            }
            "tensor.relu" | "tensor.gelu" | "tensor.sigmoid" | "tensor.tanh" | "tensor.exp"
            | "tensor.log" | "tensor.sqrt" => {
                let op = match name {
                    "tensor.relu" => TensorOp::ReLU,
                    "tensor.gelu" => TensorOp::GELU,
                    "tensor.sigmoid" => TensorOp::Sigmoid,
                    "tensor.tanh" => TensorOp::Tanh,
                    "tensor.exp" => TensorOp::Exp,
                    "tensor.log" => TensorOp::Log,
                    _ => TensorOp::Sqrt,
                };
                let a = self.value_to_tref(args.first().ok_or("missing operand")?)?;
                Ok(Value::Tensor(self.apply_op(&op, &[a])?))
            }
            "tensor.softmax" => {
                let a = args.first().ok_or("`softmax` expects a tensor")?;
                let t = a.as_tensor()?;
                let rank = t.borrow().data.rank();
                let axis = kw_usize(&kwargs, "axis").unwrap_or(rank.saturating_sub(1));
                Ok(Value::Tensor(self.apply_op(&TensorOp::Softmax { axis }, &[t])?))
            }
            "tensor.transpose" => {
                let t = args.first().ok_or("`transpose` expects a tensor")?.as_tensor()?;
                let rank = t.borrow().data.rank();
                if rank < 2 {
                    return Err("`transpose` requires a tensor of rank >= 2".into());
                }
                let d0 = pos_usize(&args, 1).unwrap_or(rank - 2);
                let d1 = pos_usize(&args, 2).unwrap_or(rank - 1);
                Ok(Value::Tensor(self.apply_op(&TensorOp::Transpose { dim0: d0, dim1: d1 }, &[t])?))
            }
            "tensor.reshape" => {
                let t = args.first().ok_or("`reshape` expects a tensor")?.as_tensor()?;
                let dims = dims_from_args(&args[1..])?;
                Ok(Value::Tensor(self.apply_op(&TensorOp::Reshape { dims }, &[t])?))
            }
            "tensor.flatten" => {
                let t = args.first().ok_or("`flatten` expects a tensor")?.as_tensor()?;
                Ok(Value::Tensor(self.apply_op(&TensorOp::Flatten, &[t])?))
            }
            "tensor.sum" | "tensor.mean" => {
                let t = args.first().ok_or("reduction expects a tensor")?.as_tensor()?;
                let rank = t.borrow().data.rank();
                let keepdim = kw_bool(&kwargs, "keepdim").unwrap_or(false);
                let axes = match kw_usize(&kwargs, "axis") {
                    Some(a) => vec![a],
                    None => (0..rank).collect(),
                };
                let op = if name == "tensor.sum" {
                    TensorOp::Sum { axes, keepdim }
                } else {
                    TensorOp::Mean { axes, keepdim }
                };
                Ok(Value::Tensor(self.apply_op(&op, &[t])?))
            }
            "tensor.conv2d" | "tensor.conv1d" => {
                let x = args.first().ok_or("conv expects an input tensor")?.as_tensor()?;
                let w = args.get(1).ok_or("conv expects a weight tensor")?.as_tensor()?;
                let stride = kw_usize(&kwargs, "stride").unwrap_or(1);
                let padding = kw_usize(&kwargs, "padding").unwrap_or(0);
                let op = if name == "tensor.conv2d" {
                    TensorOp::Conv2D { stride: (stride, stride), padding: (padding, padding) }
                } else {
                    TensorOp::Conv1D { stride, padding }
                };
                Ok(Value::Tensor(self.apply_op(&op, &[x, w])?))
            }
            // ---- neural network layers ------------------------------------
            "nn.Linear" => {
                let i = usize_of(args.first().ok_or("`Linear` expects in_features")?)?;
                let o = usize_of(args.get(1).ok_or("`Linear` expects out_features")?)?;
                let bias = kw_bool(&kwargs, "bias").unwrap_or(true);
                let seed = self.next_seed();
                Ok(Value::Layer(Rc::new(Layer::linear(i, o, bias, seed))))
            }
            "nn.ReLU" => Ok(layer_value(Layer::Activation(Activation::ReLU))),
            "nn.GELU" => Ok(layer_value(Layer::Activation(Activation::GELU))),
            "nn.Sigmoid" => Ok(layer_value(Layer::Activation(Activation::Sigmoid))),
            "nn.Tanh" => Ok(layer_value(Layer::Activation(Activation::Tanh))),
            "nn.Softmax" => {
                let axis = kw_usize(&kwargs, "axis");
                Ok(layer_value(Layer::Softmax { axis }))
            }
            "nn.Dropout" => {
                let p = kw_f64(&kwargs, "p")
                    .or_else(|| args.first().and_then(|v| v.as_f64().ok()))
                    .unwrap_or(0.5);
                if !(0.0..1.0).contains(&p) {
                    return Err(format!("Dropout probability must be in [0, 1), found {}", p));
                }
                Ok(layer_value(Layer::Dropout { p }))
            }
            "nn.LayerNorm" => {
                let dim = width_arg(&args, &kwargs)?;
                let eps = kw_f64(&kwargs, "eps").unwrap_or(1e-5);
                Ok(layer_value(Layer::layernorm(dim, eps)))
            }
            "nn.BatchNorm" => {
                let dim = width_arg(&args, &kwargs)?;
                let eps = kw_f64(&kwargs, "eps").unwrap_or(1e-5);
                Ok(layer_value(Layer::batchnorm(dim, eps)))
            }
            "nn.Embedding" => {
                let vocab = usize_of(args.first().ok_or("`Embedding` expects num_embeddings")?)?;
                let dim = usize_of(args.get(1).ok_or("`Embedding` expects embedding_dim")?)?;
                let seed = self.next_seed();
                Ok(layer_value(Layer::embedding(vocab, dim, seed)))
            }
            "nn.Conv1D" => {
                let ic = usize_of(args.first().ok_or("`Conv1D` expects in_channels")?)?;
                let oc = usize_of(args.get(1).ok_or("`Conv1D` expects out_channels")?)?;
                let k = usize_of(args.get(2).ok_or("`Conv1D` expects kernel_size")?)?;
                let stride = kw_usize(&kwargs, "stride").unwrap_or(1);
                let padding = kw_usize(&kwargs, "padding").unwrap_or(0);
                let bias = kw_bool(&kwargs, "bias").unwrap_or(true);
                let seed = self.next_seed();
                Ok(layer_value(Layer::conv1d(ic, oc, k, stride, padding, bias, seed)))
            }
            "nn.Conv2D" => {
                let ic = usize_of(args.first().ok_or("`Conv2D` expects in_channels")?)?;
                let oc = usize_of(args.get(1).ok_or("`Conv2D` expects out_channels")?)?;
                let k = match args.get(2) {
                    Some(v) => pair_usize(v)?,
                    None => return Err("`Conv2D` expects kernel_size".into()),
                };
                let stride = kw_pair(&kwargs, "stride")?.unwrap_or((1, 1));
                let padding = kw_pair(&kwargs, "padding")?.unwrap_or((0, 0));
                let bias = kw_bool(&kwargs, "bias").unwrap_or(true);
                let seed = self.next_seed();
                Ok(layer_value(Layer::conv2d(ic, oc, k, stride, padding, bias, seed)))
            }
            "nn.Sequential" => {
                let list = args
                    .first()
                    .ok_or("`Sequential` expects a list of layers")?
                    .as_list()?;
                let mut ls: Vec<Rc<Layer>> = Vec::new();
                for item in list.borrow().iter() {
                    match item {
                        Value::Layer(l) => ls.push(l.clone()),
                        other => {
                            return Err(format!(
                                "`Sequential` expects layers, found {}",
                                other.type_name()
                            ))
                        }
                    }
                }
                Ok(Value::Layer(Rc::new(Layer::Sequential(ls))))
            }
            // ---- data ------------------------------------------------------
            "nn.load_dataset" | "nn.load_csv" | "data.load_csv" => {
                let path = args.first().ok_or("dataset loader expects a path")?.as_str()?;
                Ok(Value::Dataset(Rc::new(Dataset::load(path)?)))
            }
            // ---- losses / training ----------------------------------------
            "nn.cross_entropy" => {
                let logits = args.first().ok_or("cross_entropy expects logits")?.as_tensor()?;
                let targets = args.get(1).ok_or("cross_entropy expects targets")?.as_tensor()?;
                Ok(Value::Tensor(layers::cross_entropy(&logits, &targets)?))
            }
            "nn.parameters" => {
                let l = layer_arg(&args, 0)?;
                Ok(Value::list(
                    l.params().into_iter().map(Value::Tensor).collect::<Vec<_>>(),
                ))
            }
            "nn.train" => {
                let l = layer_arg(&args, 0)?;
                let ds = dataset_arg(&args, 1)?;
                self.train_layer(&l, &ds, &args[2..], &kwargs)
            }
            "nn.forward" | "nn.predict" | "inference.predict" => {
                let l = layer_arg(&args, 0)?;
                let x = args.get(1).ok_or("forward expects an input tensor")?.as_tensor()?;
                let seed = self.next_seed();
                Ok(Value::Tensor(l.forward(&x, self.options.training, seed)?))
            }
            "nn.save" | "serialization.save" => {
                let l = layer_arg(&args, 0)?;
                let path = args.get(1).ok_or("save expects a path")?.as_str()?;
                l.save(&path)?;
                Ok(Value::Unit)
            }
            "nn.load" | "serialization.load" => {
                let path = args.first().ok_or("load expects a path")?.as_str()?;
                Ok(Value::Layer(Rc::new(Layer::load(path)?)))
            }
            // ---- optimizers ------------------------------------------------
            "optim.SGD" | "optim.Adam" | "optim.AdamW" | "optim.RMSProp" => {
                let params = params_arg(args.first().ok_or("optimizer expects parameters")?)?;
                let lr = kw_f64(&kwargs, "learning_rate")
                    .or_else(|| args.get(1).and_then(|v| v.as_f64().ok()))
                    .unwrap_or(0.001);
                if !(lr.is_finite() && lr > 0.0) {
                    return Err(format!("learning_rate must be positive, found {}", lr));
                }
                let kind = match name {
                    "optim.SGD" => OptKind::Sgd { momentum: kw_f64(&kwargs, "momentum").unwrap_or(0.0) },
                    "optim.Adam" => OptKind::Adam {
                        beta1: 0.9,
                        beta2: 0.999,
                        eps: kw_f64(&kwargs, "eps").unwrap_or(1e-8),
                    },
                    "optim.AdamW" => OptKind::AdamW {
                        beta1: 0.9,
                        beta2: 0.999,
                        eps: kw_f64(&kwargs, "eps").unwrap_or(1e-8),
                        weight_decay: kw_f64(&kwargs, "weight_decay").unwrap_or(0.01),
                    },
                    _ => OptKind::RmsProp {
                        alpha: kw_f64(&kwargs, "alpha").unwrap_or(0.99),
                        eps: kw_f64(&kwargs, "eps").unwrap_or(1e-8),
                        momentum: kw_f64(&kwargs, "momentum").unwrap_or(0.0),
                    },
                };
                Ok(Value::Optimizer(optim::shared(kind, lr, params)))
            }
            // ---- math ------------------------------------------------------
            "math.sqrt" => scalar_or_tensor(&args, f64::sqrt, TensorOp::Sqrt, false),
            "math.exp" => scalar_or_tensor(&args, f64::exp, TensorOp::Exp, false),
            "math.log" => scalar_or_tensor(&args, f64::ln, TensorOp::Log, false),
            "math.tanh" => scalar_or_tensor(&args, f64::tanh, TensorOp::Tanh, false),
            // ---- autograd --------------------------------------------------
            "autograd.backward" => {
                let t = args.first().ok_or("backward expects a tensor")?.as_tensor()?;
                autograd::backward(&t)?;
                Ok(Value::Unit)
            }
            // ---- device ----------------------------------------------------
            "device.available" => {
                let names: Vec<Value> = device::discover()
                    .into_iter()
                    .filter(|d| d.available)
                    .map(|d| Value::Str(d.name))
                    .collect();
                Ok(Value::list(names))
            }
            "device.count" => Ok(Value::Int(
                device::discover().iter().filter(|d| d.available).count() as i64,
            )),
            "device.default" => Ok(Value::Str("cpu".into())),
            "distributed.rank" => Ok(Value::Int(0)),
            "distributed.world_size" => Ok(Value::Int(1)),
            other => Err(format!("builtin `{}` is not implemented by the runtime", other)),
        }
    }

    fn value_to_tref(&self, v: &Value) -> Result<TRef, String> {
        match v {
            Value::Tensor(t) => Ok(t.clone()),
            Value::Int(_) | Value::Float(_) | Value::Bool(_) => {
                let f = v.as_f64()?;
                let dtype = if matches!(v, Value::Int(_)) { DType::I64 } else { DType::F64 };
                Ok(tensor_value(Tensor::from_f64(dtype, vec![], vec![f])?))
            }
            other => Err(format!("expected a tensor, found {}", other.type_name())),
        }
    }

    fn train_layer(
        &mut self,
        model: &Rc<Layer>,
        dataset: &Rc<Dataset>,
        extra: &[Value],
        kwargs: &[(String, Value)],
    ) -> Result<Value, String> {
        let epochs = kw_usize(kwargs, "epochs")
            .or_else(|| extra.first().map(usize_of).transpose().ok().flatten())
            .unwrap_or(1);
        let batch_size = kw_usize(kwargs, "batch_size")
            .or_else(|| extra.get(1).map(usize_of).transpose().ok().flatten())
            .unwrap_or(32);
        let optimizer = kwargs
            .iter()
            .find(|(k, _)| k == "optimizer")
            .map(|(_, v)| v)
            .or_else(|| extra.iter().find(|v| matches!(v, Value::Optimizer(_))));
        let opt = match optimizer {
            Some(Value::Optimizer(o)) => o.clone(),
            Some(other) => {
                return Err(format!(
                    "model.train expects an optimizer, found {}",
                    other.type_name()
                ))
            }
            None => return Err("model.train requires an optimizer".into()),
        };
        let seed = self.next_seed();
        let mut report = Vec::new();
        layers::train(
            model.as_ref(),
            dataset.as_ref(),
            epochs,
            batch_size,
            &mut opt.borrow_mut(),
            seed,
            &mut report,
        )?;
        for line in &report {
            if !self.options.quiet {
                println!("{}", line);
            }
            self.output.push(line.clone());
        }
        Ok(Value::Unit)
    }

    fn call_method(
        &mut self,
        recv: Value,
        method: &str,
        args: Vec<Value>,
        kwargs: Vec<(String, Value)>,
    ) -> Result<Value, String> {
        match recv {
            Value::Tensor(t) => match method {
                "backward" => {
                    autograd::backward(&t)?;
                    Ok(Value::Unit)
                }
                "item" => Ok(Value::Float(t.borrow().data.item()?)),
                "detach" => Ok(Value::Tensor(TNode::leaf(t.borrow().data.clone(), false))),
                other => Err(format!("tensor has no runtime method `{}`", other)),
            },
            Value::Layer(l) => match method {
                "forward" | "predict" | "inference" => {
                    let x = args.first().ok_or("forward expects an input tensor")?.as_tensor()?;
                    let seed = self.next_seed();
                    Ok(Value::Tensor(l.forward(&x, self.options.training, seed)?))
                }
                "parameters" => Ok(Value::list(
                    l.params().into_iter().map(Value::Tensor).collect::<Vec<_>>(),
                )),
                "train" => {
                    let ds = dataset_arg(&args, 0)?;
                    self.train_layer(&l, &ds, &args[1..], &kwargs)
                }
                "save" => {
                    let path = args.first().ok_or("save expects a path")?.as_str()?;
                    l.save(&path)?;
                    Ok(Value::Unit)
                }
                "load" => {
                    let path = args.first().ok_or("load expects a path")?.as_str()?;
                    let loaded = Layer::load(&path)?;
                    let src = loaded.params();
                    let dst = l.params();
                    if src.len() != dst.len() {
                        return Err(format!(
                            "model file has {} parameter tensors, this model has {}",
                            src.len(),
                            dst.len()
                        ));
                    }
                    for (a, b) in src.iter().zip(dst.iter()) {
                        let data = a.borrow().data.clone();
                        if data.shape != b.borrow().data.shape {
                            return Err(format!(
                                "parameter shape mismatch: model file {:?}, layer {:?}",
                                data.shape,
                                b.borrow().data.shape
                            ));
                        }
                        b.borrow_mut().data = data;
                    }
                    Ok(Value::Unit)
                }
                "describe" => Ok(Value::Str(l.describe())),
                other => Err(format!("layer has no runtime method `{}`", other)),
            },
            Value::Optimizer(o) => match method {
                "step" => {
                    o.borrow_mut().step()?;
                    Ok(Value::Unit)
                }
                "zero_grad" => {
                    o.borrow_mut().zero_grad();
                    Ok(Value::Unit)
                }
                "describe" => Ok(Value::Str(o.borrow().describe())),
                other => Err(format!("optimizer has no runtime method `{}`", other)),
            },
            Value::Dataset(d) => match method {
                "len" => Ok(Value::Int(d.len() as i64)),
                "shuffle" => {
                    let seed = args.first().map(|v| v.as_i64()).transpose()?.unwrap_or(0) as u64;
                    Ok(Value::Dataset(Rc::new(d.shuffle(seed))))
                }
                "split" => {
                    let f = args.first().ok_or("split expects a fraction")?.as_f64()?;
                    let (a, _) = d.split(f)?;
                    Ok(Value::Dataset(Rc::new(a)))
                }
                "batch" | "batches" => {
                    let n = args.first().map(usize_of).transpose()?.unwrap_or(32).max(1);
                    let rows = n.min(d.len());
                    let cols = d.features.shape[1];
                    let feats = d.features.slice(0, 0, rows)?;
                    let labels = d.labels.slice(0, 0, rows)?;
                    let _ = cols;
                    Ok(Value::Dataset(Rc::new(Dataset::new(feats, labels)?)))
                }
                other => Err(format!("dataset has no runtime method `{}`", other)),
            },
            Value::List(l) => match method {
                "len" => Ok(Value::Int(l.borrow().len() as i64)),
                other => Err(format!("list has no runtime method `{}`", other)),
            },
            other => Err(format!(
                "{} has no runtime method `{}`",
                other.type_name(),
                method
            )),
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn read_line() -> Result<String, String> {
    let mut line = String::new();
    std::io::stdin()
        .read_line(&mut line)
        .map_err(|e| format!("cannot read input: {}", e))?;
    while line.ends_with('\n') || line.ends_with('\r') {
        line.pop();
    }
    Ok(line)
}

fn parse_input(line: &str, ty: &Ty) -> Result<Value, String> {
    match ty {
        Ty::Int => line
            .trim()
            .parse::<i64>()
            .map(Value::Int)
            .map_err(|_| format!("expected an Int input, found {:?}", line)),
        Ty::Float => line
            .trim()
            .parse::<f64>()
            .map(Value::Float)
            .map_err(|_| format!("expected a Float input, found {:?}", line)),
        Ty::Bool => match line.trim().to_ascii_lowercase().as_str() {
            "true" | "1" => Ok(Value::Bool(true)),
            "false" | "0" => Ok(Value::Bool(false)),
            _ => Err(format!(
                "expected a Bool input (true/false), found {:?}",
                line
            )),
        },
        _ => Ok(Value::Str(line.to_string())),
    }
}

fn layer_value(l: Layer) -> Value {
    Value::Layer(Rc::new(l))
}

fn build_tensor(v: &Value, ty: &TensorTy) -> Result<Tensor, String> {
    match v {
        Value::List(_) | Value::Tensor(_) => tensor_from_value(v, ty.dtype),
        Value::Int(_) | Value::Float(_) | Value::Bool(_) => {
            Tensor::from_f64(ty.dtype, vec![], vec![v.as_f64()?])
        }
        Value::Unit => Err("cannot build a tensor from an empty value".into()),
        other => Err(format!("cannot build a tensor from {}", other.type_name())),
    }
}

fn take_broadcast(t: &Tensor, out_idx: &[usize]) -> f64 {
    if t.rank() == 0 {
        return t.f64_at(0);
    }
    let pad = out_idx.len().saturating_sub(t.rank());
    let mut flat = 0usize;
    let mut stride = 1usize;
    for i in (0..t.rank()).rev() {
        let d = t.shape[i];
        let idx = if d == 1 { 0 } else { out_idx.get(pad + i).copied().unwrap_or(0) };
        flat += idx * stride;
        stride *= d;
    }
    t.f64_at(flat)
}

/// Rebuild a fused elementwise expression as an ordinary differentiable graph.
fn rebuild_fused(e: &FusedExpr, args: &[TRef]) -> Result<TRef, String> {
    let op = |op: TensorOp, a: TRef, b: Option<TRef>| -> Result<TRef, String> {
        match b {
            Some(b) => apply(&op, &[a, b], false, 0),
            None => apply(&op, &[a], false, 0),
        }
    };
    Ok(match e {
        FusedExpr::Input(i) => args
            .get(*i as usize)
            .cloned()
            .ok_or_else(|| format!("fused expression references input %{}", i))?,
        FusedExpr::Const(v) => tensor_value(Tensor::from_f64(DType::F64, vec![], vec![*v])?),
        FusedExpr::Add(a, b) => {
            op(TensorOp::Add, rebuild_fused(a, args)?, Some(rebuild_fused(b, args)?))?
        }
        FusedExpr::Sub(a, b) => {
            op(TensorOp::Sub, rebuild_fused(a, args)?, Some(rebuild_fused(b, args)?))?
        }
        FusedExpr::Mul(a, b) => {
            op(TensorOp::Mul, rebuild_fused(a, args)?, Some(rebuild_fused(b, args)?))?
        }
        FusedExpr::Div(a, b) => {
            op(TensorOp::Div, rebuild_fused(a, args)?, Some(rebuild_fused(b, args)?))?
        }
        FusedExpr::Neg(a) => op(TensorOp::Neg, rebuild_fused(a, args)?, None)?,
        FusedExpr::ReLU(a) => op(TensorOp::ReLU, rebuild_fused(a, args)?, None)?,
        FusedExpr::GELU(a) => op(TensorOp::GELU, rebuild_fused(a, args)?, None)?,
        FusedExpr::Sigmoid(a) => op(TensorOp::Sigmoid, rebuild_fused(a, args)?, None)?,
        FusedExpr::Tanh(a) => op(TensorOp::Tanh, rebuild_fused(a, args)?, None)?,
        FusedExpr::Exp(a) => op(TensorOp::Exp, rebuild_fused(a, args)?, None)?,
        FusedExpr::Log(a) => op(TensorOp::Log, rebuild_fused(a, args)?, None)?,
        FusedExpr::Sqrt(a) => op(TensorOp::Sqrt, rebuild_fused(a, args)?, None)?,
    })
}

fn resolve_dynamic(op: &TensorOp, dyn_dims: &[Reg], frame: &Frame) -> Result<TensorOp, String> {
    if dyn_dims.is_empty() {
        return Ok(op.clone());
    }
    match op {
        TensorOp::Reshape { dims } => {
            let mut it = dyn_dims.iter();
            let mut out = Vec::with_capacity(dims.len());
            for d in dims {
                match d {
                    Dim::Static(n) => out.push(Dim::Static(*n)),
                    _ => match it.next() {
                        Some(r) => match frame.regs.get(*r as usize) {
                            Some(Value::Int(i)) if *i >= 0 => out.push(Dim::Static(*i as usize)),
                            _ => out.push(Dim::Unknown),
                        },
                        None => out.push(Dim::Unknown),
                    },
                }
            }
            Ok(TensorOp::Reshape { dims: out })
        }
        other => Ok(other.clone()),
    }
}

fn binary(op: BinOp, a: &Value, b: &Value, ty: &neura_ir::types::Ty) -> Result<Value, String> {
    if let (Value::Str(x), Value::Str(y)) = (a, b) {
        if op == BinOp::Add {
            return Ok(Value::Str(format!("{}{}", x, y)));
        }
    }
    match op {
        BinOp::Eq => return Ok(Value::Bool(eq_values(a, b)?)),
        BinOp::Ne => return Ok(Value::Bool(!eq_values(a, b)?)),
        BinOp::And => return Ok(Value::Bool(a.as_bool()? && b.as_bool()?)),
        BinOp::Or => return Ok(Value::Bool(a.as_bool()? || b.as_bool()?)),
        BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge => {
            let x = a.as_f64()?;
            let y = b.as_f64()?;
            return Ok(Value::Bool(match op {
                BinOp::Lt => x < y,
                BinOp::Le => x <= y,
                BinOp::Gt => x > y,
                _ => x >= y,
            }));
        }
        _ => {}
    }
    let ints = matches!(a, Value::Int(_)) && matches!(b, Value::Int(_)) && !matches!(ty, neura_ir::types::Ty::Float);
    if ints {
        let x = a.as_i64()?;
        let y = b.as_i64()?;
        let v = match op {
            BinOp::Add => x.wrapping_add(y),
            BinOp::Sub => x.wrapping_sub(y),
            BinOp::Mul => x.wrapping_mul(y),
            BinOp::Div => {
                if y == 0 {
                    return Err("integer division by zero".into());
                }
                x / y
            }
            BinOp::FloorDiv => {
                if y == 0 {
                    return Err("integer floor division by zero".into());
                }
                floor_div(x, y)
            }
            BinOp::Pow => {
                if y < 0 {
                    return Err("negative exponent is not supported for integers".into());
                }
                if y > u32::MAX as i64 {
                    return Err("integer exponent is too large".into());
                }
                x.wrapping_pow(y as u32)
            }
            BinOp::Mod => {
                if y == 0 {
                    return Err("integer modulo by zero".into());
                }
                x % y
            }
            _ => return Err(format!("operator `{}` is not defined for integers", op.symbol())),
        };
        Ok(Value::Int(v))
    } else {
        let x = a.as_f64()?;
        let y = b.as_f64()?;
        let v = match op {
            BinOp::Add => x + y,
            BinOp::Sub => x - y,
            BinOp::Mul => x * y,
            BinOp::Div => x / y,
            BinOp::FloorDiv => (x / y).floor(),
            BinOp::Pow => x.powf(y),
            BinOp::Mod => x % y,
            _ => return Err(format!("operator `{}` is not defined for numbers", op.symbol())),
        };
        // Integer-typed results (e.g. `Int / Int` when the checker says Float)
        // keep the checker's declared type.
        if matches!(ty, neura_ir::types::Ty::Int) {
            Ok(Value::Int(v as i64))
        } else {
            Ok(Value::Float(v))
        }
    }
}

/// Floor division for integers (Rust's `/` truncates toward zero).
fn floor_div(x: i64, y: i64) -> i64 {
    let q = x / y;
    let r = x % y;
    if r != 0 && ((r < 0) != (y < 0)) {
        q - 1
    } else {
        q
    }
}

fn eq_values(a: &Value, b: &Value) -> Result<bool, String> {
    Ok(match (a, b) {
        (Value::Str(x), Value::Str(y)) => x == y,
        (Value::Bool(x), Value::Bool(y)) => x == y,
        _ => a.as_f64()? == b.as_f64()?,
    })
}

fn unary(op: UnOp, a: &Value, ty: &neura_ir::types::Ty) -> Result<Value, String> {
    match op {
        UnOp::Not => Ok(Value::Bool(!a.as_bool()?)),
        UnOp::Neg => {
            if let Value::Tensor(t) = a {
                let out = apply(&TensorOp::Neg, &[t.clone()], false, 0)?;
                return Ok(Value::Tensor(out));
            }
            if matches!(ty, neura_ir::types::Ty::Int) || matches!(a, Value::Int(_)) {
                Ok(Value::Int(-a.as_i64()?))
            } else {
                Ok(Value::Float(-a.as_f64()?))
            }
        }
    }
}

fn get_attr(v: &Value, name: &str) -> Result<Value, String> {
    match v {
        Value::Tensor(t) => {
            let node = t.borrow();
            match name {
                "shape" => Ok(Value::list(
                    node.data.shape.iter().map(|d| Value::Int(*d as i64)).collect(),
                )),
                "grad" => {
                    let g = node.grad.clone().unwrap_or_else(|| {
                        Tensor::zeros(node.data.dtype, node.data.shape.clone())
                    });
                    Ok(Value::Tensor(tensor_value(g)))
                }
                "requires_grad" => Ok(Value::Bool(node.requires_grad)),
                "dtype" => Ok(Value::Str(node.data.dtype.name().to_string())),
                "device" => Ok(Value::Str("cpu".into())),
                other => Err(format!("tensor has no property `{}`", other)),
            }
        }
        Value::Struct { fields, .. } => fields
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, v)| v.clone())
            .ok_or_else(|| format!("struct has no field `{}`", name)),
        Value::Layer(l) => match name {
            "parameters" => Ok(Value::list(
                l.params().into_iter().map(Value::Tensor).collect::<Vec<_>>(),
            )),
            other => Err(format!("layer has no property `{}`", other)),
        },
        Value::Dataset(d) => match name {
            "len" => Ok(Value::Int(d.len() as i64)),
            other => Err(format!("dataset has no property `{}`", other)),
        },
        other => Err(format!("{} has no property `{}`", other.type_name(), name)),
    }
}

fn index_value(obj: &Value, index: &Value) -> Result<Value, String> {
    let i = index.as_i64()?;
    match obj {
        Value::List(l) => {
            let items = l.borrow();
            let n = items.len() as i64;
            let pos = if i < 0 { n + i } else { i };
            if pos < 0 || pos >= n {
                return Err(format!("index {} is out of range for a list of length {}", i, n));
            }
            Ok(items[pos as usize].clone())
        }
        Value::Str(s) => {
            let chars: Vec<char> = s.chars().collect();
            let n = chars.len() as i64;
            let pos = if i < 0 { n + i } else { i };
            if pos < 0 || pos >= n {
                return Err(format!("index {} is out of range for a string of length {}", i, n));
            }
            Ok(Value::Str(chars[pos as usize].to_string()))
        }
        Value::Tensor(t) => {
            let (shape, rank) = {
                let b = t.borrow();
                (b.data.shape.clone(), b.data.rank())
            };
            if rank == 0 {
                return Err("cannot index a scalar tensor".into());
            }
            let n = shape[0] as i64;
            let pos = if i < 0 { n + i } else { i };
            if pos < 0 || pos >= n {
                return Err(format!(
                    "index {} is out of range for a tensor with {} rows",
                    i, n
                ));
            }
            let sliced = apply(
                &TensorOp::Slice { axis: 0, start: pos as usize, end: pos as usize + 1 },
                &[t.clone()],
                false,
                0,
            )?;
            let dims: Vec<Dim> = shape[1..].iter().map(|d| Dim::Static(*d)).collect();
            Ok(Value::Tensor(apply(&TensorOp::Reshape { dims }, &[sliced], false, 0)?))
        }
        other => Err(format!("cannot index {}", other.type_name())),
    }
}

fn length_of(v: &Value) -> Result<i64, String> {
    Ok(match v {
        Value::List(l) => l.borrow().len() as i64,
        Value::Str(s) => s.chars().count() as i64,
        Value::Tensor(t) => {
            let b = t.borrow();
            if b.data.rank() == 0 {
                1
            } else {
                b.data.shape[0] as i64
            }
        }
        Value::Range(a, b) => (b - a).max(0),
        Value::Dataset(d) => d.len() as i64,
        other => return Err(format!("`len` is not defined for {}", other.type_name())),
    })
}

fn scalar_or_tensor(
    args: &[Value],
    f: fn(f64) -> f64,
    op: TensorOp,
    is_abs: bool,
) -> Result<Value, String> {
    let v = args.first().ok_or("expected 1 argument")?;
    match v {
        Value::Tensor(t) => {
            if is_abs {
                let neg = apply(&TensorOp::Neg, &[t.clone()], false, 0)?;
                let pos = apply(&TensorOp::ReLU, &[t.clone()], false, 0)?;
                let n = apply(&TensorOp::ReLU, &[neg], false, 0)?;
                Ok(Value::Tensor(apply(&TensorOp::Add, &[pos, n], false, 0)?))
            } else {
                Ok(Value::Tensor(apply(&op, &[t.clone()], false, 0)?))
            }
        }
        other => Ok(Value::Float(f(other.as_f64()?))),
    }
}

fn layer_arg(args: &[Value], i: usize) -> Result<Rc<Layer>, String> {
    match args.get(i) {
        Some(Value::Layer(l)) => Ok(l.clone()),
        Some(other) => Err(format!("expected a layer, found {}", other.type_name())),
        None => Err(format!("missing argument #{}: expected a layer", i + 1)),
    }
}

fn dataset_arg(args: &[Value], i: usize) -> Result<Rc<Dataset>, String> {
    match args.get(i) {
        Some(Value::Dataset(d)) => Ok(d.clone()),
        Some(other) => Err(format!("expected a dataset, found {}", other.type_name())),
        None => Err(format!("missing argument #{}: expected a dataset", i + 1)),
    }
}

fn params_arg(v: &Value) -> Result<Vec<TRef>, String> {
    let list = v.as_list()?;
    let mut out = Vec::new();
    for item in list.borrow().iter() {
        match item {
            Value::Tensor(t) => out.push(t.clone()),
            other => return Err(format!("expected a parameter tensor, found {}", other.type_name())),
        }
    }
    if out.is_empty() {
        return Err("optimizer received an empty parameter list".into());
    }
    Ok(out)
}

fn usize_of(v: &Value) -> Result<usize, String> {
    let i = v.as_i64()?;
    if i < 0 {
        return Err(format!("expected a non-negative integer, found {}", i));
    }
    Ok(i as usize)
}

fn pos_usize(args: &[Value], i: usize) -> Option<usize> {
    args.get(i).and_then(|v| v.as_i64().ok()).map(|v| v.max(0) as usize)
}

fn pair_usize(v: &Value) -> Result<(usize, usize), String> {
    match v {
        Value::List(l) => {
            let items = l.borrow();
            if items.len() != 2 {
                return Err("expected a pair of integers".into());
            }
            Ok((usize_of(&items[0])?, usize_of(&items[1])?))
        }
        other => {
            let n = usize_of(other)?;
            Ok((n, n))
        }
    }
}

fn kw_f64(kwargs: &[(String, Value)], name: &str) -> Option<f64> {
    kwargs.iter().find(|(k, _)| k == name).and_then(|(_, v)| v.as_f64().ok())
}

fn kw_usize(kwargs: &[(String, Value)], name: &str) -> Option<usize> {
    kwargs
        .iter()
        .find(|(k, _)| k == name)
        .and_then(|(_, v)| v.as_i64().ok())
        .map(|v| v.max(0) as usize)
}

fn kw_bool(kwargs: &[(String, Value)], name: &str) -> Option<bool> {
    kwargs.iter().find(|(k, _)| k == name).and_then(|(_, v)| v.as_bool().ok())
}

fn kw_pair(kwargs: &[(String, Value)], name: &str) -> Result<Option<(usize, usize)>, String> {
    match kwargs.iter().find(|(k, _)| k == name) {
        Some((_, v)) => Ok(Some(pair_usize(v)?)),
        None => Ok(None),
    }
}

fn dtype_kw(kwargs: &[(String, Value)]) -> Option<DType> {
    kwargs
        .iter()
        .find(|(k, _)| k == "dtype")
        .and_then(|(_, v)| v.as_str().ok())
        .and_then(|s| DType::parse(&s))
}

fn width_arg(args: &[Value], kwargs: &[(String, Value)]) -> Result<usize, String> {
    let direct = args
        .first()
        .or_else(|| kwargs.iter().find(|(k, _)| k == "normalized_shape" || k == "num_features").map(|(_, v)| v));
    match direct {
        Some(Value::List(l)) => {
            let items = l.borrow();
            usize_of(items.last().ok_or("normalized shape cannot be empty")?)
        }
        Some(v) => usize_of(v),
        None => Err("expected a feature width".into()),
    }
}

fn dims_from_args(args: &[Value]) -> Result<Vec<Dim>, String> {
    let mut out = Vec::new();
    for a in args {
        match a {
            Value::List(l) => {
                for item in l.borrow().iter() {
                    push_dim(&mut out, item)?;
                }
            }
            other => push_dim(&mut out, other)?,
        }
    }
    if out.is_empty() {
        return Err("reshape expects at least one dimension".into());
    }
    Ok(out)
}

fn push_dim(out: &mut Vec<Dim>, v: &Value) -> Result<(), String> {
    let i = v.as_i64()?;
    if i == -1 {
        out.push(Dim::Unknown);
    } else if i < 0 {
        return Err(format!("invalid reshape dimension {}", i));
    } else {
        out.push(Dim::Static(i as usize));
    }
    Ok(())
}

fn inst_key(inst: &Inst) -> &'static str {
    match inst {
        Inst::TensorOp { .. } => "tensor_op",
        Inst::FusedOp { .. } => "fused_op",
        Inst::Call { .. } => "call",
        Inst::CallBuiltin { name, .. } => match name.split('.').next() {
            Some("nn") => "nn",
            Some("optim") => "optim",
            Some("tensor") => "tensor",
            Some("math") => "math",
            Some("data") => "data",
            Some("autograd") => "autograd",
            Some("device") => "device",
            _ => "builtin",
        },
        Inst::MethodCall { .. } => "method_call",
        Inst::Input { .. } => "input",
        _ => "control",
    }
}

fn disasm(i: &Inst) -> String {
    match i {
        Inst::ConstF64 { dst, v } => format!("%{} = {}f", dst, v),
        Inst::ConstInt { dst, v } => format!("%{} = {}", dst, v),
        Inst::ConstBool { dst, v } => format!("%{} = {}", dst, v),
        Inst::ConstStr { dst, s } => format!("%{} = {:?}", dst, s),
        Inst::ConstUnit { dst } => format!("%{} = ()", dst),
        Inst::Copy { dst, src } => format!("%{} = %{}", dst, src),
        Inst::Move { dst, src } => format!("%{} = move %{}", dst, src),
        Inst::MakeList { dst, elems } => format!("%{} = [{:?}]", dst, elems),
        Inst::MakeTensor { dst, requires_grad, .. } => {
            format!("%{} = tensor(requires_grad={})", dst, requires_grad)
        }
        Inst::MakeStruct { dst, struct_id, .. } => format!("%{} = struct#{}", dst, struct_id),
        Inst::TensorOp { dst, op, .. } => format!("%{} = tir.{}", dst, op.name()),
        Inst::FusedOp { dst, spec, .. } => format!("%{} = fused {}", dst, spec.expr),
        Inst::BinOp { dst, op, .. } => format!("%{} = {} ...", dst, op.symbol()),
        Inst::UnOp { dst, op, .. } => format!("%{} = {}{{...}}", dst, op.symbol()),
        Inst::Call { dst, func, .. } => format!("%{} = call @{}", dst, func),
        Inst::CallBuiltin { dst, name, .. } => format!("%{} = {}(...)", dst, name),
        Inst::MethodCall { dst, recv, method, .. } => {
            format!("%{} = %{}.{}(...)", dst, recv, method)
        }
        Inst::GetAttr { dst, obj, name, .. } => format!("%{} = %{}.{}", dst, obj, name),
        Inst::Index { dst, obj, index, .. } => format!("%{} = %{}[%{}]", dst, obj, index),
        Inst::Len { dst, obj } => format!("%{} = len(%{})", dst, obj),
        Inst::IterInit { dst, iter } => format!("%{} = iter(%{})", dst, iter),
        Inst::IterNext { dst, state, end } => {
            format!("%{} = next(%{}) else .L{}", dst, state, end)
        }
        Inst::Jump(t) => format!("jump .L{}", t),
        Inst::JumpIfFalse { cond, target } => format!("jump_if_false %{} .L{}", cond, target),
        Inst::JumpIfTrue { cond, target } => format!("jump_if_true %{} .L{}", cond, target),
        Inst::Return(Some(r)) => format!("return %{}", r),
        Inst::Return(None) => "return".to_string(),
        Inst::Drop(r) => format!("drop %{}", r),
        Inst::Print(r) => format!("print %{}", r),
        Inst::Input { dst, prompt, ty } => match prompt {
            Some(p) => format!("%{}: {} = input(%{})", dst, ty.name(), p),
            None => format!("%{}: {} = input()", dst, ty.name()),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use neura_ir::eir::{CompilePolicies, EirFunction, EirProgram};
    use neura_ir::ops::BinOp;
    use neura_ir::types::Ty;

    fn program(code: Vec<Inst>, types: Vec<Ty>) -> EirProgram {
        EirProgram {
            name: "test".into(),
            functions: vec![EirFunction {
                name: "main".into(),
                params: vec![],
                locals: vec![],
                types,
                code,
                ret_ty: Ty::Unit,
                spans: vec![],
            }],
            main: 0,
            structs: vec![],
            policies: CompilePolicies::default(),
            source_hash: 7,
        }
    }

    #[test]
    fn runs_arithmetic_and_records_output() {
        let p = program(
            vec![
                Inst::ConstInt { dst: 0, v: 2 },
                Inst::ConstInt { dst: 1, v: 5 },
                Inst::BinOp {
                    dst: 2,
                    op: BinOp::Add,
                    lhs: 0,
                    rhs: 1,
                    ty: Ty::Int,
                },
                Inst::Print(2),
                Inst::Return(None),
            ],
            vec![Ty::Int, Ty::Int, Ty::Int],
        );
        let out = run(&p, VmOptions { quiet: true, ..Default::default() }).unwrap();
        assert_eq!(out.output, vec!["7".to_string()]);
        assert!(matches!(out.value, Value::Unit));
    }

    #[test]
    fn disassembly_is_human_readable() {
        assert_eq!(
            disasm(&Inst::BinOp {
                dst: 3,
                op: BinOp::Mul,
                lhs: 1,
                rhs: 2,
                ty: Ty::Float,
            }),
            "%3 = * ..."
        );
        assert_eq!(disasm(&Inst::Drop(4)), "drop %4");
    }
}