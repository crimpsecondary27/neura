//! EIR — the executable register IR consumed by the Neura VM.
//!
//! Every instruction is typed; tensor instructions keep shape, element type
//! and dtype metadata. Last-use information and buffer-slot assignments from
//! NAOE are attached to locals and tensor instructions.

use crate::ops::{BinOp, FusedSpec, TensorOp, UnOp};
use crate::types::{DType, TensorTy, Ty};
use std::fmt::Write as _;

pub type Reg = u32;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum OptimizeMode {
    Auto,
    Aggressive,
    Debug,
    Off,
}

impl OptimizeMode {
    pub fn parse(s: &str) -> Option<OptimizeMode> {
        Some(match s {
            "auto" => OptimizeMode::Auto,
            "aggressive" => OptimizeMode::Aggressive,
            "debug" => OptimizeMode::Debug,
            "off" => OptimizeMode::Off,
            _ => return None,
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            OptimizeMode::Auto => "auto",
            OptimizeMode::Aggressive => "aggressive",
            OptimizeMode::Debug => "debug",
            OptimizeMode::Off => "off",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DevicePolicy {
    Auto,
    Cpu,
    Cuda,
    Rocm,
}

impl DevicePolicy {
    pub fn parse(s: &str) -> Option<DevicePolicy> {
        Some(match s {
            "auto" => DevicePolicy::Auto,
            "cpu" => DevicePolicy::Cpu,
            "cuda" => DevicePolicy::Cuda,
            "rocm" => DevicePolicy::Rocm,
            _ => return None,
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            DevicePolicy::Auto => "auto",
            DevicePolicy::Cpu => "cpu",
            DevicePolicy::Cuda => "cuda",
            DevicePolicy::Rocm => "rocm",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PrecisionPolicy {
    Strict,
    MixedBf16,
    MixedFp16,
}

impl PrecisionPolicy {
    pub fn parse(s: &str) -> Option<PrecisionPolicy> {
        Some(match s {
            "strict" => PrecisionPolicy::Strict,
            "mixed_bf16" | "bf16" => PrecisionPolicy::MixedBf16,
            "mixed_fp16" | "fp16" => PrecisionPolicy::MixedFp16,
            _ => return None,
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            PrecisionPolicy::Strict => "strict",
            PrecisionPolicy::MixedBf16 => "mixed_bf16",
            PrecisionPolicy::MixedFp16 => "mixed_fp16",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct CompilePolicies {
    pub optimize: OptimizeMode,
    pub device: DevicePolicy,
    pub precision: PrecisionPolicy,
}

impl Default for CompilePolicies {
    fn default() -> Self {
        CompilePolicies {
            optimize: OptimizeMode::Auto,
            device: DevicePolicy::Auto,
            precision: PrecisionPolicy::Strict,
        }
    }
}

#[derive(Clone, Debug)]
pub struct EirStruct {
    pub name: String,
    pub fields: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct EirLocal {
    pub name: String,
    pub ty: Ty,
    pub mutable: bool,
    pub is_param: bool,
    /// Index of the last instruction that reads this local (or None).
    pub last_use: Option<usize>,
    /// True when the value participates in autograd and must survive to
    /// the matching backward pass.
    pub needs_autograd: bool,
}

#[derive(Clone, Debug)]
pub struct EirFunction {
    pub name: String,
    pub params: Vec<Reg>,
    pub locals: Vec<EirLocal>,
    /// Types indexed by register.
    pub types: Vec<Ty>,
    pub code: Vec<Inst>,
    pub ret_ty: Ty,
    pub spans: Vec<(usize, usize)>,
}

#[derive(Clone, Debug)]
pub struct EirProgram {
    pub name: String,
    pub functions: Vec<EirFunction>,
    pub main: usize,
    pub structs: Vec<EirStruct>,
    pub policies: CompilePolicies,
    pub source_hash: u64,
}

#[derive(Clone, Debug)]
pub enum Inst {
    ConstF64 { dst: Reg, v: f64 },
    ConstInt { dst: Reg, v: i64 },
    ConstBool { dst: Reg, v: bool },
    ConstStr { dst: Reg, s: String },
    ConstUnit { dst: Reg },
    Copy { dst: Reg, src: Reg },
    Move { dst: Reg, src: Reg },
    MakeList { dst: Reg, elems: Vec<Reg> },
    MakeTensor { dst: Reg, values: Reg, requires_grad: bool, ty: TensorTy },
    MakeStruct { dst: Reg, struct_id: usize, fields: Vec<Reg> },
    TensorOp { dst: Reg, op: TensorOp, args: Vec<Reg>, dyn_dims: Vec<Reg>, ty: TensorTy },
    FusedOp { dst: Reg, spec: FusedSpec, args: Vec<Reg>, ty: TensorTy },
    BinOp { dst: Reg, op: BinOp, lhs: Reg, rhs: Reg, ty: Ty },
    UnOp { dst: Reg, op: UnOp, src: Reg, ty: Ty },
    Call { dst: Reg, func: usize, args: Vec<Reg>, ty: Ty },
    CallBuiltin { dst: Reg, name: String, args: Vec<Reg>, kwargs: Vec<(String, Reg)>, ty: Ty },
    MethodCall {
        dst: Reg,
        recv: Reg,
        method: String,
        args: Vec<Reg>,
        kwargs: Vec<(String, Reg)>,
        ty: Ty,
    },
    GetAttr { dst: Reg, obj: Reg, name: String, ty: Ty },
    Index { dst: Reg, obj: Reg, index: Reg, ty: Ty },
    Len { dst: Reg, obj: Reg },
    IterInit { dst: Reg, iter: Reg },
    IterNext { dst: Reg, state: Reg, end: usize },
    Jump(usize),
    JumpIfFalse { cond: Reg, target: usize },
    JumpIfTrue { cond: Reg, target: usize },
    Return(Option<Reg>),
    Drop(Reg),
    Print(Reg),
    Input { dst: Reg, prompt: Option<Reg>, ty: Ty },
}

impl Inst {
    pub fn dst(&self) -> Option<Reg> {
        match self {
            Inst::ConstF64 { dst, .. }
            | Inst::ConstInt { dst, .. }
            | Inst::ConstBool { dst, .. }
            | Inst::ConstStr { dst, .. }
            | Inst::ConstUnit { dst }
            | Inst::Copy { dst, .. }
            | Inst::Move { dst, .. }
            | Inst::MakeList { dst, .. }
            | Inst::MakeTensor { dst, .. }
            | Inst::MakeStruct { dst, .. }
            | Inst::TensorOp { dst, .. }
            | Inst::FusedOp { dst, .. }
            | Inst::BinOp { dst, .. }
            | Inst::UnOp { dst, .. }
            | Inst::Call { dst, .. }
            | Inst::CallBuiltin { dst, .. }
            | Inst::MethodCall { dst, .. }
            | Inst::GetAttr { dst, .. }
            | Inst::Index { dst, .. }
            | Inst::Len { dst, .. }
            | Inst::IterInit { dst, .. }
            | Inst::IterNext { dst, .. }
            | Inst::Input { dst, .. } => Some(*dst),
            _ => None,
        }
    }

    pub fn reads(&self) -> Vec<Reg> {
        match self {
            Inst::Copy { src, .. } | Inst::Move { src, .. } | Inst::UnOp { src, .. } => vec![*src],
            Inst::MakeList { elems, .. } => elems.clone(),
            Inst::MakeTensor { values, .. } => vec![*values],
            Inst::MakeStruct { fields, .. } => fields.clone(),
            Inst::TensorOp { args, dyn_dims, .. } => {
                let mut v = args.clone();
                v.extend(dyn_dims.iter().copied());
                v
            }
            Inst::FusedOp { args, .. } => args.clone(),
            Inst::BinOp { lhs, rhs, .. } => vec![*lhs, *rhs],
            Inst::Call { args, .. } => args.clone(),
            Inst::CallBuiltin { args, kwargs, .. } => {
                let mut v = args.clone();
                v.extend(kwargs.iter().map(|(_, r)| *r));
                v
            }
            Inst::MethodCall { recv, args, kwargs, .. } => {
                let mut v = vec![*recv];
                v.extend(args.iter().copied());
                v.extend(kwargs.iter().map(|(_, r)| *r));
                v
            }
            Inst::GetAttr { obj, .. } => vec![*obj],
            Inst::Index { obj, index, .. } => vec![*obj, *index],
            Inst::Len { obj, .. } => vec![*obj],
            Inst::IterInit { iter, .. } => vec![*iter],
            Inst::IterNext { state, .. } => vec![*state],
            Inst::JumpIfFalse { cond, .. } | Inst::JumpIfTrue { cond, .. } => vec![*cond],
            Inst::Return(Some(r)) => vec![*r],
            Inst::Print(r) => vec![*r],
            Inst::Input { prompt: Some(r), .. } => vec![*r],
            Inst::Drop(r) => vec![*r],
            _ => vec![],
        }
    }
}

pub fn tensor_dtype(ty: &Ty) -> DType {
    ty.tensor().map(|t| t.dtype).unwrap_or(DType::F32)
}

impl EirProgram {
    pub fn disassemble(&self) -> String {
        let mut out = String::new();
        let _ = writeln!(out, "; Neura EIR v0.1  program={}", self.name);
        let _ = writeln!(
            out,
            "; policies: optimize={} device={} precision={}",
            self.policies.optimize.name(),
            self.policies.device.name(),
            self.policies.precision.name()
        );
        for s in &self.structs {
            let _ = writeln!(out, "struct {} {{ {} }}", s.name, s.fields.join(", "));
        }
        for f in &self.functions {
            let params: Vec<String> = f
                .params
                .iter()
                .map(|r| format!("%{}: {}", r, ty_name(&f.types, *r)))
                .collect();
            let _ = writeln!(out, "\nfn {} @{} ({}) -> {} {{", f.name, self.functions.iter().position(|x| std::ptr::eq(x, f)).unwrap_or(0), params.join(", "), f.ret_ty);
            for (i, inst) in f.code.iter().enumerate() {
                let _ = writeln!(out, "  .L{}: {}", i, disasm_inst(inst, &f.types));
            }
            let _ = writeln!(out, "}}");
        }
        out
    }

    pub fn find_function(&self, name: &str) -> Option<usize> {
        self.functions.iter().position(|f| f.name == name)
    }
}

fn ty_name(types: &[Ty], r: Reg) -> String {
    types.get(r as usize).map(|t| t.name()).unwrap_or_else(|| "?".into())
}

fn disasm_inst(inst: &Inst, types: &[Ty]) -> String {
    match inst {
        Inst::ConstF64 { dst, v } => format!("%{}: f64 = const {}", dst, v),
        Inst::ConstInt { dst, v } => format!("%{}: i64 = const {}", dst, v),
        Inst::ConstBool { dst, v } => format!("%{}: bool = const {}", dst, v),
        Inst::ConstStr { dst, s } => format!("%{} = const {:?}", dst, s),
        Inst::ConstUnit { dst } => format!("%{} = unit", dst),
        Inst::Copy { dst, src } => format!("%{} = copy %{}", dst, src),
        Inst::Move { dst, src } => format!("%{} = move %{}", dst, src),
        Inst::MakeList { dst, elems } => format!(
            "%{}: {} = list [{}]",
            dst,
            ty_name(types, *dst),
            elems.iter().map(|e| format!("%{}", e)).collect::<Vec<_>>().join(", ")
        ),
        Inst::MakeTensor { dst, values, requires_grad, ty } => format!(
            "%{}: {} = tensor(%{}, requires_grad={})",
            dst, ty, values, requires_grad
        ),
        Inst::MakeStruct { dst, struct_id, fields } => format!(
            "%{} = struct#{} {{{}}}",
            dst,
            struct_id,
            fields.iter().map(|r| format!("%{}", r)).collect::<Vec<_>>().join(", ")
        ),
        Inst::TensorOp { dst, op, args, dyn_dims, ty } => format!(
            "%{}: {} = tir.{} ({}){}",
            dst,
            ty,
            op.name(),
            args.iter().map(|r| format!("%{}", r)).collect::<Vec<_>>().join(", "),
            if dyn_dims.is_empty() {
                String::new()
            } else {
                format!(" dyn({})", dyn_dims.iter().map(|r| format!("%{}", r)).collect::<Vec<_>>().join(", "))
            }
        ),
        Inst::FusedOp { dst, spec, args, ty } => format!(
            "%{}: {} = fused {} [{}]",
            dst,
            ty,
            spec.expr,
            args.iter().map(|r| format!("%{}", r)).collect::<Vec<_>>().join(", ")
        ),
        Inst::BinOp { dst, op, lhs, rhs, .. } => {
            format!("%{} = {} %{} %{}", dst, op.symbol(), lhs, rhs)
        }
        Inst::UnOp { dst, op, src, .. } => format!("%{} = {}%{}", dst, op.symbol(), src),
        Inst::Call { dst, func, args, .. } => format!(
            "%{} = call @{} ({})",
            dst,
            func,
            args.iter().map(|r| format!("%{}", r)).collect::<Vec<_>>().join(", ")
        ),
        Inst::CallBuiltin { dst, name, args, kwargs, .. } => {
            let mut parts: Vec<String> = args.iter().map(|r| format!("%{}", r)).collect();
            for (k, r) in kwargs {
                parts.push(format!("{} = %{}", k, r));
            }
            format!("%{} = call_builtin {:?}({})", dst, name, parts.join(", "))
        }
        Inst::MethodCall { dst, recv, method, args, kwargs, .. } => {
            let mut parts: Vec<String> = args.iter().map(|r| format!("%{}", r)).collect();
            for (k, r) in kwargs {
                parts.push(format!("{} = %{}", k, r));
            }
            format!("%{} = method %{}.{}({})", dst, recv, method, parts.join(", "))
        }
        Inst::GetAttr { dst, obj, name, .. } => format!("%{} = getattr %{}.{}", dst, obj, name),
        Inst::Index { dst, obj, index, .. } => format!("%{} = index %{}[%{}]", dst, obj, index),
        Inst::Len { dst, obj } => format!("%{} = len %{}", dst, obj),
        Inst::IterInit { dst, iter } => format!("%{} = iter_init %{}", dst, iter),
        Inst::IterNext { dst, state, end } => format!("%{} = iter_next %{} else .L{}", dst, state, end),
        Inst::Jump(t) => format!("jump .L{}", t),
        Inst::JumpIfFalse { cond, target } => format!("branch_if_false %{} .L{}", cond, target),
        Inst::JumpIfTrue { cond, target } => format!("branch_if_true %{} .L{}", cond, target),
        Inst::Return(Some(r)) => format!("return %{}", r),
        Inst::Return(None) => "return".into(),
        Inst::Drop(r) => format!("drop %{}", r),
        Inst::Print(r) => format!("print %{}", r),
        Inst::Input { dst, prompt, ty } => match prompt {
            Some(p) => format!("%{}: {} = input %{}", dst, ty.name(), p),
            None => format!("%{}: {} = input", dst, ty.name()),
        },
    }
}

/// Names used in diagnostic / introspection output.
pub fn ty_debug(types: &[Ty], r: Reg) -> String {
    ty_name(types, r)
}

pub fn tensor_ty_of(types: &[Ty], r: Reg) -> Option<TensorTy> {
    types.get(r as usize).and_then(|t| t.tensor().cloned())
}
