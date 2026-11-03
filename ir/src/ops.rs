//! Tensor and scalar operation definitions shared by TIR and EIR.

use crate::types::{Dim, DType, TensorTy};
use std::fmt;

/// Scalar binary operators.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    FloorDiv,
    Mod,
    Pow,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    And,
    Or,
}

impl BinOp {
    pub fn symbol(self) -> &'static str {
        match self {
            BinOp::Add => "+",
            BinOp::Sub => "-",
            BinOp::Mul => "*",
            BinOp::Div => "/",
            BinOp::FloorDiv => "//",
            BinOp::Mod => "%",
            BinOp::Pow => "**",
            BinOp::Eq => "==",
            BinOp::Ne => "!=",
            BinOp::Lt => "<",
            BinOp::Le => "<=",
            BinOp::Gt => ">",
            BinOp::Ge => ">=",
            BinOp::And => "&&",
            BinOp::Or => "||",
        }
    }
}

/// Scalar unary operators.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum UnOp {
    Neg,
    Not,
}

impl UnOp {
    pub fn symbol(self) -> &'static str {
        match self {
            UnOp::Neg => "-",
            UnOp::Not => "!",
        }
    }
}

/// Every tensor operation understood by TIR, the optimizer and the runtime.
///
/// Shape parameters are symbolic dimensions so dynamic and symbolic shapes
/// survive the whole pipeline. Operations that need runtime-computed shape
/// values carry them out-of-band (see `eir::Inst::TensorOp::dyn_dims`).
#[derive(Clone, PartialEq, Debug)]
pub enum TensorOp {
    Add,
    Sub,
    Mul,
    Div,
    MatMul,
    Transpose { dim0: usize, dim1: usize },
    Reshape { dims: Vec<Dim> },
    Flatten,
    Slice { axis: usize, start: usize, end: usize },
    Sum { axes: Vec<usize>, keepdim: bool },
    Mean { axes: Vec<usize>, keepdim: bool },
    ReduceMax { axis: Option<usize>, keepdim: bool },
    ReduceMin { axis: Option<usize>, keepdim: bool },
    ReLU,
    GELU,
    Sigmoid,
    Tanh,
    Exp,
    Log,
    Sqrt,
    Neg,
    Softmax { axis: usize },
    Conv2D { stride: (usize, usize), padding: (usize, usize) },
    Conv1D { stride: usize, padding: usize },
    Gather { axis: usize },
    BroadcastTo { shape: Vec<Dim> },
    Dropout { p: f64 },
}

impl TensorOp {
    /// True when the op is a pure elementwise map with no shape-changing
    /// parameters (fusion candidates).
    pub fn is_elementwise(&self) -> bool {
        matches!(
            self,
            TensorOp::Add
                | TensorOp::Sub
                | TensorOp::Mul
                | TensorOp::Div
                | TensorOp::ReLU
                | TensorOp::GELU
                | TensorOp::Sigmoid
                | TensorOp::Tanh
                | TensorOp::Exp
                | TensorOp::Log
                | TensorOp::Sqrt
                | TensorOp::Neg
        )
    }

    pub fn is_unary_elementwise(&self) -> bool {
        matches!(
            self,
            TensorOp::ReLU
                | TensorOp::GELU
                | TensorOp::Sigmoid
                | TensorOp::Tanh
                | TensorOp::Exp
                | TensorOp::Log
                | TensorOp::Sqrt
                | TensorOp::Neg
        )
    }

    pub fn name(&self) -> String {
        match self {
            TensorOp::Add => "add".into(),
            TensorOp::Sub => "sub".into(),
            TensorOp::Mul => "mul".into(),
            TensorOp::Div => "div".into(),
            TensorOp::MatMul => "matmul".into(),
            TensorOp::Transpose { dim0, dim1 } => format!("transpose<{}:{}>", dim0, dim1),
            TensorOp::Reshape { dims } => format!("reshape<{}>", fmt_dims(dims)),
            TensorOp::Flatten => "flatten".into(),
            TensorOp::Slice { axis, start, end } => format!("slice<{}:{}:{}>", axis, start, end),
            TensorOp::Sum { axes, keepdim } => {
                format!("sum<{}:{}>", fmt_axes(axes), keepdim)
            }
            TensorOp::Mean { axes, keepdim } => {
                format!("mean<{}:{}>", fmt_axes(axes), keepdim)
            }
            TensorOp::ReduceMax { axis, keepdim } => {
                format!("max<{}:{}>", fmt_axis(*axis), keepdim)
            }
            TensorOp::ReduceMin { axis, keepdim } => {
                format!("min<{}:{}>", fmt_axis(*axis), keepdim)
            }
            TensorOp::ReLU => "relu".into(),
            TensorOp::GELU => "gelu".into(),
            TensorOp::Sigmoid => "sigmoid".into(),
            TensorOp::Tanh => "tanh".into(),
            TensorOp::Exp => "exp".into(),
            TensorOp::Log => "log".into(),
            TensorOp::Sqrt => "sqrt".into(),
            TensorOp::Neg => "neg".into(),
            TensorOp::Softmax { axis } => format!("softmax<{}>", axis),
            TensorOp::Conv2D { stride, padding } => {
                format!("conv2d<{}x{}:{}x{}>", stride.0, stride.1, padding.0, padding.1)
            }
            TensorOp::Conv1D { stride, padding } => format!("conv1d<{}:{}>", stride, padding),
            TensorOp::Gather { axis } => format!("gather<{}>", axis),
            TensorOp::BroadcastTo { shape } => format!("broadcast_to<{}>", fmt_dims(shape)),
            TensorOp::Dropout { p } => format!("dropout<{}>", p),
        }
    }

    /// MLIR-style mnemonic used by the textual Neura dialect printer.
    pub fn mlir_mnemonic(&self) -> String {
        let base = match self {
            TensorOp::Add => "add",
            TensorOp::Sub => "sub",
            TensorOp::Mul => "mul",
            TensorOp::Div => "div",
            TensorOp::MatMul => "matmul",
            TensorOp::Transpose { .. } => "transpose",
            TensorOp::Reshape { .. } => "reshape",
            TensorOp::Flatten => "flatten",
            TensorOp::Slice { .. } => "slice",
            TensorOp::Sum { .. } => "sum",
            TensorOp::Mean { .. } => "mean",
            TensorOp::ReduceMax { .. } => "reduce_max",
            TensorOp::ReduceMin { .. } => "reduce_min",
            TensorOp::ReLU => "relu",
            TensorOp::GELU => "gelu",
            TensorOp::Sigmoid => "sigmoid",
            TensorOp::Tanh => "tanh",
            TensorOp::Exp => "exp",
            TensorOp::Log => "log",
            TensorOp::Sqrt => "sqrt",
            TensorOp::Neg => "neg",
            TensorOp::Softmax { .. } => "softmax",
            TensorOp::Conv2D { .. } => "conv2d",
            TensorOp::Conv1D { .. } => "conv1d",
            TensorOp::Gather { .. } => "gather",
            TensorOp::BroadcastTo { .. } => "broadcast_to",
            TensorOp::Dropout { .. } => "dropout",
        };
        format!("neura_tir.{}", base)
    }
}

fn fmt_dims(dims: &[Dim]) -> String {
    dims.iter().map(|d| d.to_string()).collect::<Vec<_>>().join("x")
}

fn fmt_axes(axes: &[usize]) -> String {
    axes.iter().map(|a| a.to_string()).collect::<Vec<_>>().join(",")
}

fn fmt_axis(axis: Option<usize>) -> String {
    match axis {
        Some(a) => a.to_string(),
        None => "all".into(),
    }
}

/// Elementwise expression tree used by fused kernels.
///
/// This is intentionally a small, explicit tree (not a closure) so fused
/// kernels can be inspected, printed and compared.
#[derive(Clone, PartialEq, Debug)]
pub enum FusedExpr {
    Input(u32),
    Const(f64),
    Add(Box<FusedExpr>, Box<FusedExpr>),
    Sub(Box<FusedExpr>, Box<FusedExpr>),
    Mul(Box<FusedExpr>, Box<FusedExpr>),
    Div(Box<FusedExpr>, Box<FusedExpr>),
    Neg(Box<FusedExpr>),
    ReLU(Box<FusedExpr>),
    GELU(Box<FusedExpr>),
    Sigmoid(Box<FusedExpr>),
    Tanh(Box<FusedExpr>),
    Exp(Box<FusedExpr>),
    Log(Box<FusedExpr>),
    Sqrt(Box<FusedExpr>),
}

impl FusedExpr {
    pub fn eval(&self, inputs: &[f64]) -> f64 {
        match self {
            FusedExpr::Input(i) => inputs.get(*i as usize).copied().unwrap_or(0.0),
            FusedExpr::Const(v) => *v,
            FusedExpr::Add(a, b) => a.eval(inputs) + b.eval(inputs),
            FusedExpr::Sub(a, b) => a.eval(inputs) - b.eval(inputs),
            FusedExpr::Mul(a, b) => a.eval(inputs) * b.eval(inputs),
            FusedExpr::Div(a, b) => a.eval(inputs) / b.eval(inputs),
            FusedExpr::Neg(a) => -a.eval(inputs),
            FusedExpr::ReLU(a) => a.eval(inputs).max(0.0),
            FusedExpr::GELU(a) => gelu(a.eval(inputs)),
            FusedExpr::Sigmoid(a) => sigmoid(a.eval(inputs)),
            FusedExpr::Tanh(a) => a.eval(inputs).tanh(),
            FusedExpr::Exp(a) => a.eval(inputs).exp(),
            FusedExpr::Log(a) => a.eval(inputs).ln(),
            FusedExpr::Sqrt(a) => a.eval(inputs).sqrt(),
        }
    }

    /// Number of leaf inputs referenced (not necessarily dense).
    pub fn input_count(&self) -> u32 {
        fn walk(e: &FusedExpr, max: &mut u32) {
            match e {
                FusedExpr::Input(i) => *max = (*max).max(*i + 1),
                FusedExpr::Const(_) => {}
                FusedExpr::Add(a, b)
                | FusedExpr::Sub(a, b)
                | FusedExpr::Mul(a, b)
                | FusedExpr::Div(a, b) => {
                    walk(a, max);
                    walk(b, max);
                }
                FusedExpr::Neg(a)
                | FusedExpr::ReLU(a)
                | FusedExpr::GELU(a)
                | FusedExpr::Sigmoid(a)
                | FusedExpr::Tanh(a)
                | FusedExpr::Exp(a)
                | FusedExpr::Log(a)
                | FusedExpr::Sqrt(a) => walk(a, max),
            }
        }
        let mut m = 0;
        walk(self, &mut m);
        m
    }
}

impl fmt::Display for FusedExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FusedExpr::Input(i) => write!(f, "%{}", i),
            FusedExpr::Const(v) => write!(f, "{}", v),
            FusedExpr::Add(a, b) => write!(f, "({} + {})", a, b),
            FusedExpr::Sub(a, b) => write!(f, "({} - {})", a, b),
            FusedExpr::Mul(a, b) => write!(f, "({} * {})", a, b),
            FusedExpr::Div(a, b) => write!(f, "({} / {})", a, b),
            FusedExpr::Neg(a) => write!(f, "(-{})", a),
            FusedExpr::ReLU(a) => write!(f, "relu({})", a),
            FusedExpr::GELU(a) => write!(f, "gelu({})", a),
            FusedExpr::Sigmoid(a) => write!(f, "sigmoid({})", a),
            FusedExpr::Tanh(a) => write!(f, "tanh({})", a),
            FusedExpr::Exp(a) => write!(f, "exp({})", a),
            FusedExpr::Log(a) => write!(f, "log({})", a),
            FusedExpr::Sqrt(a) => write!(f, "sqrt({})", a),
        }
    }
}

#[derive(Clone, PartialEq, Debug)]
pub struct FusedSpec {
    pub expr: FusedExpr,
    pub ty: TensorTy,
}

/// GELU (tanh approximation), used by both eager and fused kernels.
pub fn gelu(x: f64) -> f64 {
    const C: f64 = 0.797_884_560_802_865_4; // sqrt(2/pi)
    0.5 * x * (1.0 + (C * (x + 0.044715 * x * x * x)).tanh())
}

pub fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Dtype-aware elementwise map for eager kernels.
pub fn unary_elem(op: &TensorOp, x: f64) -> f64 {
    match op {
        TensorOp::ReLU => x.max(0.0),
        TensorOp::GELU => gelu(x),
        TensorOp::Sigmoid => sigmoid(x),
        TensorOp::Tanh => x.tanh(),
        TensorOp::Exp => x.exp(),
        TensorOp::Log => x.ln(),
        TensorOp::Sqrt => x.sqrt(),
        TensorOp::Neg => -x,
        _ => x,
    }
}

/// Tensor result shape/dtype after applying `op` to `inputs`.
pub fn infer_shape(op: &TensorOp, inputs: &[TensorTy]) -> Result<TensorTy, String> {
    use crate::shape;
    match op {
        TensorOp::Add | TensorOp::Sub | TensorOp::Mul | TensorOp::Div => {
            if inputs.len() != 2 {
                return Err(format!("{} expects 2 inputs", op.name()));
            }
            let a = &inputs[0];
            let b = &inputs[1];
            Ok(TensorTy {
                dtype: a.dtype.promote(b.dtype),
                shape: shape::broadcast_shapes(&a.shape, &b.shape)?,
            })
        }
        TensorOp::MatMul => {
            if inputs.len() != 2 {
                return Err("matmul expects 2 inputs".into());
            }
            let (s, _) = shape::matmul_shape(&inputs[0].shape, &inputs[1].shape)?;
            Ok(TensorTy {
                dtype: inputs[0].dtype.promote(inputs[1].dtype),
                shape: s,
            })
        }
        TensorOp::Transpose { dim0, dim1 } => {
            let t = inputs.first().ok_or("transpose expects 1 input")?;
            let mut sh = t.shape.clone();
            if *dim0 >= sh.len() || *dim1 >= sh.len() {
                return Err("transpose dimension out of range".into());
            }
            sh.swap(*dim0, *dim1);
            Ok(TensorTy::new(t.dtype, sh))
        }
        TensorOp::Reshape { dims } => {
            let t = inputs.first().ok_or("reshape expects 1 input")?;
            let resolved = shape::resolve_reshape(&t.shape, dims)?;
            Ok(TensorTy::new(t.dtype, resolved))
        }
        TensorOp::Flatten => {
            let t = inputs.first().ok_or("flatten expects 1 input")?;
            let n = if t.shape.iter().all(|d| d.concrete().is_some()) {
                Dim::Static(t.shape.iter().map(|d| d.concrete().unwrap()).product())
            } else {
                Dim::Unknown
            };
            Ok(TensorTy::new(t.dtype, vec![n]))
        }
        TensorOp::Slice { axis, start, end } => {
            let t = inputs.first().ok_or("slice expects 1 input")?;
            if *axis >= t.shape.len() {
                return Err("slice axis out of range".into());
            }
            let mut sh = t.shape.clone();
            sh[*axis] = Dim::Static(end.saturating_sub(*start));
            Ok(TensorTy::new(t.dtype, sh))
        }
        TensorOp::Sum { axes, keepdim } | TensorOp::Mean { axes, keepdim } => {
            let t = inputs.first().ok_or("reduction expects 1 input")?;
            Ok(TensorTy::new(t.dtype, shape::reduce_shape(&t.shape, axes, *keepdim)))
        }
        TensorOp::ReduceMax { axis, keepdim } | TensorOp::ReduceMin { axis, keepdim } => {
            let t = inputs.first().ok_or("reduction expects 1 input")?;
            let axes: Vec<usize> = axis.map(|a| vec![a]).unwrap_or_else(|| (0..t.shape.len()).collect());
            Ok(TensorTy::new(t.dtype, shape::reduce_shape(&t.shape, &axes, *keepdim)))
        }
        TensorOp::ReLU
        | TensorOp::GELU
        | TensorOp::Sigmoid
        | TensorOp::Tanh
        | TensorOp::Exp
        | TensorOp::Log
        | TensorOp::Sqrt
        | TensorOp::Neg
        | TensorOp::Dropout { .. } => {
            let t = inputs.first().ok_or("unary op expects 1 input")?;
            Ok(t.clone())
        }
        TensorOp::Softmax { axis } => {
            let t = inputs.first().ok_or("softmax expects 1 input")?;
            if *axis >= t.shape.len() && !t.shape.is_empty() {
                return Err("softmax axis out of range".into());
            }
            Ok(TensorTy::new(
                if t.dtype.is_float() { t.dtype } else { DType::F32 },
                t.shape.clone(),
            ))
        }
        TensorOp::Conv2D { stride, padding } => {
            if inputs.len() != 2 {
                return Err("conv2d expects input and weight".into());
            }
            let (x, w) = (&inputs[0], &inputs[1]);
            if x.shape.len() != 4 || w.shape.len() != 4 {
                return Err("conv2d expects NCHW input and OIHW weight".into());
            }
            let (h, wd) = (x.shape[2].clone(), x.shape[3].clone());
            let (kh, kw) = (w.shape[2].clone(), w.shape[3].clone());
            let out_h = shape::conv_out_dim(&h, &kh, stride.0, padding.0);
            let out_w = shape::conv_out_dim(&wd, &kw, stride.1, padding.1);
            Ok(TensorTy::new(
                x.dtype.promote(w.dtype),
                vec![x.shape[0].clone(), w.shape[0].clone(), out_h, out_w],
            ))
        }
        TensorOp::Conv1D { stride, padding } => {
            if inputs.len() != 2 {
                return Err("conv1d expects input and weight".into());
            }
            let (x, w) = (&inputs[0], &inputs[1]);
            if x.shape.len() != 3 || w.shape.len() != 3 {
                return Err("conv1d expects NCW input and OCW weight".into());
            }
            let out = shape::conv_out_dim(&x.shape[2], &w.shape[2], *stride, *padding);
            Ok(TensorTy::new(
                x.dtype.promote(w.dtype),
                vec![x.shape[0].clone(), w.shape[0].clone(), out],
            ))
        }
        TensorOp::Gather { axis } => {
            if inputs.len() != 2 {
                return Err("gather expects data and indices".into());
            }
            let (data, idx) = (&inputs[0], &inputs[1]);
            if *axis >= data.shape.len() {
                return Err("gather axis out of range".into());
            }
            if idx.shape.len() != data.shape.len() {
                return Err("gather expects indices with the same rank as the data".into());
            }
            Ok(TensorTy::new(data.dtype, idx.shape.clone()))
        }
        TensorOp::BroadcastTo { shape } => {
            let t = inputs.first().ok_or("broadcast_to expects 1 input")?;
            Ok(TensorTy::new(t.dtype, shape.clone()))
        }
    }
}
