//! Standard-library signatures used by the type checker.
//!
//! Library calls are resolved statically to canonical names (`nn.Linear`,
//! `tensor.matmul`, ...). The runtime implements the same names; there is no
//! dynamic Python-side dispatch.

use neura_ir::ops::{infer_shape, TensorOp};
use neura_ir::types::{Dim, DType, TensorTy, Ty};

pub const MODULES: &[&str] = &[
    "tensor",
    "nn",
    "optim",
    "math",
    "data",
    "autograd",
    "distributed",
    "inference",
    "serialization",
    "device",
];

pub fn is_known_module(m: &str) -> bool {
    MODULES.contains(&m)
}

pub fn is_known_module_path(path: &str) -> bool {
    let last = path.split('.').next_back().unwrap_or(path);
    path.starts_with("neura") && is_known_module(last)
}

fn f32_tensor(shape: Vec<Dim>) -> Ty {
    Ty::Tensor(TensorTy::new(DType::F32, shape))
}

fn arg_ty<'a>(args: &'a [Ty], i: usize) -> Option<&'a Ty> {
    args.get(i)
}

fn int_ty(t: &Ty) -> bool {
    matches!(t, Ty::Int | Ty::Any)
}

fn float_ty(t: &Ty) -> bool {
    matches!(t, Ty::Float | Ty::Int | Ty::Any)
}

fn kw<'a>(kwargs: &'a [(String, Ty)], name: &str) -> Option<&'a Ty> {
    kwargs.iter().find(|(k, _)| k == name).map(|(_, t)| t)
}

fn scalar_tensor(ty: &Ty) -> Option<TensorTy> {
    match ty {
        Ty::Tensor(t) => Some(t.clone()),
        Ty::Int => Some(TensorTy::new(DType::I64, vec![])),
        Ty::Float => Some(TensorTy::new(DType::F64, vec![])),
        Ty::Bool => Some(TensorTy::new(DType::Bool, vec![])),
        _ => None,
    }
}

fn require_tensor(name: &str, args: &[Ty], i: usize) -> Result<TensorTy, String> {
    match arg_ty(args, i) {
        Some(Ty::Tensor(t)) => Ok(t.clone()),
        Some(Ty::Any) => Ok(TensorTy::new(DType::F32, vec![Dim::Unknown])),
        other => Err(format!(
            "`{}` expects a tensor for argument {}, found {}",
            name,
            i + 1,
            other.map(|t| t.name()).unwrap_or_else(|| "nothing".into())
        )),
    }
}

fn tensor_bin_op(op: TensorOp, name: &str, args: &[Ty]) -> Result<Ty, String> {
    if args.len() != 2 {
        return Err(format!("`{}` expects 2 arguments", name));
    }
    let a = scalar_tensor(&args[0]);
    let b = scalar_tensor(&args[1]);
    match (a, b) {
        (Some(a), Some(b)) => {
            let ty = infer_shape(&op, &[a, b])?;
            Ok(Ty::Tensor(ty))
        }
        _ => Err(format!("`{}` expects numeric tensor arguments", name)),
    }
}

fn tensor_unary_op(op: TensorOp, name: &str, args: &[Ty]) -> Result<Ty, String> {
    let t = require_tensor(name, args, 0)?;
    Ok(Ty::Tensor(infer_shape(&op, &[t])?))
}

/// Parse `Linear<512>`-style opaque layer types.
pub fn opaque_param(ty: &Ty, prefix: &str) -> Option<usize> {
    if let Ty::Opaque(s) = ty {
        if let Some(rest) = s.strip_prefix(prefix) {
            let rest = rest.trim_start_matches('<').trim_end_matches('>');
            return rest.parse().ok();
        }
    }
    None
}

fn literal_int(_ty: &Ty) -> Option<usize> {
    None
}

/// Type of a standard-library call.
pub fn call_builtin(name: &str, args: &[Ty], kwargs: &[(String, Ty)]) -> Result<Ty, String> {
    let out = match name {
        // ---- core builtins -------------------------------------------------
        "print" => Ty::Unit,
        "input" => {
            if args.len() > 1 {
                return Err("`input` expects at most 1 argument".into());
            }
            if let Some(prompt) = args.first() {
                if !matches!(prompt, Ty::Str | Ty::Any) {
                    return Err("`input` expects a string prompt".into());
                }
            }
            Ty::Str
        }
        "len" => {
            match arg_ty(args, 0) {
                Some(Ty::List(_))
                | Some(Ty::Str)
                | Some(Ty::Tensor(_))
                | Some(Ty::Opaque(_))
                | Some(Ty::Any) => {}
                other => {
                    return Err(format!(
                        "`len` expects a list, string, tensor or dataset, found {}",
                        other.map(|t| t.name()).unwrap_or_else(|| "nothing".into())
                    ))
                }
            }
            Ty::Int
        }
        "range" => {
            if args.is_empty() || args.len() > 2 {
                return Err("`range` expects 1 or 2 integer arguments".into());
            }
            if !args.iter().all(int_ty) {
                return Err("`range` arguments must be integers".into());
            }
            Ty::Opaque("range".into())
        }
        "abs" | "sqrt" | "exp" | "log" | "tanh" => {
            let a = arg_ty(args, 0).ok_or_else(|| format!("`{}` expects 1 argument", name))?;
            match a {
                Ty::Int | Ty::Float => Ty::Float,
                Ty::Tensor(_) => {
                    let op = match name {
                        "sqrt" => TensorOp::Sqrt,
                        "exp" => TensorOp::Exp,
                        "log" => TensorOp::Log,
                        "tanh" => TensorOp::Tanh,
                        _ => TensorOp::ReLU,
                    };
                    return tensor_unary_op(op, name, args);
                }
                _ => return Err(format!("`{}` expects a number or tensor", name)),
            }
        }
        "int" => Ty::Int,
        "float" => Ty::Float,
        // ---- tensor --------------------------------------------------------
        "tensor.Tensor" => {
            let t = match arg_ty(args, 0) {
                Some(Ty::List(_)) | Some(Ty::Int) | Some(Ty::Float) | Some(Ty::Any) => {}
                _ => return Err("`Tensor` expects a list of numbers".into()),
            };
            let _ = t;
            if let Some(d) = kw(kwargs, "dtype") {
                if let Ty::Str = d {
                } else {
                    return Err("`dtype` must be a string literal such as \"f32\"".into());
                }
            }
            f32_tensor(vec![Dim::Unknown])
        }
        "tensor.scalar" => f32_tensor(vec![]),
        "tensor.zeros" | "tensor.ones" | "tensor.randn" => {
            let mut dims = Vec::new();
            for a in args {
                if !int_ty(a) {
                    return Err(format!("`{}` expects integer dimensions", name));
                }
                dims.push(Dim::Unknown);
            }
            f32_tensor(dims)
        }
        "tensor.arange" => f32_tensor(vec![Dim::Unknown]),
        "tensor.matmul" => tensor_bin_op(TensorOp::MatMul, name, args)?,
        "tensor.add" => tensor_bin_op(TensorOp::Add, name, args)?,
        "tensor.sub" => tensor_bin_op(TensorOp::Sub, name, args)?,
        "tensor.mul" => tensor_bin_op(TensorOp::Mul, name, args)?,
        "tensor.div" => tensor_bin_op(TensorOp::Div, name, args)?,
        "tensor.relu" => tensor_unary_op(TensorOp::ReLU, name, args)?,
        "tensor.gelu" => tensor_unary_op(TensorOp::GELU, name, args)?,
        "tensor.sigmoid" => tensor_unary_op(TensorOp::Sigmoid, name, args)?,
        "tensor.tanh" => tensor_unary_op(TensorOp::Tanh, name, args)?,
        "tensor.exp" => tensor_unary_op(TensorOp::Exp, name, args)?,
        "tensor.log" => tensor_unary_op(TensorOp::Log, name, args)?,
        "tensor.sqrt" => tensor_unary_op(TensorOp::Sqrt, name, args)?,
        "tensor.softmax" => {
            let t = require_tensor(name, args, 0)?;
            Ty::Tensor(t)
        }
        "tensor.transpose" => {
            let t = require_tensor(name, args, 0)?;
            Ty::Tensor(t)
        }
        "tensor.reshape" | "tensor.flatten" => {
            let t = require_tensor(name, args, 0)?;
            Ty::Tensor(t)
        }
        "tensor.sum" | "tensor.mean" => {
            let t = require_tensor(name, args, 0)?;
            Ty::Tensor(TensorTy::new(t.dtype, vec![]))
        }
        "tensor.conv2d" | "tensor.conv1d" => {
            let _ = require_tensor(name, args, 0)?;
            let _ = require_tensor(name, args, 1)?;
            f32_tensor(vec![Dim::Unknown, Dim::Unknown, Dim::Unknown, Dim::Unknown])
        }
        // ---- nn ------------------------------------------------------------
        "nn.Linear" => {
            if args.len() < 2 || args.len() > 3 {
                return Err("`nn.Linear` expects (in_features, out_features, bias=True)".into());
            }
            if !int_ty(&args[0]) || !int_ty(&args[1]) {
                return Err("`nn.Linear` features must be integers".into());
            }
            match literal_int(&args[1]) {
                Some(n) => Ty::Opaque(format!("Linear<{}>", n)),
                None => Ty::Opaque("Linear<?>".into()),
            }
        }
        "nn.ReLU" => Ty::Opaque("ReLU".into()),
        "nn.GELU" => Ty::Opaque("GELU".into()),
        "nn.Sigmoid" => Ty::Opaque("Sigmoid".into()),
        "nn.Softmax" => Ty::Opaque("Softmax".into()),
        "nn.Tanh" => Ty::Opaque("Tanh".into()),
        "nn.Dropout" => Ty::Opaque("Dropout".into()),
        "nn.LayerNorm" => Ty::Opaque("LayerNorm".into()),
        "nn.BatchNorm" => Ty::Opaque("BatchNorm".into()),
        "nn.Embedding" => {
            if args.len() < 2 {
                return Err("`nn.Embedding` expects (num_embeddings, embedding_dim)".into());
            }
            Ty::Opaque("Embedding".into())
        }
        "nn.Conv1D" => {
            if args.len() < 3 {
                return Err("`nn.Conv1D` expects (in_channels, out_channels, kernel_size)".into());
            }
            Ty::Opaque("Conv1D".into())
        }
        "nn.Conv2D" => {
            if args.len() < 3 {
                return Err("`nn.Conv2D` expects (in_channels, out_channels, kernel_size)".into());
            }
            Ty::Opaque("Conv2D".into())
        }
        "nn.Sequential" => {
            match arg_ty(args, 0) {
                Some(Ty::List(_)) | Some(Ty::Any) => {}
                _ => return Err("`nn.Sequential` expects a list of layers".into()),
            }
            Ty::Opaque("Sequential<?>".into())
        }
        "nn.load_dataset" | "nn.load_csv" | "data.load_csv" => {
            match arg_ty(args, 0) {
                Some(Ty::Str) | Some(Ty::Any) => {}
                _ => return Err("dataset loaders expect a string path".into()),
            }
            Ty::Opaque("Dataset".into())
        }
        "nn.cross_entropy" => {
            let _ = require_tensor(name, args, 0)?;
            let _ = require_tensor(name, args, 1)?;
            f32_tensor(vec![])
        }
        "nn.save" | "serialization.save" => Ty::Unit,
        "nn.load" | "serialization.load" => Ty::Unit,
        "nn.parameters" => Ty::List(Box::new(f32_tensor(vec![Dim::Unknown]))),
        "nn.train" => Ty::Unit,
        "nn.forward" | "nn.predict" | "inference.predict" => {
            if args.len() >= 2 {
                scalar_tensor(&args[1]).map(Ty::Tensor).unwrap_or_else(|| f32_tensor(vec![Dim::Unknown]))
            } else {
                f32_tensor(vec![Dim::Unknown])
            }
        }
        // ---- optim ---------------------------------------------------------
        "optim.SGD" | "optim.Adam" | "optim.AdamW" | "optim.RMSProp" => {
            if args.is_empty() {
                return Err(format!("`{}` expects a parameter list", name));
            }
            if let Some(rate) = kw(kwargs, "learning_rate") {
                if !float_ty(rate) {
                    return Err("`learning_rate` must be a number".into());
                }
            }
            Ty::Opaque(name.split('.').next_back().unwrap().to_string())
        }
        // ---- math ----------------------------------------------------------
        "math.sqrt" => return tensor_unary_op(TensorOp::Sqrt, name, args),
        "math.exp" => return tensor_unary_op(TensorOp::Exp, name, args),
        "math.log" => return tensor_unary_op(TensorOp::Log, name, args),
        "math.tanh" => return tensor_unary_op(TensorOp::Tanh, name, args),
        // ---- autograd ------------------------------------------------------
        "autograd.backward" => Ty::Unit,
        // ---- device --------------------------------------------------------
        "device.available" => Ty::List(Box::new(Ty::Str)),
        "device.count" => Ty::Int,
        "device.default" => Ty::Str,
        "distributed.rank" | "distributed.world_size" => Ty::Int,
        _ => return Err(format!("unknown function `{}`", name)),
    };
    Ok(out)
}

/// Tensor elementwise binary operation between a tensor and a scalar/tensor.
pub fn binary_tensor_ty(op: TensorOp, lhs: &Ty, rhs: &Ty) -> Result<Ty, String> {
    let a = scalar_tensor(lhs);
    let b = scalar_tensor(rhs);
    match (a, b) {
        (Some(a), Some(b)) => {
            let out = infer_shape(&op, &[a, b])?;
            Ok(Ty::Tensor(out))
        }
        _ => Err("operator requires numeric operands".into()),
    }
}

/// Replace the last dimension of a tensor type with `n` (layer output width).
pub fn replace_last(ty: &Ty, n: Option<usize>) -> Ty {
    match ty {
        Ty::Tensor(t) => {
            let mut shape = t.shape.clone();
            if shape.is_empty() {
                shape.push(Dim::Static(1));
            }
            if let Some(n) = n {
                let idx = shape.len() - 1;
                shape[idx] = Dim::Static(n);
            } else {
                let idx = shape.len() - 1;
                shape[idx] = Dim::Unknown;
            }
            Ty::Tensor(TensorTy::new(t.dtype, shape))
        }
        _ => f32_tensor(vec![Dim::Unknown]),
    }
}
