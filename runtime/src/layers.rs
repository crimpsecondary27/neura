//! Neural-network layers built from primitive tensor operations.
//!
//! Every layer's forward pass is expressed with the same differentiable ops the
//! language exposes, so gradients come from the single autograd engine rather
//! than per-layer hand-written backward code.

use crate::autograd::{self, apply};
use crate::data::Dataset;
use crate::optim::Optimizer;
use crate::value::{tensor, trainable, TRef, TNode};
use neura_backends::tensor::Tensor;
use neura_ir::ops::TensorOp;
use neura_ir::types::{Dim, DType};
use std::cell::RefCell;
use std::fs;
use std::path::Path;
use std::rc::Rc;

pub enum Layer {
    Linear { w: TRef, b: Option<TRef>, in_features: usize, out_features: usize },
    Activation(Activation),
    Softmax { axis: Option<usize> },
    Dropout { p: f64 },
    LayerNorm { gamma: TRef, beta: TRef, dim: usize, eps: f64 },
    BatchNorm {
        gamma: TRef,
        beta: TRef,
        running_mean: RefCell<Tensor>,
        running_var: RefCell<Tensor>,
        dim: usize,
        eps: f64,
    },
    Embedding { table: TRef, vocab: usize, dim: usize },
    Conv2D {
        w: TRef,
        b: Option<TRef>,
        in_channels: usize,
        out_channels: usize,
        kernel: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    },
    Conv1D {
        w: TRef,
        b: Option<TRef>,
        in_channels: usize,
        out_channels: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
    },
    Sequential(Vec<Rc<Layer>>),
}

/// One step of a streaming forward pass.
#[derive(Clone, Debug)]
pub struct StreamStep {
    pub layer: String,
    pub shape: Vec<usize>,
    pub bytes: usize,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    ReLU,
    GELU,
    Sigmoid,
    Tanh,
}

impl Activation {
    pub fn op(self) -> TensorOp {
        match self {
            Activation::ReLU => TensorOp::ReLU,
            Activation::GELU => TensorOp::GELU,
            Activation::Sigmoid => TensorOp::Sigmoid,
            Activation::Tanh => TensorOp::Tanh,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Activation::ReLU => "ReLU",
            Activation::GELU => "GELU",
            Activation::Sigmoid => "Sigmoid",
            Activation::Tanh => "Tanh",
        }
    }
}

fn matmul(a: &TRef, b: &TRef) -> Result<TRef, String> {
    apply(&TensorOp::MatMul, &[a.clone(), b.clone()], true, 0)
}

fn bin(op: TensorOp, a: &TRef, b: &TRef) -> Result<TRef, String> {
    apply(&op, &[a.clone(), b.clone()], true, 0)
}

fn unary(op: TensorOp, a: &TRef) -> Result<TRef, String> {
    apply(&op, &[a.clone()], true, 0)
}

fn constant(v: f64, shape: Vec<usize>, dtype: DType) -> TRef {
    tensor(Tensor::full(dtype, shape, v))
}

fn add_scalar(t: &TRef, v: f64) -> Result<TRef, String> {
    let c = constant(v, vec![], t.borrow().data.dtype);
    bin(TensorOp::Add, t, &c)
}

fn reshape(t: &TRef, shape: &[usize]) -> Result<TRef, String> {
    let dims: Vec<Dim> = shape.iter().map(|d| Dim::Static(*d)).collect();
    apply(&TensorOp::Reshape { dims }, &[t.clone()], true, 0)
}

fn mean_last(t: &TRef, rank: usize, keepdim: bool) -> Result<TRef, String> {
    apply(
        &TensorOp::Mean { axes: vec![rank.saturating_sub(1)], keepdim },
        &[t.clone()],
        true,
        0,
    )
}

fn sqrt(t: &TRef) -> Result<TRef, String> {
    unary(TensorOp::Sqrt, t)
}

impl Layer {
    pub fn linear(in_features: usize, out_features: usize, bias: bool, seed: u64) -> Layer {
        let scale = (2.0 / in_features.max(1) as f64).sqrt();
        let mut w = Tensor::randn(vec![out_features, in_features], seed);
        for i in 0..w.numel() {
            let v = w.f64_at(i) * scale;
            let mut vals = w.to_f64_vec();
            vals[i] = v;
            w = Tensor::from_f64(w.dtype, w.shape.clone(), vals).unwrap();
        }
        let b = if bias {
            Some(trainable(Tensor::zeros(DType::F32, vec![out_features])))
        } else {
            None
        };
        Layer::Linear { w: trainable(w), b, in_features, out_features }
    }

    pub fn layernorm(dim: usize, eps: f64) -> Layer {
        Layer::LayerNorm {
            gamma: trainable(Tensor::ones(DType::F32, vec![dim])),
            beta: trainable(Tensor::zeros(DType::F32, vec![dim])),
            dim,
            eps,
        }
    }

    pub fn batchnorm(dim: usize, eps: f64) -> Layer {
        Layer::BatchNorm {
            gamma: trainable(Tensor::ones(DType::F32, vec![dim])),
            beta: trainable(Tensor::zeros(DType::F32, vec![dim])),
            running_mean: RefCell::new(Tensor::zeros(DType::F32, vec![dim])),
            running_var: RefCell::new(Tensor::ones(DType::F32, vec![dim])),
            dim,
            eps,
        }
    }

    pub fn embedding(vocab: usize, dim: usize, seed: u64) -> Layer {
        let t = Tensor::randn(vec![vocab, dim], seed);
        Layer::Embedding { table: trainable(t), vocab, dim }
    }

    pub fn sequential(layers: Vec<Layer>) -> Layer {
        Layer::Sequential(layers.into_iter().map(Rc::new).collect())
    }

    pub fn conv2d(
        in_channels: usize,
        out_channels: usize,
        kernel: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        bias: bool,
        seed: u64,
    ) -> Layer {
        let fan_in = in_channels * kernel.0 * kernel.1;
        let scale = (2.0 / fan_in.max(1) as f64).sqrt();
        let mut t = Tensor::randn(vec![out_channels, in_channels, kernel.0, kernel.1], seed);
        let vals: Vec<f64> = (0..t.numel()).map(|i| t.f64_at(i) * scale).collect();
        t = Tensor::from_f64(t.dtype, t.shape.clone(), vals).unwrap();
        Layer::Conv2D {
            w: trainable(t),
            b: if bias { Some(trainable(Tensor::zeros(DType::F32, vec![out_channels]))) } else { None },
            in_channels,
            out_channels,
            kernel,
            stride,
            padding,
        }
    }

    pub fn conv1d(
        in_channels: usize,
        out_channels: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
        bias: bool,
        seed: u64,
    ) -> Layer {
        let fan_in = in_channels * kernel;
        let scale = (2.0 / fan_in.max(1) as f64).sqrt();
        let mut t = Tensor::randn(vec![out_channels, in_channels, kernel], seed);
        let vals: Vec<f64> = (0..t.numel()).map(|i| t.f64_at(i) * scale).collect();
        t = Tensor::from_f64(t.dtype, t.shape.clone(), vals).unwrap();
        Layer::Conv1D {
            w: trainable(t),
            b: if bias { Some(trainable(Tensor::zeros(DType::F32, vec![out_channels]))) } else { None },
            in_channels,
            out_channels,
            kernel,
            stride,
            padding,
        }
    }

    pub fn forward(&self, x: &TRef, training: bool, seed: u64) -> Result<TRef, String> {
        match self {
            Layer::Linear { w, b, in_features, out_features } => {
                let input = x.borrow().data.clone();
                if input.rank() == 0 {
                    return Err("Linear expects a tensor of rank >= 1".into());
                }
                let width = *input.shape.last().unwrap();
                if width != *in_features {
                    return Err(format!(
                        "Linear layer expects input width {}, found {} (shape {:?})",
                        in_features, width, input.shape
                    ));
                }
                let promoted = input.rank() == 1;
                let xv = if promoted { reshape(x, &[1, width])? } else { x.clone() };
                let wt = apply(&TensorOp::Transpose { dim0: 0, dim1: 1 }, &[w.clone()], true, 0)?;
                let mut out = matmul(&xv, &wt)?;
                if let Some(bias) = b {
                    out = bin(TensorOp::Add, &out, bias)?;
                }
                if promoted {
                    out = reshape(&out, &[*out_features])?;
                }
                Ok(out)
            }
            Layer::Activation(a) => unary(a.op(), x),
            Layer::Softmax { axis } => {
                let rank = x.borrow().data.rank();
                let ax = axis.unwrap_or(rank.saturating_sub(1));
                apply(&TensorOp::Softmax { axis: ax }, &[x.clone()], true, seed)
            }
            Layer::Dropout { p } => {
                apply(&TensorOp::Dropout { p: *p }, &[x.clone()], training, seed)
            }
            Layer::LayerNorm { gamma, beta, dim, eps } => {
                let rank = x.borrow().data.rank();
                if rank == 0 {
                    return Err("LayerNorm expects a tensor of rank >= 1".into());
                }
                let last = *x.borrow().data.shape.last().unwrap();
                if last != *dim {
                    return Err(format!("LayerNorm expects width {}, found {}", dim, last));
                }
                let mu = mean_last(x, rank, true)?;
                let xc = bin(TensorOp::Sub, x, &mu)?;
                let var = mean_last(&bin(TensorOp::Mul, &xc, &xc)?, rank, true)?;
                let denom = sqrt(&add_scalar(&var, *eps)?)?;
                let norm = bin(TensorOp::Div, &xc, &denom)?;
                let scaled = bin(TensorOp::Mul, &norm, gamma)?;
                bin(TensorOp::Add, &scaled, beta)
            }
            Layer::BatchNorm { gamma, beta, running_mean, running_var, dim, eps } => {
                let rank = x.borrow().data.rank();
                if rank == 0 {
                    return Err("BatchNorm expects a tensor of rank >= 1".into());
                }
                let width = *x.borrow().data.shape.last().unwrap();
                if width != *dim {
                    return Err(format!("BatchNorm expects width {}, found {}", dim, width));
                }
                let mut axes: Vec<usize> = (0..rank).collect();
                axes.pop();
                let mu = apply(
                    &TensorOp::Mean { axes: axes.clone(), keepdim: true },
                    &[x.clone()],
                    true,
                    0,
                )?;
                let xc = bin(TensorOp::Sub, x, &mu)?;
                let var = apply(
                    &TensorOp::Mean { axes: axes.clone(), keepdim: true },
                    &[bin(TensorOp::Mul, &xc, &xc)?],
                    true,
                    0,
                )?;
                let (mean_ref, var_ref) = if training {
                    if rank > 1 {
                        let m = mu.borrow().data.to_f64_vec();
                        let v = var.borrow().data.to_f64_vec();
                        let rv = running_var.borrow().clone().to_f64_vec();
                        let rm = running_mean.borrow().clone().to_f64_vec();
                        *running_mean.borrow_mut() = Tensor::from_f64(
                            DType::F32,
                            vec![*dim],
                            rm.iter().enumerate().map(|(i, x)| 0.9 * x + 0.1 * m.get(i).copied().unwrap_or(0.0)).collect(),
                        )?;
                        *running_var.borrow_mut() = Tensor::from_f64(
                            DType::F32,
                            vec![*dim],
                            rv.iter().enumerate().map(|(i, x)| 0.9 * x + 0.1 * v.get(i).copied().unwrap_or(0.0)).collect(),
                        )?;
                    }
                    (mu, var)
                } else {
                    let keep_shape = mu.borrow().data.shape.clone();
                    let rm = running_mean.borrow().clone().reshape(
                        &keep_shape.iter().map(|d| Dim::Static(*d)).collect::<Vec<_>>(),
                    )?;
                    let rv = running_var.borrow().clone().reshape(
                        &keep_shape.iter().map(|d| Dim::Static(*d)).collect::<Vec<_>>(),
                    )?;
                    (tensor(rm), tensor(rv))
                };
                let denom = sqrt(&add_scalar(&var_ref, *eps)?)?;
                let norm = bin(TensorOp::Div, &xc, &denom)?;
                let scaled = bin(TensorOp::Mul, &norm, gamma)?;
                let _ = mean_ref;
                bin(TensorOp::Add, &scaled, beta)
            }
            Layer::Embedding { table, vocab, dim } => {
                let idx = x.borrow().data.clone();
                if !idx.dtype.is_int() {
                    return Err("Embedding expects an integer index tensor".into());
                }
                if idx.numel() > 0 {
                    let max = (0..idx.numel()).map(|i| idx.i64_at(i)).max().unwrap_or(0);
                    let min = (0..idx.numel()).map(|i| idx.i64_at(i)).min().unwrap_or(0);
                    if min < 0 || max >= *vocab as i64 {
                        return Err(format!("Embedding index out of range [0, {})", vocab));
                    }
                }
                // Gather needs same-rank indices: expand every token id across
                // the embedding dimension so each output row keeps all `dim`
                // values, then restore the leading index-shaped dimensions.
                let n = idx.numel();
                let column = reshape(x, &[n, 1])?;
                let expanded = apply(
                    &TensorOp::BroadcastTo {
                        shape: vec![Dim::Static(n), Dim::Static(*dim)],
                    },
                    &[column],
                    false,
                    0,
                )?;
                let gathered = apply(&TensorOp::Gather { axis: 0 }, &[table.clone(), expanded], true, 0)?;
                let mut out_shape = idx.shape.clone();
                out_shape.push(*dim);
                reshape(&gathered, &out_shape)
            }
            Layer::Conv2D { w, b, in_channels, kernel, stride, padding, .. } => {
                let xd = x.borrow().data.clone();
                if xd.rank() != 4 {
                    return Err(format!("Conv2D expects NCHW input, found shape {:?}", xd.shape));
                }
                if xd.shape[1] != *in_channels {
                    return Err(format!(
                        "Conv2D expects {} channels, found {}",
                        in_channels, xd.shape[1]
                    ));
                }
                if xd.shape[2] < kernel.0 || xd.shape[3] < kernel.1 {
                    return Err("Conv2D input is smaller than the kernel".into());
                }
                let out = apply(
                    &TensorOp::Conv2D { stride: *stride, padding: *padding },
                    &[x.clone(), w.clone()],
                    true,
                    0,
                )?;
                match b {
                    Some(bias) => {
                        let o = out.borrow().data.shape[1];
                        let b4 = reshape(bias, &[1, o, 1, 1])?;
                        bin(TensorOp::Add, &out, &b4)
                    }
                    None => Ok(out),
                }
            }
            Layer::Conv1D { w, b, in_channels, kernel, stride, padding, .. } => {
                let xd = x.borrow().data.clone();
                if xd.rank() != 3 {
                    return Err(format!("Conv1D expects NCW input, found shape {:?}", xd.shape));
                }
                if xd.shape[1] != *in_channels {
                    return Err(format!(
                        "Conv1D expects {} channels, found {}",
                        in_channels, xd.shape[1]
                    ));
                }
                if xd.shape[2] < *kernel {
                    return Err("Conv1D input is shorter than the kernel".into());
                }
                let out = apply(
                    &TensorOp::Conv1D { stride: *stride, padding: *padding },
                    &[x.clone(), w.clone()],
                    true,
                    0,
                )?;
                match b {
                    Some(bias) => {
                        let o = out.borrow().data.shape[1];
                        let b3 = reshape(bias, &[1, o, 1])?;
                        bin(TensorOp::Add, &out, &b3)
                    }
                    None => Ok(out),
                }
            }
            Layer::Sequential(layers) => {
                let mut cur = x.clone();
                for (i, l) in layers.iter().enumerate() {
                    cur = l.forward(&cur, training, seed.wrapping_add(i as u64))?;
                }
                Ok(cur)
            }
        }
    }

    /// Streaming inference: run a sequential model one layer at a time and
    /// detach the activation between layers, so only the current activation
    /// and the executing layer's parameters stay live. The returned trace
    /// records the shape and bytes each layer produced.
    pub fn forward_streaming(
        &self,
        x: &TRef,
        seed: u64,
    ) -> Result<(TRef, Vec<StreamStep>), String> {
        let layers = match self {
            Layer::Sequential(layers) => layers,
            other => {
                let out = other.forward(x, false, seed)?;
                return Ok((out, Vec::new()));
            }
        };
        let mut cur = TNode::leaf(x.borrow().data.clone(), false);
        let mut trace = Vec::with_capacity(layers.len());
        for (i, layer) in layers.iter().enumerate() {
            let out = layer.forward(&cur, false, seed.wrapping_add(i as u64))?;
            let data = out.borrow().data.clone();
            trace.push(StreamStep {
                layer: layer.describe(),
                shape: data.shape.clone(),
                bytes: data.numel() * data.dtype.size(),
            });
            // Inference never calls backward, so the autograd graph built by
            // this layer can be released before the next one runs.
            cur = TNode::leaf(data, false);
        }
        Ok((cur, trace))
    }

    pub fn params(&self) -> Vec<TRef> {
        match self {
            Layer::Linear { w, b, .. } => {
                let mut v = vec![w.clone()];
                if let Some(b) = b {
                    v.push(b.clone());
                }
                v
            }
            Layer::LayerNorm { gamma, beta, .. } => vec![gamma.clone(), beta.clone()],
            Layer::BatchNorm { gamma, beta, .. } => vec![gamma.clone(), beta.clone()],
            Layer::Embedding { table, .. } => vec![table.clone()],
            Layer::Conv2D { w, b, .. } | Layer::Conv1D { w, b, .. } => {
                let mut v = vec![w.clone()];
                if let Some(b) = b {
                    v.push(b.clone());
                }
                v
            }
            Layer::Sequential(ls) => ls.iter().flat_map(|l| l.params()).collect(),
            _ => Vec::new(),
        }
    }

    pub fn describe(&self) -> String {
        match self {
            Layer::Linear { in_features, out_features, b, .. } => {
                format!("Linear({}, {}, bias={})", in_features, out_features, b.is_some())
            }
            Layer::Activation(a) => format!("{}()", a.name()),
            Layer::Softmax { axis } => format!("Softmax(axis={})", axis.map(|a| a.to_string()).unwrap_or_else(|| "-1".into())),
            Layer::Dropout { p } => format!("Dropout(p={})", p),
            Layer::LayerNorm { dim, eps, .. } => format!("LayerNorm({}, eps={})", dim, eps),
            Layer::BatchNorm { dim, eps, .. } => format!("BatchNorm({}, eps={})", dim, eps),
            Layer::Embedding { vocab, dim, .. } => format!("Embedding({}, {})", vocab, dim),
            Layer::Conv2D { in_channels, out_channels, kernel, stride, padding, .. } => format!(
                "Conv2D({}, {}, kernel={}x{}, stride={}x{}, padding={}x{})",
                in_channels, out_channels, kernel.0, kernel.1, stride.0, stride.1, padding.0, padding.1
            ),
            Layer::Conv1D { in_channels, out_channels, kernel, stride, padding, .. } => format!(
                "Conv1D({}, {}, kernel={}, stride={}, padding={})",
                in_channels, out_channels, kernel, stride, padding
            ),
            Layer::Sequential(ls) => {
                let parts: Vec<String> = ls.iter().map(|l| l.describe()).collect();
                format!("Sequential[{}]", parts.join(", "))
            }
        }
    }

    /// Save parameters in a small, documented text format.
    pub fn save(&self, path: &str) -> Result<(), String> {
        let mut out = String::from("neura-model 1\n");
        write_layer(self, &mut out);
        fs::write(path, out).map_err(|e| format!("cannot write model to {}: {}", path, e))
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Layer, String> {
        let text = fs::read_to_string(&path)
            .map_err(|e| format!("cannot read model {}: {}", path.as_ref().display(), e))?;
        let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
        if lines.first().map(|l| l.trim()) != Some("neura-model 1") {
            return Err("not a Neura model file (missing header)".into());
        }
        let mut idx = 1usize;
        let layer = read_layer(&lines, &mut idx)?;
        if idx != lines.len() {
            return Err(format!("trailing data in model file at line {}", idx + 1));
        }
        Ok(layer)
    }
}

fn fmt_params(t: &Tensor) -> String {
    let vals: Vec<String> = (0..t.numel()).map(|i| format!("{}", t.f64_at(i))).collect();
    let dims: Vec<String> = t.shape.iter().map(|d| d.to_string()).collect();
    format!("param {} {} = {}", t.dtype.name(), dims.join(" "), vals.join(" "))
}

fn write_layer(l: &Layer, out: &mut String) {
    match l {
        Layer::Linear { w, b, in_features, out_features } => {
            out.push_str(&format!("linear {} {} {}\n", in_features, out_features, b.is_some() as u8));
            out.push_str(&fmt_params(&w.borrow().data));
            out.push('\n');
            if let Some(b) = b {
                out.push_str(&fmt_params(&b.borrow().data));
                out.push('\n');
            }
        }
        Layer::Activation(a) => out.push_str(&format!("{}\n", a.name().to_lowercase())),
        Layer::Softmax { axis } => {
            out.push_str(&format!("softmax {}\n", axis.map(|a| a as i64).unwrap_or(-1)))
        }
        Layer::Dropout { p } => out.push_str(&format!("dropout {}\n", p)),
        Layer::LayerNorm { gamma, beta, dim, eps } => {
            out.push_str(&format!("layernorm {} {}\n", dim, eps));
            out.push_str(&fmt_params(&gamma.borrow().data));
            out.push('\n');
            out.push_str(&fmt_params(&beta.borrow().data));
            out.push('\n');
        }
        Layer::BatchNorm { gamma, beta, running_mean, running_var, dim, eps } => {
            out.push_str(&format!("batchnorm {} {}\n", dim, eps));
            out.push_str(&fmt_params(&gamma.borrow().data));
            out.push('\n');
            out.push_str(&fmt_params(&beta.borrow().data));
            out.push('\n');
            out.push_str(&fmt_params(&running_mean.borrow()));
            out.push('\n');
            out.push_str(&fmt_params(&running_var.borrow()));
            out.push('\n');
        }
        Layer::Embedding { table, vocab, dim } => {
            out.push_str(&format!("embedding {} {}\n", vocab, dim));
            out.push_str(&fmt_params(&table.borrow().data));
            out.push('\n');
        }
        Layer::Conv2D { w, b, in_channels, out_channels, kernel, stride, padding } => {
            out.push_str(&format!(
                "conv2d {} {} {} {} {} {} {} {} {}\n",
                in_channels, out_channels, kernel.0, kernel.1, stride.0, stride.1, padding.0, padding.1, b.is_some() as u8
            ));
            out.push_str(&fmt_params(&w.borrow().data));
            out.push('\n');
            if let Some(b) = b {
                out.push_str(&fmt_params(&b.borrow().data));
                out.push('\n');
            }
        }
        Layer::Conv1D { w, b, in_channels, out_channels, kernel, stride, padding } => {
            out.push_str(&format!(
                "conv1d {} {} {} {} {} {}\n",
                in_channels, out_channels, kernel, stride, padding, b.is_some() as u8
            ));
            out.push_str(&fmt_params(&w.borrow().data));
            out.push('\n');
            if let Some(b) = b {
                out.push_str(&fmt_params(&b.borrow().data));
                out.push('\n');
            }
        }
        Layer::Sequential(ls) => {
            out.push_str(&format!("sequential {}\n", ls.len()));
            for l in ls {
                write_layer(l.as_ref(), out);
            }
        }
    }
}

fn parse_param(line: &str) -> Result<Tensor, String> {
    let rest = line
        .trim()
        .strip_prefix("param ")
        .ok_or_else(|| format!("expected a parameter line, found `{}`", line))?;
    let (header, data) = rest
        .split_once(" = ")
        .ok_or_else(|| format!("malformed parameter line `{}`", line))?;
    let mut parts = header.split_whitespace();
    let dtype = parts
        .next()
        .and_then(DType::parse)
        .ok_or_else(|| format!("invalid dtype in `{}`", line))?;
    let mut shape = Vec::new();
    for tok in parts {
        shape.push(
            tok.parse::<usize>()
                .map_err(|_| format!("invalid dimension `{}` in `{}`", tok, line))?,
        );
    }
    let values: Vec<f64> = data
        .split_whitespace()
        .map(|t| t.parse::<f64>().map_err(|_| format!("invalid number `{}`", t)))
        .collect::<Result<_, _>>()?;
    let expected: usize = shape.iter().product();
    if expected != values.len() {
        return Err(format!(
            "parameter shape {:?} needs {} values, found {}",
            shape,
            expected,
            values.len()
        ));
    }
    Tensor::from_f64(dtype, shape, values)
}

fn read_layer(lines: &[&str], idx: &mut usize) -> Result<Layer, String> {
    if *idx >= lines.len() {
        return Err("unexpected end of model file".into());
    }
    let line = lines[*idx].trim();
    *idx += 1;
    let mut parts = line.split_whitespace();
    let kind = parts.next().unwrap_or("");
    match kind {
        "linear" => {
            let in_features: usize = parts.next().and_then(|s| s.parse().ok()).ok_or("invalid linear layer")?;
            let out_features: usize = parts.next().and_then(|s| s.parse().ok()).ok_or("invalid linear layer")?;
            let bias: u8 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(1);
            if *idx >= lines.len() {
                return Err("missing linear weights".into());
            }
            let w = parse_param(lines[*idx])?;
            *idx += 1;
            let b = if bias != 0 {
                if *idx >= lines.len() {
                    return Err("missing linear bias".into());
                }
                let t = parse_param(lines[*idx])?;
                *idx += 1;
                Some(trainable(t))
            } else {
                None
            };
            Ok(Layer::Linear { w: trainable(w), b, in_features, out_features })
        }
        "relu" => Ok(Layer::Activation(Activation::ReLU)),
        "gelu" => Ok(Layer::Activation(Activation::GELU)),
        "sigmoid" => Ok(Layer::Activation(Activation::Sigmoid)),
        "tanh" => Ok(Layer::Activation(Activation::Tanh)),
        "softmax" => {
            let axis: i64 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(-1);
            Ok(Layer::Softmax { axis: if axis < 0 { None } else { Some(axis as usize) } })
        }
        "dropout" => {
            let p: f64 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0.0);
            Ok(Layer::Dropout { p })
        }
        "layernorm" => {
            let dim: usize = parts.next().and_then(|s| s.parse().ok()).ok_or("invalid layernorm")?;
            let eps: f64 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(1e-5);
            let gamma = parse_param(lines[*idx])?;
            *idx += 1;
            let beta = parse_param(lines[*idx])?;
            *idx += 1;
            Ok(Layer::LayerNorm { gamma: trainable(gamma), beta: trainable(beta), dim, eps })
        }
        "batchnorm" => {
            let dim: usize = parts.next().and_then(|s| s.parse().ok()).ok_or("invalid batchnorm")?;
            let eps: f64 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(1e-5);
            let gamma = parse_param(lines[*idx])?;
            *idx += 1;
            let beta = parse_param(lines[*idx])?;
            *idx += 1;
            let rm = parse_param(lines[*idx])?;
            *idx += 1;
            let rv = parse_param(lines[*idx])?;
            *idx += 1;
            Ok(Layer::BatchNorm {
                gamma: trainable(gamma),
                beta: trainable(beta),
                running_mean: RefCell::new(rm),
                running_var: RefCell::new(rv),
                dim,
                eps,
            })
        }
        "embedding" => {
            let vocab: usize = parts.next().and_then(|s| s.parse().ok()).ok_or("invalid embedding")?;
            let dim: usize = parts.next().and_then(|s| s.parse().ok()).ok_or("invalid embedding")?;
            let table = parse_param(lines[*idx])?;
            *idx += 1;
            Ok(Layer::Embedding { table: trainable(table), vocab, dim })
        }
        "conv2d" => {
            let nums: Vec<usize> = parts.filter_map(|p| p.parse().ok()).collect();
            if nums.len() != 9 {
                return Err("invalid conv2d header".into());
            }
            let w = parse_param(lines[*idx])?;
            *idx += 1;
            let b = if nums[8] != 0 {
                if *idx >= lines.len() {
                    return Err("missing conv2d bias".into());
                }
                let t = parse_param(lines[*idx])?;
                *idx += 1;
                Some(trainable(t))
            } else {
                None
            };
            Ok(Layer::Conv2D {
                w: trainable(w),
                b,
                in_channels: nums[0],
                out_channels: nums[1],
                kernel: (nums[2], nums[3]),
                stride: (nums[4], nums[5]),
                padding: (nums[6], nums[7]),
            })
        }
        "conv1d" => {
            let nums: Vec<usize> = parts.filter_map(|p| p.parse().ok()).collect();
            if nums.len() != 6 {
                return Err("invalid conv1d header".into());
            }
            let w = parse_param(lines[*idx])?;
            *idx += 1;
            let b = if nums[5] != 0 {
                if *idx >= lines.len() {
                    return Err("missing conv1d bias".into());
                }
                let t = parse_param(lines[*idx])?;
                *idx += 1;
                Some(trainable(t))
            } else {
                None
            };
            Ok(Layer::Conv1D {
                w: trainable(w),
                b,
                in_channels: nums[0],
                out_channels: nums[1],
                kernel: nums[2],
                stride: nums[3],
                padding: nums[4],
            })
        }
        "sequential" => {
            let n: usize = parts.next().and_then(|s| s.parse().ok()).ok_or("invalid sequential")?;
            let mut ls = Vec::with_capacity(n);
            for _ in 0..n {
                ls.push(Rc::new(read_layer(lines, idx)?));
            }
            Ok(Layer::Sequential(ls))
        }
        other => Err(format!("unknown layer `{}` in model file", other)),
    }
}

/// Differentiable cross entropy over `[batch, classes]` logits.
pub fn cross_entropy(logits: &TRef, targets: &TRef) -> Result<TRef, String> {
    let shape = logits.borrow().data.shape.clone();
    if shape.len() != 2 {
        return Err(format!("cross_entropy expects logits of shape [batch, classes], found {:?}", shape));
    }
    let targets_data = targets.borrow().data.clone();
    if targets_data.dtype.is_float() {
        return Err("cross_entropy targets must be integer class indices".into());
    }
    if targets_data.numel() != shape[0] {
        return Err(format!(
            "cross_entropy expects {} targets for {} rows",
            shape[0],
            shape[0]
        ));
    }
    let max = apply(
        &TensorOp::ReduceMax { axis: Some(1), keepdim: true },
        &[logits.clone()],
        true,
        0,
    )?;
    let shifted = bin(TensorOp::Sub, logits, &max)?;
    let exps = unary(TensorOp::Exp, &shifted)?;
    let sum = apply(
        &TensorOp::Sum { axes: vec![1], keepdim: true },
        &[exps],
        true,
        0,
    )?;
    let logsum = unary(TensorOp::Log, &sum)?;
    let logsum = bin(TensorOp::Add, &logsum, &max)?;
    // Gather selects one class per row: with `[batch, 1]` indices the result
    // keeps that shape, so `picked[i, 0] == logits[i, targets[i]]`.
    let batch = shape[0];
    let index = reshape(targets, &[batch, 1])?;
    let picked = apply(&TensorOp::Gather { axis: 1 }, &[logits.clone(), index], true, 0)?;
    let logp = bin(TensorOp::Sub, &picked, &logsum)?;
    let mean = apply(&TensorOp::Mean { axes: vec![0, 1], keepdim: false }, &[logp], true, 0)?;
    // `mean(log p_target)` is the negative cross entropy; gradients must
    // descend on `-mean(log p_target)`, so the reported loss is the negation.
    unary(TensorOp::Neg, &mean)
}

/// Train `model` on `dataset` for `epochs`, reporting per-epoch loss/accuracy.
pub fn train(
    model: &Layer,
    dataset: &Dataset,
    epochs: usize,
    batch_size: usize,
    optimizer: &mut Optimizer,
    seed: u64,
    report: &mut Vec<String>,
) -> Result<(), String> {
    if dataset.len() == 0 {
        return Err("cannot train on an empty dataset".into());
    }
    let batch_size = batch_size.max(1);
    for epoch in 0..epochs {
        let batches = dataset.batches(batch_size, seed.wrapping_add(epoch as u64));
        let mut total_loss = 0.0;
        let mut correct = 0usize;
        let mut seen = 0usize;
        for (x, y) in &batches {
            let xr = tensor(x.clone());
            let yr = tensor(y.clone());
            optimizer.zero_grad();
            let logits = model.forward(&xr, true, seed.wrapping_add(seen as u64))?;
            let loss = cross_entropy(&logits, &yr)?;
            autograd::backward(&loss)?;
            optimizer.step()?;
            total_loss += loss.borrow().data.scalar_f64().unwrap_or(0.0);
            seen += y.numel();
            correct += count_correct(&logits, y);
        }
        let n = batches.len().max(1) as f64;
        let acc = if seen == 0 { 0.0 } else { correct as f64 / seen as f64 };
        report.push(format!(
            "epoch {}/{}  loss={:.6}  accuracy={:.4}",
            epoch + 1,
            epochs,
            total_loss / n,
            acc
        ));
    }
    Ok(())
}

pub fn count_correct(logits: &TRef, targets: &Tensor) -> usize {
    let t = logits.borrow().data.clone();
    if t.rank() != 2 {
        return 0;
    }
    let (rows, cols) = (t.shape[0], t.shape[1]);
    let mut correct = 0;
    for r in 0..rows {
        let mut best = 0usize;
        let mut bestv = f64::NEG_INFINITY;
        for c in 0..cols {
            let v = t.f64_at(r * cols + c);
            if v > bestv {
                bestv = v;
                best = c;
            }
        }
        if r < targets.numel() && best as i64 == targets.i64_at(r) {
            correct += 1;
        }
    }
    correct
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd;
    use crate::data::Dataset;
    use crate::optim::{OptKind, Optimizer};
    use crate::value::{tensor, trainable};
    use neura_backends::tensor::Tensor;
    use neura_ir::types::DType;

    #[test]
    fn cross_entropy_matches_the_analytic_value_and_gradient() {
        let logits = trainable(Tensor::from_f64(DType::F64, vec![2, 2], vec![0.0; 4]).unwrap());
        let targets = tensor(Tensor::from_i64(DType::I64, vec![2], vec![0, 1]).unwrap());
        let loss = cross_entropy(&logits, &targets).unwrap();
        let value = loss.borrow().data.item().unwrap();
        assert!((value - 2f64.ln()).abs() < 1e-9, "loss was {}", value);
        autograd::backward(&loss).unwrap();
        let g = logits.borrow().grad.clone().expect("no gradient");
        // dL/dlogit[i][j] = (p_j - 1[j == target_i]) / batch with p = 1/2.
        assert!((g.f64_at(0) + 0.25).abs() < 1e-9, "target logit gradient was {}", g.f64_at(0));
        assert!((g.f64_at(1) - 0.25).abs() < 1e-9);
        assert!((g.f64_at(2) - 0.25).abs() < 1e-9);
        assert!((g.f64_at(3) + 0.25).abs() < 1e-9);
    }

    #[test]
    fn linear_model_learns_a_separable_problem() {
        let features = vec![0.0, 0.0, 1.0, 1.0, 4.0, 4.0, 5.0, 5.0];
        let ds = Dataset::new(
            Tensor::from_f64(DType::F32, vec![4, 2], features).unwrap(),
            Tensor::from_i64(DType::I64, vec![4], vec![0, 0, 1, 1]).unwrap(),
        )
        .unwrap();
        let model = Layer::sequential(vec![Layer::linear(2, 2, true, 7)]);
        let mut opt = Optimizer::new(OptKind::Sgd { momentum: 0.0 }, 0.5, model.params());
        let mut report = Vec::new();
        train(&model, &ds, 60, 4, &mut opt, 3, &mut report).unwrap();
        let xs = tensor(ds.features.clone());
        let logits = model.forward(&xs, false, 0).unwrap();
        let preds = tensor(logits.borrow().data.clone());
        assert_eq!(count_correct(&preds, &ds.labels), 4);
    }
}

