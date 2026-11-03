//! Native CPU tensor engine.
//!
//! Tensors are dense, row-major, and tagged with an element type. Floating
//! point element types are stored internally as `f64` and quantised to their
//! declared precision on creation and after every operation, so `f16`/`bf16`
//! results carry the rounding they would have on real hardware rather than
//! pretending to be exactly `f64`.

use neura_ir::ops::BinOp;
use neura_ir::shape;
use neura_ir::types::{Dim, DType};
use std::fmt;

#[derive(Clone, Debug)]
pub enum Storage {
    F(Vec<f64>),
    I(Vec<i64>),
    B(Vec<bool>),
}

#[derive(Clone, Debug)]
pub struct Tensor {
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub data: Storage,
}

pub type TResult<T> = Result<T, String>;

impl Tensor {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    pub fn strides(&self) -> Vec<usize> {
        let mut s = vec![1usize; self.shape.len()];
        for i in (0..self.shape.len().saturating_sub(1)).rev() {
            s[i] = s[i + 1] * self.shape[i + 1];
        }
        s
    }

    pub fn from_f64(dtype: DType, shape: Vec<usize>, values: Vec<f64>) -> TResult<Tensor> {
        let n: usize = shape.iter().product();
        if n != values.len() {
            return Err(format!(
                "tensor shape {:?} needs {} elements, got {}",
                shape,
                n,
                values.len()
            ));
        }
        let data = match dtype {
            DType::I8 | DType::I32 | DType::I64 => {
                Storage::I(values.iter().map(|v| int_store(*v, dtype)).collect())
            }
            DType::Bool => Storage::B(values.iter().map(|v| *v != 0.0).collect()),
            _ => Storage::F(values.iter().map(|v| quantize(*v, dtype)).collect()),
        };
        Ok(Tensor { dtype, shape, data })
    }

    pub fn from_i64(dtype: DType, shape: Vec<usize>, values: Vec<i64>) -> TResult<Tensor> {
        let n: usize = shape.iter().product();
        if n != values.len() {
            return Err(format!("tensor shape {:?} needs {} elements", shape, n));
        }
        let data = match dtype {
            DType::Bool => Storage::B(values.iter().map(|v| *v != 0).collect()),
            DType::I8 | DType::I32 | DType::I64 => {
                Storage::I(values.iter().map(|v| int_store(*v as f64, dtype)).collect())
            }
            _ => Storage::F(values.iter().map(|v| quantize(*v as f64, dtype)).collect()),
        };
        Ok(Tensor { dtype, shape, data })
    }

    pub fn zeros(dtype: DType, shape: Vec<usize>) -> Tensor {
        let n: usize = shape.iter().product();
        match dtype {
            DType::I8 | DType::I32 | DType::I64 => Tensor {
                dtype,
                shape,
                data: Storage::I(vec![0; n]),
            },
            DType::Bool => Tensor { dtype, shape, data: Storage::B(vec![false; n]) },
            _ => Tensor { dtype, shape, data: Storage::F(vec![0.0; n]) },
        }
    }

    pub fn ones(dtype: DType, shape: Vec<usize>) -> Tensor {
        let n: usize = shape.iter().product();
        match dtype {
            DType::I8 | DType::I32 | DType::I64 => Tensor {
                dtype,
                shape,
                data: Storage::I(vec![1; n]),
            },
            DType::Bool => Tensor { dtype, shape, data: Storage::B(vec![true; n]) },
            _ => Tensor { dtype, shape, data: Storage::F(vec![1.0; n]) },
        }
    }

    pub fn full(dtype: DType, shape: Vec<usize>, v: f64) -> Tensor {
        let n: usize = shape.iter().product();
        match dtype {
            DType::I8 | DType::I32 | DType::I64 => Tensor {
                dtype,
                shape,
                data: Storage::I(vec![v as i64; n]),
            },
            DType::Bool => Tensor { dtype, shape, data: Storage::B(vec![v != 0.0; n]) },
            _ => Tensor {
                dtype,
                shape,
                data: Storage::F(vec![quantize(v, dtype); n]),
            },
        }
    }

    pub fn arange(start: f64, end: f64, step: f64) -> TResult<Tensor> {
        if step == 0.0 {
            return Err("arange step must be non-zero".into());
        }
        let mut v = Vec::new();
        let mut x = start;
        if step > 0.0 {
            while x < end {
                v.push(x);
                x += step;
            }
        } else {
            while x > end {
                v.push(x);
                x += step;
            }
        }
        Tensor::from_f64(DType::F32, vec![v.len()], v)
    }

    /// Deterministic pseudo-random normal tensor (xorshift + Box-Muller).
    pub fn randn(shape: Vec<usize>, seed: u64) -> Tensor {
        let n: usize = shape.iter().product();
        let mut state = seed | 1;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        let mut vals = Vec::with_capacity(n);
        while vals.len() < n {
            let u1 = next().max(1e-12);
            let u2 = next();
            let r = (-2.0 * u1.ln()).sqrt();
            vals.push(r * (std::f64::consts::TAU * u2).cos());
            if vals.len() < n {
                vals.push(r * (std::f64::consts::TAU * u2).sin());
            }
        }
        Tensor::from_f64(DType::F32, shape, vals).unwrap()
    }

    pub fn f64_at(&self, flat: usize) -> f64 {
        match &self.data {
            Storage::F(v) => v[flat],
            Storage::I(v) => v[flat] as f64,
            Storage::B(v) => {
                if v[flat] {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }

    pub fn i64_at(&self, flat: usize) -> i64 {
        match &self.data {
            Storage::F(v) => v[flat] as i64,
            Storage::I(v) => v[flat],
            Storage::B(v) => v[flat] as i64,
        }
    }

    pub fn to_f64_vec(&self) -> Vec<f64> {
        (0..self.numel()).map(|i| self.f64_at(i)).collect()
    }

    pub fn item(&self) -> TResult<f64> {
        if self.numel() != 1 {
            return Err(format!("item() needs a one-element tensor, got shape {:?}", self.shape));
        }
        Ok(self.f64_at(0))
    }

    pub fn scalar_f64(&self) -> Option<f64> {
        if self.numel() == 1 {
            Some(self.f64_at(0))
        } else {
            None
        }
    }

    pub fn cast(&self, dtype: DType) -> Tensor {
        match dtype {
            DType::I8 | DType::I32 | DType::I64 => {
                Tensor::from_i64(dtype, self.shape.clone(), (0..self.numel()).map(|i| self.i64_at(i)).collect()).unwrap()
            }
            DType::Bool => {
                let v: Vec<bool> = (0..self.numel()).map(|i| self.f64_at(i) != 0.0).collect();
                Tensor { dtype, shape: self.shape.clone(), data: Storage::B(v) }
            }
            _ => Tensor::from_f64(dtype, self.shape.clone(), self.to_f64_vec()).unwrap(),
        }
    }

    fn flat_index(&self, idx: &[usize]) -> usize {
        let strides = self.strides();
        idx.iter().zip(strides.iter()).map(|(i, s)| i * s).sum()
    }

    fn multi_index(&self, mut flat: usize) -> Vec<usize> {
        let strides = self.strides();
        let mut idx = vec![0; self.shape.len()];
        for i in 0..self.shape.len() {
            if self.shape[i] == 0 {
                continue;
            }
            idx[i] = flat / strides[i];
            flat %= strides[i];
        }
        idx
    }

    pub fn reshape(&self, target: &[Dim]) -> TResult<Tensor> {
        let resolved = shape::resolve_reshape(
            &self.shape.iter().map(|d| Dim::Static(*d)).collect::<Vec<_>>(),
            target,
        )?;
        let mut dims = Vec::new();
        for d in resolved {
            match d.concrete() {
                Some(n) => dims.push(n),
                None => {
                    return Err("reshape target contains an unresolved dimension".into());
                }
            }
        }
        let n: usize = dims.iter().product();
        if n != self.numel() {
            return Err(format!(
                "cannot reshape tensor with {} elements into {:?}",
                self.numel(),
                dims
            ));
        }
        Ok(Tensor { dtype: self.dtype, shape: dims, data: self.data.clone() })
    }

    pub fn flatten(&self) -> Tensor {
        Tensor {
            dtype: self.dtype,
            shape: vec![self.numel()],
            data: self.data.clone(),
        }
    }

    pub fn transpose(&self, d0: usize, d1: usize) -> TResult<Tensor> {
        if d0 >= self.rank() || d1 >= self.rank() {
            return Err("transpose dimension out of range".into());
        }
        let mut shape = self.shape.clone();
        shape.swap(d0, d1);
        let n = self.numel();
        let mut vals = vec![0.0; n];
        for flat in 0..n {
            let out_idx = self.multi_index_in(&shape, flat);
            let mut src = out_idx;
            src.swap(d0, d1);
            vals[flat] = self.f64_at(self.flat_index(&src));
        }
        let out = if self.dtype.is_int() {
            Tensor::from_i64(self.dtype, shape, vals.iter().map(|v| int_store(*v, self.dtype)).collect())?
        } else if self.dtype == DType::Bool {
            Tensor { dtype: DType::Bool, shape, data: Storage::B(vals.iter().map(|v| *v != 0.0).collect()) }
        } else {
            Tensor::from_f64(self.dtype, shape, vals)?
        };
        Ok(out)
    }

    pub fn slice(&self, axis: usize, start: usize, end: usize) -> TResult<Tensor> {
        if axis >= self.rank() {
            return Err("slice axis out of range".into());
        }
        let len = self.shape[axis];
        let start = start.min(len);
        let end = end.min(len).max(start);
        let mut shape = self.shape.clone();
        shape[axis] = end - start;
        let n: usize = shape.iter().product();
        let mut vals = Vec::with_capacity(n);
        let mut idx = vec![0usize; self.rank()];
        for flat in 0..n {
            idx = self.multi_index_in(&shape, flat);
            let mut src = idx.clone();
            src[axis] += start;
            vals.push(self.f64_at(self.flat_index(&src)));
        }
        self.rebuild(shape, vals)
    }

    fn multi_index_in(&self, shape: &[usize], mut flat: usize) -> Vec<usize> {
        let mut strides = vec![1usize; shape.len()];
        for i in (0..shape.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        let mut idx = vec![0; shape.len()];
        for i in 0..shape.len() {
            if shape[i] == 0 {
                continue;
            }
            idx[i] = flat / strides[i];
            flat %= strides[i];
        }
        idx
    }

    fn rebuild(&self, shape: Vec<usize>, vals: Vec<f64>) -> TResult<Tensor> {
        if self.dtype.is_int() {
            Tensor::from_i64(self.dtype, shape, vals.iter().map(|v| int_store(*v, self.dtype)).collect())
        } else if self.dtype == DType::Bool {
            Ok(Tensor {
                dtype: DType::Bool,
                shape,
                data: Storage::B(vals.iter().map(|v| *v != 0.0).collect()),
            })
        } else {
            Tensor::from_f64(self.dtype, shape, vals)
        }
    }

    // ---- broadcasting ---------------------------------------------------

    pub fn broadcast_pair(a: &[usize], b: &[usize]) -> TResult<Vec<usize>> {
        let rank = a.len().max(b.len());
        let mut out = Vec::with_capacity(rank);
        for i in 0..rank {
            let da = if i < rank - a.len() { 1 } else { a[i - (rank - a.len())] };
            let db = if i < rank - b.len() { 1 } else { b[i - (rank - b.len())] };
            if da == db {
                out.push(da);
            } else if da == 1 {
                out.push(db);
            } else if db == 1 {
                out.push(da);
            } else {
                return Err(format!("cannot broadcast shapes {:?} and {:?}", a, b));
            }
        }
        Ok(out)
    }

    fn broadcast_index(&self, out_idx: &[usize], out_rank: usize) -> Vec<usize> {
        let pad = out_rank - self.rank();
        let mut idx = Vec::with_capacity(self.rank());
        for i in 0..self.rank() {
            let oi = if self.shape[i] == 1 { 0 } else { out_idx[pad + i] };
            idx.push(oi);
        }
        idx
    }

    pub fn broadcast_to(&self, target: &[usize]) -> TResult<Tensor> {
        let out = Tensor::broadcast_pair(&self.shape, target)?;
        if out != target {
            return Err(format!("cannot broadcast {:?} to {:?}", self.shape, target));
        }
        let n: usize = target.iter().product();
        let mut vals = Vec::with_capacity(n);
        for flat in 0..n {
            let idx = self.multi_index_in(target, flat);
            let src = self.broadcast_index(&idx, target.len());
            vals.push(self.f64_at(self.flat_index(&src)));
        }
        self.rebuild(target.to_vec(), vals)
    }

    // ---- elementwise ----------------------------------------------------

    pub fn elementwise(&self, op: &neura_ir::ops::TensorOp, other: &Tensor) -> TResult<Tensor> {
        use neura_ir::ops::TensorOp as T;
        let f: fn(f64, f64) -> f64 = match op {
            T::Add => |a, b| a + b,
            T::Sub => |a, b| a - b,
            T::Mul => |a, b| a * b,
            T::Div => |a, b| {
                if b == 0.0 {
                    if a == 0.0 {
                        f64::NAN
                    } else {
                        a.signum() * f64::INFINITY
                    }
                } else {
                    a / b
                }
            },
            other => return Err(format!("`{}` is not an elementwise binary op", other.name())),
        };
        let out_shape = Tensor::broadcast_pair(&self.shape, &other.shape)?;
        let n: usize = out_shape.iter().product();
        let mut vals = Vec::with_capacity(n);
        for flat in 0..n {
            let idx = self.multi_index_in(&out_shape, flat);
            let ai = self.broadcast_index(&idx, out_shape.len());
            let bi = other.broadcast_index(&idx, out_shape.len());
            vals.push(f(self.f64_at(self.flat_index(&ai)), other.f64_at(other.flat_index(&bi))));
        }
        let mut dtype = self.dtype.promote(other.dtype);
        if matches!(op, T::Div) && dtype.is_int() {
            dtype = DType::F64;
        }
        if dtype.is_int() {
            Tensor::from_i64(dtype, out_shape, vals.iter().map(|v| *v as i64).collect())
        } else if dtype == DType::Bool {
            Ok(Tensor { dtype, shape: out_shape, data: Storage::B(vals.iter().map(|v| *v != 0.0).collect()) })
        } else {
            Tensor::from_f64(dtype, out_shape, vals)
        }
    }

    pub fn binop_scalar(&self, op: BinOp, rhs: f64) -> TResult<Tensor> {
        let f: fn(f64, f64) -> f64 = match op {
            BinOp::Add => |a, b| a + b,
            BinOp::Sub => |a, b| a - b,
            BinOp::Mul => |a, b| a * b,
            BinOp::Div => |a, b| a / b,
            BinOp::FloorDiv => |a, b| (a / b).floor(),
            BinOp::Pow => |a, b| a.powf(b),
            BinOp::Mod => |a, b| a % b,
            BinOp::Eq => |a, b| (a == b) as i32 as f64,
            BinOp::Ne => |a, b| (a != b) as i32 as f64,
            BinOp::Lt => |a, b| (a < b) as i32 as f64,
            BinOp::Le => |a, b| (a <= b) as i32 as f64,
            BinOp::Gt => |a, b| (a > b) as i32 as f64,
            BinOp::Ge => |a, b| (a >= b) as i32 as f64,
            BinOp::And => |a, b| ((a != 0.0) && (b != 0.0)) as i32 as f64,
            BinOp::Or => |a, b| ((a != 0.0) || (b != 0.0)) as i32 as f64,
        };
        let vals: Vec<f64> = (0..self.numel()).map(|i| f(self.f64_at(i), rhs)).collect();
        let dtype = if matches!(op, BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge | BinOp::And | BinOp::Or) {
            DType::Bool
        } else {
            self.dtype
        };
        if dtype == DType::Bool {
            Ok(Tensor { dtype, shape: self.shape.clone(), data: Storage::B(vals.iter().map(|v| *v != 0.0).collect()) })
        } else {
            Tensor::from_f64(dtype, self.shape.clone(), vals)
        }
    }

    pub fn unary(&self, op: &neura_ir::ops::TensorOp) -> TResult<Tensor> {
        use neura_ir::ops::TensorOp as T;
        let f: fn(f64) -> f64 = match op {
            T::Neg => |x| -x,
            T::ReLU => |x| x.max(0.0),
            T::GELU => neura_ir::ops::gelu,
            T::Sigmoid => neura_ir::ops::sigmoid,
            T::Tanh => |x| x.tanh(),
            T::Exp => |x| x.exp(),
            T::Log => |x| x.ln(),
            T::Sqrt => |x| x.sqrt(),
            other => return Err(format!("`{}` is not a unary op", other.name())),
        };
        let vals: Vec<f64> = (0..self.numel()).map(|i| f(self.f64_at(i))).collect();
        let dtype = if self.dtype.is_float() { self.dtype } else { DType::F32 };
        Tensor::from_f64(dtype, self.shape.clone(), vals)
    }

    // ---- reductions -----------------------------------------------------

    pub fn reduce(&self, op: &neura_ir::ops::TensorOp) -> TResult<Tensor> {
        use neura_ir::ops::TensorOp as T;
        let (axes, keepdim, kind) = match op {
            T::Sum { axes, keepdim } => (axes.clone(), *keepdim, "sum"),
            T::Mean { axes, keepdim } => (axes.clone(), *keepdim, "mean"),
            T::ReduceMax { axis, keepdim } => (
                axis.map(|a| vec![a]).unwrap_or_else(|| (0..self.rank()).collect()),
                *keepdim,
                "max",
            ),
            T::ReduceMin { axis, keepdim } => (
                axis.map(|a| vec![a]).unwrap_or_else(|| (0..self.rank()).collect()),
                *keepdim,
                "min",
            ),
            other => return Err(format!("`{}` is not a reduction", other.name())),
        };
        for a in &axes {
            if *a >= self.rank() {
                return Err("reduction axis out of range".into());
            }
        }
        let mut out_shape = Vec::new();
        for (i, d) in self.shape.iter().enumerate() {
            if axes.contains(&i) {
                if keepdim {
                    out_shape.push(1);
                }
            } else {
                out_shape.push(*d);
            }
        }
        let out_n: usize = out_shape.iter().product();
        let mut sums = vec![0.0f64; out_n.max(1)];
        let mut counts = vec![0usize; out_n.max(1)];
        let mut extrema = vec![
            if kind == "max" { f64::NEG_INFINITY } else { f64::INFINITY };
            out_n.max(1)
        ];
        for flat in 0..self.numel() {
            let idx = self.multi_index(flat);
            let mut oidx = Vec::new();
            for (i, v) in idx.iter().enumerate() {
                if axes.contains(&i) {
                    if keepdim {
                        oidx.push(0);
                    }
                } else {
                    oidx.push(*v);
                }
            }
            let o = if out_n == 0 {
                0
            } else {
                let strides = strides_of(&out_shape);
                oidx.iter().zip(strides.iter()).map(|(i, s)| i * s).sum()
            };
            let x = self.f64_at(flat);
            sums[o] += x;
            counts[o] += 1;
            if kind == "max" {
                extrema[o] = extrema[o].max(x);
            } else if kind == "min" {
                extrema[o] = extrema[o].min(x);
            }
        }
        let vals: Vec<f64> = match kind {
            "sum" => sums,
            "mean" => sums.iter().zip(counts.iter()).map(|(s, c)| if *c == 0 { 0.0 } else { s / *c as f64 }).collect(),
            _ => extrema,
        };
        let dtype = if kind == "mean" {
            if self.dtype.is_float() { self.dtype } else { DType::F32 }
        } else {
            self.dtype
        };
        if dtype.is_int() {
            Tensor::from_i64(dtype, out_shape, vals.iter().map(|v| *v as i64).collect())
        } else {
            Tensor::from_f64(dtype, out_shape, vals)
        }
    }

    pub fn sum_all(&self) -> f64 {
        (0..self.numel()).map(|i| self.f64_at(i)).sum()
    }

    // ---- matmul / convolution ------------------------------------------

    pub fn matmul(&self, other: &Tensor) -> TResult<Tensor> {
        if self.rank() < 2 || other.rank() < 2 {
            return Err("matmul requires both operands to have rank >= 2".into());
        }
        let (m, k1) = (self.shape[self.rank() - 2], self.shape[self.rank() - 1]);
        let (k2, n) = (other.shape[other.rank() - 2], other.shape[other.rank() - 1]);
        if k1 != k2 {
            return Err(format!("matmul inner dimension mismatch {} vs {}", k1, k2));
        }
        let batch = Tensor::broadcast_pair(
            &self.shape[..self.rank() - 2],
            &other.shape[..other.rank() - 2],
        )?;
        let mut out_shape = batch.clone();
        out_shape.push(m);
        out_shape.push(n);
        let batch_n: usize = batch.iter().product();
        let mut vals = vec![0.0; batch_n * m * n];
        let bstrides_a = strides_of(&self.shape[..self.rank() - 2]);
        let bstrides_b = strides_of(&other.shape[..other.rank() - 2]);
        for b in 0..batch_n.max(1) {
            let bidx = multi_index_in(&batch, b);
            let a_off = if bidx.is_empty() {
                0
            } else {
                bidx.iter()
                    .zip(bstrides_a.iter())
                    .map(|(i, s)| i * s)
                    .sum::<usize>()
                    * m
                    * k1
            };
            let b_off = if bidx.is_empty() {
                0
            } else {
                bidx.iter()
                    .zip(bstrides_b.iter())
                    .map(|(i, s)| i * s)
                    .sum::<usize>()
                    * k2
                    * n
            };
            for i in 0..m {
                for j in 0..n {
                    let mut acc = 0.0;
                    for k in 0..k1 {
                        acc += self.f64_at(a_off + i * k1 + k) * other.f64_at(b_off + k * n + j);
                    }
                    vals[b * m * n + i * n + j] = acc;
                }
            }
        }
        let dtype = self.dtype.promote(other.dtype);
        if dtype.is_int() {
            Tensor::from_i64(dtype, out_shape, vals.iter().map(|v| *v as i64).collect())
        } else {
            Tensor::from_f64(dtype, out_shape, vals)
        }
    }

    pub fn conv2d(&self, weight: &Tensor, stride: (usize, usize), padding: (usize, usize)) -> TResult<Tensor> {
        if self.rank() != 4 || weight.rank() != 4 {
            return Err("conv2d expects NCHW input and OIHW weight".into());
        }
        let (n, c, h, w) = (self.shape[0], self.shape[1], self.shape[2], self.shape[3]);
        let (o, cw, kh, kw) = (weight.shape[0], weight.shape[1], weight.shape[2], weight.shape[3]);
        if c != cw {
            return Err(format!("conv2d channel mismatch: input {} vs weight {}", c, cw));
        }
        let out_h = (h + 2 * padding.0).saturating_sub(kh) / stride.0.max(1) + 1;
        let out_w = (w + 2 * padding.1).saturating_sub(kw) / stride.1.max(1) + 1;
        let mut vals = vec![0.0; n * o * out_h * out_w];
        for ni in 0..n {
            for oi in 0..o {
                for y in 0..out_h {
                    for x in 0..out_w {
                        let mut acc = 0.0;
                        for ci in 0..c {
                            for ky in 0..kh {
                                for kx in 0..kw {
                                    let iy = (y * stride.0 + ky) as isize - padding.0 as isize;
                                    let ix = (x * stride.1 + kx) as isize - padding.1 as isize;
                                    if iy < 0 || ix < 0 || iy >= h as isize || ix >= w as isize {
                                        continue;
                                    }
                                    let iv = self.f64_at(((ni * c + ci) * h + iy as usize) * w + ix as usize);
                                    let wv = weight.f64_at(((oi * c + ci) * kh + ky) * kw + kx);
                                    acc += iv * wv;
                                }
                            }
                        }
                        vals[((ni * o + oi) * out_h + y) * out_w + x] = acc;
                    }
                }
            }
        }
        let dtype = self.dtype.promote(weight.dtype);
        Tensor::from_f64(dtype, vec![n, o, out_h, out_w], vals)
    }

    pub fn conv1d(&self, weight: &Tensor, stride: usize, padding: usize) -> TResult<Tensor> {
        if self.rank() != 3 || weight.rank() != 3 {
            return Err("conv1d expects NCW input and OCW weight".into());
        }
        let (n, c, w) = (self.shape[0], self.shape[1], self.shape[2]);
        let (o, cw, kw) = (weight.shape[0], weight.shape[1], weight.shape[2]);
        if c != cw {
            return Err("conv1d channel mismatch".into());
        }
        let out_w = (w + 2 * padding).saturating_sub(kw) / stride.max(1) + 1;
        let mut vals = vec![0.0; n * o * out_w];
        for ni in 0..n {
            for oi in 0..o {
                for x in 0..out_w {
                    let mut acc = 0.0;
                    for ci in 0..c {
                        for k in 0..kw {
                            let ix = (x * stride + k) as isize - padding as isize;
                            if ix < 0 || ix >= w as isize {
                                continue;
                            }
                            acc += self.f64_at((ni * c + ci) * w + ix as usize)
                                * weight.f64_at((oi * c + ci) * kw + k);
                        }
                    }
                    vals[(ni * o + oi) * out_w + x] = acc;
                }
            }
        }
        let dtype = self.dtype.promote(weight.dtype);
        Tensor::from_f64(dtype, vec![n, o, out_w], vals)
    }

    /// Select elements along `axis` using same-rank integer `indices`.
    ///
    /// The result has the shape of `indices`; position `p` is copied from
    /// `self[p]` with its `axis` coordinate replaced by `indices[p]`. Every
    /// non-`axis` dimension must match, so the index tensor determines the
    /// output shape exactly.
    pub fn gather(&self, indices: &Tensor, axis: usize) -> TResult<Tensor> {
        if axis >= self.rank() {
            return Err("gather axis out of range".into());
        }
        if indices.rank() != self.rank() {
            return Err(format!(
                "gather expects rank-{} indices, found {:?}",
                self.rank(),
                indices.shape
            ));
        }
        for (d, (data_dim, idx_dim)) in self.shape.iter().zip(indices.shape.iter()).enumerate() {
            if d != axis && data_dim != idx_dim {
                return Err(format!(
                    "gather index dim {} is {} but data dim is {}",
                    d, idx_dim, data_dim
                ));
            }
        }
        let out_shape = indices.shape.clone();
        let n: usize = out_shape.iter().product();
        let data_strides = self.strides();
        let mut vals = Vec::with_capacity(n);
        for flat in 0..n {
            let mut idx = multi_index_in(&out_shape, flat);
            let pos = indices.i64_at(flat);
            if pos < 0 || pos as usize >= self.shape[axis] {
                return Err(format!("gather index {} out of range", pos));
            }
            idx[axis] = pos as usize;
            vals.push(self.f64_at(dot(&idx, &data_strides)));
        }
        self.rebuild(out_shape, vals)
    }

    pub fn softmax(&self, axis: usize) -> TResult<Tensor> {
        if axis >= self.rank() {
            return Err("softmax axis out of range".into());
        }
        let outer: usize = self.shape[..axis].iter().product();
        let len = self.shape[axis];
        let inner: usize = self.shape[axis + 1..].iter().product();
        let mut vals = vec![0.0; self.numel()];
        for o in 0..outer {
            for i in 0..inner {
                let base = o * len * inner + i;
                let mut maxv = f64::NEG_INFINITY;
                for j in 0..len {
                    maxv = maxv.max(self.f64_at(base + j * inner));
                }
                let mut sum = 0.0;
                let mut exps = Vec::with_capacity(len);
                for j in 0..len {
                    let e = (self.f64_at(base + j * inner) - maxv).exp();
                    exps.push(e);
                    sum += e;
                }
                for j in 0..len {
                    vals[base + j * inner] = if sum == 0.0 { 0.0 } else { exps[j] / sum };
                }
            }
        }
        Tensor::from_f64(if self.dtype.is_float() { self.dtype } else { DType::F32 }, self.shape.clone(), vals)
    }

    pub fn dropout(&self, p: f64, training: bool, seed: u64) -> Tensor {
        if !training || p <= 0.0 {
            return self.clone();
        }
        let mut state = seed | 1;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        let scale = 1.0 / (1.0 - p).max(1e-12);
        let vals: Vec<f64> = (0..self.numel())
            .map(|i| {
                if next() < p {
                    0.0
                } else {
                    self.f64_at(i) * scale
                }
            })
            .collect();
        self.rebuild(self.shape.clone(), vals).unwrap()
    }

    /// Numerically stable softmax + negative log likelihood for integer labels.
    pub fn cross_entropy(&self, targets: &Tensor) -> TResult<f64> {
        if self.rank() != 2 {
            return Err("cross_entropy expects logits of shape [batch, classes]".into());
        }
        let (batch, classes) = (self.shape[0], self.shape[1]);
        let mut total = 0.0;
        for b in 0..batch {
            let mut maxv = f64::NEG_INFINITY;
            for c in 0..classes {
                maxv = maxv.max(self.f64_at(b * classes + c));
            }
            let mut sum = 0.0;
            for c in 0..classes {
                sum += (self.f64_at(b * classes + c) - maxv).exp();
            }
            let t = targets.i64_at(b.min(targets.numel().saturating_sub(1)));
            if t < 0 || t as usize >= classes {
                return Err(format!("cross_entropy target {} out of range", t));
            }
            let logprob = self.f64_at(b * classes + t as usize) - maxv - sum.ln();
            total -= logprob;
        }
        Ok(total / batch as f64)
    }

    pub fn reshape_2d(&self, rows: usize, cols: usize) -> TResult<Tensor> {
        if self.numel() != rows * cols {
            return Err(format!(
                "cannot view a tensor with {} elements as {}x{}",
                self.numel(),
                rows,
                cols
            ));
        }
        Ok(Tensor { dtype: self.dtype, shape: vec![rows, cols], data: self.data.clone() })
    }
}

fn strides_of(shape: &[usize]) -> Vec<usize> {
    let mut s = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        s[i] = s[i + 1] * shape[i + 1];
    }
    s
}

fn dot(a: &[usize], b: &[usize]) -> usize {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn multi_index_in(shape: &[usize], mut flat: usize) -> Vec<usize> {
    let strides = strides_of(shape);
    let mut idx = vec![0; shape.len()];
    for i in 0..shape.len() {
        if shape[i] == 0 {
            continue;
        }
        idx[i] = flat / strides[i];
        flat %= strides[i];
    }
    idx
}

/// Convert a host value to the integer representation of `dtype`.
///
/// Integer storage is transparent about wrapping: storing 300 in an `i8`
/// tensor yields 44, exactly as a narrowing cast would on hardware.
fn int_store(v: f64, dtype: DType) -> i64 {
    match dtype {
        DType::I8 => (v as i64) as i8 as i64,
        DType::I32 => (v as i64) as i32 as i64,
        DType::I64 => v as i64,
        _ => v as i64,
    }
}

/// Round a value to the precision of its declared element type.
pub fn quantize(v: f64, dtype: DType) -> f64 {
    match dtype {
        DType::F64 => v,
        DType::F32 => v as f32 as f64,
        DType::F16 => f16_round(v),
        DType::BF16 => {
            let bits = (v as f32).to_bits() & 0xffff_0000;
            f32::from_bits(bits) as f64
        }
        DType::I8 => ((v as i8) as f64).clamp(-128.0, 127.0),
        DType::I32 => v as i32 as f64,
        DType::I64 => v as i64 as f64,
        DType::Bool => {
            if v != 0.0 {
                1.0
            } else {
                0.0
            }
        }
    }
}

fn f16_round(v: f64) -> f64 {
    let f = v as f32;
    let bits = f.to_bits();
    let sign = bits & 0x8000_0000;
    let exp = (bits >> 23) & 0xff;
    let mant = bits & 0x7f_ffff;
    // Flush values outside the f16 range to +/-inf or 0 like hardware would.
    if exp == 0xff {
        return f as f64;
    }
    let e = exp as i32 - 127;
    if e > 15 {
        return if sign != 0 { f64::NEG_INFINITY } else { f64::INFINITY };
    }
    if e < -24 {
        return 0.0 * if sign != 0 { -1.0 } else { 1.0 };
    }
    // Keep 10 mantissa bits with round-to-nearest-even.
    if e >= -14 {
        let keep = 10u32;
        let shift = 23 - keep;
        let mut m = mant >> shift;
        let rem = mant & ((1u32 << shift) - 1);
        let half = 1u32 << (shift - 1);
        if rem > half || (rem == half && (m & 1) == 1) {
            m += 1;
        }
        let out = ((e + 127) as u32) << 23 | (m << shift);
        return f32::from_bits(sign | out) as f64;
    }
    // Subnormal f16: represent via f32 arithmetic.
    let scale = 2f64.powi(-24);
    (v / scale).round() * scale
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tensor{:?};{}", self.shape, self.dtype)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use neura_ir::ops::TensorOp;

    fn t(shape: Vec<usize>, vals: Vec<f64>) -> Tensor {
        Tensor::from_f64(DType::F32, shape, vals).unwrap()
    }

    #[test]
    fn elementwise_broadcasting() {
        let a = t(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = t(vec![3], vec![10.0, 20.0, 30.0]);
        let c = a.elementwise(&TensorOp::Add, &b).unwrap();
        assert_eq!(c.shape, vec![2, 3]);
        assert_eq!(c.to_f64_vec(), vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
        assert!(a.elementwise(&TensorOp::Add, &t(vec![2], vec![1.0, 2.0])).is_err());
    }

    #[test]
    fn matmul_is_correct() {
        let a = t(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = t(vec![3, 2], vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let c = a.matmul(&b).unwrap();
        assert_eq!(c.shape, vec![2, 2]);
        assert_eq!(c.to_f64_vec(), vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn reductions_and_softmax() {
        let a = t(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let s = a.reduce(&TensorOp::Sum { axes: vec![1], keepdim: false }).unwrap();
        assert_eq!(s.shape, vec![2]);
        assert_eq!(s.to_f64_vec(), vec![6.0, 15.0]);
        let m = a.reduce(&TensorOp::Mean { axes: vec![0], keepdim: true }).unwrap();
        assert_eq!(m.shape, vec![1, 3]);
        assert_eq!(m.to_f64_vec(), vec![2.5, 3.5, 4.5]);
        let p = a.softmax(1).unwrap();
        for row in p.to_f64_vec().chunks(3) {
            assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn reshape_transpose_slice() {
        let a = t(vec![2, 6], (0..12).map(|x| x as f64).collect());
        let r = a.reshape(&[Dim::Static(3), Dim::Static(4)]).unwrap();
        assert_eq!(r.shape, vec![3, 4]);
        let tr = a.transpose(0, 1).unwrap();
        assert_eq!(tr.shape, vec![6, 2]);
        let s = a.slice(1, 1, 4).unwrap();
        assert_eq!(s.shape, vec![2, 3]);
        assert_eq!(s.to_f64_vec(), vec![1.0, 2.0, 3.0, 7.0, 8.0, 9.0]);
        let f = a.flatten();
        assert_eq!(f.shape, vec![12]);
    }

    #[test]
    fn conv2d_matches_hand_computation() {
        let x = t(vec![1, 1, 3, 3], (0..9).map(|v| v as f64).collect());
        let w = t(vec![1, 1, 2, 2], vec![1.0, 0.0, 0.0, -1.0]);
        let y = x.conv2d(&w, (1, 1), (0, 0)).unwrap();
        assert_eq!(y.shape, vec![1, 1, 2, 2]);
        assert_eq!(y.to_f64_vec(), vec![-4.0, -4.0, -4.0, -4.0]);
    }

    #[test]
    fn dtype_precision_is_honest() {
        let f = Tensor::from_f64(DType::F16, vec![1], vec![1.0009765625]).unwrap();
        assert_eq!(f.f64_at(0), 1.0009765625);
        let rounded = Tensor::from_f64(DType::BF16, vec![1], vec![1.0009765625]).unwrap();
        assert_eq!(rounded.f64_at(0), 1.0);
        let i = Tensor::from_f64(DType::I8, vec![2], vec![300.0, -300.0]).unwrap();
        assert_eq!(i.to_f64_vec(), vec![44.0, -44.0]);
    }

    #[test]
    fn cross_entropy_is_stable() {
        let logits = t(vec![1, 3], vec![1000.0, 1000.0, 1000.0]);
        let target = Tensor::from_i64(DType::I64, vec![1], vec![2]).unwrap();
        let loss = logits.cross_entropy(&target).unwrap();
        assert!((loss - 3f64.ln()).abs() < 1e-9, "loss={}", loss);
    }
}
