//! Graph-level optimisation passes.
//!
//! All passes operate on straight-line regions of the EIR bytecode and stop at
//! control-flow boundaries, so a rewrite never changes which values are live on
//! a given path. Jump targets are remapped whenever instructions are removed.

use crate::PassOutcome;
use neura_ir::eir::{EirProgram, Inst, Reg};
use neura_ir::ops::BinOp;
use neura_ir::types::Ty;
use std::collections::{BTreeSet, HashMap, HashSet};

/// True when an instruction produces a value without observable side effects.
pub(crate) fn is_pure(inst: &Inst) -> bool {
    matches!(
        inst,
        Inst::ConstF64 { .. }
            | Inst::ConstInt { .. }
            | Inst::ConstBool { .. }
            | Inst::ConstStr { .. }
            | Inst::ConstUnit { .. }
            | Inst::Copy { .. }
            | Inst::MakeList { .. }
            | Inst::MakeTensor { .. }
            | Inst::MakeStruct { .. }
            | Inst::TensorOp { .. }
            | Inst::FusedOp { .. }
            | Inst::BinOp { .. }
            | Inst::UnOp { .. }
            | Inst::GetAttr { .. }
            | Inst::Index { .. }
            | Inst::Len { .. }
    )
}

pub(crate) fn use_counts(code: &[Inst]) -> HashMap<Reg, u32> {
    let mut counts = HashMap::new();
    for inst in code {
        for r in inst.reads() {
            *counts.entry(r).or_insert(0) += 1;
        }
    }
    counts
}

/// Instruction indices that a jump or loop terminator can land on.
pub(crate) fn jump_targets(code: &[Inst]) -> HashSet<usize> {
    let mut targets = HashSet::new();
    for inst in code {
        match inst {
            Inst::Jump(t) => {
                targets.insert(*t);
            }
            Inst::JumpIfFalse { target, .. } | Inst::JumpIfTrue { target, .. } => {
                targets.insert(*target);
            }
            Inst::IterNext { end, .. } => {
                targets.insert(*end);
            }
            _ => {}
        }
    }
    targets
}

/// Remove instructions and rewrite every jump so targets still point at the
/// same logical position.
pub(crate) fn remove_indices(code: &[Inst], remove: &BTreeSet<usize>) -> Vec<Inst> {
    let n = code.len();
    let kept: Vec<usize> = (0..n).filter(|i| !remove.contains(i)).collect();
    let map = |old: usize| kept.iter().filter(|&&i| i < old).count();
    let map_target = |old: usize| if old >= n { kept.len() } else { map(old) };
    let mut out = Vec::with_capacity(kept.len());
    for &i in &kept {
        let mut inst = code[i].clone();
        match &mut inst {
            Inst::Jump(t) => *t = map_target(*t),
            Inst::JumpIfFalse { target, .. } | Inst::JumpIfTrue { target, .. } => {
                *target = map_target(*target)
            }
            Inst::IterNext { end, .. } => *end = map_target(*end),
            _ => {}
        }
        out.push(inst);
    }
    out
}

fn rewrite_reads(inst: &mut Inst, subst: &HashMap<Reg, Reg>) {
    if subst.is_empty() {
        return;
    }
    let apply = |r: &mut Reg| {
        if let Some(new) = subst.get(r) {
            *r = *new;
        }
    };
    match inst {
        Inst::Copy { src, .. } | Inst::Move { src, .. } | Inst::UnOp { src, .. } => apply(src),
        Inst::MakeList { elems, .. } => elems.iter_mut().for_each(apply),
        Inst::MakeTensor { values, .. } => apply(values),
        Inst::MakeStruct { fields, .. } => fields.iter_mut().for_each(apply),
        Inst::TensorOp { args, dyn_dims, .. } => {
            args.iter_mut().for_each(apply);
            dyn_dims.iter_mut().for_each(apply);
        }
        Inst::FusedOp { args, .. } => args.iter_mut().for_each(apply),
        Inst::BinOp { lhs, rhs, .. } => {
            apply(lhs);
            apply(rhs);
        }
        Inst::Call { args, .. } => args.iter_mut().for_each(apply),
        Inst::CallBuiltin { args, kwargs, .. } => {
            args.iter_mut().for_each(apply);
            for (_, r) in kwargs {
                apply(r);
            }
        }
        Inst::MethodCall { recv, args, kwargs, .. } => {
            apply(recv);
            args.iter_mut().for_each(apply);
            for (_, r) in kwargs {
                apply(r);
            }
        }
        Inst::GetAttr { obj, .. } => apply(obj),
        Inst::Index { obj, index, .. } => {
            apply(obj);
            apply(index);
        }
        Inst::Len { obj, .. } => apply(obj),
        Inst::IterInit { iter, .. } => apply(iter),
        Inst::IterNext { state, .. } => apply(state),
        Inst::JumpIfFalse { cond, .. } | Inst::JumpIfTrue { cond, .. } => apply(cond),
        Inst::Return(Some(r)) => apply(r),
        Inst::Print(r) | Inst::Drop(r) => apply(r),
        Inst::Input { prompt: Some(r), .. } => apply(r),
        _ => {}
    }
}

#[derive(Clone, Debug)]
enum ConstVal {
    F64(f64),
    Int(i64),
    Bool(bool),
    Str(String),
    Unit,
}

impl ConstVal {
    fn from_inst(inst: &Inst) -> Option<ConstVal> {
        Some(match inst {
            Inst::ConstF64 { v, .. } => ConstVal::F64(*v),
            Inst::ConstInt { v, .. } => ConstVal::Int(*v),
            Inst::ConstBool { v, .. } => ConstVal::Bool(*v),
            Inst::ConstStr { s, .. } => ConstVal::Str(s.clone()),
            Inst::ConstUnit { .. } => ConstVal::Unit,
            _ => return None,
        })
    }

    fn to_inst(&self, dst: Reg) -> Inst {
        match self {
            ConstVal::F64(v) => Inst::ConstF64 { dst, v: *v },
            ConstVal::Int(v) => Inst::ConstInt { dst, v: *v },
            ConstVal::Bool(v) => Inst::ConstBool { dst, v: *v },
            ConstVal::Str(s) => Inst::ConstStr { dst, s: s.clone() },
            ConstVal::Unit => Inst::ConstUnit { dst },
        }
    }

    fn as_f64(&self) -> Option<f64> {
        match self {
            ConstVal::F64(v) => Some(*v),
            ConstVal::Int(v) => Some(*v as f64),
            ConstVal::Bool(v) => Some(*v as i64 as f64),
            _ => None,
        }
    }

    fn as_i64(&self) -> Option<i64> {
        match self {
            ConstVal::F64(v) => Some(*v as i64),
            ConstVal::Int(v) => Some(*v),
            ConstVal::Bool(v) => Some(*v as i64),
            _ => None,
        }
    }
}

fn const_eq(a: &ConstVal, b: &ConstVal) -> Option<bool> {
    Some(match (a, b) {
        (ConstVal::Str(x), ConstVal::Str(y)) => x == y,
        (ConstVal::Bool(x), ConstVal::Bool(y)) => x == y,
        (ConstVal::Unit, ConstVal::Unit) => true,
        _ => return None,
    })
}

fn fold_bin(op: BinOp, ty: &Ty, a: &ConstVal, b: &ConstVal) -> Option<ConstVal> {
    use BinOp::*;
    match op {
        Eq | Ne => {
            let equal = match const_eq(a, b) {
                Some(v) => v,
                None => a.as_f64()? == b.as_f64()?,
            };
            return Some(ConstVal::Bool(if op == Eq { equal } else { !equal }));
        }
        Lt | Le | Gt | Ge => {
            let (x, y) = (a.as_f64()?, b.as_f64()?);
            return Some(ConstVal::Bool(match op {
                Lt => x < y,
                Le => x <= y,
                Gt => x > y,
                _ => x >= y,
            }));
        }
        And | Or => {
            let (x, y) = match (a, b) {
                (ConstVal::Bool(x), ConstVal::Bool(y)) => (*x, *y),
                _ => return None,
            };
            return Some(ConstVal::Bool(if op == And { x && y } else { x || y }));
        }
        Add => {
            if let (ConstVal::Str(x), ConstVal::Str(y)) = (a, b) {
                return Some(ConstVal::Str(format!("{}{}", x, y)));
            }
        }
        _ => {}
    }
    let ints = matches!((a, b), (ConstVal::Int(_), ConstVal::Int(_))) && !matches!(ty, Ty::Float);
    if ints {
        let (x, y) = (a.as_i64()?, b.as_i64()?);
        let v = match op {
            Add => x.wrapping_add(y),
            Sub => x.wrapping_sub(y),
            Mul => x.wrapping_mul(y),
            Div => {
                if y == 0 {
                    return None;
                }
                x / y
            }
            FloorDiv => {
                if y == 0 {
                    return None;
                }
                let q = x / y;
                let r = x % y;
                if r != 0 && ((r < 0) != (y < 0)) {
                    q - 1
                } else {
                    q
                }
            }
            Pow => {
                if y < 0 || y > u32::MAX as i64 {
                    return None;
                }
                x.wrapping_pow(y as u32)
            }
            Mod => {
                if y == 0 {
                    return None;
                }
                x % y
            }
            _ => return None,
        };
        return Some(ConstVal::Int(v));
    }
    let (x, y) = (a.as_f64()?, b.as_f64()?);
    let v = match op {
        Add => x + y,
        Sub => x - y,
        Mul => x * y,
        Div => x / y,
        FloorDiv => (x / y).floor(),
        Pow => x.powf(y),
        Mod => x % y,
        _ => return None,
    };
    Some(if matches!(ty, Ty::Int) { ConstVal::Int(v as i64) } else { ConstVal::F64(v) })
}

fn fold_un(op: neura_ir::ops::UnOp, ty: &Ty, a: &ConstVal) -> Option<ConstVal> {
    use neura_ir::ops::UnOp::*;
    Some(match op {
        Not => match a {
            ConstVal::Bool(v) => ConstVal::Bool(!v),
            _ => return None,
        },
        Neg => {
            if matches!(ty, Ty::Int) || matches!(a, ConstVal::Int(_)) {
                ConstVal::Int(a.as_i64()?.wrapping_neg())
            } else {
                ConstVal::F64(-a.as_f64()?)
            }
        }
    })
}

fn is_boundary(inst: &Inst) -> bool {
    matches!(
        inst,
        Inst::Jump(_) | Inst::JumpIfFalse { .. } | Inst::JumpIfTrue { .. } | Inst::IterNext { .. }
    )
}

/// Constant propagation plus folding of scalar arithmetic.
pub fn fold_constants(program: &mut EirProgram) -> PassOutcome {
    let mut folded = 0usize;
    for f in &mut program.functions {
        let targets = jump_targets(&f.code);
        let mut env: HashMap<Reg, ConstVal> = HashMap::new();
        for i in 0..f.code.len() {
            if targets.contains(&i) {
                env.clear();
            }
            let inst = f.code[i].clone();
            if let Some(c) = ConstVal::from_inst(&inst) {
                if let Some(d) = inst.dst() {
                    env.insert(d, c);
                }
                continue;
            }
            match &inst {
                Inst::Copy { dst, src } => {
                    if let Some(c) = env.get(src).cloned() {
                        env.insert(*dst, c);
                    } else {
                        env.remove(dst);
                    }
                }
                Inst::BinOp { dst, op, lhs, rhs, ty } => {
                    let value = match (env.get(lhs), env.get(rhs)) {
                        (Some(a), Some(b)) => fold_bin(*op, ty, a, b),
                        _ => None,
                    };
                    match value {
                        Some(c) => {
                            f.code[i] = c.to_inst(*dst);
                            env.insert(*dst, c);
                            folded += 1;
                        }
                        None => {
                            env.remove(dst);
                        }
                    }
                }
                Inst::UnOp { dst, op, src, ty } => {
                    let value = env.get(src).and_then(|a| fold_un(*op, ty, a));
                    match value {
                        Some(c) => {
                            f.code[i] = c.to_inst(*dst);
                            env.insert(*dst, c);
                            folded += 1;
                        }
                        None => {
                            env.remove(dst);
                        }
                    }
                }
                other => {
                    if let Some(d) = other.dst() {
                        env.remove(&d);
                    }
                    if is_boundary(other) {
                        env.clear();
                    }
                }
            }
        }
    }
    PassOutcome {
        name: "constant-folding",
        changed: folded,
        detail: format!("folded {} scalar expressions", folded),
    }
}

fn const_of(env: &HashMap<Reg, ConstVal>, r: Reg) -> Option<f64> {
    env.get(&r).and_then(|c| c.as_f64())
}

/// Identities such as `x * 1`, `x + 0` and `x - x`.
///
/// Float identities are limited to multiplication/division by one so that
/// signed zero and NaN behaviour is preserved.
pub fn simplify_algebraic(program: &mut EirProgram) -> PassOutcome {
    use BinOp::*;
    let mut changed = 0usize;
    for f in &mut program.functions {
        let targets = jump_targets(&f.code);
        let mut env: HashMap<Reg, ConstVal> = HashMap::new();
        for i in 0..f.code.len() {
            if targets.contains(&i) {
                env.clear();
            }
            let inst = f.code[i].clone();
            if let Some(c) = ConstVal::from_inst(&inst) {
                if let Some(d) = inst.dst() {
                    env.insert(d, c);
                }
                continue;
            }
            if let Inst::BinOp { dst, op, lhs, rhs, ty } = &inst {
                let int_ty = matches!(ty, Ty::Int);
                let l = const_of(&env, *lhs);
                let r = const_of(&env, *rhs);
                let mut replacement: Option<Inst> = None;
                match op {
                    Add if int_ty => {
                        if l == Some(0.0) {
                            replacement = Some(Inst::Copy { dst: *dst, src: *rhs });
                        } else if r == Some(0.0) {
                            replacement = Some(Inst::Copy { dst: *dst, src: *lhs });
                        }
                    }
                    Sub if int_ty => {
                        if r == Some(0.0) {
                            replacement = Some(Inst::Copy { dst: *dst, src: *lhs });
                        } else if lhs == rhs {
                            replacement = Some(Inst::ConstInt { dst: *dst, v: 0 });
                        }
                    }
                    Mul => {
                        if l == Some(1.0) {
                            replacement = Some(Inst::Copy { dst: *dst, src: *rhs });
                        } else if r == Some(1.0) {
                            replacement = Some(Inst::Copy { dst: *dst, src: *lhs });
                        } else if int_ty && (l == Some(0.0) || r == Some(0.0)) {
                            replacement = Some(Inst::ConstInt { dst: *dst, v: 0 });
                        }
                    }
                    Div => {
                        if r == Some(1.0) {
                            replacement = Some(Inst::Copy { dst: *dst, src: *lhs });
                        }
                    }
                    _ => {}
                }
                if let Some(new) = replacement {
                    f.code[i] = new;
                    env.remove(dst);
                    changed += 1;
                    continue;
                }
                env.remove(dst);
                continue;
            }
            if let Some(d) = inst.dst() {
                env.remove(&d);
            }
            if is_boundary(&inst) {
                env.clear();
            }
        }
    }
    PassOutcome {
        name: "algebraic-simplification",
        changed,
        detail: format!("applied {} algebraic identities", changed),
    }
}

fn expr_key(inst: &Inst) -> Option<(String, Vec<Reg>)> {
    let reads = inst.reads();
    let key = match inst {
        Inst::ConstF64 { v, .. } => format!("f64:{}", v),
        Inst::ConstInt { v, .. } => format!("int:{}", v),
        Inst::ConstBool { v, .. } => format!("bool:{}", v),
        Inst::ConstStr { s, .. } => format!("str:{:?}", s),
        Inst::ConstUnit { .. } => "unit".to_string(),
        Inst::Copy { src, .. } => format!("copy:%{}", src),
        Inst::BinOp { op, lhs, rhs, ty, .. } => {
            format!("bin:{}:%{}:%{}:{}", op.symbol(), lhs, rhs, ty.name())
        }
        Inst::UnOp { op, src, ty, .. } => format!("un:{}:%{}:{}", op.symbol(), src, ty.name()),
        Inst::TensorOp { op, args, dyn_dims, ty, .. } => format!(
            "tir:{:?}:{:?}:{:?}:{}",
            op, args, dyn_dims, ty
        ),
        Inst::FusedOp { spec, args, ty, .. } => {
            format!("fused:{}:{:?}:{}", spec.expr, args, ty)
        }
        Inst::GetAttr { obj, name, .. } => format!("getattr:%{}.{}", obj, name),
        Inst::Index { obj, index, .. } => format!("index:%{}[%{}]", obj, index),
        Inst::Len { obj, .. } => format!("len:%{}", obj),
        _ => return None,
    };
    Some((key, reads))
}

/// Available-expression elimination inside basic blocks.
pub fn eliminate_common_subexpressions(program: &mut EirProgram) -> PassOutcome {
    let mut removed = 0usize;
    for f in &mut program.functions {
        let targets = jump_targets(&f.code);
        let mut available: HashMap<String, (Reg, Vec<Reg>)> = HashMap::new();
        let mut subst: HashMap<Reg, Reg> = HashMap::new();
        for i in 0..f.code.len() {
            let boundary = targets.contains(&i);
            if boundary {
                available.clear();
                subst.clear();
            }
            let mut inst = f.code[i].clone();
            rewrite_reads(&mut inst, &subst);
            f.code[i] = inst.clone();

            if is_boundary(&inst) {
                available.clear();
                subst.clear();
                continue;
            }
            if matches!(
                inst,
                Inst::Call { .. } | Inst::CallBuiltin { .. } | Inst::MethodCall { .. }
            ) {
                // Library calls can mutate state, so nothing is known afterwards.
                available.clear();
                subst.clear();
                continue;
            }
            let Some((key, reads)) = expr_key(&inst) else {
                if let Some(d) = inst.dst() {
                    available.retain(|_, (reg, rs)| *reg != d && !rs.contains(&d));
                    subst.retain(|_, v| *v != d);
                }
                continue;
            };
            let dst = match inst.dst() {
                Some(d) => d,
                None => continue,
            };
            // A new definition of `dst` invalidates expressions using it.
            available.retain(|_, (reg, rs)| *reg != dst && !rs.contains(&dst));
            subst.retain(|_, v| *v != dst);
            if reads.contains(&dst) {
                available.insert(key, (dst, reads));
                continue;
            }
            if let Some((existing, _)) = available.get(&key).cloned() {
                if existing != dst {
                    f.code[i] = Inst::Copy { dst, src: existing };
                    subst.insert(dst, existing);
                    removed += 1;
                    continue;
                }
            }
            available.insert(key, (dst, reads));
        }
    }
    PassOutcome {
        name: "common-subexpression-elimination",
        changed: removed,
        detail: format!("reused {} previously computed values", removed),
    }
}

/// Remove pure instructions whose result is never read.
pub fn eliminate_dead_code(program: &mut EirProgram) -> PassOutcome {
    let mut removed_total = 0usize;
    for f in &mut program.functions {
        let protected = jump_targets(&f.code);
        let mut used: HashSet<Reg> = HashSet::new();
        let mut remove: BTreeSet<usize> = BTreeSet::new();
        for i in (0..f.code.len()).rev() {
            let inst = &f.code[i];
            let dead = is_pure(inst)
                && !protected.contains(&i)
                && inst.dst().map(|d| !used.contains(&d)).unwrap_or(false);
            if dead {
                remove.insert(i);
            } else {
                for r in inst.reads() {
                    used.insert(r);
                }
            }
        }
        if !remove.is_empty() {
            removed_total += remove.len();
            f.code = remove_indices(&f.code, &remove);
        }
    }
    PassOutcome {
        name: "dead-code-elimination",
        changed: removed_total,
        detail: format!("removed {} unread instructions", removed_total),
    }
}
