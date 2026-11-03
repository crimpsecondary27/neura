//! Lowering from typed NIR to executable EIR (register VM bytecode).
//!
//! The lowering is total for the language subset accepted by the type
//! checker. Every register is typed, tensor instructions keep their shape and
//! element type, and last-use information is computed afterwards for memory
//! planning.

use neura_ir::eir::{
    CompilePolicies, EirFunction, EirLocal, EirProgram, EirStruct, Inst, Reg,
};
use neura_ir::nir::{
    CallArg, LocalId, NirBlock, NirExpr, NirFunction, NirModule, NirStmt,
};
use neura_ir::types::{DType, TensorTy, Ty};
use std::collections::HashMap;

pub struct LowerResult {
    pub program: EirProgram,
    pub diags: Vec<crate::diagnostics::Diagnostic>,
}

struct LoopCtx {
    continue_target: usize,
    break_jumps: Vec<usize>,
}

struct FnLower<'a> {
    f: &'a NirFunction,
    struct_ids: &'a HashMap<String, usize>,
    pub types: Vec<Ty>,
    pub code: Vec<Inst>,
    pub spans: Vec<(usize, usize)>,
    local_reg: HashMap<LocalId, Reg>,
    loops: Vec<LoopCtx>,
    diags: Vec<crate::diagnostics::Diagnostic>,
    saw_backward: bool,
}

fn fnv1a(src: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in src.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Lower a whole NIR module using the given compile policies.
pub fn lower(module: &NirModule, policies: CompilePolicies, source_hash: u64) -> LowerResult {
    let mut struct_ids = HashMap::new();
    let mut structs = Vec::new();
    for (i, s) in module.structs.iter().enumerate() {
        struct_ids.insert(s.name.clone(), i);
        structs.push(EirStruct {
            name: s.name.clone(),
            fields: s.fields.iter().map(|(n, _)| n.clone()).collect(),
        });
    }
    let mut functions = Vec::new();
    for f in &module.functions {
        functions.push(FnLower::new(f, &struct_ids).run());
    }
    let main = functions.iter().position(|f| f.name == "main").unwrap_or(0);
    LowerResult {
        program: EirProgram {
            name: module.name.clone(),
            functions,
            main,
            structs,
            policies,
            source_hash,
        },
        diags: Vec::new(),
    }
}

impl<'a> FnLower<'a> {
    fn new(f: &'a NirFunction, struct_ids: &'a HashMap<String, usize>) -> Self {
        let mut types = Vec::new();
        let mut local_reg = HashMap::new();
        for (i, l) in f.locals.iter().enumerate() {
            types.push(l.ty.clone());
            local_reg.insert(i as LocalId, i as Reg);
        }
        FnLower {
            f,
            struct_ids,
            types,
            code: Vec::new(),
            spans: Vec::new(),
            local_reg,
            loops: Vec::new(),
            diags: Vec::new(),
            saw_backward: false,
        }
    }

    fn run(mut self) -> EirFunction {
        self.walk_block(&self.f.body);
        // Functions without an explicit return yield Unit; functions with one
        // have already emitted it, so a trailing `return` is harmless (the VM
        // stops at the first one).
        let f = self.f;
        let mut locals: Vec<EirLocal> = f
            .locals
            .iter()
            .enumerate()
            .map(|(_i, l)| EirLocal {
                name: l.name.clone(),
                ty: l.ty.clone(),
                mutable: l.mutable,
                is_param: l.is_param,
                last_use: None,
                needs_autograd: false,
            })
            .collect();
        for (i, l) in locals.iter_mut().enumerate() {
            l.last_use = last_use_of(&self.code, i as Reg);
            l.needs_autograd = self.saw_backward && matches!(l.ty, Ty::Tensor(_));
        }
        let params = f
            .params
            .iter()
            .map(|p| *self.local_reg.get(&p.local).unwrap_or(&0))
            .collect();
        EirFunction {
            name: f.name.clone(),
            params,
            locals,
            types: self.types,
            code: self.code,
            ret_ty: f.ret.clone(),
            spans: self.spans,
        }
    }

    fn alloc(&mut self, ty: Ty) -> Reg {
        let r = self.types.len() as Reg;
        self.types.push(ty);
        r
    }

    fn emit(&mut self, inst: Inst, span: (usize, usize)) -> usize {
        let at = self.code.len();
        self.code.push(inst);
        self.spans.push(span);
        at
    }

    fn walk_block(&mut self, b: &NirBlock) {
        for s in &b.stmts {
            self.walk_stmt(s);
        }
    }

    fn walk_stmt(&mut self, s: &NirStmt) {
        match s {
            NirStmt::Let { local, expr, span } => {
                let dst = *self.local_reg.get(local).unwrap_or(&0);
                let src = self.lower_expr(expr, *span);
                if src != dst {
                    self.emit(Inst::Copy { dst, src }, *span);
                }
            }
            NirStmt::Assign { local, expr, span } => {
                let dst = *self.local_reg.get(local).unwrap_or(&0);
                let src = self.lower_expr(expr, *span);
                if src != dst {
                    self.emit(Inst::Copy { dst, src }, *span);
                }
            }
            NirStmt::Expr(e) => {
                self.lower_expr(e, (0, 0));
            }
            NirStmt::If { cond, then_block, else_block, span } => {
                let c = self.lower_expr(cond, *span);
                let jf = self.emit(Inst::JumpIfFalse { cond: c, target: 0 }, *span);
                self.walk_block(then_block);
                let jump_end = if else_block.is_some() {
                    Some(self.emit(Inst::Jump(0), *span))
                } else {
                    None
                };
                let else_start = self.code.len();
                if let Some(b) = else_block {
                    self.walk_block(b);
                }
                if let Inst::JumpIfFalse { target, .. } = &mut self.code[jf] {
                    *target = else_start;
                }
                if let Some(j) = jump_end {
                    let end = self.code.len();
                    if let Inst::Jump(t) = &mut self.code[j] {
                        *t = end;
                    }
                }
            }
            NirStmt::While { cond, body, span } => {
                let start = self.code.len();
                let c = self.lower_expr(cond, *span);
                let jf = self.emit(Inst::JumpIfFalse { cond: c, target: 0 }, *span);
                self.loops.push(LoopCtx { continue_target: start, break_jumps: Vec::new() });
                self.walk_block(body);
                self.emit(Inst::Jump(start), *span);
                let end = self.code.len();
                if let Inst::JumpIfFalse { target, .. } = &mut self.code[jf] {
                    *target = end;
                }
                let ctx = self.loops.pop().unwrap();
                for j in ctx.break_jumps {
                    if let Inst::Jump(t) = &mut self.code[j] {
                        *t = end;
                    }
                }
            }
            NirStmt::For { var, iter, body, span } => {
                let it = self.lower_expr(iter, *span);
                let state = self.alloc(Ty::Any);
                self.emit(Inst::IterInit { dst: state, iter: it }, *span);
                let var_reg = *self.local_reg.get(var).unwrap_or(&0);
                let start = self.code.len();
                let next = self.emit(
                    Inst::IterNext { dst: var_reg, state, end: 0 },
                    *span,
                );
                self.loops.push(LoopCtx { continue_target: start, break_jumps: Vec::new() });
                self.walk_block(body);
                self.emit(Inst::Jump(start), *span);
                let end = self.code.len();
                if let Inst::IterNext { end: e, .. } = &mut self.code[next] {
                    *e = end;
                }
                let ctx = self.loops.pop().unwrap();
                for j in ctx.break_jumps {
                    if let Inst::Jump(t) = &mut self.code[j] {
                        *t = end;
                    }
                }
            }
            NirStmt::Return(value, span) => {
                let r = value.as_ref().map(|e| self.lower_expr(e, *span));
                self.emit(Inst::Return(r), *span);
            }
            NirStmt::Break(span) => {
                let j = self.emit(Inst::Jump(0), *span);
                if let Some(ctx) = self.loops.last_mut() {
                    ctx.break_jumps.push(j);
                }
            }
            NirStmt::Continue(span) => {
                let target = self.loops.last().map(|c| c.continue_target).unwrap_or(0);
                self.emit(Inst::Jump(target), *span);
            }
            NirStmt::Pass => {}
        }
    }

    fn lower_expr(&mut self, e: &NirExpr, span: (usize, usize)) -> Reg {
        match e {
            NirExpr::Int(v) => {
                let d = self.alloc(Ty::Int);
                self.emit(Inst::ConstInt { dst: d, v: *v }, span);
                d
            }
            NirExpr::Float(v) => {
                let d = self.alloc(Ty::Float);
                self.emit(Inst::ConstF64 { dst: d, v: *v }, span);
                d
            }
            NirExpr::Bool(v) => {
                let d = self.alloc(Ty::Bool);
                self.emit(Inst::ConstBool { dst: d, v: *v }, span);
                d
            }
            NirExpr::Str(s) => {
                let d = self.alloc(Ty::Str);
                self.emit(Inst::ConstStr { dst: d, s: s.clone() }, span);
                d
            }
            NirExpr::Unit => {
                let d = self.alloc(Ty::Unit);
                self.emit(Inst::ConstUnit { dst: d }, span);
                d
            }
            NirExpr::Local(id) | NirExpr::Move(id) | NirExpr::Borrow { local: id, .. } => {
                *self.local_reg.get(id).unwrap_or(&0)
            }
            NirExpr::Unary { op, expr, ty } => {
                let src = self.lower_expr(expr, span);
                let d = self.alloc(ty.clone());
                self.emit(Inst::UnOp { dst: d, op: *op, src, ty: ty.clone() }, span);
                d
            }
            NirExpr::Binary { op, lhs, rhs, ty } => {
                let l = self.lower_expr(lhs, span);
                let r = self.lower_expr(rhs, span);
                let d = self.alloc(ty.clone());
                self.emit(
                    Inst::BinOp { dst: d, op: *op, lhs: l, rhs: r, ty: ty.clone() },
                    span,
                );
                d
            }
            NirExpr::List { elems, ty } => {
                let elems: Vec<Reg> = elems.iter().map(|x| self.lower_expr(x, span)).collect();
                let d = self.alloc(ty.clone());
                self.emit(Inst::MakeList { dst: d, elems }, span);
                d
            }
            NirExpr::Module(name) => {
                let d = self.alloc(Ty::Module(name.clone()));
                self.emit(Inst::ConstUnit { dst: d }, span);
                d
            }
            NirExpr::Builtin { ty, .. } => {
                let d = self.alloc(ty.clone());
                self.emit(Inst::ConstUnit { dst: d }, span);
                d
            }
            NirExpr::FunctionRef { ty, .. } => {
                let d = self.alloc(ty.clone());
                self.emit(Inst::ConstUnit { dst: d }, span);
                d
            }
            NirExpr::Attr { obj, name, ty } => {
                let o = self.lower_expr(obj, span);
                let d = self.alloc(ty.clone());
                self.emit(
                    Inst::GetAttr { dst: d, obj: o, name: name.clone(), ty: ty.clone() },
                    span,
                );
                d
            }
            NirExpr::Index { obj, index, ty } => {
                let o = self.lower_expr(obj, span);
                let i = self.lower_expr(index, span);
                let d = self.alloc(ty.clone());
                self.emit(
                    Inst::Index { dst: d, obj: o, index: i, ty: ty.clone() },
                    span,
                );
                d
            }
            NirExpr::TensorOp { op, args, dyn_dims, ty } => {
                let a: Vec<Reg> = args.iter().map(|x| self.lower_expr(x, span)).collect();
                let dd: Vec<Reg> = dyn_dims.iter().map(|x| self.lower_expr(x, span)).collect();
                let d = self.alloc(Ty::Tensor(ty.clone()));
                self.emit(
                    Inst::TensorOp {
                        dst: d,
                        op: op.clone(),
                        args: a,
                        dyn_dims: dd,
                        ty: ty.clone(),
                    },
                    span,
                );
                d
            }
            NirExpr::Call { callee, args, ty, span: call_span } => {
                self.lower_call(callee, args, ty, *call_span)
            }
        }
    }

    fn lower_call(
        &mut self,
        callee: &NirExpr,
        args: &[CallArg],
        ty: &Ty,
        span: (usize, usize),
    ) -> Reg {
        let (pos, kwargs) = self.lower_args(args, span);
        match callee {
            NirExpr::Builtin { name, .. } => {
                if name == "input" {
                    let prompt = pos.first().copied();
                    let d = self.alloc(ty.clone());
                    self.emit(Inst::Input { dst: d, prompt, ty: ty.clone() }, span);
                    return d;
                }
                if name == "print" {
                    if let Some(r) = pos.first() {
                        self.emit(Inst::Print(*r), span);
                    } else {
                        let u = self.alloc(Ty::Unit);
                        self.emit(Inst::Print(u), span);
                    }
                    let u = self.alloc(Ty::Unit);
                    self.emit(Inst::ConstUnit { dst: u }, span);
                    return u;
                }
                if let Some(struct_name) = name.strip_prefix("struct.") {
                    let id = *self.struct_ids.get(struct_name).unwrap_or(&0);
                    let d = self.alloc(ty.clone());
                    self.emit(
                        Inst::MakeStruct { dst: d, struct_id: id, fields: pos },
                        span,
                    );
                    return d;
                }
                if name == "tensor.Tensor" {
                    let values = pos.first().copied().unwrap_or_else(|| {
                        let d = self.alloc(Ty::Unit);
                        self.emit(Inst::ConstUnit { dst: d }, span);
                        d
                    });
                    let requires_grad = kwargs
                        .iter()
                        .find(|(k, _)| k == "requires_grad")
                        .and_then(|(_, r)| const_bool_at(&self.code, *r))
                        .unwrap_or(false);
                    let tensor_ty = ty
                        .tensor()
                        .cloned()
                        .unwrap_or_else(|| TensorTy::new(DType::F32, vec![]));
                    let d = self.alloc(ty.clone());
                    self.emit(
                        Inst::MakeTensor {
                            dst: d,
                            values,
                            requires_grad,
                            ty: tensor_ty,
                        },
                        span,
                    );
                    return d;
                }
                let d = self.alloc(ty.clone());
                self.emit(
                    Inst::CallBuiltin {
                        dst: d,
                        name: name.clone(),
                        args: pos,
                        kwargs,
                        ty: ty.clone(),
                    },
                    span,
                );
                d
            }
            NirExpr::FunctionRef { index, .. } => {
                let d = self.alloc(ty.clone());
                self.emit(
                    Inst::Call { dst: d, func: *index, args: pos, ty: ty.clone() },
                    span,
                );
                d
            }
            NirExpr::Attr { obj, name, .. } => {
                let recv = self.lower_expr(obj, span);
                if name == "backward" {
                    self.saw_backward = true;
                }
                let d = self.alloc(ty.clone());
                self.emit(
                    Inst::MethodCall {
                        dst: d,
                        recv,
                        method: name.clone(),
                        args: pos,
                        kwargs,
                        ty: ty.clone(),
                    },
                    span,
                );
                d
            }
            other => {
                let _ = self.lower_expr(other, span);
                self.diags.push(crate::diagnostics::Diagnostic::error(
                    "E0501",
                    "calling a function value is not supported; call a named function",
                    Some(crate::diagnostics::Span { start: span.0, end: span.1, line: 0, col: 0 }),
                ));
                let d = self.alloc(Ty::Any);
                self.emit(Inst::ConstUnit { dst: d }, span);
                d
            }
        }
    }

    fn lower_args(&mut self, args: &[CallArg], span: (usize, usize)) -> (Vec<Reg>, Vec<(String, Reg)>) {
        let mut pos = Vec::new();
        let mut kwargs = Vec::new();
        for a in args {
            let r = self.lower_expr(&a.expr, span);
            match &a.name {
                Some(n) => kwargs.push((n.clone(), r)),
                None => pos.push(r),
            }
        }
        (pos, kwargs)
    }
}

fn const_bool_at(code: &[Inst], reg: Reg) -> Option<bool> {
    code.iter().rev().find_map(|i| match i {
        Inst::ConstBool { dst, v } if *dst == reg => Some(*v),
        Inst::Copy { dst, src } if *dst == reg => {
            let _ = src;
            None
        }
        _ => None,
    })
}

fn last_use_of(code: &[Inst], reg: Reg) -> Option<usize> {
    code.iter()
        .enumerate()
        .filter(|(_, inst)| inst.reads().contains(&reg))
        .map(|(i, _)| i)
        .next_back()
}

pub fn hash_source(src: &str) -> u64 {
    fnv1a(src)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse;
    use crate::typecheck::check as typecheck;

    fn lower_src(src: &str) -> (EirProgram, Vec<crate::diagnostics::Diagnostic>) {
        let (p, d) = parse(src);
        assert!(d.is_empty(), "{:?}", d);
        let r = typecheck(&p);
        assert!(!r.diags.iter().any(|d| d.severity == crate::diagnostics::Severity::Error), "{:#?}", r.diags);
        let lowered = lower(&r.module, CompilePolicies::default(), hash_source(src));
        (lowered.program, lowered.diags)
    }

    #[test]
    fn lowers_arithmetic_to_register_code() {
        let (p, d) = lower_src("fn main():\n    x = 1 + 2 * 3\n    print(x)\n");
        assert!(d.is_empty(), "{:?}", d);
        assert_eq!(p.functions.len(), 1);
        assert!(p.functions[0].code.iter().any(|i| matches!(i, Inst::BinOp { .. })));
        assert!(p.functions[0].code.iter().any(|i| matches!(i, Inst::Print(_))));
        assert_eq!(p.main, 0);
    }

    #[test]
    fn lowers_control_flow_and_loops() {
        let (p, _) = lower_src(
            "fn main():\n    total = 0\n    for i in range(4):\n        if i > 1:\n            total = total + i\n    print(total)\n",
        );
        let code = &p.functions[0].code;
        assert!(code.iter().any(|i| matches!(i, Inst::IterInit { .. })));
        assert!(code.iter().any(|i| matches!(i, Inst::JumpIfFalse { .. })));
    }

    #[test]
    fn lowers_tensor_ops_with_metadata() {
        let (p, _) = lower_src(
            "fn main():\n    a = Tensor([[1.0, 2.0], [3.0, 4.0]])\n    b = a @ a\n    print(b.shape)\n",
        );
        let code = &p.functions[0].code;
        let t = code.iter().find_map(|i| match i {
            Inst::TensorOp { ty, .. } => Some(ty.clone()),
            _ => None,
        }).unwrap();
        assert_eq!(t.static_shape(), Some(vec![2, 2]));
    }

    #[test]
    fn computes_last_use_for_locals() {
        let (p, _) = lower_src("fn main():\n    x = Tensor([1.0])\n    y = x\n    print(y)\n");
        let f = &p.functions[0];
        assert!(f.locals.iter().all(|l| l.last_use.is_some() || l.is_param));
    }
}
