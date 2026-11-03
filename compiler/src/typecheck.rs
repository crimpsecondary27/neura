//! Semantic analysis and static type checking.
//!
//! Resolves names, infers local/function types, performs shape inference,
//! validates library calls, and produces a fully typed NIR module.

use crate::ast::*;
use crate::builtins::{call_builtin, is_known_module_path};
use crate::diagnostics::{Diagnostic, Span};
use neura_ir::nir::{
    CallArg, LocalId, NirBlock, NirExpr, NirFunction, NirLocal, NirModule, NirParam, NirStmt,
    NirStruct,
};
use neura_ir::ops::{infer_shape, BinOp, TensorOp};
use neura_ir::types::{Dim, DType, FnSig, TensorTy, Ty};
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct FunctionReport {
    pub name: String,
    pub params: Vec<(String, Ty)>,
    pub ret: Ty,
    pub locals: Vec<(String, Ty)>,
}

#[derive(Clone, Debug, Default)]
pub struct TypeReport {
    pub functions: Vec<FunctionReport>,
    pub imports: Vec<String>,
    pub structs: Vec<(String, Vec<(String, Ty)>)>,
}

pub struct CheckResult {
    pub module: NirModule,
    pub diags: Vec<Diagnostic>,
    pub report: TypeReport,
}

#[derive(Clone, Debug)]
pub(crate) struct Var {
    pub ty: Ty,
    pub mutable: bool,
    pub local: LocalId,
}

pub(crate) struct FnState {
    pub ret_ann: Option<Ty>,
    pub ret_span: Span,
    pub returns: Vec<Ty>,
    pub loop_depth: usize,
}

pub struct Checker {
    pub(crate) structs: HashMap<String, Vec<(String, Ty)>>,
    struct_order: Vec<String>,
    pub(crate) fns: HashMap<String, FnSig>,
    fn_idx: HashMap<String, usize>,
    fn_names: Vec<String>,
    pub(crate) imports: HashMap<String, String>,
    import_paths: Vec<String>,
    pub(crate) diags: Vec<Diagnostic>,
    scopes: Vec<HashMap<String, Var>>,
    locals: Vec<NirLocal>,
    pub(crate) fn_state: Option<FnState>,
    report: TypeReport,
    all_locals: Vec<(String, Ty)>,
    pending: Vec<NirFunction>,
}

pub fn check(program: &Program) -> CheckResult {
    let mut c = Checker {
        structs: HashMap::new(),
        struct_order: Vec::new(),
        fns: HashMap::new(),
        fn_idx: HashMap::new(),
        fn_names: Vec::new(),
        imports: HashMap::new(),
        import_paths: Vec::new(),
        diags: Vec::new(),
        scopes: Vec::new(),
        locals: Vec::new(),
        fn_state: None,
        report: TypeReport::default(),
        all_locals: Vec::new(),
        pending: Vec::new(),
    };
    c.run(program);
    let (module, diags, report) = c.into_parts(program);
    CheckResult { module, diags, report }
}

impl Checker {
    fn run(&mut self, program: &Program) {
        for item in &program.items {
            if let Item::Import(imp) = item {
                self.declare_import(imp);
            }
        }
        for item in &program.items {
            if let Item::Struct(s) = item {
                self.declare_struct(s);
            }
        }
        for item in &program.items {
            if let Item::Fn(f) = item {
                if self.fns.contains_key(&f.name) {
                    self.diags.push(Diagnostic::error(
                        "E0201",
                        format!("function `{}` is defined twice", f.name),
                        Some(f.span),
                    ));
                    continue;
                }
                let params: Vec<Ty> = f
                    .params
                    .iter()
                    .map(|p| p.ty.as_ref().map(|t| self.type_expr(t)).unwrap_or(Ty::Any))
                    .collect();
                let ret = f.ret.as_ref().map(|t| self.type_expr(t)).unwrap_or(Ty::Any);
                let idx = self.fn_names.len();
                self.fn_names.push(f.name.clone());
                self.fn_idx.insert(f.name.clone(), idx);
                self.fns.insert(f.name.clone(), FnSig { params, ret: Box::new(ret) });
            }
        }
        for item in &program.items {
            if let Item::Fn(f) = item {
                self.check_fn(f);
            }
        }
    }

    fn into_parts(self, program: &Program) -> (NirModule, Vec<Diagnostic>, TypeReport) {
        let mut structs = Vec::new();
        for name in &self.struct_order {
            if let Some(fields) = self.structs.get(name) {
                structs.push(NirStruct { name: name.clone(), fields: fields.clone() });
            }
        }
        let _ = program;
        let module = NirModule {
            name: "main".into(),
            functions: self.pending,
            structs,
            streaming: program.items.iter().any(|i| matches!(i, Item::AddLs)),
        };
        (module, self.diags, self.report)
    }

    fn declare_import(&mut self, imp: &ImportDecl) {
        let path = imp.path.join(".");
        if !is_known_module_path(&path) {
            self.diags.push(
                Diagnostic::error("E0250", format!("unknown module `{}`", path), Some(imp.span))
                    .help("available modules: neura.tensor, neura.nn, neura.optim, neura.math, neura.data, neura.autograd, neura.distributed, neura.inference, neura.serialization, neura.device"),
            );
            return;
        }
        let alias = imp
            .alias
            .clone()
            .unwrap_or_else(|| imp.path.last().cloned().unwrap_or_default());
        let module = imp.path.last().cloned().unwrap_or_default();
        if self.imports.contains_key(&alias) {
            self.diags.push(Diagnostic::error(
                "E0251",
                format!("module alias `{}` is already bound", alias),
                Some(imp.span),
            ));
            return;
        }
        self.imports.insert(alias, module);
        self.import_paths.push(path.clone());
        self.report.imports.push(path);
    }

    fn declare_struct(&mut self, s: &StructDecl) {
        let mut fields = Vec::new();
        for f in &s.fields {
            let ty = self.type_expr(&f.ty);
            if fields.iter().any(|(n, _): &(String, Ty)| *n == f.name) {
                self.diags.push(Diagnostic::error(
                    "E0202",
                    format!("duplicate field `{}` in struct `{}`", f.name, s.name),
                    Some(f.span),
                ));
            }
            fields.push((f.name.clone(), ty));
        }
        if self.structs.contains_key(&s.name) {
            self.diags.push(Diagnostic::error(
                "E0202",
                format!("struct `{}` is defined twice", s.name),
                Some(s.span),
            ));
            return;
        }
        self.structs.insert(s.name.clone(), fields.clone());
        self.struct_order.push(s.name.clone());
        self.report.structs.push((s.name.clone(), fields));
    }

    pub(crate) fn type_expr(&mut self, te: &TypeExpr) -> Ty {
        match te {
            TypeExpr::Named { name, span } => match name.as_str() {
                "Int" | "int" => Ty::Int,
                "Float" | "float" => Ty::Float,
                "Bool" | "bool" => Ty::Bool,
                "Str" | "str" | "String" => Ty::Str,
                "Unit" | "unit" => Ty::Unit,
                "Any" | "any" => Ty::Any,
                "Tensor" => Ty::Tensor(TensorTy::new(DType::F32, vec![Dim::Unknown])),
                "List" => Ty::List(Box::new(Ty::Any)),
                other => {
                    if self.structs.contains_key(other) {
                        Ty::Struct(other.to_string())
                    } else {
                        self.diags.push(
                            Diagnostic::error("E0321", format!("unknown type `{}`", other), Some(*span))
                                .help("use Int, Float, Bool, Str, Unit, Tensor[...] or a declared struct"),
                        );
                        Ty::Any
                    }
                }
            },
            TypeExpr::List { inner, .. } => Ty::List(Box::new(self.type_expr(inner))),
            TypeExpr::Tensor { dims, dtype, span } => {
                let dt = match dtype.as_deref() {
                    Some(d) => match DType::parse(d) {
                        Some(dt) => dt,
                        None => {
                            self.diags.push(Diagnostic::error(
                                "E0322",
                                format!("unknown element type `{}`", d),
                                Some(*span),
                            ));
                            DType::F32
                        }
                    },
                    None => DType::F32,
                };
                let shape = dims
                    .iter()
                    .map(|d| match d {
                        TypeDim::Int(n) => Dim::Static(*n),
                        TypeDim::Symbol(s) => Dim::Symbol(s.clone()),
                        TypeDim::Unknown => Dim::Unknown,
                    })
                    .collect();
                Ty::Tensor(TensorTy::new(dt, shape))
            }
        }
    }

    fn check_fn(&mut self, f: &FnDecl) {
        self.locals.clear();
        self.all_locals.clear();
        self.scopes.clear();
        self.scopes.push(HashMap::new());
        let declared = self.fns.get(&f.name).cloned().unwrap_or(FnSig {
            params: vec![],
            ret: Box::new(Ty::Any),
        });
        let ret_ann = if f.ret.is_some() { Some((*declared.ret).clone()) } else { None };
        self.fn_state = Some(FnState {
            ret_ann,
            ret_span: f.ret.as_ref().map(|t| t.span()).unwrap_or(f.span),
            returns: Vec::new(),
            loop_depth: 0,
        });
        let mut params = Vec::new();
        let mut param_report = Vec::new();
        for (i, p) in f.params.iter().enumerate() {
            let ty = if p.ty.is_some() {
                declared.params.get(i).cloned().unwrap_or(Ty::Any)
            } else {
                self.diags.push(
                    Diagnostic::warning(
                        "W0100",
                        format!(
                            "parameter `{}` has no type annotation and is treated as dynamically typed",
                            p.name
                        ),
                        Some(p.span),
                    )
                    .help(format!("annotate it, e.g. `{}: Float`", p.name)),
                );
                Ty::Any
            };
            let local = self.alloc_local(&p.name, ty.clone(), false, true);
            param_report.push((p.name.clone(), ty.clone()));
            params.push(NirParam {
                name: p.name.clone(),
                local,
                ty,
                is_borrow: p.borrow.is_some(),
            });
        }
        let body = self.check_block(&f.body);
        let state = self.fn_state.take().unwrap();
        let ret = match state.ret_ann.clone() {
            Some(r) => r,
            None => {
                if state.returns.is_empty() {
                    Ty::Unit
                } else {
                    let mut acc = state.returns[0].clone();
                    for t in &state.returns[1..] {
                        acc = merge_types(&acc, t);
                    }
                    acc
                }
            }
        };
        if let Some(last) = f.body.stmts.last() {
            if ret != Ty::Unit
                && !matches!(last.kind, StmtKind::Return(_))
                && !matches!(last.kind, StmtKind::If { .. } | StmtKind::While { .. } | StmtKind::For { .. })
            {
                self.diags.push(Diagnostic::warning(
                    "W0201",
                    format!("function `{}` may fall through without returning a value", f.name),
                    Some(last.span),
                ));
            }
        }
        if let Some(sig) = self.fns.get_mut(&f.name) {
            sig.ret = Box::new(ret.clone());
        }
        let _ = state.ret_span;
        self.report.functions.push(FunctionReport {
            name: f.name.clone(),
            params: param_report,
            ret: ret.clone(),
            locals: self.all_locals.clone(),
        });
        self.scopes.pop();
        self.pending.push(NirFunction {
            name: f.name.clone(),
            params,
            ret,
            locals: self.locals.clone(),
            body,
            span: (f.span.start, f.span.end),
        });
    }

    fn alloc_local(&mut self, name: &str, ty: Ty, mutable: bool, is_param: bool) -> LocalId {
        let id = self.locals.len() as LocalId;
        self.locals.push(NirLocal {
            name: name.to_string(),
            ty: ty.clone(),
            mutable,
            is_param,
        });
        self.all_locals.push((name.to_string(), ty.clone()));
        self.scopes
            .last_mut()
            .unwrap()
            .insert(name.to_string(), Var { ty, mutable, local: id });
        id
    }

    fn lookup(&self, name: &str) -> Option<Var> {
        for scope in self.scopes.iter().rev() {
            if let Some(v) = scope.get(name) {
                return Some(v.clone());
            }
        }
        None
    }

    fn lookup_in_current(&self, name: &str) -> Option<Var> {
        self.scopes.last().and_then(|s| s.get(name)).cloned()
    }

    fn check_block(&mut self, b: &Block) -> NirBlock {
        self.scopes.push(HashMap::new());
        let mut stmts = Vec::new();
        for stmt in &b.stmts {
            if let Some(s) = self.check_stmt(stmt) {
                stmts.push(s);
            }
        }
        self.scopes.pop();
        NirBlock { stmts }
    }

    fn check_stmt(&mut self, stmt: &Stmt) -> Option<NirStmt> {
        let span = (stmt.span.start, stmt.span.end);
        match &stmt.kind {
            StmtKind::Let { name, mutable, ty, value } => {
                let (expr, ety) = self.check_expr(value);
                let declared = ty.as_ref().map(|t| self.type_expr(t));
                let final_ty = match &declared {
                    Some(d) => {
                        if !compatible(d, &ety) {
                            self.type_mismatch(d, &ety, value.span, "initializer");
                        }
                        d.clone()
                    }
                    None => ety,
                };
                if self.lookup_in_current(name).is_some() {
                    self.diags.push(Diagnostic::error(
                        "E0201",
                        format!("`{}` is already declared in this scope", name),
                        Some(stmt.span),
                    ));
                    return None;
                }
                let local = self.alloc_local(name, final_ty, *mutable, false);
                Some(NirStmt::Let { local, expr, span })
            }
            StmtKind::Assign { target, value } => {
                let (vexpr, vty) = self.check_expr(value);
                self.check_assign(target, vexpr, vty, value.span, span)
            }
            StmtKind::OpAssign { target, op, value } => {
                let (vexpr, vty) = self.check_binary(*op, target, value, stmt.span);
                self.check_assign(target, vexpr, vty, stmt.span, span)
            }
            StmtKind::Expr(e) => {
                let (expr, _) = self.check_expr(e);
                Some(NirStmt::Expr(expr))
            }
            StmtKind::If { cond, then_block, else_block } => {
                let (cond_expr, cty) = self.check_expr(cond);
                self.expect_cond(&cty, cond.span);
                let then_block = self.check_block(then_block);
                let else_block = else_block.as_ref().map(|b| self.check_block(b));
                Some(NirStmt::If { cond: cond_expr, then_block, else_block, span })
            }
            StmtKind::While { cond, body } => {
                let (cond_expr, cty) = self.check_expr(cond);
                self.expect_cond(&cty, cond.span);
                if let Some(s) = &mut self.fn_state {
                    s.loop_depth += 1;
                }
                let body = self.check_block(body);
                if let Some(s) = &mut self.fn_state {
                    s.loop_depth -= 1;
                }
                Some(NirStmt::While { cond: cond_expr, body, span })
            }
            StmtKind::For { var, iter, body } => {
                let (iter_expr, ity) = self.check_expr(iter);
                let elem = match &ity {
                    Ty::List(t) => (**t).clone(),
                    Ty::Opaque(o) if o == "range" => Ty::Int,
                    Ty::Opaque(o) if o == "Dataset" => Ty::Any,
                    Ty::Any => Ty::Any,
                    other => {
                        self.diags.push(
                            Diagnostic::error(
                                "E0323",
                                format!("cannot iterate over `{}`", other.name()),
                                Some(iter.span),
                            )
                            .help("iterables are lists, range(...) and datasets"),
                        );
                        Ty::Any
                    }
                };
                self.scopes.push(HashMap::new());
                let local = self.alloc_local(var, elem, false, false);
                if let Some(s) = &mut self.fn_state {
                    s.loop_depth += 1;
                }
                let body = self.check_block(body);
                if let Some(s) = &mut self.fn_state {
                    s.loop_depth -= 1;
                }
                self.scopes.pop();
                Some(NirStmt::For { var: local, iter: iter_expr, body, span })
            }
            StmtKind::Return(value) => {
                if let Some(e) = value {
                    let (expr, ety) = self.check_expr(e);
                    let ann = self.fn_state.as_ref().and_then(|s| s.ret_ann.clone());
                    if let Some(ann) = &ann {
                        if !compatible(ann, &ety) {
                            self.type_mismatch(ann, &ety, e.span, "return value");
                        }
                    }
                    if let Some(s) = &mut self.fn_state {
                        s.returns.push(ety);
                    }
                    Some(NirStmt::Return(Some(expr), span))
                } else {
                    let ann = self.fn_state.as_ref().and_then(|s| s.ret_ann.clone());
                    if let Some(ann) = &ann {
                        if *ann != Ty::Unit && *ann != Ty::Any {
                            self.type_mismatch(ann, &Ty::Unit, stmt.span, "return value");
                        }
                    }
                    if let Some(s) = &mut self.fn_state {
                        s.returns.push(Ty::Unit);
                    }
                    Some(NirStmt::Return(None, span))
                }
            }
            StmtKind::Break => {
                if self.fn_state.as_ref().map(|s| s.loop_depth).unwrap_or(0) == 0 {
                    self.diags.push(Diagnostic::error(
                        "E0324",
                        "`break` outside of a loop",
                        Some(stmt.span),
                    ));
                }
                Some(NirStmt::Break(span))
            }
            StmtKind::Continue => {
                if self.fn_state.as_ref().map(|s| s.loop_depth).unwrap_or(0) == 0 {
                    self.diags.push(Diagnostic::error(
                        "E0324",
                        "`continue` outside of a loop",
                        Some(stmt.span),
                    ));
                }
                Some(NirStmt::Continue(span))
            }
            StmtKind::Pass => Some(NirStmt::Pass),
        }
    }

    fn check_assign(
        &mut self,
        target: &Expr,
        vexpr: NirExpr,
        vty: Ty,
        value_span: Span,
        span: (usize, usize),
    ) -> Option<NirStmt> {
        match &target.kind {
            ExprKind::Ident(name) => {
                if let Some(var) = self.lookup(name) {
                    if !var.mutable {
                        self.diags.push(
                            Diagnostic::error(
                                "E0203",
                                format!("cannot assign to immutable variable `{}`", name),
                                Some(target.span),
                            )
                            .help("declare it with `let mut` or use a bare `x = ...` declaration"),
                        );
                    }
                    if !compatible(&var.ty, &vty) {
                        self.type_mismatch(&var.ty, &vty, value_span, "assignment");
                    }
                    Some(NirStmt::Assign { local: var.local, expr: vexpr, span })
                } else {
                    let local = self.alloc_local(name, vty, true, false);
                    Some(NirStmt::Let { local, expr: vexpr, span })
                }
            }
            ExprKind::Attr { .. } | ExprKind::Index { .. } => {
                self.diags.push(
                    Diagnostic::error(
                        "E0311",
                        "assignment to a field or element is not supported",
                        Some(target.span),
                    )
                    .help("assign a new value to a variable instead"),
                );
                None
            }
            _ => {
                self.diags.push(Diagnostic::error(
                    "E0312",
                    "invalid assignment target",
                    Some(target.span),
                ));
                None
            }
        }
    }

    fn expect_cond(&mut self, ty: &Ty, span: Span) {
        if !matches!(ty, Ty::Bool | Ty::Any) {
            self.diags.push(
                Diagnostic::error(
                    "E0325",
                    format!("condition must have type Bool, found `{}`", ty.name()),
                    Some(span),
                )
                .help("use a comparison such as `x > 0`"),
            );
        }
    }

    fn type_mismatch(&mut self, expected: &Ty, found: &Ty, span: Span, ctx: &str) {
        self.diags.push(
            Diagnostic::error(
                "E0320",
                format!(
                    "type mismatch in {}: expected `{}`, found `{}`",
                    ctx,
                    expected.name(),
                    found.name()
                ),
                Some(span),
            )
            .help("add a type annotation or convert the value explicitly"),
        );
    }

    fn check_expr(&mut self, e: &Expr) -> (NirExpr, Ty) {
        match &e.kind {
            ExprKind::Int(v) => (NirExpr::Int(*v), Ty::Int),
            ExprKind::Float(v) => (NirExpr::Float(*v), Ty::Float),
            ExprKind::Bool(v) => (NirExpr::Bool(*v), Ty::Bool),
            ExprKind::Str(s) => (NirExpr::Str(s.clone()), Ty::Str),
            ExprKind::Ident(name) => self.check_ident(name, e.span),
            ExprKind::List(items) => self.check_list(items, e.span),
            ExprKind::Unary { op, expr } => {
                let (inner, ty) = self.check_expr(expr);
                match op {
                    UnaryOp::Neg => match &ty {
                        Ty::Int => (
                            NirExpr::Unary { op: neura_ir::UnOp::Neg, expr: Box::new(inner), ty: Ty::Int },
                            Ty::Int,
                        ),
                        Ty::Float => (
                            NirExpr::Unary { op: neura_ir::UnOp::Neg, expr: Box::new(inner), ty: Ty::Float },
                            Ty::Float,
                        ),
                        Ty::Tensor(t) => {
                            let tensor_ty = t.clone();
                            (
                                NirExpr::TensorOp {
                                    op: TensorOp::Neg,
                                    args: vec![inner],
                                    dyn_dims: vec![],
                                    ty: tensor_ty.clone(),
                                },
                                Ty::Tensor(tensor_ty),
                            )
                        }
                        Ty::Any => (
                            NirExpr::Unary { op: neura_ir::UnOp::Neg, expr: Box::new(inner), ty: Ty::Any },
                            Ty::Any,
                        ),
                        other => {
                            self.diags.push(Diagnostic::error(
                                "E0330",
                                format!("cannot negate `{}`", other.name()),
                                Some(expr.span),
                            ));
                            (inner, Ty::Any)
                        }
                    },
                    UnaryOp::Not => {
                        if !matches!(ty, Ty::Bool | Ty::Any) {
                            self.diags.push(Diagnostic::error(
                                "E0330",
                                format!("`!` expects Bool, found `{}`", ty.name()),
                                Some(expr.span),
                            ));
                        }
                        (
                            NirExpr::Unary { op: neura_ir::UnOp::Not, expr: Box::new(inner), ty: Ty::Bool },
                            Ty::Bool,
                        )
                    }
                }
            }
            ExprKind::Binary { op, lhs, rhs } => self.check_binary(*op, lhs, rhs, e.span),
            ExprKind::Call { callee, args } => self.check_call(callee, args, e.span),
            ExprKind::Attr { obj, name } => self.check_attr(obj, name, e.span),
            ExprKind::Index { obj, index } => {
                let (o, oty) = self.check_expr(obj);
                let (i, ity) = self.check_expr(index);
                if !matches!(ity, Ty::Int | Ty::Any) {
                    self.diags.push(Diagnostic::error(
                        "E0331",
                        "index must be an integer",
                        Some(index.span),
                    ));
                }
                match &oty {
                    Ty::List(t) => (
                        NirExpr::Index { obj: Box::new(o), index: Box::new(i), ty: (**t).clone() },
                        (**t).clone(),
                    ),
                    Ty::Str => (
                        NirExpr::Index { obj: Box::new(o), index: Box::new(i), ty: Ty::Str },
                        Ty::Str,
                    ),
                    Ty::Tensor(t) => {
                        let mut shape = t.shape.clone();
                        if !shape.is_empty() {
                            shape.remove(0);
                        }
                        let rty = Ty::Tensor(TensorTy::new(t.dtype, shape));
                        (
                            NirExpr::Index { obj: Box::new(o), index: Box::new(i), ty: rty.clone() },
                            rty,
                        )
                    }
                    Ty::Any => (
                        NirExpr::Index { obj: Box::new(o), index: Box::new(i), ty: Ty::Any },
                        Ty::Any,
                    ),
                    other => {
                        self.diags.push(Diagnostic::error(
                            "E0331",
                            format!("cannot index `{}`", other.name()),
                            Some(e.span),
                        ));
                        (NirExpr::Unit, Ty::Any)
                    }
                }
            }
            ExprKind::Slice { obj, start, end } => {
                self.check_slice(obj, start.as_deref(), end.as_deref(), e.span)
            }
            ExprKind::Borrow { expr, mutable } => {
                if let ExprKind::Ident(name) = &expr.kind {
                    if let Some(var) = self.lookup(name) {
                        (NirExpr::Borrow { local: var.local, mutable: *mutable }, var.ty.clone())
                    } else {
                        self.diags.push(Diagnostic::error(
                            "E0300",
                            format!("unknown variable `{}`", name),
                            Some(expr.span),
                        ));
                        (NirExpr::Unit, Ty::Any)
                    }
                } else {
                    self.diags.push(Diagnostic::error(
                        "E0335",
                        "can only borrow a named variable",
                        Some(expr.span),
                    ));
                    self.check_expr(expr)
                }
            }
        }
    }

    fn check_ident(&mut self, name: &str, span: Span) -> (NirExpr, Ty) {
        if let Some(v) = self.lookup(name) {
            return (NirExpr::Local(v.local), v.ty);
        }
        if let Some(module) = self.imports.get(name) {
            return (NirExpr::Module(module.clone()), Ty::Module(module.clone()));
        }
        if let Some(idx) = self.fn_idx.get(name) {
            let sig = self.fns.get(name).cloned().unwrap_or(FnSig {
                params: vec![],
                ret: Box::new(Ty::Any),
            });
            return (
                NirExpr::FunctionRef { index: *idx, name: name.to_string(), ty: Ty::Func(sig.clone()) },
                Ty::Func(sig),
            );
        }
        if self.structs.contains_key(name) {
            return (NirExpr::Builtin { name: format!("struct.{}", name), ty: Ty::Any }, Ty::Any);
        }
        if is_builtin_global(name) {
            return (NirExpr::Builtin { name: name.to_string(), ty: Ty::Any }, Ty::Any);
        }
        self.diags.push(
            Diagnostic::error("E0300", format!("unknown identifier `{}`", name), Some(span))
                .help("declare it with `let`, import a module, or define a function"),
        );
        (NirExpr::Unit, Ty::Any)
    }

    fn check_list(&mut self, items: &[Expr], _span: Span) -> (NirExpr, Ty) {
        let mut elems = Vec::new();
        let mut elem_ty = Ty::Any;
        let mut first = true;
        for item in items {
            let (e, t) = self.check_expr(item);
            if first {
                elem_ty = t.clone();
                first = false;
            } else {
                elem_ty = merge_types(&elem_ty, &t);
            }
            elems.push(e);
        }
        let ty = Ty::List(Box::new(elem_ty));
        (NirExpr::List { elems, ty: ty.clone() }, ty)
    }

    fn check_binary(&mut self, op: BinaryOp, lhs: &Expr, rhs: &Expr, span: Span) -> (NirExpr, Ty) {
        let (l, lt) = self.check_expr(lhs);
        let (r, rt) = self.check_expr(rhs);
        let ir_op = match op {
            BinaryOp::Add => BinOp::Add,
            BinaryOp::Sub => BinOp::Sub,
            BinaryOp::Mul => BinOp::Mul,
            BinaryOp::Div => BinOp::Div,
            BinaryOp::FloorDiv => BinOp::FloorDiv,
            BinaryOp::Mod => BinOp::Mod,
            BinaryOp::Pow => BinOp::Pow,
            BinaryOp::Eq => BinOp::Eq,
            BinaryOp::Ne => BinOp::Ne,
            BinaryOp::Lt => BinOp::Lt,
            BinaryOp::Le => BinOp::Le,
            BinaryOp::Gt => BinOp::Gt,
            BinaryOp::Ge => BinOp::Ge,
            BinaryOp::And => BinOp::And,
            BinaryOp::Or => BinOp::Or,
            BinaryOp::MatMul => BinOp::Mul,
        };
        if matches!(op, BinaryOp::MatMul) {
            return self.tensor_binary_expr(TensorOp::MatMul, l, lt, r, rt, span);
        }
        if matches!(
            op,
            BinaryOp::Eq | BinaryOp::Ne | BinaryOp::Lt | BinaryOp::Le | BinaryOp::Gt | BinaryOp::Ge
        ) {
            if matches!((&lt, &rt), (Ty::Tensor(_), _) | (_, Ty::Tensor(_))) {
                self.diags.push(Diagnostic::error(
                    "E0333",
                    "tensor comparisons are not supported; reduce to a scalar first",
                    Some(span),
                ));
            }
            return (
                NirExpr::Binary { op: ir_op, lhs: Box::new(l), rhs: Box::new(r), ty: Ty::Bool },
                Ty::Bool,
            );
        }
        if matches!(op, BinaryOp::And | BinaryOp::Or) {
            if !matches!((&lt, &rt), (Ty::Bool | Ty::Any, Ty::Bool | Ty::Any)) {
                self.diags.push(Diagnostic::error(
                    "E0334",
                    "logical operators expect Bool operands",
                    Some(span),
                ));
            }
            return (
                NirExpr::Binary { op: ir_op, lhs: Box::new(l), rhs: Box::new(r), ty: Ty::Bool },
                Ty::Bool,
            );
        }
        if matches!(op, BinaryOp::Add) && matches!((&lt, &rt), (Ty::Str, Ty::Str)) {
            return (
                NirExpr::Binary { op: ir_op, lhs: Box::new(l), rhs: Box::new(r), ty: Ty::Str },
                Ty::Str,
            );
        }
        if matches!(lt, Ty::Tensor(_)) || matches!(rt, Ty::Tensor(_)) {
            let t_op = match op {
                BinaryOp::Add => TensorOp::Add,
                BinaryOp::Sub => TensorOp::Sub,
                BinaryOp::Mul => TensorOp::Mul,
                BinaryOp::Div => TensorOp::Div,
                _ => {
                    self.diags.push(Diagnostic::error(
                        "E0332",
                        format!("operator `{}` is not supported on tensors", ir_op.symbol()),
                        Some(span),
                    ));
                    return (NirExpr::Unit, Ty::Any);
                }
            };
            return self.tensor_binary_expr(t_op, l, lt, r, rt, span);
        }
        let ty = match (&lt, &rt) {
            (Ty::Int, Ty::Int) => Ty::Int,
            (Ty::Int | Ty::Float | Ty::Any, Ty::Float) | (Ty::Float, Ty::Int | Ty::Any) => Ty::Float,
            (Ty::Any, _) | (_, Ty::Any) => Ty::Any,
            (a, b) if a == b => a.clone(),
            _ => {
                self.diags.push(Diagnostic::error(
                    "E0332",
                    format!(
                        "operator `{}` cannot be applied to `{}` and `{}`",
                        ir_op.symbol(),
                        lt.name(),
                        rt.name()
                    ),
                    Some(span),
                ));
                Ty::Any
            }
        };
        (NirExpr::Binary { op: ir_op, lhs: Box::new(l), rhs: Box::new(r), ty: ty.clone() }, ty)
    }

    fn tensor_binary_expr(
        &mut self,
        op: TensorOp,
        l: NirExpr,
        lt: Ty,
        r: NirExpr,
        rt: Ty,
        span: Span,
    ) -> (NirExpr, Ty) {
        match crate::builtins::binary_tensor_ty(op.clone(), &lt, &rt) {
            Ok(ty) => {
                let args = vec![self.as_tensor_expr(l, &lt), self.as_tensor_expr(r, &rt)];
                let args: Vec<NirExpr> = args.into_iter().flatten().collect();
                let tensor_ty = match &ty {
                    Ty::Tensor(t) => t.clone(),
                    _ => TensorTy::new(DType::F32, vec![Dim::Unknown]),
                };
                (
                    NirExpr::TensorOp { op, args, dyn_dims: vec![], ty: tensor_ty },
                    ty,
                )
            }
            Err(msg) => {
                self.diags.push(Diagnostic::error("E0332", msg, Some(span)));
                (NirExpr::Unit, Ty::Any)
            }
        }
    }

    fn as_tensor_expr(&mut self, expr: NirExpr, ty: &Ty) -> Option<NirExpr> {
        match ty {
            Ty::Tensor(_) => Some(expr),
            Ty::Int | Ty::Float | Ty::Bool => {
                let dtype = match ty {
                    Ty::Int => DType::I64,
                    Ty::Float => DType::F64,
                    _ => DType::Bool,
                };
                Some(NirExpr::Call {
                    callee: Box::new(NirExpr::Builtin { name: "tensor.scalar".into(), ty: Ty::Any }),
                    args: vec![CallArg { name: None, expr }],
                    ty: Ty::Tensor(TensorTy::new(dtype, vec![])),
                    span: (0, 0),
                })
            }
            Ty::Any => Some(expr),
            _ => None,
        }
    }

    fn check_slice(
        &mut self,
        obj: &Expr,
        start: Option<&Expr>,
        end: Option<&Expr>,
        span: Span,
    ) -> (NirExpr, Ty) {
        let (o, oty) = self.check_expr(obj);
        let _ = (start, end);
        match &oty {
            Ty::List(t) => (
                NirExpr::Index { obj: Box::new(o), index: Box::new(NirExpr::Unit), ty: (**t).clone() },
                (**t).clone(),
            ),
            Ty::Tensor(t) => {
                let start_i = start.and_then(const_int).unwrap_or(0).max(0) as usize;
                let end_i = end
                    .and_then(const_int)
                    .map(|v| v.max(0) as usize)
                    .unwrap_or_else(|| t.shape.first().and_then(|d| d.concrete()).unwrap_or(usize::MAX));
                let mut sh = t.shape.clone();
                if sh.is_empty() {
                    self.diags.push(Diagnostic::error(
                        "E0336",
                        "cannot slice a scalar tensor",
                        Some(span),
                    ));
                    return (o, oty);
                }
                sh[0] = if end_i == usize::MAX {
                    Dim::Unknown
                } else {
                    Dim::Static(end_i.saturating_sub(start_i))
                };
                let ty = Ty::Tensor(TensorTy::new(t.dtype, sh.clone()));
                (
                    NirExpr::TensorOp {
                        op: TensorOp::Slice { axis: 0, start: start_i, end: end_i },
                        args: vec![o],
                        dyn_dims: vec![],
                        ty: TensorTy::new(t.dtype, sh),
                    },
                    ty,
                )
            }
            Ty::Str => (o, Ty::Str),
            Ty::Any => (o, Ty::Any),
            other => {
                self.diags.push(Diagnostic::error(
                    "E0336",
                    format!("cannot slice `{}`", other.name()),
                    Some(span),
                ));
                (o, Ty::Any)
            }
        }
    }

    fn check_attr(&mut self, obj: &Expr, name: &str, span: Span) -> (NirExpr, Ty) {
        let (o, oty) = self.check_expr(obj);
        match &oty {
            Ty::Module(m) => (
                NirExpr::Builtin { name: format!("{}.{}", m, name), ty: Ty::Any },
                Ty::Any,
            ),
            Ty::Tensor(t) => match name {
                "T" => {
                    if t.rank() < 2 {
                        self.diags.push(Diagnostic::error(
                            "E0337",
                            "`.T` requires a tensor of rank >= 2",
                            Some(span),
                        ));
                        return (o, oty);
                    }
                    let mut sh = t.shape.clone();
                    let n = sh.len();
                    sh.swap(n - 1, n - 2);
                    let ty = Ty::Tensor(TensorTy::new(t.dtype, sh.clone()));
                    (
                        NirExpr::TensorOp {
                            op: TensorOp::Transpose { dim0: n - 2, dim1: n - 1 },
                            args: vec![o],
                            dyn_dims: vec![],
                            ty: TensorTy::new(t.dtype, sh),
                        },
                        ty,
                    )
                }
                "shape" => (
                    NirExpr::Attr { obj: Box::new(o), name: name.into(), ty: Ty::List(Box::new(Ty::Int)) },
                    Ty::List(Box::new(Ty::Int)),
                ),
                "grad" => (
                    NirExpr::Attr { obj: Box::new(o), name: name.into(), ty: oty.clone() },
                    oty.clone(),
                ),
                "requires_grad" => (
                    NirExpr::Attr { obj: Box::new(o), name: name.into(), ty: Ty::Bool },
                    Ty::Bool,
                ),
                "dtype" => (
                    NirExpr::Attr { obj: Box::new(o), name: name.into(), ty: Ty::Str },
                    Ty::Str,
                ),
                "device" => (
                    NirExpr::Attr { obj: Box::new(o), name: name.into(), ty: Ty::Str },
                    Ty::Str,
                ),
                other => {
                    self.diags.push(
                        Diagnostic::error("E0337", format!("unknown tensor property `{}`", other), Some(span))
                            .help("known properties: T, shape, grad, requires_grad, dtype, device"),
                    );
                    (o, Ty::Any)
                }
            },
            Ty::Struct(s) => {
                if let Some(fields) = self.structs.get(s) {
                    if let Some((_, ft)) = fields.iter().find(|(n, _)| n == name) {
                        return (
                            NirExpr::Attr { obj: Box::new(o), name: name.into(), ty: ft.clone() },
                            ft.clone(),
                        );
                    }
                }
                self.diags.push(Diagnostic::error(
                    "E0337",
                    format!("struct `{}` has no field `{}`", s, name),
                    Some(span),
                ));
                (o, Ty::Any)
            }
            Ty::Any => (
                NirExpr::Attr { obj: Box::new(o), name: name.into(), ty: Ty::Any },
                Ty::Any,
            ),
            other => {
                self.diags.push(Diagnostic::error(
                    "E0337",
                    format!("type `{}` has no attribute `{}`", other.name(), name),
                    Some(span),
                ));
                (o, Ty::Any)
            }
        }
    }

    fn check_call(&mut self, callee: &Expr, args: &[Arg], span: Span) -> (NirExpr, Ty) {
        if let ExprKind::Attr { obj, name } = &callee.kind {
            let (nobj, oty) = self.check_expr(obj);
            if let Ty::Module(module) = &oty {
                let full = format!("{}.{}", module, name);
                let checked = self.check_args(args);
                match call_builtin(&full, &checked.1, &checked.2) {
                    Ok(mut ret) => {
                        ret = self.refine_constructor(&full, args, ret);
                        return (
                            NirExpr::Call {
                                callee: Box::new(NirExpr::Builtin { name: full, ty: Ty::Any }),
                                args: checked.0,
                                ty: ret.clone(),
                                span: (span.start, span.end),
                            },
                            ret,
                        );
                    }
                    Err(msg) => {
                        self.diags.push(Diagnostic::error("E0301", msg, Some(span)));
                        return (NirExpr::Unit, Ty::Any);
                    }
                }
            }
            return self.check_method(nobj, oty, name, args, span);
        }
        if let ExprKind::Ident(name) = &callee.kind {
            let is_local = self.lookup(name).is_some();
            if !is_local && self.imports.get(name).is_none() {
                if let Some(idx) = self.fn_idx.get(name).cloned() {
                    return self.check_user_call(idx, name, args, span);
                }
                if self.structs.contains_key(name) {
                    return self.check_struct_ctor(name, args, span);
                }
                if is_builtin_global(name) {
                    if name == "input" {
                        return self.check_input_call(args, span);
                    }
                    let checked = self.check_args(args);
                    let builtin_name = if name == "Tensor" {
                        "tensor.Tensor".to_string()
                    } else {
                        name.to_string()
                    };
                    match call_builtin(&builtin_name, &checked.1, &checked.2) {
                        Ok(mut ret) => {
                            if name == "Tensor" {
                                ret = self.refine_tensor_ctor(args, ret);
                            }
                            return (
                                NirExpr::Call {
                                    callee: Box::new(NirExpr::Builtin { name: builtin_name, ty: Ty::Any }),
                                    args: checked.0,
                                    ty: ret.clone(),
                                    span: (span.start, span.end),
                                },
                                ret,
                            );
                        }
                        Err(msg) => {
                            self.diags.push(Diagnostic::error("E0301", msg, Some(span)));
                            return (NirExpr::Unit, Ty::Any);
                        }
                    }
                }
            }
        }
        let (ncallee, cty) = self.check_expr(callee);
        let checked = self.check_args(args);
        match cty {
            Ty::Func(sig) => {
                if checked.1.len() != sig.params.len() {
                    self.diags.push(Diagnostic::error(
                        "E0302",
                        format!(
                            "function expects {} arguments, found {}",
                            sig.params.len(),
                            checked.1.len()
                        ),
                        Some(span),
                    ));
                }
                let ret = (*sig.ret).clone();
                (
                    NirExpr::Call {
                        callee: Box::new(ncallee),
                        args: checked.0,
                        ty: ret.clone(),
                        span: (span.start, span.end),
                    },
                    ret,
                )
            }
            Ty::Any => (
                NirExpr::Call {
                    callee: Box::new(ncallee),
                    args: checked.0,
                    ty: Ty::Any,
                    span: (span.start, span.end),
                },
                Ty::Any,
            ),
            other => {
                self.diags.push(Diagnostic::error(
                    "E0303",
                    format!("`{}` is not callable", other.name()),
                    Some(span),
                ));
                (NirExpr::Unit, Ty::Any)
            }
        }
    }

    /// `input()` reads a line and returns `Str`; `input(TYPE)` and
    /// `input(prompt, TYPE)` parse it as `Int`, `Float`, `Bool` or `Str`.
    fn check_input_call(&mut self, args: &[Arg], span: Span) -> (NirExpr, Ty) {
        let mut target = Ty::Str;
        let mut target_set = false;
        let mut prompt_seen = false;
        let mut checked = Vec::new();
        for a in args {
            if a.name.is_none() {
                if let ExprKind::Ident(t) = &a.value.kind {
                    if self.lookup(t).is_none() && matches!(t.as_str(), "Int" | "Float" | "Bool" | "Str") {
                        if target_set {
                            self.diags.push(Diagnostic::error(
                                "E0301",
                                "`input` expects exactly one type argument".to_string(),
                                Some(span),
                            ));
                        }
                        target_set = true;
                        target = match t.as_str() {
                            "Int" => Ty::Int,
                            "Float" => Ty::Float,
                            "Bool" => Ty::Bool,
                            _ => Ty::Str,
                        };
                        continue;
                    }
                    if self.lookup(t).is_none()
                        && matches!(t.as_str(), "Unit" | "Tensor" | "List" | "Any")
                    {
                        self.diags.push(
                            Diagnostic::error(
                                "E0301",
                                format!("`input` does not support type `{}`", t),
                                Some(span),
                            )
                            .help("input supports Int, Float, Bool and Str"),
                        );
                        continue;
                    }
                }
            }
            if prompt_seen {
                self.diags.push(Diagnostic::error(
                    "E0301",
                    "`input` expects at most one prompt".to_string(),
                    Some(span),
                ));
            }
            prompt_seen = true;
            let (expr, ty) = self.check_expr(&a.value);
            if !matches!(ty, Ty::Str | Ty::Any) {
                self.diags.push(Diagnostic::error(
                    "E0301",
                    format!("`input` expects a string prompt, found {}", ty.name()),
                    Some(a.span),
                ));
            }
            checked.push(CallArg { name: a.name.clone(), expr });
        }
        (
            NirExpr::Call {
                callee: Box::new(NirExpr::Builtin { name: "input".into(), ty: Ty::Any }),
                args: checked,
                ty: target.clone(),
                span: (span.start, span.end),
            },
            target,
        )
    }

    fn check_args(&mut self, args: &[Arg]) -> (Vec<CallArg>, Vec<Ty>, Vec<(String, Ty)>) {
        let mut nargs = Vec::new();
        let mut tys = Vec::new();
        let mut kwargs = Vec::new();
        for a in args {
            let (e, t) = self.check_expr(&a.value);
            match &a.name {
                Some(name) => {
                    if kwargs.iter().any(|(n, _): &(String, Ty)| n == name) {
                        self.diags.push(Diagnostic::error(
                            "E0304",
                            format!("duplicate keyword argument `{}`", name),
                            Some(a.span),
                        ));
                    }
                    kwargs.push((name.clone(), t.clone()));
                    nargs.push(CallArg { name: Some(name.clone()), expr: e });
                }
                None => {
                    tys.push(t.clone());
                    nargs.push(CallArg { name: None, expr: e });
                }
            }
        }
        (nargs, tys, kwargs)
    }

    fn check_user_call(&mut self, idx: usize, name: &str, args: &[Arg], span: Span) -> (NirExpr, Ty) {
        let sig = match self.fns.get(name) {
            Some(s) => s.clone(),
            None => return (NirExpr::Unit, Ty::Any),
        };
        let (call_args, tys, _kwargs) = self.check_args(args);
        if tys.len() != sig.params.len() {
            self.diags.push(Diagnostic::error(
                "E0302",
                format!(
                    "function `{}` expects {} arguments, found {}",
                    name,
                    sig.params.len(),
                    tys.len()
                ),
                Some(span),
            ));
        }
        for (i, (a, p)) in tys.iter().zip(sig.params.iter()).enumerate() {
            if !compatible(p, a) {
                self.diags.push(Diagnostic::error(
                    "E0305",
                    format!(
                        "argument {} of `{}`: expected `{}`, found `{}`",
                        i + 1,
                        name,
                        p.name(),
                        a.name()
                    ),
                    Some(args.get(i).map(|x| x.span).unwrap_or(span)),
                ));
            }
        }
        let ret = (*sig.ret).clone();
        (
            NirExpr::Call {
                callee: Box::new(NirExpr::FunctionRef {
                    index: idx,
                    name: name.to_string(),
                    ty: Ty::Func(sig),
                }),
                args: call_args,
                ty: ret.clone(),
                span: (span.start, span.end),
            },
            ret,
        )
    }

    fn check_struct_ctor(&mut self, name: &str, args: &[Arg], span: Span) -> (NirExpr, Ty) {
        let fields = self.structs.get(name).cloned().unwrap_or_default();
        let (call_args, _tys, _) = self.check_args(args);
        let ty = Ty::Struct(name.to_string());
        if args.len() != fields.len() {
            self.diags.push(Diagnostic::error(
                "E0306",
                format!("struct `{}` expects {} fields, found {}", name, fields.len(), args.len()),
                Some(span),
            ));
        }
        for (a, (fname, fty)) in args.iter().zip(fields.iter()) {
            let at = self.check_expr(&a.value).1;
            if !compatible(fty, &at) {
                self.diags.push(Diagnostic::error(
                    "E0306",
                    format!(
                        "field `{}` of `{}`: expected `{}`, found `{}`",
                        fname,
                        name,
                        fty.name(),
                        at.name()
                    ),
                    Some(a.span),
                ));
            }
        }
        (
            NirExpr::Call {
                callee: Box::new(NirExpr::Builtin { name: format!("struct.{}", name), ty: Ty::Any }),
                args: call_args,
                ty: ty.clone(),
                span: (span.start, span.end),
            },
            ty,
        )
    }

    // ---- constructor refinement -----------------------------------------

    /// Refine the coarse type returned by `call_builtin` using literal
    /// constructor arguments, e.g. `nn.Linear(784, 512)`.
    fn refine_constructor(&mut self, full: &str, args: &[Arg], ret: Ty) -> Ty {
        match full {
            "nn.Linear" => {
                let inp = positional_const_int(args, 0);
                let out = positional_const_int(args, 1);
                match (inp, out) {
                    (Some(i), Some(o)) if i >= 0 && o >= 0 => {
                        Ty::Opaque(format!("Linear<{},{}>", i, o))
                    }
                    (None, Some(o)) if o >= 0 => Ty::Opaque(format!("Linear<?,{}>", o)),
                    _ => ret,
                }
            }
            "nn.Sequential" => {
                let out = args
                    .first()
                    .and_then(|a| layer_output_dim(&a.value, &self.imports));
                match out {
                    Some(n) => Ty::Opaque(format!("Sequential<{}>", n)),
                    None => ret,
                }
            }
            _ => ret,
        }
    }

    /// Refine `Tensor(...)` using the literal element structure.
    fn refine_tensor_ctor(&mut self, args: &[Arg], ret: Ty) -> Ty {
        let dtype = args
            .iter()
            .find(|a| a.name.as_deref() == Some("dtype"))
            .and_then(|a| const_str(&a.value))
            .and_then(|s| DType::parse(&s))
            .unwrap_or(DType::F32);
        let shape = match positional_arg(args, 0).and_then(|a| list_shape(&a.value)) {
            Some(s) => s,
            None => return ret,
        };
        if shape.is_empty() {
            return ret;
        }
        Ty::Tensor(TensorTy::new(dtype, shape))
    }

    // ---- method calls ----------------------------------------------------

    fn check_method(
        &mut self,
        obj: NirExpr,
        oty: Ty,
        name: &str,
        args: &[Arg],
        span: Span,
    ) -> (NirExpr, Ty) {
        match &oty {
            Ty::Tensor(t) => self.check_tensor_method(obj, t.clone(), name, args, span),
            Ty::Opaque(o) => self.check_opaque_method(obj, o.clone(), name, args, span),
            Ty::Any => self.method_call(obj, name, args, Ty::Any, span),
            other => {
                self.diags.push(
                    Diagnostic::error(
                        "E0340",
                        format!("type `{}` has no method `{}`", other.name(), name),
                        Some(span),
                    ),
                    );
                // Still type-check the arguments so nested errors surface.
                let _ = self.check_args(args);
                (NirExpr::Unit, Ty::Any)
            }
        }
    }

    fn method_call(
        &mut self,
        obj: NirExpr,
        name: &str,
        args: &[Arg],
        ret: Ty,
        span: Span,
    ) -> (NirExpr, Ty) {
        let (call_args, _, _) = self.check_args(args);
        (
            NirExpr::Call {
                callee: Box::new(NirExpr::Attr {
                    obj: Box::new(obj),
                    name: name.to_string(),
                    ty: Ty::Any,
                }),
                args: call_args,
                ty: ret.clone(),
                span: (span.start, span.end),
            },
            ret,
        )
    }

    fn unary_tensor_method(
        &mut self,
        obj: NirExpr,
        t: TensorTy,
        op: TensorOp,
        span: Span,
    ) -> (NirExpr, Ty) {
        match infer_shape(&op, &[t]) {
            Ok(out) => {
                let ty = Ty::Tensor(out.clone());
                (
                    NirExpr::TensorOp { op, args: vec![obj], dyn_dims: vec![], ty: out },
                    ty,
                )
            }
            Err(msg) => {
                self.diags.push(Diagnostic::error("E0341", msg, Some(span)));
                (NirExpr::Unit, Ty::Any)
            }
        }
    }

    fn check_tensor_method(
        &mut self,
        obj: NirExpr,
        t: TensorTy,
        name: &str,
        args: &[Arg],
        span: Span,
    ) -> (NirExpr, Ty) {
        // Methods that stay as explicit calls in EIR.
        match name {
            "backward" => return self.method_call(obj, name, args, Ty::Unit, span),
            "item" => return self.method_call(obj, name, args, Ty::Float, span),
            "detach" => return self.method_call(obj, name, args, Ty::Tensor(t), span),
            _ => {}
        }
        let _ = self.check_args(args);
        match name {
            "matmul" => {
                let other = match positional_arg(args, 0) {
                    Some(a) => self.check_expr(&a.value),
                    None => {
                        self.diags.push(Diagnostic::error(
                            "E0341",
                            "`matmul` expects one argument",
                            Some(span),
                        ));
                        return (NirExpr::Unit, Ty::Any);
                    }
                };
                self.tensor_binary_expr(
                    TensorOp::MatMul,
                    obj,
                    Ty::Tensor(t),
                    other.0,
                    other.1,
                    span,
                )
            }
            "transpose" => {
                if t.rank() < 2 {
                    self.diags.push(Diagnostic::error(
                        "E0341",
                        "`transpose` requires a tensor of rank >= 2",
                        Some(span),
                    ));
                    return (NirExpr::Unit, Ty::Any);
                }
                let n = t.rank();
                let d0 = positional_const_int(args, 0).unwrap_or((n - 2) as i64);
                let d1 = positional_const_int(args, 1).unwrap_or((n - 1) as i64);
                if d0 < 0 || d1 < 0 || d0 as usize >= n || d1 as usize >= n {
                    self.diags.push(Diagnostic::error(
                        "E0341",
                        "transpose dimensions out of range",
                        Some(span),
                    ));
                    return (NirExpr::Unit, Ty::Any);
                }
                self.unary_tensor_method(
                    obj,
                    t,
                    TensorOp::Transpose { dim0: d0 as usize, dim1: d1 as usize },
                    span,
                )
            }
            "reshape" => {
                let dims = reshape_dims(args);
                if dims.is_empty() {
                    self.diags.push(
                        Diagnostic::error("E0341", "`reshape` expects target dimensions", Some(span))
                            .help("for example `x.reshape(2, -1)` or `x.reshape(shape=[2, -1])`"),
                    );
                    return (NirExpr::Unit, Ty::Any);
                }
                self.unary_tensor_method(obj, t, TensorOp::Reshape { dims }, span)
            }
            "flatten" => self.unary_tensor_method(obj, t, TensorOp::Flatten, span),
            "sum" | "mean" | "max" | "min" => {
                let axis = method_axis(args);
                let keepdim = kw_arg(args, "keepdim")
                    .and_then(|a| const_bool(&a.value))
                    .unwrap_or(false);
                match (name, axis) {
                    ("sum", _) => {
                        let axes = axis.map(|a| vec![a]).unwrap_or_else(|| (0..t.rank()).collect());
                        self.unary_tensor_method(obj, t, TensorOp::Sum { axes, keepdim }, span)
                    }
                    ("mean", _) => {
                        let axes = axis.map(|a| vec![a]).unwrap_or_else(|| (0..t.rank()).collect());
                        self.unary_tensor_method(obj, t, TensorOp::Mean { axes, keepdim }, span)
                    }
                    ("max", _) => {
                        self.unary_tensor_method(obj, t, TensorOp::ReduceMax { axis, keepdim }, span)
                    }
                    ("min", _) => {
                        self.unary_tensor_method(obj, t, TensorOp::ReduceMin { axis, keepdim }, span)
                    }
                    _ => unreachable!(),
                }
            }
            "relu" => self.unary_tensor_method(obj, t, TensorOp::ReLU, span),
            "gelu" => self.unary_tensor_method(obj, t, TensorOp::GELU, span),
            "sigmoid" => self.unary_tensor_method(obj, t, TensorOp::Sigmoid, span),
            "tanh" => self.unary_tensor_method(obj, t, TensorOp::Tanh, span),
            "exp" => self.unary_tensor_method(obj, t, TensorOp::Exp, span),
            "log" => self.unary_tensor_method(obj, t, TensorOp::Log, span),
            "sqrt" => self.unary_tensor_method(obj, t, TensorOp::Sqrt, span),
            "softmax" => {
                let axis = method_axis(args).unwrap_or_else(|| t.rank().saturating_sub(1));
                self.unary_tensor_method(obj, t, TensorOp::Softmax { axis }, span)
            }
            other => {
                self.diags.push(
                    Diagnostic::error(
                        "E0340",
                        format!("tensor has no method `{}`", other),
                        Some(span),
                    )
                    .help("known tensor methods: matmul, transpose, reshape, flatten, sum, mean, max, min, relu, gelu, sigmoid, tanh, exp, log, sqrt, softmax, backward, item, detach"),
                );
                (NirExpr::Unit, Ty::Any)
            }
        }
    }

    fn check_opaque_method(
        &mut self,
        obj: NirExpr,
        opaque: String,
        name: &str,
        args: &[Arg],
        span: Span,
    ) -> (NirExpr, Ty) {
        let base = opaque
            .split('<')
            .next()
            .unwrap_or(opaque.as_str())
            .to_string();
        let is_layer = matches!(
            base.as_str(),
            "Linear"
                | "Sequential"
                | "ReLU"
                | "GELU"
                | "Sigmoid"
                | "Softmax"
                | "Tanh"
                | "Dropout"
                | "LayerNorm"
                | "BatchNorm"
                | "Embedding"
                | "Conv1D"
                | "Conv2D"
        );
        match name {
            "forward" | "predict" | "inference" if is_layer => {
                let (call_args, tys, _) = self.check_args(args);
                let input = tys.first().cloned();
                let out = match (&base[..], &input) {
                    ("Linear", Some(Ty::Tensor(t))) => {
                        if let Some((inn, outlet)) = parse_linear(&opaque) {
                            if let (Some(i), Some(Dim::Static(w))) =
                                (inn, t.shape.last().cloned())
                            {
                                if i != w {
                                    self.diags.push(Diagnostic::error(
                                        "E0342",
                                        format!(
                                            "Linear layer expects input width {}, found {}",
                                            i, w
                                        ),
                                        Some(span),
                                    ));
                                }
                            }
                            crate::builtins::replace_last(&Ty::Tensor(t.clone()), outlet)
                        } else {
                            Ty::Tensor(t.clone())
                        }
                    }
                    ("Sequential", Some(Ty::Tensor(t))) => {
                        let outlet = parse_sequential(&opaque);
                        crate::builtins::replace_last(&Ty::Tensor(t.clone()), outlet)
                    }
                    ("Conv2D", Some(_)) => Ty::Tensor(TensorTy::new(
                        DType::F32,
                        vec![Dim::Unknown, Dim::Unknown, Dim::Unknown, Dim::Unknown],
                    )),
                    ("Conv1D", Some(_)) => Ty::Tensor(TensorTy::new(
                        DType::F32,
                        vec![Dim::Unknown, Dim::Unknown, Dim::Unknown],
                    )),
                    ("Embedding", Some(_)) => Ty::Tensor(TensorTy::new(
                        DType::F32,
                        vec![Dim::Unknown, Dim::Unknown],
                    )),
                    (_, Some(Ty::Tensor(t))) => Ty::Tensor(t.clone()),
                    (_, Some(Ty::Any)) | (_, None) => Ty::Tensor(TensorTy::new(
                        DType::F32,
                        vec![Dim::Unknown],
                    )),
                    (_, Some(other)) => {
                        let name = other.name();
                        self.diags.push(Diagnostic::error(
                            "E0343",
                            format!("`{}` expects a tensor input, found `{}`", base, name),
                            Some(span),
                        ));
                        Ty::Any
                    }
                };
                (
                    NirExpr::Call {
                        callee: Box::new(NirExpr::Attr {
                            obj: Box::new(obj),
                            name: "forward".into(),
                            ty: Ty::Any,
                        }),
                        args: call_args,
                        ty: out.clone(),
                        span: (span.start, span.end),
                    },
                    out,
                )
            }
            "parameters" => {
                let _ = self.check_args(args);
                let list = Ty::List(Box::new(Ty::Tensor(TensorTy::new(
                    DType::F32,
                    vec![Dim::Unknown],
                ))));
                (
                    NirExpr::Call {
                        callee: Box::new(NirExpr::Attr {
                            obj: Box::new(obj),
                            name: name.into(),
                            ty: Ty::Any,
                        }),
                        args: vec![],
                        ty: list.clone(),
                        span: (span.start, span.end),
                    },
                    list,
                )
            }
            "train" => self.method_call(obj, name, args, Ty::Unit, span),
            "save" | "load" | "step" | "zero_grad" | "backward" => {
                self.method_call(obj, name, args, Ty::Unit, span)
            }
            "batch" | "shuffle" | "split" => {
                self.method_call(obj, name, args, Ty::Opaque("Dataset".into()), span)
            }
            "len" => self.method_call(obj, name, args, Ty::Int, span),
            other => {
                let _ = self.check_args(args);
                self.diags.push(
                    Diagnostic::error(
                        "E0340",
                        format!("`{}` has no method `{}`", base, other),
                        Some(span),
                    )
                    .help("known methods: forward, predict, parameters, train, save, load, step, zero_grad, backward"),
                );
                (NirExpr::Unit, Ty::Any)
            }
        }
    }
}

fn positional_arg<'a>(args: &'a [Arg], i: usize) -> Option<&'a Arg> {
    args.iter().filter(|a| a.name.is_none()).nth(i)
}

fn positional_const_int(args: &[Arg], i: usize) -> Option<i64> {
    positional_arg(args, i).and_then(|a| const_int(&a.value))
}

fn kw_arg<'a>(args: &'a [Arg], name: &str) -> Option<&'a Arg> {
    args.iter().find(|a| a.name.as_deref() == Some(name))
}

fn method_axis(args: &[Arg]) -> Option<usize> {
    kw_arg(args, "axis")
        .or_else(|| kw_arg(args, "dim"))
        .and_then(|a| const_int(&a.value))
        .or_else(|| positional_const_int(args, 0))
        .and_then(|v| if v >= 0 { Some(v as usize) } else { None })
}

fn reshape_dims(args: &[Arg]) -> Vec<Dim> {
    let from_expr = |e: &Expr| -> Dim {
        match const_int(e) {
            Some(-1) => Dim::Unknown,
            Some(n) if n >= 0 => Dim::Static(n as usize),
            _ => Dim::Unknown,
        }
    };
    if let Some(list) = kw_arg(args, "shape") {
        if let ExprKind::List(items) = &list.value.kind {
            return items.iter().map(from_expr).collect();
        }
    }
    args.iter()
        .filter(|a| a.name.is_none())
        .map(|a| from_expr(&a.value))
        .collect()
}

fn parse_linear(opaque: &str) -> Option<(Option<usize>, Option<usize>)> {
    let inner = opaque.strip_prefix("Linear<")?.strip_suffix('>')?;
    let (a, b) = inner.split_once(',')?;
    Some((a.trim().parse().ok(), b.trim().parse().ok()))
}

fn parse_sequential(opaque: &str) -> Option<usize> {
    opaque
        .strip_prefix("Sequential<")?
        .strip_suffix('>')?
        .parse()
        .ok()
}

/// Best-effort output width of a `Sequential` layer list: the last
/// `nn.Linear(_, out)` constructor wins.
fn layer_output_dim(e: &Expr, imports: &HashMap<String, String>) -> Option<usize> {
    match &e.kind {
        ExprKind::List(items) => items.iter().rev().find_map(|i| layer_output_dim(i, imports)),
        ExprKind::Call { callee, args } => {
            if let ExprKind::Attr { obj, name } = &callee.kind {
                if let ExprKind::Ident(alias) = &obj.kind {
                    let module = imports.get(alias)?;
                    if module == "nn" && name == "Linear" {
                        return positional_arg(args, 1)
                            .and_then(|a| const_int(&a.value))
                            .filter(|n| *n >= 0)
                            .map(|n| n as usize);
                    }
                }
            }
            None
        }
        _ => None,
    }
}

/// Shape of a (possibly nested) list literal. Returns `None` for ragged lists.
fn list_shape(e: &Expr) -> Option<Vec<Dim>> {
    match &e.kind {
        ExprKind::List(items) => {
            if items.is_empty() {
                return Some(vec![Dim::Static(0)]);
            }
            let nested = items
                .iter()
                .filter(|i| matches!(i.kind, ExprKind::List(_)))
                .count();
            if nested == 0 {
                return Some(vec![Dim::Static(items.len())]);
            }
            if nested != items.len() {
                return None;
            }
            let inner = list_shape(&items[0])?;
            if !items.iter().all(|i| list_shape(i).as_ref() == Some(&inner)) {
                return None;
            }
            let mut shape = vec![Dim::Static(items.len())];
            shape.extend(inner);
            Some(shape)
        }
        _ => None,
    }
}

fn is_builtin_global(name: &str) -> bool {
    matches!(
        name,
        "print" | "input" | "len" | "range" | "abs" | "sqrt" | "exp" | "log" | "tanh" | "int" | "float" | "Tensor"
    )
}

pub(crate) fn const_int(e: &Expr) -> Option<i64> {
    match &e.kind {
        ExprKind::Int(v) => Some(*v),
        ExprKind::Unary { op: UnaryOp::Neg, expr } => const_int(expr).map(|v| -v),
        _ => None,
    }
}

pub(crate) fn const_bool(e: &Expr) -> Option<bool> {
    match &e.kind {
        ExprKind::Bool(v) => Some(*v),
        _ => None,
    }
}

pub(crate) fn const_str(e: &Expr) -> Option<String> {
    match &e.kind {
        ExprKind::Str(s) => Some(s.clone()),
        _ => None,
    }
}

pub(crate) fn merge_types(a: &Ty, b: &Ty) -> Ty {
    if a == b {
        return a.clone();
    }
    match (a, b) {
        (Ty::Any, other) | (other, Ty::Any) => other.clone(),
        (Ty::Int, Ty::Float) | (Ty::Float, Ty::Int) => Ty::Float,
        (Ty::Tensor(x), Ty::Tensor(y)) => {
            let dtype = x.dtype.promote(y.dtype);
            let shape = if x.shape.len() == y.shape.len() {
                x.shape
                    .iter()
                    .zip(y.shape.iter())
                    .map(|(d1, d2)| merge_dim(d1, d2))
                    .collect()
            } else {
                vec![Dim::Unknown]
            };
            Ty::Tensor(TensorTy::new(dtype, shape))
        }
        (Ty::List(x), Ty::List(y)) => Ty::List(Box::new(merge_types(x, y))),
        _ => Ty::Any,
    }
}

fn merge_dim(a: &Dim, b: &Dim) -> Dim {
    if a == b {
        a.clone()
    } else if matches!(a, Dim::Unknown) {
        b.clone()
    } else if matches!(b, Dim::Unknown) {
        a.clone()
    } else {
        Dim::Unknown
    }
}

pub(crate) fn compatible(expected: &Ty, actual: &Ty) -> bool {
    match (expected, actual) {
        (Ty::Any, _) | (_, Ty::Any) | (_, Ty::Never) => true,
        (a, b) if a == b => true,
        (Ty::Float, Ty::Int) => true,
        (Ty::Tensor(e), Ty::Tensor(a)) => {
            if e.dtype != a.dtype && e.dtype.is_float() && a.dtype.is_float() {
                // allow narrowing annotations only when explicitly written
            } else if e.dtype != a.dtype {
                return false;
            }
            if e.shape.len() != a.shape.len() {
                return false;
            }
            e.shape.iter().zip(a.shape.iter()).all(|(d1, d2)| match (d1, d2) {
                (Dim::Static(x), Dim::Static(y)) => x == y,
                _ => true,
            })
        }
        (Ty::List(e), Ty::List(a)) => compatible(e, a),
        (Ty::Func(e), Ty::Func(a)) => {
            e.params.len() == a.params.len()
                && e.params.iter().zip(a.params.iter()).all(|(x, y)| compatible(x, y))
                && compatible(&e.ret, &a.ret)
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse;

    fn chk(src: &str) -> CheckResult {
        let (p, d) = parse(src);
        assert!(d.is_empty(), "{:?}", d);
        check(&p)
    }

    #[test]
    fn infers_arithmetic_and_tensor_shapes() {
        let r = chk("fn f(x: Tensor[2, 3], y: Tensor[3, 4]) -> Tensor[2, 4]:\n    z = x @ y\n    return z\n");
        assert!(!r.diags.iter().any(|d| d.severity == crate::diagnostics::Severity::Error), "{:#?}", r.diags);
        assert_eq!(r.module.functions.len(), 1);
        let f = &r.module.functions[0];
        assert_eq!(f.ret.name(), "Tensor[2, 4; f32]");
    }

    #[test]
    fn catches_shape_mismatch() {
        let r = chk("fn f(x: Tensor[2, 3], y: Tensor[4, 5]) -> Tensor[2, 5]:\n    return x @ y\n");
        assert!(r.diags.iter().any(|d| d.message.contains("inner dimension")));
    }

    #[test]
    fn infers_function_and_local_types() {
        let r = chk("fn add(a: Int, b: Int) -> Int:\n    return a + b\n\nfn main():\n    x = add(1, 2)\n    print(x)\n");
        let main = r.module.functions.iter().find(|f| f.name == "main").unwrap();
        assert_eq!(main.locals[0].ty, Ty::Int);
        assert_eq!(main.ret, Ty::Unit);
    }

    #[test]
    fn rejects_immutable_assignment() {
        let r = chk("fn main():\n    let x = 1\n    x = 2\n");
        assert!(r.diags.iter().any(|d| d.code == "E0203"));
    }

    #[test]
    fn resolves_nn_shapes() {
        let r = chk("import neura.nn as nn\n\nfn main():\n    model = nn.Sequential([nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 2)])\n    out = model.forward(Tensor([1.0, 2.0, 3.0, 4.0]))\n    print(out.shape)\n");
        assert!(!r.diags.iter().any(|d| d.severity == crate::diagnostics::Severity::Error), "{:#?}", r.diags);
    }
}
