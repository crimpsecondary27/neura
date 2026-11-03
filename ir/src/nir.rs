//! NIR — the Neura IR at language level.
//!
//! NIR is a typed, ownership-annotated tree. Every local carries its resolved
//! type; moves and borrows are explicit; library calls are resolved to builtin
//! names so later layers never need source-level name resolution.

use crate::types::{TensorTy, Ty};

pub type LocalId = u32;

#[derive(Clone, Debug)]
pub struct NirModule {
    pub name: String,
    pub functions: Vec<NirFunction>,
    pub structs: Vec<NirStruct>,
    /// `add ls`: stream sequential models layer-by-layer during inference.
    pub streaming: bool,
}

#[derive(Clone, Debug)]
pub struct NirStruct {
    pub name: String,
    pub fields: Vec<(String, Ty)>,
}

#[derive(Clone, Debug)]
pub struct NirFunction {
    pub name: String,
    pub params: Vec<NirParam>,
    pub ret: Ty,
    pub locals: Vec<NirLocal>,
    pub body: NirBlock,
    pub span: (usize, usize),
}

#[derive(Clone, Debug)]
pub struct NirParam {
    pub name: String,
    pub local: LocalId,
    pub ty: Ty,
    pub is_borrow: bool,
}

#[derive(Clone, Debug)]
pub struct NirLocal {
    pub name: String,
    pub ty: Ty,
    pub mutable: bool,
    pub is_param: bool,
}

#[derive(Clone, Debug, Default)]
pub struct NirBlock {
    pub stmts: Vec<NirStmt>,
}

#[derive(Clone, Debug)]
pub enum NirStmt {
    Let { local: LocalId, expr: NirExpr, span: (usize, usize) },
    Assign { local: LocalId, expr: NirExpr, span: (usize, usize) },
    Expr(NirExpr),
    If {
        cond: NirExpr,
        then_block: NirBlock,
        else_block: Option<NirBlock>,
        span: (usize, usize),
    },
    While { cond: NirExpr, body: NirBlock, span: (usize, usize) },
    For {
        var: LocalId,
        iter: NirExpr,
        body: NirBlock,
        span: (usize, usize),
    },
    Return(Option<NirExpr>, (usize, usize)),
    Break((usize, usize)),
    Continue((usize, usize)),
    Pass,
}

#[derive(Clone, Debug)]
pub struct CallArg {
    pub name: Option<String>,
    pub expr: NirExpr,
}

#[derive(Clone, Debug)]
pub enum NirExpr {
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(String),
    Unit,
    Local(LocalId),
    /// Transfer ownership out of a local (non-Copy values).
    Move(LocalId),
    /// Immutable or mutable borrow of a local.
    Borrow { local: LocalId, mutable: bool },
    Unary { op: crate::UnOp, expr: Box<NirExpr>, ty: Ty },
    Binary { op: crate::BinOp, lhs: Box<NirExpr>, rhs: Box<NirExpr>, ty: Ty },
    List { elems: Vec<NirExpr>, ty: Ty },
    Call {
        callee: Box<NirExpr>,
        args: Vec<CallArg>,
        ty: Ty,
        span: (usize, usize),
    },
    Module(String),
    /// Resolved standard-library function, e.g. `nn.Linear` or `tensor.relu`.
    Builtin { name: String, ty: Ty },
    Attr { obj: Box<NirExpr>, name: String, ty: Ty },
    Index { obj: Box<NirExpr>, index: Box<NirExpr>, ty: Ty },
    /// Typed tensor op produced by operator overloading / method sugar.
    TensorOp {
        op: crate::TensorOp,
        args: Vec<NirExpr>,
        dyn_dims: Vec<NirExpr>,
        ty: TensorTy,
    },
    FunctionRef { index: usize, name: String, ty: Ty },
}

impl NirExpr {
    pub fn ty(&self) -> Ty {
        match self {
            NirExpr::Int(_) => Ty::Int,
            NirExpr::Float(_) => Ty::Float,
            NirExpr::Bool(_) => Ty::Bool,
            NirExpr::Str(_) => Ty::Str,
            NirExpr::Unit => Ty::Unit,
            NirExpr::Local(_) | NirExpr::Move(_) => Ty::Any,
            NirExpr::Borrow { .. } => Ty::Any,
            NirExpr::Unary { ty, .. }
            | NirExpr::Binary { ty, .. }
            | NirExpr::List { ty, .. }
            | NirExpr::Call { ty, .. }
            | NirExpr::Builtin { ty, .. }
            | NirExpr::Attr { ty, .. }
            | NirExpr::Index { ty, .. }
            | NirExpr::FunctionRef { ty, .. } => ty.clone(),
            NirExpr::Module(m) => Ty::Module(m.clone()),
            NirExpr::TensorOp { ty, .. } => Ty::Tensor(ty.clone()),
        }
    }
}
