//! MLIR-compatible textual emission.
//!
//! Neura does not re-implement MLIR. This module prints the Neura dialects
//! (`neura_tir`) together with standard `func`/`arith`/`tensor` constructs in
//! generic MLIR syntax, so a real `mlir-opt` toolchain can consume the output
//! when one is installed. The runtime VM is independent of this printer.

use crate::eir::{EirProgram, Inst};
use crate::tir::{TirGraph, TirNode};
use crate::types::{Dim, TensorTy, Ty};
use std::fmt::Write as _;

pub fn mlir_tensor_type(t: &TensorTy) -> String {
    let mut s = String::from("tensor<");
    for d in &t.shape {
        match d {
            Dim::Static(n) => {
                let _ = write!(s, "{}x", n);
            }
            Dim::Symbol(n) => {
                let _ = write!(s, "?{}x", n);
            }
            Dim::Unknown => s.push_str("?x"),
        }
    }
    let _ = write!(s, "{}>", t.dtype);
    s
}

pub fn mlir_ty(ty: &Ty) -> String {
    match ty {
        Ty::Int => "i64".into(),
        Ty::Float => "f64".into(),
        Ty::Bool => "i1".into(),
        Ty::Str => "!neura.string".into(),
        Ty::Unit => "()".into(),
        Ty::Tensor(t) => mlir_tensor_type(t),
        Ty::List(t) => format!("!neura.list<{}>", mlir_ty(t)),
        Ty::Struct(s) | Ty::Opaque(s) => format!("!neura.obj<\"{}\">", s),
        Ty::Module(m) => format!("!neura.module<\"{}\">", m),
        Ty::Func(_) => "!neura.fn".into(),
        Ty::Any => "!neura.any".into(),
        Ty::Never => "none".into(),
    }
}

/// Print a TIR graph as an MLIR module containing one `func.func`
/// parameterised by the graph inputs.
pub fn print_tir(fn_name: &str, graph: &TirGraph) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "// Neura TIR -> MLIR textual form");
    let _ = writeln!(out, "module attributes {{neura.version = \"0.1.0\"}} {{");
    let mut args = String::new();
    for (i, input) in graph.inputs.iter().enumerate() {
        let _ = write!(args, "%arg{}: {}, ", i, mlir_tensor_type(&graph.nodes[*input].ty));
    }
    let args = args.trim_end_matches(", ").to_string();
    let outs: Vec<String> = graph
        .outputs
        .iter()
        .map(|o| mlir_tensor_type(&graph.nodes[*o].ty))
        .collect();
    let ret = if outs.is_empty() {
        "()".to_string()
    } else {
        format!("({})", outs.join(", "))
    };
    let _ = writeln!(out, "  func.func @{} ({}) -> {} {{", fn_name, args, ret);
    let mut names: Vec<Option<String>> = vec![None; graph.nodes.len()];
    for input in &graph.inputs {
        if let Some(pos) = graph.inputs.iter().position(|x| x == input) {
            names[*input] = Some(format!("%arg{}", pos));
        }
    }
    for id in graph.topo_order() {
        let n = &graph.nodes[id];
        let name = format!("%t{}", id);
        if n.args.is_empty() {
            let _ = writeln!(out, "    {} = \"neura_tir.arg\"() : () -> {}", name, mlir_tensor_type(&n.ty));
        } else {
            let operands: Vec<String> = n
                .args
                .iter()
                .map(|a| names[*a].clone().unwrap_or_else(|| format!("%t{}", a)))
                .collect();
            let operand_tys: Vec<String> = n
                .args
                .iter()
                .map(|a| mlir_tensor_type(&graph.nodes[*a].ty))
                .collect();
            let _ = writeln!(
                out,
                "    {} = \"{}\"({}) {{op = \"{}\"}} : ({}) -> {}",
                name,
                n.op.mlir_mnemonic(),
                operands.join(", "),
                n.op.name(),
                operand_tys.join(", "),
                mlir_tensor_type(&n.ty)
            );
        }
        names[id] = Some(name);
    }
    let results: Vec<String> = graph
        .outputs
        .iter()
        .map(|o| names[*o].clone().unwrap_or_else(|| "%t0".into()))
        .collect();
    let _ = writeln!(out, "    func.return {} : {}", results.join(", "), ret);
    let _ = writeln!(out, "  }}");
    let _ = writeln!(out, "}}");
    out
}

/// Print a whole EIR program. Scalar instructions use `arith`/`scf`; tensor
/// instructions use the `neura_tir` dialect. Control-flow lowering is
/// structural and intended for inspection plus downstream MLIR tooling.
pub fn print_eir(program: &EirProgram) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "// Neura {} -> MLIR textual form", program.name);
    let _ = writeln!(out, "module attributes {{neura.version = \"0.1.0\"}} {{");
    for f in &program.functions {
        let params: Vec<String> = f
            .params
            .iter()
            .map(|r| format!("%p{}: {}", r, mlir_ty(&f.types[*r as usize])))
            .collect();
        let ret = mlir_ty(&f.ret_ty);
        let ret_str = if ret == "()" { "()".to_string() } else { format!("({})", ret) };
        let _ = writeln!(out, "  func.func @{} ({}) -> {} {{", f.name, params.join(", "), ret_str);
        let mut tensor_counter = 0usize;
        for inst in &f.code {
            print_inst(&mut out, inst, &f.types, &mut tensor_counter);
        }
        let _ = writeln!(out, "  }}");
    }
    let _ = writeln!(out, "}}");
    out
}

fn print_inst(out: &mut String, inst: &Inst, _types: &[Ty], counter: &mut usize) {
    match inst {
        Inst::ConstF64 { dst, v } => {
            let _ = writeln!(out, "    %r{} = arith.constant {} : f64", dst, v);
        }
        Inst::ConstInt { dst, v } => {
            let _ = writeln!(out, "    %r{} = arith.constant {} : i64", dst, v);
        }
        Inst::ConstBool { dst, v } => {
            let _ = writeln!(out, "    %r{} = arith.constant {} : i1", dst, if *v { "true" } else { "false" });
        }
        Inst::ConstUnit { dst } => {
            let _ = writeln!(out, "    // %r{} = unit", dst);
        }
        Inst::Copy { dst, src } | Inst::Move { dst, src } => {
            let _ = writeln!(out, "    // %r{} = copy %r{}", dst, src);
        }
        Inst::MakeList { dst, elems } => {
            let _ = writeln!(out, "    // %r{} = list({} elems)", dst, elems.len());
        }
        Inst::MakeTensor { dst, requires_grad, ty, .. } => {
            let _ = writeln!(
                out,
                "    %r{} = \"neura_tir.alloc\"() {{requires_grad = {}}} : () -> {}",
                dst,
                requires_grad,
                mlir_tensor_type(ty)
            );
        }
        Inst::MakeStruct { dst, struct_id, .. } => {
            let _ = writeln!(out, "    // %r{} = neura.struct #{}", dst, struct_id);
        }
        Inst::TensorOp { dst, op, ty, .. } => {
            let _ = writeln!(
                out,
                "    %r{} = \"{}\" {{op = \"{}\"}} : () -> {}",
                dst,
                op.mlir_mnemonic(),
                op.name(),
                mlir_tensor_type(ty)
            );
        }
        Inst::FusedOp { dst, spec, ty, args } => {
            let _ = writeln!(
                out,
                "    %r{} = \"neura_tir.fused\"({}) {{expr = \"{}\"}} : () -> {}",
                dst,
                args.iter().map(|a| format!("%r{}", a)).collect::<Vec<_>>().join(", "),
                spec.expr,
                mlir_tensor_type(ty)
            );
        }
        Inst::BinOp { dst, op, lhs, rhs, ty } => {
            let m = match op {
                crate::BinOp::Add => "arith.addf",
                crate::BinOp::Sub => "arith.subf",
                crate::BinOp::Mul => "arith.mulf",
                crate::BinOp::Div => "arith.divf",
                crate::BinOp::FloorDiv => "arith.divf",
                crate::BinOp::Pow => "arith.powf",
                crate::BinOp::Mod => "arith.remf",
                crate::BinOp::Eq => "arith.cmpf oeq",
                crate::BinOp::Ne => "arith.cmpf one",
                crate::BinOp::Lt => "arith.cmpf olt",
                crate::BinOp::Le => "arith.cmpf ole",
                crate::BinOp::Gt => "arith.cmpf ogt",
                crate::BinOp::Ge => "arith.cmpf oge",
                crate::BinOp::And => "arith.andi",
                crate::BinOp::Or => "arith.ori",
            };
            let _ = writeln!(
                out,
                "    %r{} = {} %r{}, %r{} : {}",
                dst,
                m,
                lhs,
                rhs,
                mlir_ty(ty)
            );
        }
        Inst::UnOp { dst, op, src, ty } => {
            let m = match op {
                crate::UnOp::Neg => "arith.negf",
                crate::UnOp::Not => "arith.xori",
            };
            let _ = writeln!(out, "    %r{} = {} %r{} : {}", dst, m, src, mlir_ty(ty));
        }
        Inst::Call { dst, func, .. } => {
            let _ = writeln!(out, "    // %r{} = call func#{}", dst, func);
        }
        Inst::CallBuiltin { dst, name, .. } => {
            let _ = writeln!(out, "    // %r{} = \"neura.std.{}\"", dst, name.replace('.', "_"));
        }
        Inst::MethodCall { dst, method, .. } => {
            let _ = writeln!(out, "    // %r{} = neura.method \"{}\"", dst, method);
        }
        Inst::GetAttr { dst, name, .. } => {
            let _ = writeln!(out, "    // %r{} = neura.getattr \"{}\"", dst, name);
        }
        _ => {
            *counter += 1;
            let _ = writeln!(out, "    // {} : {} operands", inst_name(inst), inst.reads().len());
        }
    }
}

fn inst_name(inst: &Inst) -> &'static str {
    match inst {
        Inst::Index { .. } => "index",
        Inst::Len { .. } => "len",
        Inst::IterInit { .. } => "iter_init",
        Inst::IterNext { .. } => "iter_next",
        Inst::Jump(_) => "jump",
        Inst::JumpIfFalse { .. } => "branch",
        Inst::JumpIfTrue { .. } => "branch",
        Inst::Return(_) => "return",
        Inst::Drop(_) => "drop",
        Inst::Print(_) => "print",
        Inst::Input { .. } => "input",
        _ => "op",
    }
}

pub fn tir_node_summary(n: &TirNode) -> String {
    format!("{} : {}", n.op.name(), n.ty)
}
