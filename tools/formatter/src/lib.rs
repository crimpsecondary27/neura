//! Canonical formatting for Neura source files.
//!
//! The formatter re-prints the parsed AST with one fixed layout: four-space
//! indentation, one statement per line, a blank line between top-level items
//! and minimal parentheses that still preserve the original grouping.
//! Formatting is idempotent: formatting already-formatted source produces
//! byte-identical output, which is what `neura fmt --check` relies on.

use neura_compiler::ast::{
    Arg, BinaryOp, Block, BorrowKind, Expr, ExprKind, FieldDecl, FnDecl, ImportDecl, Item, Param,
    Program, Stmt, StmtKind, StructDecl, TypeDim, TypeExpr, UnaryOp,
};
use neura_compiler::diagnostics::Severity;
use neura_compiler::parser;

const INDENT: &str = "    ";

/// Format Neura source. Returns the parser diagnostics when the input does not
/// parse, so callers can show the same errors the compiler would.
pub fn format_source(src: &str) -> Result<String, String> {
    let (program, diags) = parser::parse(src);
    let errors: Vec<String> = diags
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .map(|d| d.render(src))
        .collect();
    if !errors.is_empty() {
        return Err(errors.join("\n"));
    }
    Ok(format_program(&program))
}

/// Print a parsed program in canonical form.
pub fn format_program(program: &Program) -> String {
    let mut out = String::new();
    let mut previous_was_import = false;
    for (i, item) in program.items.iter().enumerate() {
        let is_import = matches!(item, Item::Import(_));
        if i > 0 && !(is_import && previous_was_import) {
            out.push('\n');
        }
        previous_was_import = is_import;
        match item {
            Item::Import(imp) => {
                out.push_str(&format_import(imp));
                out.push('\n');
            }
            Item::Struct(s) => {
                out.push_str(&format_struct(s));
            }
            Item::Fn(f) => {
                out.push_str(&format_fn(f, 0));
            }
            Item::AddLs => {
                out.push_str("add ls\n");
            }
        }
    }
    out
}

fn format_import(imp: &ImportDecl) -> String {
    let path = imp.path.join(".");
    match &imp.alias {
        Some(alias) if Some(alias) != imp.path.last() => format!("import {} as {}", path, alias),
        _ => format!("import {}", path),
    }
}

fn format_struct(s: &StructDecl) -> String {
    let mut out = format!("struct {}:\n", s.name);
    if s.fields.is_empty() {
        out.push_str(INDENT);
        out.push_str("pass\n");
        return out;
    }
    for FieldDecl { name, ty, .. } in &s.fields {
        out.push_str(INDENT);
        out.push_str(&format!("{}: {}\n", name, format_type(ty)));
    }
    out
}

fn format_fn(f: &FnDecl, depth: usize) -> String {
    let pad = INDENT.repeat(depth);
    let params: Vec<String> = f.params.iter().map(format_param).collect();
    let mut out = format!("{}fn {}({})", pad, f.name, params.join(", "));
    if let Some(ret) = &f.ret {
        out.push_str(&format!(" -> {}", format_type(ret)));
    }
    out.push_str(":\n");
    out.push_str(&format_block(&f.body, depth + 1));
    out
}

fn format_param(p: &Param) -> String {
    let mut out = String::new();
    match p.borrow {
        Some(BorrowKind::Shared) => out.push_str("&"),
        Some(BorrowKind::Mutable) => out.push_str("&mut "),
        None => {}
    }
    out.push_str(&p.name);
    if let Some(ty) = &p.ty {
        out.push_str(&format!(": {}", format_type(ty)));
    }
    out
}

fn format_type(ty: &TypeExpr) -> String {
    match ty {
        TypeExpr::Named { name, .. } => name.clone(),
        TypeExpr::List { inner, .. } => format!("List[{}]", format_type(inner)),
        TypeExpr::Tensor { dims, dtype, .. } => {
            let mut parts: Vec<String> = dims
                .iter()
                .map(|d| match d {
                    TypeDim::Int(v) => v.to_string(),
                    TypeDim::Symbol(s) => s.clone(),
                    TypeDim::Unknown => "dyn".to_string(),
                })
                .collect();
            if let Some(dt) = dtype {
                parts.push(dt.clone());
            }
            format!("Tensor[{}]", parts.join(", "))
        }
    }
}

fn format_block(block: &Block, depth: usize) -> String {
    let pad = INDENT.repeat(depth);
    if block.stmts.is_empty() {
        return format!("{}pass\n", pad);
    }
    let mut out = String::new();
    for stmt in &block.stmts {
        out.push_str(&format_stmt(stmt, depth));
    }
    out
}

fn format_stmt(stmt: &Stmt, depth: usize) -> String {
    let pad = INDENT.repeat(depth);
    match &stmt.kind {
        StmtKind::Let { name, mutable, ty, value } => {
            let mut out = format!("{}let ", pad);
            if *mutable {
                out.push_str("mut ");
            }
            out.push_str(name);
            if let Some(t) = ty {
                out.push_str(&format!(": {}", format_type(t)));
            }
            out.push_str(&format!(" = {}\n", format_expr(value, 0, false)));
            out
        }
        StmtKind::Assign { target, value } => format!(
            "{}{} = {}\n",
            pad,
            format_expr(target, 0, false),
            format_expr(value, 0, false)
        ),
        StmtKind::OpAssign { target, op, value } => format!(
            "{}{} {}= {}\n",
            pad,
            format_expr(target, 0, false),
            binary_symbol(*op),
            format_expr(value, 0, false)
        ),
        StmtKind::Expr(e) => format!("{}{}\n", pad, format_expr(e, 0, false)),
        StmtKind::Return(None) => format!("{}return\n", pad),
        StmtKind::Return(Some(e)) => format!("{}return {}\n", pad, format_expr(e, 0, false)),
        StmtKind::Break => format!("{}break\n", pad),
        StmtKind::Continue => format!("{}continue\n", pad),
        StmtKind::Pass => format!("{}pass\n", pad),
        StmtKind::While { cond, body } => format!(
            "{}while {}:\n{}",
            pad,
            format_expr(cond, 0, false),
            format_block(body, depth + 1)
        ),
        StmtKind::For { var, iter, body } => format!(
            "{}for {} in {}:\n{}",
            pad,
            var,
            format_expr(iter, 0, false),
            format_block(body, depth + 1)
        ),
        StmtKind::If { cond, then_block, else_block } => {
            let mut out = format!(
                "{}if {}:\n{}",
                pad,
                format_expr(cond, 0, false),
                format_block(then_block, depth + 1)
            );
            if let Some(else_block) = else_block {
                // `elif` chains are nested single-statement ifs in the AST.
                if let [Stmt { kind: StmtKind::If { .. }, .. }] = else_block.stmts.as_slice() {
                    let nested = format_stmt(&else_block.stmts[0], depth);
                    let trimmed = nested.trim_start_matches(' ');
                    out.push_str(&format!("{}elif{}", pad, trimmed.strip_prefix("if").unwrap_or(trimmed)));
                } else {
                    out.push_str(&format!("{}else:\n{}", pad, format_block(else_block, depth + 1)));
                }
            }
            out
        }
    }
}

fn precedence(e: &Expr) -> u8 {
    match &e.kind {
        ExprKind::Binary { op, .. } => match op {
            BinaryOp::Or => 1,
            BinaryOp::And => 2,
            BinaryOp::Eq | BinaryOp::Ne => 3,
            BinaryOp::Lt | BinaryOp::Le | BinaryOp::Gt | BinaryOp::Ge => 4,
            BinaryOp::Add | BinaryOp::Sub => 5,
            BinaryOp::Mul | BinaryOp::Div | BinaryOp::FloorDiv | BinaryOp::Mod | BinaryOp::MatMul => 6,
            BinaryOp::Pow => 8,
        },
        ExprKind::Unary { .. } => 7,
        _ => 9,
    }
}

fn binary_symbol(op: BinaryOp) -> &'static str {
    match op {
        BinaryOp::Add => "+",
        BinaryOp::Sub => "-",
        BinaryOp::Mul => "*",
        BinaryOp::Div => "/",
        BinaryOp::FloorDiv => "//",
        BinaryOp::Mod => "%",
        BinaryOp::Pow => "**",
        BinaryOp::Eq => "==",
        BinaryOp::Ne => "!=",
        BinaryOp::Lt => "<",
        BinaryOp::Le => "<=",
        BinaryOp::Gt => ">",
        BinaryOp::Ge => ">=",
        BinaryOp::And => "&&",
        BinaryOp::Or => "||",
        BinaryOp::MatMul => "@",
    }
}

/// Render an expression. `min_prec` is the precedence of the enclosing
/// operator; a child with lower precedence is wrapped in parentheses.
fn format_expr(e: &Expr, min_prec: u8, right_of_same: bool) -> String {
    let own = precedence(e);
    let rendered = match &e.kind {
        ExprKind::Int(v) => v.to_string(),
        ExprKind::Float(v) => format_float(*v),
        ExprKind::Bool(true) => "true".to_string(),
        ExprKind::Bool(false) => "false".to_string(),
        ExprKind::Str(s) => format_string(s),
        ExprKind::Ident(name) => name.clone(),
        ExprKind::List(items) => {
            let inner: Vec<String> = items.iter().map(|i| format_expr(i, 0, false)).collect();
            format!("[{}]", inner.join(", "))
        }
        ExprKind::Unary { op, expr } => match op {
            UnaryOp::Neg => format!("-{}", format_expr(expr, 7, false)),
            UnaryOp::Not => format!("!{}", format_expr(expr, 7, false)),
        },
        ExprKind::Borrow { expr, mutable } => {
            if *mutable {
                format!("&mut {}", format_expr(expr, 7, false))
            } else {
                format!("&{}", format_expr(expr, 7, false))
            }
        }
        ExprKind::Binary { op, lhs, rhs } => {
            let lhs_prec = if matches!(op, BinaryOp::Pow) { own + 1 } else { own };
            format!(
                "{} {} {}",
                format_expr(lhs, lhs_prec, false),
                binary_symbol(*op),
                format_expr(rhs, own, true)
            )
        }
        ExprKind::Call { callee, args } => {
            let inner: Vec<String> = args.iter().map(format_arg).collect();
            format!("{}({})", format_expr(callee, 9, false), inner.join(", "))
        }
        ExprKind::Attr { obj, name } => format!("{}.{}", format_expr(obj, 9, false), name),
        ExprKind::Index { obj, index } => format!(
            "{}[{}]",
            format_expr(obj, 9, false),
            format_expr(index, 0, false)
        ),
        ExprKind::Slice { obj, start, end } => {
            let s = start.as_ref().map(|e| format_expr(e, 0, false)).unwrap_or_default();
            let t = end.as_ref().map(|e| format_expr(e, 0, false)).unwrap_or_default();
            format!("{}[{}:{}]", format_expr(obj, 9, false), s, t)
        }
    };
    let wraps = own < min_prec || (right_of_same && own == min_prec && own <= 6);
    if wraps {
        format!("({})", rendered)
    } else {
        rendered
    }
}

fn format_arg(arg: &Arg) -> String {
    match &arg.name {
        Some(name) => format!("{}={}", name, format_expr(&arg.value, 0, false)),
        None => format_expr(&arg.value, 0, false),
    }
}

fn format_float(v: f64) -> String {
    if v.is_nan() {
        return "0.0 / 0.0".to_string();
    }
    if v.is_infinite() {
        return if v < 0.0 { "-1.0 / 0.0" } else { "1.0 / 0.0" }.to_string();
    }
    let mut s = format!("{}", v);
    if !s.contains('.') && !s.contains('e') && !s.contains('E') {
        s.push_str(".0");
    }
    s
}

fn format_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const SOURCE: &str = "import neura.nn as nn\nfn main():\n  x=1+2*3\n  if x>=7:\n    print(x)\n  else:\n    pass\n  for i in range(2):\n    print(i)\n";

    #[test]
    fn produces_canonical_layout() {
        let out = format_source(SOURCE).unwrap();
        assert!(out.starts_with("import neura.nn\n\nfn main():\n"));
        assert!(out.contains("    x = 1 + 2 * 3\n"));
        assert!(out.contains("    if x >= 7:\n        print(x)\n    else:\n        pass\n"));
        assert!(out.contains("    for i in range(2):\n        print(i)\n"));
    }

    #[test]
    fn formatting_is_idempotent() {
        let once = format_source(SOURCE).unwrap();
        let twice = format_source(&once).unwrap();
        assert_eq!(once, twice);
    }

    #[test]
    fn preserves_grouping_with_parentheses() {
        let out = format_source("fn main():\n  x = (1 + 2) * 3 - 4 / (5 - 6)\n").unwrap();
        assert!(out.contains("x = (1 + 2) * 3 - 4 / (5 - 6)"));
    }

    #[test]
    fn emits_elif_and_typed_tensors() {
        let src = "fn f(a: Tensor[2, 3, f32]) -> Tensor[2, 3, f32]:\n  if a@a == a:\n    return a\n  elif true:\n    return a\n  else:\n    return a\n";
        let out = format_source(src).unwrap();
        assert!(out.contains("fn f(a: Tensor[2, 3, f32]) -> Tensor[2, 3, f32]:"));
        assert!(out.contains("    elif true:"));
    }

    #[test]
    fn formats_power_floor_div_and_compound_assignment() {
        let src = "fn main():\n  x = 2 ** 3 ** 2\n  y = (2 ** 3) ** 2\n  z = -2 ** 2\n  w = 7 // 2\n  x **= 2\n  w //= 2\n";
        let out = format_source(src).unwrap();
        assert!(out.contains("x = 2 ** 3 ** 2"), "{}", out);
        assert!(out.contains("y = (2 ** 3) ** 2"), "{}", out);
        assert!(out.contains("z = -2 ** 2"), "{}", out);
        assert!(out.contains("w = 7 // 2"), "{}", out);
        assert!(out.contains("x **= 2"), "{}", out);
        assert!(out.contains("w //= 2"), "{}", out);
    }

    #[test]
    fn formats_add_ls_directive() {
        let out = format_source("add ls\n\nfn main():\n  pass\n").unwrap();
        assert_eq!(out, "add ls\n\nfn main():\n    pass\n");
    }

    #[test]
    fn rejects_invalid_source() {
        assert!(format_source("fn main(:\n").is_err());
    }
}
