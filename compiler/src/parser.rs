//! Recursive-descent parser for the indentation-based Neura grammar.

use crate::ast::*;
use crate::diagnostics::{Diagnostic, Span};
use crate::lexer::{Lexer, Token, TokenKind};

pub fn parse(src: &str) -> (Program, Vec<Diagnostic>) {
    let (tokens, mut diags) = Lexer::new(src).lex();
    let mut p = Parser { tokens, pos: 0, diags: Vec::new() };
    let program = p.parse_program();
    diags.extend(p.diags);
    (program, diags)
}

struct Parser {
    tokens: Vec<Token>,
    pos: usize,
    diags: Vec<Diagnostic>,
}

impl Parser {
    fn peek(&self) -> &TokenKind {
        &self.tokens[self.pos.min(self.tokens.len() - 1)].kind
    }

    fn peek_at(&self, n: usize) -> &TokenKind {
        &self.tokens[(self.pos + n).min(self.tokens.len() - 1)].kind
    }

    fn token(&self) -> &Token {
        &self.tokens[self.pos.min(self.tokens.len() - 1)]
    }

    fn advance(&mut self) -> TokenKind {
        let k = self.peek().clone();
        if self.pos < self.tokens.len() - 1 {
            self.pos += 1;
        }
        k
    }

    fn check(&self, kind: &TokenKind) -> bool {
        self.peek() == kind
    }

    fn eat(&mut self, kind: &TokenKind) -> bool {
        if self.check(kind) {
            self.advance();
            true
        } else {
            false
        }
    }

    fn expect(&mut self, kind: TokenKind, ctx: &str) -> Option<Span> {
        if self.check(&kind) {
            let sp = self.token().span;
            self.advance();
            Some(sp)
        } else {
            let sp = self.token().span;
            self.diags.push(
                Diagnostic::error(
                    "E0100",
                    format!("expected {} {}, found {}", kind.describe(), ctx, self.peek().describe()),
                    Some(sp),
                )
                .help("check for missing punctuation or indentation"),
            );
            None
        }
    }

    fn error_here(&mut self, msg: impl Into<String>) {
        let sp = self.token().span;
        self.diags.push(Diagnostic::error("E0101", msg, Some(sp)));
    }

    fn skip_newlines(&mut self) {
        while matches!(self.peek(), TokenKind::Newline | TokenKind::Semi) {
            self.advance();
        }
    }

    fn parse_program(&mut self) -> Program {
        let mut items = Vec::new();
        loop {
            self.skip_newlines();
            match self.peek() {
                TokenKind::Eof => break,
                TokenKind::Fn => items.push(Item::Fn(self.parse_fn())),
                TokenKind::Struct => items.push(Item::Struct(self.parse_struct())),
                TokenKind::Import => items.push(Item::Import(self.parse_import())),
                TokenKind::Ident(name)
                    if name == "add"
                        && matches!(self.peek_at(1), TokenKind::Ident(t) if t == "ls") =>
                {
                    self.advance();
                    self.advance();
                    items.push(Item::AddLs);
                }
                TokenKind::Pub => {
                    self.advance();
                    match self.peek() {
                        TokenKind::Fn => {
                            let mut f = self.parse_fn();
                            f.public = true;
                            items.push(Item::Fn(f));
                        }
                        TokenKind::Struct => items.push(Item::Struct(self.parse_struct())),
                        _ => {
                            self.error_here("`pub` must precede a function or struct");
                            self.synchronize();
                        }
                    }
                }
                other => {
                    self.error_here(format!("expected top-level item, found {}", other.describe()));
                    self.synchronize();
                }
            }
        }
        Program { items }
    }

    fn parse_import(&mut self) -> ImportDecl {
        let start = self.token().span;
        self.advance();
        let mut path = Vec::new();
        match self.advance() {
            TokenKind::Ident(s) => path.push(s),
            _ => self.error_here("expected module name after `import`"),
        }
        while self.eat(&TokenKind::Dot) {
            match self.advance() {
                TokenKind::Ident(s) => path.push(s),
                _ => {
                    self.error_here("expected module segment after `.`");
                    break;
                }
            }
        }
        let alias = if self.eat(&TokenKind::As) {
            match self.advance() {
                TokenKind::Ident(s) => Some(s),
                _ => {
                    self.error_here("expected alias after `as`");
                    None
                }
            }
        } else {
            path.last().cloned()
        };
        let end = self.token().span;
        if matches!(self.peek(), TokenKind::Newline) {
            self.advance();
        }
        ImportDecl { path, alias, span: start.merge(end) }
    }

    fn parse_fn(&mut self) -> FnDecl {
        let start = self.token().span;
        self.advance(); // fn
        let name = match self.advance() {
            TokenKind::Ident(s) => s,
            _ => {
                self.error_here("expected function name");
                "<error>".into()
            }
        };
        self.expect(TokenKind::LParen, "after function name");
        let mut params = Vec::new();
        while !self.check(&TokenKind::RParen) && !self.check(&TokenKind::Eof) {
            let pspan = self.token().span;
            let mut borrow = None;
            if self.eat(&TokenKind::Amp) {
                borrow = Some(if self.eat(&TokenKind::Mut) {
                    BorrowKind::Mutable
                } else {
                    BorrowKind::Shared
                });
            }
            let pname = match self.advance() {
                TokenKind::Ident(s) => s,
                _ => {
                    self.error_here("expected parameter name");
                    break;
                }
            };
            let ty = if self.eat(&TokenKind::Colon) {
                self.parse_type()
            } else {
                None
            };
            params.push(Param { name: pname, ty, borrow, span: pspan });
            if !self.eat(&TokenKind::Comma) {
                break;
            }
        }
        self.expect(TokenKind::RParen, "after parameters");
        let ret = if self.eat(&TokenKind::Arrow) {
            self.parse_type()
        } else {
            None
        };
        let body = self.parse_block("function body");
        FnDecl { name, params, ret, body, span: start.merge(self.token().span), public: false }
    }

    fn parse_struct(&mut self) -> StructDecl {
        let start = self.token().span;
        self.advance();
        let name = match self.advance() {
            TokenKind::Ident(s) => s,
            _ => {
                self.error_here("expected struct name");
                "<error>".into()
            }
        };
        self.expect(TokenKind::Colon, "after struct name");
        self.expect(TokenKind::Newline, "after `:`");
        self.expect(TokenKind::Indent, "to begin struct body");
        let mut fields = Vec::new();
        while !self.check(&TokenKind::Dedent) && !self.check(&TokenKind::Eof) {
            self.skip_newlines();
            if self.check(&TokenKind::Dedent) {
                break;
            }
            let fspan = self.token().span;
            let fname = match self.advance() {
                TokenKind::Ident(s) => s,
                _ => {
                    self.error_here("expected field name");
                    self.synchronize();
                    continue;
                }
            };
            self.expect(TokenKind::Colon, "after field name");
            if let Some(ty) = self.parse_type() {
                fields.push(FieldDecl { name: fname, ty, span: fspan });
            }
            if matches!(self.peek(), TokenKind::Newline) {
                self.advance();
            }
        }
        self.eat(&TokenKind::Dedent);
        StructDecl { name, fields, span: start.merge(self.token().span) }
    }

    fn parse_type(&mut self) -> Option<TypeExpr> {
        let start = self.token().span;
        let name = match self.advance() {
            TokenKind::Ident(s) => s,
            _ => {
                self.error_here("expected a type name");
                return None;
            }
        };
        if name == "Tensor" && self.eat(&TokenKind::LBracket) {
            let mut dims = Vec::new();
            let mut dtype = None;
            while !self.check(&TokenKind::RBracket) && !self.check(&TokenKind::Eof) {
                match self.advance() {
                    TokenKind::Int(v) if v >= 0 => dims.push(TypeDim::Int(v as usize)),
                    TokenKind::Ident(s) if s == "dyn" => dims.push(TypeDim::Unknown),
                    TokenKind::Ident(s) if s == "f32" || s == "f64" || s == "f16"
                        || s == "bf16" || s == "i8" || s == "i32" || s == "i64" =>
                    {
                        dtype = Some(s);
                    }
                    TokenKind::Ident(s) => dims.push(TypeDim::Symbol(s)),
                    TokenKind::Semi => {
                        if let TokenKind::Ident(d) = self.advance() {
                            dtype = Some(d);
                        }
                    }
                    _ => {
                        self.error_here("expected a tensor dimension");
                        break;
                    }
                }
                if !self.eat(&TokenKind::Comma) {
                    if self.eat(&TokenKind::Semi) {
                        if let TokenKind::Ident(d) = self.advance() {
                            dtype = Some(d);
                        }
                    }
                    break;
                }
            }
            self.expect(TokenKind::RBracket, "to close tensor type");
            return Some(TypeExpr::Tensor { dims, dtype, span: start.merge(self.token().span) });
        }
        if name == "List" && self.eat(&TokenKind::LBracket) {
            let inner = self.parse_type().unwrap_or(TypeExpr::Named { name: "Any".into(), span: start });
            self.expect(TokenKind::RBracket, "to close List type");
            return Some(TypeExpr::List { inner: Box::new(inner), span: start.merge(self.token().span) });
        }
        Some(TypeExpr::Named { name, span: start })
    }

    fn parse_block(&mut self, ctx: &str) -> Block {
        let start = self.token().span;
        self.expect(TokenKind::Colon, &format!("before {}", ctx));
        self.expect(TokenKind::Newline, "after `:`");
        self.expect(TokenKind::Indent, &format!("to begin {}", ctx));
        let mut stmts = Vec::new();
        loop {
            self.skip_newlines();
            if self.check(&TokenKind::Dedent) || self.check(&TokenKind::Eof) {
                break;
            }
            let before = self.pos;
            if let Some(s) = self.parse_stmt() {
                stmts.push(s);
            }
            if self.pos == before {
                self.synchronize();
            }
        }
        self.eat(&TokenKind::Dedent);
        if stmts.is_empty() {
            self.diags.push(Diagnostic::error(
                "E0102",
                format!("{} cannot be empty; use `pass`", ctx),
                Some(start),
            ));
        }
        Block { stmts, span: start.merge(self.token().span) }
    }

    fn parse_stmt(&mut self) -> Option<Stmt> {
        let start = self.token().span;
        match self.peek() {
            TokenKind::Let => {
                self.advance();
                let mutable = self.eat(&TokenKind::Mut);
                let name = match self.advance() {
                    TokenKind::Ident(s) => s,
                    _ => {
                        self.error_here("expected variable name after `let`");
                        return None;
                    }
                };
                let ty = if self.eat(&TokenKind::Colon) { self.parse_type() } else { None };
                self.expect(TokenKind::Assign, "in let binding");
                let value = self.parse_expr();
                self.end_stmt();
                Some(Stmt {
                    kind: StmtKind::Let { name, mutable, ty, value },
                    span: start.merge(self.token().span),
                })
            }
            TokenKind::If => Some(self.parse_if()),
            TokenKind::While => {
                self.advance();
                let cond = self.parse_expr();
                let body = self.parse_block("while body");
                Some(Stmt { kind: StmtKind::While { cond, body }, span: start.merge(self.token().span) })
            }
            TokenKind::For => {
                self.advance();
                let var = match self.advance() {
                    TokenKind::Ident(s) => s,
                    _ => {
                        self.error_here("expected loop variable after `for`");
                        return None;
                    }
                };
                self.expect(TokenKind::In, "in for loop");
                let iter = self.parse_expr();
                let body = self.parse_block("for body");
                Some(Stmt { kind: StmtKind::For { var, iter, body }, span: start.merge(self.token().span) })
            }
            TokenKind::Return => {
                self.advance();
                let value = if matches!(self.peek(), TokenKind::Newline | TokenKind::Dedent | TokenKind::Eof) {
                    None
                } else {
                    Some(self.parse_expr())
                };
                self.end_stmt();
                Some(Stmt { kind: StmtKind::Return(value), span: start })
            }
            TokenKind::Break => {
                self.advance();
                self.end_stmt();
                Some(Stmt { kind: StmtKind::Break, span: start })
            }
            TokenKind::Continue => {
                self.advance();
                self.end_stmt();
                Some(Stmt { kind: StmtKind::Continue, span: start })
            }
            TokenKind::Pass => {
                self.advance();
                self.end_stmt();
                Some(Stmt { kind: StmtKind::Pass, span: start })
            }
            TokenKind::Elif | TokenKind::Else | TokenKind::Dedent | TokenKind::Eof => None,
            _ => {
                let expr = self.parse_expr();
                let op = match self.peek() {
                    TokenKind::PlusAssign => Some(BinaryOp::Add),
                    TokenKind::MinusAssign => Some(BinaryOp::Sub),
                    TokenKind::StarAssign => Some(BinaryOp::Mul),
                    TokenKind::SlashAssign => Some(BinaryOp::Div),
                    TokenKind::SlashSlashAssign => Some(BinaryOp::FloorDiv),
                    TokenKind::PercentAssign => Some(BinaryOp::Mod),
                    TokenKind::StarStarAssign => Some(BinaryOp::Pow),
                    _ => None,
                };
                if let Some(op) = op {
                    self.advance();
                    let value = self.parse_expr();
                    self.end_stmt();
                    Some(Stmt {
                        kind: StmtKind::OpAssign { target: expr, op, value },
                        span: start.merge(self.token().span),
                    })
                } else if self.check(&TokenKind::Assign) {
                    self.advance();
                    let value = self.parse_expr();
                    self.end_stmt();
                    Some(Stmt {
                        kind: StmtKind::Assign { target: expr, value },
                        span: start.merge(self.token().span),
                    })
                } else {
                    self.end_stmt();
                    Some(Stmt { kind: StmtKind::Expr(expr), span: start.merge(self.token().span) })
                }
            }
        }
    }

    fn parse_if(&mut self) -> Stmt {
        let start = self.token().span;
        self.advance();
        let cond = self.parse_expr();
        let then_block = self.parse_block("if body");
        let else_block = if self.eat(&TokenKind::Elif) {
            self.parse_elif_chain(start)
        } else if self.eat(&TokenKind::Else) {
            Some(self.parse_block("else body"))
        } else {
            None
        };
        Stmt { kind: StmtKind::If { cond, then_block, else_block }, span: start }
    }

    fn parse_elif_chain(&mut self, span: Span) -> Option<Block> {
        // Called with `elif` already consumed.
        let cond = self.parse_expr();
        let then_block = self.parse_block("elif body");
        let else_block = if self.check(&TokenKind::Elif) {
            self.advance();
            self.parse_elif_chain(span)
        } else if self.eat(&TokenKind::Else) {
            Some(self.parse_block("else body"))
        } else {
            None
        };
        Some(Block {
            stmts: vec![Stmt {
                kind: StmtKind::If { cond, then_block, else_block },
                span,
            }],
            span,
        })
    }

    fn end_stmt(&mut self) {
        if self.check(&TokenKind::Newline) {
            self.advance();
        } else if self.check(&TokenKind::Semi) {
            self.advance();
        } else if !matches!(self.peek(), TokenKind::Dedent | TokenKind::Eof) {
            let sp = self.token().span;
            self.diags.push(
                Diagnostic::error(
                    "E0103",
                    format!("expected end of statement, found {}", self.peek().describe()),
                    Some(sp),
                )
                .help("Neura statements are newline-terminated"),
            );
            self.synchronize();
        }
    }

    fn synchronize(&mut self) {
        while !matches!(
            self.peek(),
            TokenKind::Newline | TokenKind::Dedent | TokenKind::Eof
        ) {
            self.advance();
        }
        if self.check(&TokenKind::Newline) {
            self.advance();
        }
    }

    fn parse_expr(&mut self) -> Expr {
        self.parse_or()
    }

    fn parse_or(&mut self) -> Expr {
        let mut lhs = self.parse_and();
        while self.check(&TokenKind::PipePipe) {
            let op_span = self.token().span;
            self.advance();
            let rhs = self.parse_and();
            lhs = Expr {
                kind: ExprKind::Binary { op: BinaryOp::Or, lhs: Box::new(lhs), rhs: Box::new(rhs) },
                span: op_span,
            };
        }
        lhs
    }

    fn parse_and(&mut self) -> Expr {
        let mut lhs = self.parse_equality();
        while self.check(&TokenKind::AmpAmp) {
            let op_span = self.token().span;
            self.advance();
            let rhs = self.parse_equality();
            lhs = Expr {
                kind: ExprKind::Binary { op: BinaryOp::And, lhs: Box::new(lhs), rhs: Box::new(rhs) },
                span: op_span,
            };
        }
        lhs
    }

    fn parse_equality(&mut self) -> Expr {
        let mut lhs = self.parse_comparison();
        loop {
            let op = match self.peek() {
                TokenKind::EqEq => BinaryOp::Eq,
                TokenKind::NotEq => BinaryOp::Ne,
                _ => break,
            };
            let op_span = self.token().span;
            self.advance();
            let rhs = self.parse_comparison();
            lhs = Expr {
                kind: ExprKind::Binary { op, lhs: Box::new(lhs), rhs: Box::new(rhs) },
                span: op_span,
            };
        }
        lhs
    }

    fn parse_comparison(&mut self) -> Expr {
        let mut lhs = self.parse_additive();
        loop {
            let op = match self.peek() {
                TokenKind::Lt => BinaryOp::Lt,
                TokenKind::Le => BinaryOp::Le,
                TokenKind::Gt => BinaryOp::Gt,
                TokenKind::Ge => BinaryOp::Ge,
                _ => break,
            };
            let op_span = self.token().span;
            self.advance();
            let rhs = self.parse_additive();
            lhs = Expr {
                kind: ExprKind::Binary { op, lhs: Box::new(lhs), rhs: Box::new(rhs) },
                span: op_span,
            };
        }
        lhs
    }

    fn parse_additive(&mut self) -> Expr {
        let mut lhs = self.parse_multiplicative();
        loop {
            let op = match self.peek() {
                TokenKind::Plus => BinaryOp::Add,
                TokenKind::Minus => BinaryOp::Sub,
                _ => break,
            };
            let op_span = self.token().span;
            self.advance();
            let rhs = self.parse_multiplicative();
            lhs = Expr {
                kind: ExprKind::Binary { op, lhs: Box::new(lhs), rhs: Box::new(rhs) },
                span: op_span,
            };
        }
        lhs
    }

    fn parse_multiplicative(&mut self) -> Expr {
        let mut lhs = self.parse_unary();
        loop {
            let op = match self.peek() {
                TokenKind::Star => BinaryOp::Mul,
                TokenKind::Slash => BinaryOp::Div,
                TokenKind::SlashSlash => BinaryOp::FloorDiv,
                TokenKind::Percent => BinaryOp::Mod,
                TokenKind::At => BinaryOp::MatMul,
                _ => break,
            };
            let op_span = self.token().span;
            self.advance();
            let rhs = self.parse_unary();
            lhs = Expr {
                kind: ExprKind::Binary { op, lhs: Box::new(lhs), rhs: Box::new(rhs) },
                span: op_span,
            };
        }
        lhs
    }

    fn parse_unary(&mut self) -> Expr {
        let start = self.token().span;
        match self.peek() {
            TokenKind::Minus => {
                self.advance();
                let e = self.parse_unary();
                Expr { kind: ExprKind::Unary { op: UnaryOp::Neg, expr: Box::new(e) }, span: start }
            }
            TokenKind::Bang => {
                self.advance();
                let e = self.parse_unary();
                Expr { kind: ExprKind::Unary { op: UnaryOp::Not, expr: Box::new(e) }, span: start }
            }
            TokenKind::Amp => {
                self.advance();
                let mutable = self.eat(&TokenKind::Mut);
                let e = self.parse_unary();
                Expr { kind: ExprKind::Borrow { expr: Box::new(e), mutable }, span: start }
            }
            _ => self.parse_power(),
        }
    }

    /// Exponentiation binds tighter than unary minus and is right-associative,
    /// so `-2 ** 2` is `-(2 ** 2)` and `2 ** 3 ** 2` is `2 ** (3 ** 2)`.
    fn parse_power(&mut self) -> Expr {
        let base = self.parse_postfix();
        if self.check(&TokenKind::StarStar) {
            let op_span = self.token().span;
            self.advance();
            let exp = self.parse_unary();
            return Expr {
                kind: ExprKind::Binary {
                    op: BinaryOp::Pow,
                    lhs: Box::new(base),
                    rhs: Box::new(exp),
                },
                span: op_span,
            };
        }
        base
    }

    fn parse_postfix(&mut self) -> Expr {
        let mut expr = self.parse_primary();
        loop {
            match self.peek() {
                TokenKind::Dot => {
                    self.advance();
                    let name = match self.advance() {
                        TokenKind::Ident(s) => s,
                        _ => {
                            self.error_here("expected attribute name after `.`");
                            break;
                        }
                    };
                    expr = Expr {
                        kind: ExprKind::Attr { obj: Box::new(expr), name },
                        span: self.token().span,
                    };
                }
                TokenKind::LParen => {
                    self.advance();
                    let mut args = Vec::new();
                    while !self.check(&TokenKind::RParen) && !self.check(&TokenKind::Eof) {
                        let aspan = self.token().span;
                        let name = if matches!(self.peek(), TokenKind::Ident(_))
                            && self.peek_at(1) == &TokenKind::Assign
                        {
                            let n = match self.advance() {
                                TokenKind::Ident(s) => Some(s),
                                _ => None,
                            };
                            self.advance();
                            n
                        } else {
                            None
                        };
                        let value = self.parse_expr();
                        args.push(Arg { name, value, span: aspan });
                        if !self.eat(&TokenKind::Comma) {
                            break;
                        }
                    }
                    self.expect(TokenKind::RParen, "to close argument list");
                    expr = Expr {
                        kind: ExprKind::Call { callee: Box::new(expr), args },
                        span: self.token().span,
                    };
                }
                TokenKind::LBracket => {
                    self.advance();
                    if self.check(&TokenKind::Colon) {
                        self.advance();
                        let end = if self.check(&TokenKind::RBracket) {
                            None
                        } else {
                            Some(Box::new(self.parse_expr()))
                        };
                        self.expect(TokenKind::RBracket, "to close slice");
                        expr = Expr {
                            kind: ExprKind::Slice { obj: Box::new(expr), start: None, end },
                            span: self.token().span,
                        };
                    } else {
                        let first = self.parse_expr();
                        if self.eat(&TokenKind::Colon) {
                            let end = if self.check(&TokenKind::RBracket) {
                                None
                            } else {
                                Some(Box::new(self.parse_expr()))
                            };
                            self.expect(TokenKind::RBracket, "to close slice");
                            expr = Expr {
                                kind: ExprKind::Slice {
                                    obj: Box::new(expr),
                                    start: Some(Box::new(first)),
                                    end,
                                },
                                span: self.token().span,
                            };
                        } else {
                            self.expect(TokenKind::RBracket, "to close index");
                            expr = Expr {
                                kind: ExprKind::Index { obj: Box::new(expr), index: Box::new(first) },
                                span: self.token().span,
                            };
                        }
                    }
                }
                _ => break,
            }
        }
        expr
    }

    fn parse_primary(&mut self) -> Expr {
        let start = self.token().span;
        match self.advance() {
            TokenKind::Int(v) => Expr { kind: ExprKind::Int(v), span: start },
            TokenKind::Float(v) => Expr { kind: ExprKind::Float(v), span: start },
            TokenKind::Str(s) => Expr { kind: ExprKind::Str(s), span: start },
            TokenKind::True => Expr { kind: ExprKind::Bool(true), span: start },
            TokenKind::False => Expr { kind: ExprKind::Bool(false), span: start },
            TokenKind::Ident(s) => Expr { kind: ExprKind::Ident(s), span: start },
            TokenKind::LParen => {
                let e = self.parse_expr();
                self.expect(TokenKind::RParen, "to close parenthesized expression");
                e
            }
            TokenKind::LBracket => {
                let mut elems = Vec::new();
                while !self.check(&TokenKind::RBracket) && !self.check(&TokenKind::Eof) {
                    elems.push(self.parse_expr());
                    if !self.eat(&TokenKind::Comma) {
                        break;
                    }
                }
                self.expect(TokenKind::RBracket, "to close list literal");
                Expr { kind: ExprKind::List(elems), span: start.merge(self.token().span) }
            }
            other => {
                self.error_here(format!("expected an expression, found {}", other.describe()));
                Expr { kind: ExprKind::Int(0), span: start }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_ok(src: &str) -> Program {
        let (p, d) = parse(src);
        assert!(d.is_empty(), "{:#?}", d);
        p
    }

    #[test]
    fn parses_functions_and_calls() {
        let p = parse_ok(
            "import neura.nn as nn\n\nfn main():\n    x = nn.Linear(2, 3)\n    y = x.forward([1.0, 2.0])\n",
        );
        assert_eq!(p.items.len(), 2);
        match &p.items[0] {
            Item::Import(i) => {
                assert_eq!(i.path, vec!["neura", "nn"]);
                assert_eq!(i.alias.as_deref(), Some("nn"));
            }
            _ => panic!(),
        }
        match &p.items[1] {
            Item::Fn(f) => {
                assert_eq!(f.name, "main");
                assert_eq!(f.body.stmts.len(), 2);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn parses_add_ls_directive() {
        let p = parse_ok("add ls\n\nfn main():\n    pass\n");
        assert_eq!(p.items.len(), 2);
        assert!(matches!(p.items[0], Item::AddLs));
        assert!(matches!(p.items[1], Item::Fn(_)));
    }

    #[test]
    fn parses_control_flow() {
        let p = parse_ok(
            "fn f(n: Int) -> Int:\n    total = 0\n    for i in range(n):\n        if i % 2 == 0:\n            total = total + i\n        else:\n            pass\n    return total\n",
        );
        match &p.items[0] {
            Item::Fn(f) => {
                assert_eq!(f.params.len(), 1);
                assert!(matches!(f.ret, Some(TypeExpr::Named { .. })));
                assert_eq!(f.body.stmts.len(), 3);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn parses_tensor_types_and_structs() {
        let p = parse_ok(
            "struct Pair:\n    a: Tensor[2, 3]\n    b: Int\n\nfn f(x: Tensor[2, 3; f16]) -> Tensor[2, 3]:\n    return x\n",
        );
        assert_eq!(p.items.len(), 2);
    }

    #[test]
    fn reports_missing_colon() {
        let (_, d) = parse("fn main()\n    pass\n");
        assert!(d.iter().any(|d| d.code == "E0100"));
    }

    #[test]
    fn parses_elif_chain() {
        let p = parse_ok("fn f(x: Int) -> Int:\n    if x == 1:\n        return 1\n    elif x == 2:\n        return 2\n    else:\n        return 3\n");
        match &p.items[0] {
            Item::Fn(f) => {
                assert!(matches!(f.body.stmts[0].kind, StmtKind::If { .. }));
            }
            _ => panic!(),
        }
    }
}
