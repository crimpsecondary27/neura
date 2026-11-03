//! Indentation-sensitive lexer.

use crate::diagnostics::{Diagnostic, Span};

#[derive(Clone, PartialEq, Debug)]
pub enum TokenKind {
    Ident(String),
    Int(i64),
    Float(f64),
    Str(String),
    // keywords
    Fn,
    Let,
    Mut,
    If,
    Elif,
    Else,
    While,
    For,
    In,
    Return,
    Struct,
    Import,
    As,
    True,
    False,
    Break,
    Continue,
    Pass,
    Pub,
    // punctuation
    LParen,
    RParen,
    LBracket,
    RBracket,
    LBrace,
    RBrace,
    Comma,
    Colon,
    Dot,
    Arrow,
    Semi,
    Amp,
    Pipe,
    Bang,
    At,
    // operators
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    StarStar,
    SlashSlash,
    Assign,
    PlusAssign,
    MinusAssign,
    StarAssign,
    SlashAssign,
    SlashSlashAssign,
    PercentAssign,
    StarStarAssign,
    EqEq,
    NotEq,
    Lt,
    Le,
    Gt,
    Ge,
    AmpAmp,
    PipePipe,
    // layout
    Newline,
    Indent,
    Dedent,
    Eof,
}

impl TokenKind {
    pub fn describe(&self) -> String {
        match self {
            TokenKind::Ident(s) => format!("identifier `{}`", s),
            TokenKind::Int(v) => format!("integer `{}`", v),
            TokenKind::Float(v) => format!("float `{}`", v),
            TokenKind::Str(_) => "string literal".into(),
            TokenKind::Newline => "newline".into(),
            TokenKind::Indent => "indent".into(),
            TokenKind::Dedent => "dedent".into(),
            TokenKind::Eof => "end of file".into(),
            other => format!("`{}`", other.text()),
        }
    }

    pub fn text(&self) -> &'static str {
        match self {
            TokenKind::Fn => "fn",
            TokenKind::Let => "let",
            TokenKind::Mut => "mut",
            TokenKind::If => "if",
            TokenKind::Elif => "elif",
            TokenKind::Else => "else",
            TokenKind::While => "while",
            TokenKind::For => "for",
            TokenKind::In => "in",
            TokenKind::Return => "return",
            TokenKind::Struct => "struct",
            TokenKind::Import => "import",
            TokenKind::As => "as",
            TokenKind::True => "true",
            TokenKind::False => "false",
            TokenKind::Break => "break",
            TokenKind::Continue => "continue",
            TokenKind::Pass => "pass",
            TokenKind::Pub => "pub",
            TokenKind::LParen => "(",
            TokenKind::RParen => ")",
            TokenKind::LBracket => "[",
            TokenKind::RBracket => "]",
            TokenKind::LBrace => "{",
            TokenKind::RBrace => "}",
            TokenKind::Comma => ",",
            TokenKind::Colon => ":",
            TokenKind::Dot => ".",
            TokenKind::Arrow => "->",
            TokenKind::Semi => ";",
            TokenKind::Amp => "&",
            TokenKind::Pipe => "|",
            TokenKind::Bang => "!",
            TokenKind::At => "@",
            TokenKind::Plus => "+",
            TokenKind::Minus => "-",
            TokenKind::Star => "*",
            TokenKind::Slash => "/",
            TokenKind::Percent => "%",
            TokenKind::StarStar => "**",
            TokenKind::SlashSlash => "//",
            TokenKind::Assign => "=",
            TokenKind::PlusAssign => "+=",
            TokenKind::MinusAssign => "-=",
            TokenKind::StarAssign => "*=",
            TokenKind::SlashAssign => "/=",
            TokenKind::SlashSlashAssign => "//=",
            TokenKind::PercentAssign => "%=",
            TokenKind::StarStarAssign => "**=",
            TokenKind::EqEq => "==",
            TokenKind::NotEq => "!=",
            TokenKind::Lt => "<",
            TokenKind::Le => "<=",
            TokenKind::Gt => ">",
            TokenKind::Ge => ">=",
            TokenKind::AmpAmp => "&&",
            TokenKind::PipePipe => "||",
            TokenKind::Newline => "newline",
            TokenKind::Indent => "indent",
            TokenKind::Dedent => "dedent",
            TokenKind::Eof => "eof",
            TokenKind::Ident(_) => "identifier",
            TokenKind::Int(_) => "integer",
            TokenKind::Float(_) => "float",
            TokenKind::Str(_) => "string",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
    /// Number of leading spaces (only meaningful at line starts).
    pub indent: usize,
}

pub struct Lexer<'a> {
    src: &'a str,
    chars: Vec<char>,
    pos: usize,
    line: usize,
    col: usize,
    indents: Vec<usize>,
    bracket_depth: usize,
    tokens: Vec<Token>,
    diags: Vec<Diagnostic>,
}

impl<'a> Lexer<'a> {
    pub fn new(src: &'a str) -> Lexer<'a> {
        Lexer {
            src,
            chars: src.chars().collect(),
            pos: 0,
            line: 1,
            col: 1,
            indents: vec![0],
            bracket_depth: 0,
            tokens: Vec::new(),
            diags: Vec::new(),
        }
    }

    pub fn lex(mut self) -> (Vec<Token>, Vec<Diagnostic>) {
        self.run();
        (self.tokens, self.diags)
    }

    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }

    fn peek_at(&self, n: usize) -> Option<char> {
        self.chars.get(self.pos + n).copied()
    }

    fn bump(&mut self) -> Option<char> {
        let c = self.chars.get(self.pos).copied()?;
        self.pos += 1;
        if c == '\n' {
            self.line += 1;
            self.col = 1;
        } else {
            self.col += 1;
        }
        Some(c)
    }

    fn span_from(&self, start: usize, line: usize, col: usize) -> Span {
        Span::new(start, self.pos, line, col)
    }

    fn push(&mut self, kind: TokenKind, start: usize, line: usize, col: usize, indent: usize) {
        self.tokens.push(Token {
            kind,
            span: self.span_from(start, line, col),
            indent,
        });
    }

    fn error(&mut self, code: &str, message: impl Into<String>, span: Span) {
        self.diags.push(Diagnostic::error(code, message, Some(span)));
    }

    fn run(&mut self) {
        let mut at_line_start = true;
        let mut line_had_token = false;
        let mut line_indent = 0usize;
        while self.peek().is_some() {
            if at_line_start && self.bracket_depth == 0 {
                // Measure indentation.
                let mut indent = 0usize;
                let line = self.line;
                let col = self.col;
                while let Some(ch) = self.peek() {
                    match ch {
                        ' ' => {
                            indent += 1;
                            self.bump();
                        }
                        '\t' => {
                            indent += 4;
                            self.bump();
                        }
                        '\r' => {
                            self.bump();
                        }
                        _ => break,
                    }
                }
                match self.peek() {
                    None => break,
                    Some('\n') => {
                        self.bump();
                        continue;
                    }
                    Some('#') => {
                        while let Some(ch) = self.peek() {
                            if ch == '\n' {
                                break;
                            }
                            self.bump();
                        }
                        continue;
                    }
                    _ => {}
                }
                line_indent = indent;
                at_line_start = false;
                line_had_token = false;
                let current = *self.indents.last().unwrap();
                if indent > current {
                    self.indents.push(indent);
                    self.push(TokenKind::Indent, self.pos, line, col, indent);
                } else if indent < current {
                    while indent < *self.indents.last().unwrap() {
                        self.indents.pop();
                        self.push(TokenKind::Dedent, self.pos, line, col, indent);
                    }
                    if indent != *self.indents.last().unwrap() {
                        self.error(
                            "E0002",
                            "inconsistent indentation: this line does not match any outer block",
                            Span::new(self.pos, self.pos + 1, line, col),
                        );
                        self.indents.push(indent);
                    }
                }
                if line_had_token {
                    self.push(TokenKind::Newline, self.pos, line, col, indent);
                }
                continue;
            }

            let c = match self.peek() {
                Some(c) => c,
                None => break,
            };
            let start = self.pos;
            let line = self.line;
            let col = self.col;

            match c {
                ' ' | '\t' | '\r' => {
                    self.bump();
                }
                '\n' => {
                    self.bump();
                    if self.bracket_depth == 0 {
                        if line_had_token {
                            self.push(TokenKind::Newline, start, line, col, line_indent);
                        }
                        at_line_start = true;
                        line_had_token = false;
                    }
                }
                '#' => {
                    while let Some(ch) = self.peek() {
                        if ch == '\n' {
                            break;
                        }
                        self.bump();
                    }
                }
                '(' | ')' | '[' | ']' | '{' | '}' => {
                    let kind = match c {
                        '(' => {
                            self.bracket_depth += 1;
                            TokenKind::LParen
                        }
                        ')' => {
                            self.bracket_depth = self.bracket_depth.saturating_sub(1);
                            TokenKind::RParen
                        }
                        '[' => {
                            self.bracket_depth += 1;
                            TokenKind::LBracket
                        }
                        ']' => {
                            self.bracket_depth = self.bracket_depth.saturating_sub(1);
                            TokenKind::RBracket
                        }
                        '{' => {
                            self.bracket_depth += 1;
                            TokenKind::LBrace
                        }
                        _ => {
                            self.bracket_depth = self.bracket_depth.saturating_sub(1);
                            TokenKind::RBrace
                        }
                    };
                    self.bump();
                    self.push(kind, start, line, col, line_indent);
                    line_had_token = true;
                }
                '"' => {
                    self.bump();
                    let mut s = String::new();
                    let mut closed = false;
                    while let Some(ch) = self.bump() {
                        match ch {
                            '"' => {
                                closed = true;
                                break;
                            }
                            '\\' => match self.bump() {
                                Some('n') => s.push('\n'),
                                Some('t') => s.push('\t'),
                                Some('r') => s.push('\r'),
                                Some('\\') => s.push('\\'),
                                Some('"') => s.push('"'),
                                Some(other) => {
                                    s.push('\\');
                                    s.push(other);
                                }
                                None => break,
                            },
                            '\n' => {
                                break;
                            }
                            other => s.push(other),
                        }
                    }
                    if !closed {
                        self.error(
                            "E0003",
                            "unterminated string literal",
                            Span::new(start, self.pos, line, col),
                        );
                    }
                    self.push(TokenKind::Str(s), start, line, col, line_indent);
                    line_had_token = true;
                }
                '0'..='9' => {
                    let mut num = String::new();
                    while let Some(ch) = self.peek() {
                        if ch.is_ascii_digit() {
                            num.push(ch);
                            self.bump();
                        } else {
                            break;
                        }
                    }
                    let mut is_float = false;
                    if self.peek() == Some('.') && self.peek_at(1).map(|c| c.is_ascii_digit()).unwrap_or(false) {
                        is_float = true;
                        num.push('.');
                        self.bump();
                        while let Some(ch) = self.peek() {
                            if ch.is_ascii_digit() {
                                num.push(ch);
                                self.bump();
                            } else {
                                break;
                            }
                        }
                    }
                    if matches!(self.peek(), Some('e') | Some('E')) {
                        let mut exp = String::new();
                        exp.push(self.bump().unwrap());
                        if matches!(self.peek(), Some('+') | Some('-')) {
                            exp.push(self.bump().unwrap());
                        }
                        while let Some(ch) = self.peek() {
                            if ch.is_ascii_digit() {
                                exp.push(ch);
                                self.bump();
                            } else {
                                break;
                            }
                        }
                        num.push_str(&exp);
                        is_float = true;
                    }
                    if is_float {
                        match num.parse::<f64>() {
                            Ok(v) => self.push(TokenKind::Float(v), start, line, col, line_indent),
                            Err(_) => self.error(
                                "E0004",
                                format!("invalid float literal `{}`", num),
                                Span::new(start, self.pos, line, col),
                            ),
                        }
                    } else {
                        match num.parse::<i64>() {
                            Ok(v) => self.push(TokenKind::Int(v), start, line, col, line_indent),
                            Err(_) => self.error(
                                "E0004",
                                format!("integer literal `{}` is out of range for i64", num),
                                Span::new(start, self.pos, line, col),
                            ),
                        }
                    }
                    line_had_token = true;
                }
                c if c == '_' || c.is_alphabetic() => {
                    let mut ident = String::new();
                    while let Some(ch) = self.peek() {
                        if ch == '_' || ch.is_alphanumeric() {
                            ident.push(ch);
                            self.bump();
                        } else {
                            break;
                        }
                    }
                    let kind = match ident.as_str() {
                        "fn" => TokenKind::Fn,
                        "let" => TokenKind::Let,
                        "mut" => TokenKind::Mut,
                        "if" => TokenKind::If,
                        "elif" => TokenKind::Elif,
                        "else" => TokenKind::Else,
                        "while" => TokenKind::While,
                        "for" => TokenKind::For,
                        "in" => TokenKind::In,
                        "return" => TokenKind::Return,
                        "struct" => TokenKind::Struct,
                        "import" => TokenKind::Import,
                        "as" => TokenKind::As,
                        "true" => TokenKind::True,
                        "false" => TokenKind::False,
                        "break" => TokenKind::Break,
                        "continue" => TokenKind::Continue,
                        "pass" => TokenKind::Pass,
                        "pub" => TokenKind::Pub,
                        _ => TokenKind::Ident(ident),
                    };
                    self.push(kind, start, line, col, line_indent);
                    line_had_token = true;
                }
                _ => {
                    let three = (c, self.peek_at(1), self.peek_at(2));
                    let kind3 = match three {
                        ('*', Some('*'), Some('=')) => Some(TokenKind::StarStarAssign),
                        ('/', Some('/'), Some('=')) => Some(TokenKind::SlashSlashAssign),
                        _ => None,
                    };
                    if let Some(kind) = kind3 {
                        self.bump();
                        self.bump();
                        self.bump();
                        self.push(kind, start, line, col, line_indent);
                        line_had_token = true;
                        continue;
                    }
                    let two = (c, self.peek_at(1));
                    let kind = match two {
                        ('-', Some('>')) => Some(TokenKind::Arrow),
                        ('=', Some('=')) => Some(TokenKind::EqEq),
                        ('!', Some('=')) => Some(TokenKind::NotEq),
                        ('<', Some('=')) => Some(TokenKind::Le),
                        ('>', Some('=')) => Some(TokenKind::Ge),
                        ('&', Some('&')) => Some(TokenKind::AmpAmp),
                        ('|', Some('|')) => Some(TokenKind::PipePipe),
                        ('*', Some('*')) => Some(TokenKind::StarStar),
                        ('/', Some('/')) => Some(TokenKind::SlashSlash),
                        ('+', Some('=')) => Some(TokenKind::PlusAssign),
                        ('-', Some('=')) => Some(TokenKind::MinusAssign),
                        ('*', Some('=')) => Some(TokenKind::StarAssign),
                        ('/', Some('=')) => Some(TokenKind::SlashAssign),
                        ('%', Some('=')) => Some(TokenKind::PercentAssign),
                        _ => None,
                    };
                    if let Some(kind) = kind {
                        self.bump();
                        self.bump();
                        self.push(kind, start, line, col, line_indent);
                        line_had_token = true;
                        continue;
                    }
                    let kind = match c {
                        '+' => Some(TokenKind::Plus),
                        '-' => Some(TokenKind::Minus),
                        '*' => Some(TokenKind::Star),
                        '/' => Some(TokenKind::Slash),
                        '%' => Some(TokenKind::Percent),
                        '=' => Some(TokenKind::Assign),
                        '<' => Some(TokenKind::Lt),
                        '>' => Some(TokenKind::Gt),
                        ',' => Some(TokenKind::Comma),
                        ':' => Some(TokenKind::Colon),
                        '.' => Some(TokenKind::Dot),
                        ';' => Some(TokenKind::Semi),
                        '&' => Some(TokenKind::Amp),
                        '|' => Some(TokenKind::Pipe),
                        '!' => Some(TokenKind::Bang),
                        '@' => Some(TokenKind::At),
                        _ => None,
                    };
                    match kind {
                        Some(kind) => {
                            self.bump();
                            self.push(kind, start, line, col, line_indent);
                            line_had_token = true;
                        }
                        None => {
                            self.bump();
                            self.error(
                                "E0001",
                                format!("unexpected character `{}`", c),
                                Span::new(start, self.pos, line, col),
                            );
                        }
                    }
                }
            }
        }
        // Final layout tokens.
        if self.bracket_depth > 0 {
            self.error(
                "E0005",
                "unclosed bracket at end of file",
                Span::new(self.pos, self.pos, self.line, self.col),
            );
        }
        if line_had_token {
            self.push(TokenKind::Newline, self.pos, self.line, self.col, line_indent);
        }
        while self.indents.len() > 1 {
            self.indents.pop();
            self.push(TokenKind::Dedent, self.pos, self.line, self.col, 0);
        }
        self.push(TokenKind::Eof, self.pos, self.line, self.col, 0);
        let _ = self.src;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kinds(src: &str) -> Vec<TokenKind> {
        let (tokens, diags) = Lexer::new(src).lex();
        assert!(diags.is_empty(), "{:?}", diags);
        tokens.into_iter().map(|t| t.kind).collect()
    }

    #[test]
    fn lexes_arithmetic_and_layout() {
        let k = kinds("fn main():\n    x = 1 + 2.5\n");
        assert_eq!(
            &k[..6],
            &[
                TokenKind::Fn,
                TokenKind::Ident("main".into()),
                TokenKind::LParen,
                TokenKind::RParen,
                TokenKind::Colon,
                TokenKind::Newline
            ]
        );
        assert!(k.contains(&TokenKind::Indent));
        assert!(k.contains(&TokenKind::Dedent));
        assert!(k.contains(&TokenKind::Float(2.5)));
    }

    #[test]
    fn multiline_lists_suppress_newlines() {
        let k = kinds("x = [\n  1,\n  2\n]\n");
        assert!(!k[..k.len() - 2].contains(&TokenKind::Newline));
    }

    #[test]
    fn reports_bad_indent() {
        let (_, diags) = Lexer::new("if true:\n    x = 1\n  y = 2\n").lex();
        assert!(diags.iter().any(|d| d.code == "E0002"));
    }

    #[test]
    fn lexes_math_operators() {
        let k = kinds("x **= 2 // 3 ** 4\n");
        assert!(k.contains(&TokenKind::StarStarAssign));
        assert!(k.contains(&TokenKind::SlashSlash));
        assert!(k.contains(&TokenKind::StarStar));
    }

    #[test]
    fn strings_and_comments() {
        let k = kinds("s = \"hi\\n\" # comment\n");
        assert_eq!(k[2], TokenKind::Str("hi\n".into()));
    }
}
