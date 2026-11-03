//! NAOE — the Neura Automatic Optimisation Engine.
//!
//! The engine consumes typed EIR and runs a fixed, ordered pipeline:
//!
//! 1. constant folding, algebraic simplification, common subexpressions, DCE
//! 2. elementwise kernel fusion (with bounded CPU autotuning)
//! 3. precision planning (opt-in mixed BF16/FP16)
//! 4. last-use analysis, early drops and buffer-slot planning
//! 5. device placement and dependency-level scheduling
//!
//! Optimisation is enabled by default. `debug` and `off` produce an analysis
//! report without rewriting the program. Every rewrite preserves observable
//! semantics; no pass changes a dtype unless the precision policy asks for it.

pub mod autotuning;
pub mod fusion;
pub mod graph;
pub mod memory;
pub mod precision;
pub mod scheduling;

use autotuning::Autotuner;
use neura_ir::eir::{EirProgram, OptimizeMode};

#[derive(Clone, Debug)]
pub struct PassOutcome {
    pub name: &'static str,
    pub changed: usize,
    pub detail: String,
}

impl PassOutcome {
    fn skipped(name: &'static str, reason: &str) -> PassOutcome {
        PassOutcome { name, changed: 0, detail: reason.to_string() }
    }
}

#[derive(Clone, Debug, Default)]
pub struct TuneSummary {
    pub measured: usize,
    pub fused_chosen: usize,
    pub eager_chosen: usize,
    pub total_eager_ns: u128,
    pub total_fused_ns: u128,
}

#[derive(Clone, Debug)]
pub struct OptimizationReport {
    pub mode: OptimizeMode,
    pub passes: Vec<PassOutcome>,
    pub memory: Vec<memory::MemoryPlan>,
    pub precision: precision::PrecisionPlan,
    pub scheduling: scheduling::SchedulingPlan,
    pub tuning: TuneSummary,
    pub instructions_before: usize,
    pub instructions_after: usize,
}

fn instruction_count(program: &EirProgram) -> usize {
    program.functions.iter().map(|f| f.code.len()).sum()
}

/// Run the NAOE pipeline for the policies attached to the program.
pub fn optimize(program: &mut EirProgram) -> OptimizationReport {
    let mode = program.policies.optimize;
    let before = instruction_count(program);
    let mut passes = Vec::new();
    let mut tuning = TuneSummary::default();

    match mode {
        OptimizeMode::Off => {
            passes.push(PassOutcome::skipped("pipeline", "optimisation disabled (mode=off)"));
        }
        OptimizeMode::Debug => {
            passes.push(PassOutcome::skipped(
                "pipeline",
                "debug mode: analysis only, the program was not rewritten",
            ));
        }
        OptimizeMode::Auto | OptimizeMode::Aggressive => {
            passes.push(graph::fold_constants(program));
            passes.push(graph::simplify_algebraic(program));
            passes.push(graph::eliminate_common_subexpressions(program));
            passes.push(graph::eliminate_dead_code(program));
            let mut tuner = Autotuner::new(if mode == OptimizeMode::Aggressive { 16 } else { 8 });
            let (fuse_pass, decisions) = fusion::fuse_elementwise(program, &mut tuner, true);
            for d in &decisions {
                tuning.measured += d.measured as usize;
                if d.use_fused {
                    tuning.fused_chosen += 1;
                } else {
                    tuning.eager_chosen += 1;
                }
                tuning.total_eager_ns += d.eager_ns;
                tuning.total_fused_ns += d.fused_ns;
            }
            passes.push(fuse_pass);
            passes.push(graph::eliminate_dead_code(program));
        }
    }

    let apply = matches!(mode, OptimizeMode::Auto | OptimizeMode::Aggressive);
    let precision = precision::apply(program, apply);
    let memory = memory::plan(program, apply);
    let scheduling = scheduling::plan(program);

    OptimizationReport {
        mode,
        passes,
        memory,
        precision,
        scheduling,
        tuning,
        instructions_before: before,
        instructions_after: instruction_count(program),
    }
}

fn json_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

impl OptimizationReport {
    pub fn to_json(&self) -> String {
        let mut out = String::from("{\n");
        out.push_str(&format!("  \"mode\": {},\n", json_str(self.mode.name())));
        out.push_str(&format!(
            "  \"instructions\": {{\"before\": {}, \"after\": {}}},\n",
            self.instructions_before, self.instructions_after
        ));
        out.push_str("  \"passes\": [\n");
        for (i, p) in self.passes.iter().enumerate() {
            out.push_str(&format!(
                "    {{\"name\": {}, \"changed\": {}, \"detail\": {}}}{}\n",
                json_str(p.name),
                p.changed,
                json_str(&p.detail),
                if i + 1 == self.passes.len() { "" } else { "," }
            ));
        }
        out.push_str("  ],\n");
        out.push_str("  \"precision\": {\n");
        out.push_str(&format!("    \"policy\": {},\n", json_str(self.precision.policy.name())));
        out.push_str("    \"changes\": [\n");
        for (i, c) in self.precision.changes.iter().enumerate() {
            out.push_str(&format!(
                "      {{\"function\": {}, \"pc\": {}, \"op\": {}, \"from\": {}, \"to\": {}}}{}\n",
                json_str(&c.function),
                c.pc,
                json_str(&c.op),
                json_str(c.from.name()),
                json_str(c.to.name()),
                if i + 1 == self.precision.changes.len() { "" } else { "," }
            ));
        }
        out.push_str("    ],\n");
        out.push_str(&format!(
            "    \"notes\": [{}]\n",
            self.precision
                .notes
                .iter()
                .map(|n| json_str(n))
                .collect::<Vec<_>>()
                .join(", ")
        ));
        out.push_str("  },\n");
        out.push_str("  \"memory\": [\n");
        for (i, m) in self.memory.iter().enumerate() {
            out.push_str(&format!(
                "    {{\"function\": {}, \"tensors\": {}, \"peak_bytes\": {}, \"shared_bytes\": {}, \"drops\": {}}}{}\n",
                json_str(&m.function),
                m.lives(),
                m.peak_bytes,
                m.shared_bytes,
                m.drops_inserted,
                if i + 1 == self.memory.len() { "" } else { "," }
            ));
        }
        out.push_str("  ],\n");
        out.push_str("  \"scheduling\": {\n");
        out.push_str(&format!(
            "    \"requested\": {}, \"gpu_available\": {},\n",
            json_str(self.scheduling.requested),
            self.scheduling.gpu_available
        ));
        out.push_str("    \"placements\": [\n");
        for (i, p) in self.scheduling.placements.iter().enumerate() {
            out.push_str(&format!(
                "      {{\"function\": {}, \"pc\": {}, \"op\": {}, \"device\": {}, \"elements\": {}, \"flops\": {}, \"bytes\": {}, \"reason\": {}}}{}\n",
                json_str(&p.function),
                p.pc,
                json_str(&p.op),
                json_str(p.device),
                p.elements,
                p.flops,
                p.bytes,
                json_str(p.reason),
                if i + 1 == self.scheduling.placements.len() { "" } else { "," }
            ));
        }
        out.push_str("    ],\n");
        out.push_str(&format!(
            "    \"levels\": [{}]\n",
            self.scheduling
                .schedules
                .iter()
                .map(|s| {
                    format!(
                        "{{\"function\": {}, \"width\": {}, \"levels\": {}}}",
                        json_str(&s.function),
                        s.width(),
                        s.levels.len()
                    )
                })
                .collect::<Vec<_>>()
                .join(", ")
        ));
        out.push_str("  },\n");
        out.push_str(&format!(
            "  \"autotuning\": {{\"measured\": {}, \"fused\": {}, \"eager\": {}, \"fused_ns\": {}, \"eager_ns\": {}}}\n",
            self.tuning.measured,
            self.tuning.fused_chosen,
            self.tuning.eager_chosen,
            self.tuning.total_fused_ns,
            self.tuning.total_eager_ns
        ));
        out.push_str("}\n");
        out
    }

    pub fn to_text(&self) -> String {
        let mut out = String::new();
        out.push_str("Neura automatic optimisation report\n");
        out.push_str(&format!("mode: {}\n", self.mode.name()));
        out.push_str(&format!(
            "instructions: {} -> {}\n",
            self.instructions_before, self.instructions_after
        ));
        for p in &self.passes {
            out.push_str(&format!("[{}] {}\n", p.name, p.detail));
        }
        let lives: usize = self.memory.iter().map(|m| m.lives()).sum();
        let peak: usize = self.memory.iter().map(|m| m.peak_bytes).sum();
        let shared: usize = self.memory.iter().map(|m| m.shared_bytes).sum();
        let drops: usize = self.memory.iter().map(|m| m.drops_inserted).sum();
        out.push_str(&format!(
            "memory: {} tensor values, {} live bytes at peak, {} bytes shared by buffer reuse, {} early drops\n",
            lives, peak, shared, drops
        ));
        out.push_str(&format!(
            "precision: {} ({} rewritten operations)\n",
            self.precision.policy.name(),
            self.precision.changes.len()
        ));
        for n in &self.precision.notes {
            out.push_str(&format!("  note: {}\n", n));
        }
        let (cpu, gpu) = self.scheduling.counts();
        out.push_str(&format!(
            "placement: {} tensor operations ({} cpu, {} gpu-planned), gpu runtime {}\n",
            self.scheduling.placements.len(),
            cpu,
            gpu,
            if self.scheduling.gpu_available { "available" } else { "unavailable" }
        ));
        let width = self
            .scheduling
            .schedules
            .iter()
            .map(|s| s.width())
            .max()
            .unwrap_or(0);
        out.push_str(&format!("scheduling: maximum independent width {}\n", width));
        out.push_str(&format!(
            "autotuning: {} measured chains (fused {}, eager {}), fused {}us vs eager {}us\n",
            self.tuning.measured,
            self.tuning.fused_chosen,
            self.tuning.eager_chosen,
            self.tuning.total_fused_ns as f64 / 1000.0,
            self.tuning.total_eager_ns as f64 / 1000.0
        ));
        for n in &self.scheduling.notes {
            out.push_str(&format!("  note: {}\n", n));
        }
        out
    }

    /// Decisions produced by the elementwise autotuner.
    pub fn measured(&self) -> usize {
        self.tuning.measured
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use neura_ir::eir::{
        CompilePolicies, EirFunction, EirLocal, Inst, PrecisionPolicy,
    };
    use neura_ir::ops::{BinOp, TensorOp};
    use neura_ir::types::{Dim, DType, TensorTy, Ty};

    fn tensor_ty(dtype: DType, shape: &[usize]) -> Ty {
        Ty::Tensor(TensorTy::new(
            dtype,
            shape.iter().map(|d| Dim::Static(*d)).collect(),
        ))
    }

    fn local(name: &str, ty: Ty, is_param: bool, needs_autograd: bool) -> EirLocal {
        EirLocal {
            name: name.into(),
            ty,
            mutable: true,
            is_param,
            last_use: None,
            needs_autograd,
        }
    }

    fn program(code: Vec<Inst>, types: Vec<Ty>, locals: Vec<EirLocal>) -> EirProgram {
        EirProgram {
            name: "test".into(),
            functions: vec![EirFunction {
                name: "main".into(),
                params: locals.iter().enumerate().filter(|(_, l)| l.is_param).map(|(i, _)| i as u32).collect(),
                locals,
                types,
                code,
                ret_ty: Ty::Unit,
                spans: vec![],
            }],
            main: 0,
            structs: vec![],
            policies: CompilePolicies::default(),
            source_hash: 1,
            streaming: false,
        }
    }

    #[test]
    fn folds_constants_and_drops_dead_values() {
        let mut p = program(
            vec![
                Inst::ConstInt { dst: 0, v: 2 },
                Inst::ConstInt { dst: 1, v: 3 },
                Inst::BinOp { dst: 2, op: BinOp::Add, lhs: 0, rhs: 1, ty: Ty::Int },
                Inst::Print(2),
                Inst::Return(None),
            ],
            vec![Ty::Int, Ty::Int, Ty::Int],
            vec![],
        );
        let report = optimize(&mut p);
        assert!(report.instructions_after < report.instructions_before);
        assert!(p.functions[0]
            .code
            .iter()
            .any(|i| matches!(i, Inst::ConstInt { v: 5, .. })));
        assert_eq!(report.passes[0].changed, 1);
    }

    #[test]
    fn removes_duplicate_expressions() {
        let t = tensor_ty(DType::F32, &[2, 2]);
        let mut p = program(
            vec![
                Inst::TensorOp {
                    dst: 1,
                    op: TensorOp::ReLU,
                    args: vec![0],
                    dyn_dims: vec![],
                    ty: match &t {
                        Ty::Tensor(tt) => tt.clone(),
                        _ => unreachable!(),
                    },
                },
                Inst::TensorOp {
                    dst: 2,
                    op: TensorOp::ReLU,
                    args: vec![0],
                    dyn_dims: vec![],
                    ty: match &t {
                        Ty::Tensor(tt) => tt.clone(),
                        _ => unreachable!(),
                    },
                },
                Inst::BinOp { dst: 3, op: BinOp::Add, lhs: 1, rhs: 2, ty: Ty::Int },
                Inst::Print(3),
                Inst::Return(None),
            ],
            vec![t.clone(), t.clone(), t.clone(), Ty::Int],
            vec![local("x", t, true, false)],
        );
        let report = optimize(&mut p);
        let cse = report
            .passes
            .iter()
            .find(|p| p.name == "common-subexpression-elimination")
            .unwrap();
        assert!(cse.changed >= 1);
        let relu_ops = p.functions[0]
            .code
            .iter()
            .filter(|i| matches!(i, Inst::TensorOp { op: TensorOp::ReLU, .. }))
            .count();
        assert_eq!(relu_ops, 1);
    }

    #[test]
    fn fuses_elementwise_chains() {
        let t = tensor_ty(DType::F32, &[2, 2]);
        let mut p = program(
            vec![
                Inst::TensorOp {
                    dst: 1,
                    op: TensorOp::ReLU,
                    args: vec![0],
                    dyn_dims: vec![],
                    ty: match &t {
                        Ty::Tensor(tt) => tt.clone(),
                        _ => unreachable!(),
                    },
                },
                Inst::TensorOp {
                    dst: 2,
                    op: TensorOp::Sigmoid,
                    args: vec![1],
                    dyn_dims: vec![],
                    ty: match &t {
                        Ty::Tensor(tt) => tt.clone(),
                        _ => unreachable!(),
                    },
                },
                Inst::Print(2),
                Inst::Return(None),
            ],
            vec![t.clone(), t.clone(), t.clone()],
            vec![
                local("x", t.clone(), true, false),
                local("a", t.clone(), false, false),
                local("b", t, false, false),
            ],
        );
        optimize(&mut p);
        let fused = p.functions[0]
            .code
            .iter()
            .find_map(|i| match i {
                Inst::FusedOp { spec, .. } => Some(spec.expr.to_string()),
                _ => None,
            })
            .expect("no fused kernel was emitted");
        assert_eq!(fused, "sigmoid(relu(%0))");
    }

    #[test]
    fn inserts_early_drops_but_not_for_autograd_values() {
        let t = tensor_ty(DType::F32, &[4]);
        let code = || {
            vec![
                Inst::TensorOp {
                    dst: 1,
                    op: TensorOp::ReLU,
                    args: vec![0],
                    dyn_dims: vec![],
                    ty: match &t {
                        Ty::Tensor(tt) => tt.clone(),
                        _ => unreachable!(),
                    },
                },
                Inst::Print(1),
                Inst::Return(None),
            ]
        };
        let mut p = program(
            code(),
            vec![t.clone(), t.clone()],
            vec![local("x", t.clone(), true, false), local("y", t.clone(), false, false)],
        );
        optimize(&mut p);
        assert!(p.functions[0].code.iter().any(|i| matches!(i, Inst::Drop(1))));

        let mut q = program(
            code(),
            vec![t.clone(), t.clone()],
            vec![local("x", t.clone(), true, false), local("y", t, false, true)],
        );
        optimize(&mut q);
        assert!(!q.functions[0].code.iter().any(|i| matches!(i, Inst::Drop(1))));
    }

    #[test]
    fn mixed_precision_rewrites_matmul_only() {
        let a = tensor_ty(DType::F32, &[2, 3]);
        let b = tensor_ty(DType::F32, &[3, 4]);
        let out = tensor_ty(DType::F32, &[2, 4]);
        let mut p = program(
            vec![
                Inst::TensorOp {
                    dst: 2,
                    op: TensorOp::MatMul,
                    args: vec![0, 1],
                    dyn_dims: vec![],
                    ty: TensorTy::new(DType::F32, vec![Dim::Static(2), Dim::Static(4)]),
                },
                Inst::Print(2),
                Inst::Return(None),
            ],
            vec![a.clone(), b.clone(), out.clone()],
            vec![local("a", a, true, false), local("b", b, true, false)],
        );
        p.policies.precision = PrecisionPolicy::MixedBf16;
        let report = optimize(&mut p);
        assert_eq!(report.precision.changes.len(), 1);
        let dtype = match &p.functions[0].code[0] {
            Inst::TensorOp { ty, .. } => ty.dtype,
            _ => unreachable!(),
        };
        assert_eq!(dtype, DType::BF16);
        assert!(report.to_json().contains("\"mode\""));
        assert!(report.to_text().contains("precision: mixed_bf16"));
    }
}
