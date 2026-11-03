//! End-to-end tests that drive the real `neura` binary.
//!
//! Each test runs the compiled CLI in a subprocess, so it exercises the whole
//! pipeline (lexer -> parser -> type check -> ownership -> EIR -> NAOE -> VM)
//! exactly as a user would, and checks observable output rather than internal
//! APIs.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicUsize, Ordering};

const BIN: &str = env!("CARGO_BIN_EXE_neura");
static COUNTER: AtomicUsize = AtomicUsize::new(0);

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("cli crate lives inside the workspace")
        .to_path_buf()
}

fn run(args: &[&str]) -> Output {
    Command::new(BIN)
        .args(args)
        .current_dir(workspace_root())
        .output()
        .expect("failed to launch the neura binary")
}

fn run_with_stdin(args: &[&str], stdin: &str) -> Output {
    use std::io::Write;
    use std::process::Stdio;
    let mut child = Command::new(BIN)
        .args(args)
        .current_dir(workspace_root())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("failed to launch the neura binary");
    child
        .stdin
        .as_mut()
        .expect("stdin was not piped")
        .write_all(stdin.as_bytes())
        .expect("failed to write to the neura process");
    child.wait_with_output().expect("failed to wait for the neura binary")
}

fn stdout_of(output: &Output) -> String {
    String::from_utf8_lossy(&output.stdout).to_string()
}

fn combined_of(output: &Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    )
}

fn temp_path(name: &str) -> PathBuf {
    let unique = COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!("neura-e2e-{}-{}", std::process::id(), unique));
    fs::create_dir_all(&dir).expect("cannot create the test directory");
    dir.join(name)
}

fn write_temp(name: &str, contents: &str) -> PathBuf {
    let path = temp_path(name);
    fs::write(&path, contents).unwrap();
    path
}

#[test]
fn required_autograd_example_prints_seven() {
    let output = run(&["run", "examples/autograd.neura"]);
    assert!(output.status.success(), "{}", combined_of(&output));
    assert_eq!(stdout_of(&output).trim(), "Tensor[1](7.0)");
}

#[test]
fn tensor_ops_produce_expected_values() {
    let output = run(&["run", "examples/tensors.neura"]);
    assert!(output.status.success(), "{}", combined_of(&output));
    let text = stdout_of(&output);
    assert!(text.contains("Tensor[2, 2](7.0, 10.0, 15.0, 22.0)"), "{}", text);
    assert!(text.contains("[2, 2]"), "{}", text);
    assert!(text.contains("Tensor(10.0)"), "{}", text);
    assert!(text.contains("Tensor(2.5)"), "{}", text);
}

#[test]
fn classifier_trains_saves_and_reloads() {
    let csv = workspace_root().join("examples/datasets/blobs.csv");
    let model = temp_path("blobs.neuramodel");
    let program = format!(
        "import neura.nn as nn\n\
         import neura.optim as optim\n\
         import neura.data as data\n\
         \n\
         fn main():\n\
         \x20   train = data.load_csv(\"{csv}\")\n\
         \x20   model = nn.Sequential([nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 3)])\n\
         \x20   optimizer = optim.Adam(model.parameters(), learning_rate=0.05)\n\
         \x20   model.train(train, epochs=10, batch_size=8, optimizer=optimizer)\n\
         \x20   sample = Tensor([[0.0, 0.0], [2.0, 2.0]])\n\
         \x20   print(model.predict(sample))\n\
         \x20   model.save(\"{model}\")\n\
         \x20   restored = nn.Sequential([nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 3)])\n\
         \x20   restored.load(\"{model}\")\n\
         \x20   print(restored.predict(sample))\n",
        csv = csv.display(),
        model = model.display(),
    );
    let source = write_temp("train.neura", &program);
    let output = run(&["run", &source.display().to_string()]);
    assert!(output.status.success(), "{}", combined_of(&output));

    let text = stdout_of(&output);
    let lines: Vec<&str> = text.lines().filter(|l| l.starts_with("Tensor")).collect();
    assert_eq!(lines.len(), 2, "expected two prediction lines in:\n{}", text);
    assert_eq!(lines[0], lines[1], "reloaded model must reproduce predictions");
    assert!(model.exists(), "the model file was not written");
    let _ = fs::remove_dir_all(model.parent().unwrap());
}

#[test]
fn build_artifact_runs_through_the_cli() {
    let artifact = temp_path("autograd.neurabc");
    let build = run(&[
        "build",
        "examples/autograd.neura",
        &format!("--output={}", artifact.display()),
    ]);
    assert!(build.status.success(), "{}", combined_of(&build));
    let text = fs::read_to_string(&artifact).expect("artifact was not written");
    assert!(text.starts_with("neurabc 1\n"), "{}", text);
    assert!(text.contains("source-hash "));

    let run_artifact = run(&["run", &artifact.display().to_string()]);
    assert!(run_artifact.status.success(), "{}", combined_of(&run_artifact));
    assert_eq!(stdout_of(&run_artifact).trim(), "Tensor[1](7.0)");
    let _ = fs::remove_dir_all(artifact.parent().unwrap());
}

#[test]
fn input_reads_a_line_from_stdin() {
    let source = write_temp(
        "input.neura",
        "fn main():\n    name = input(\"name: \")\n    print(\"hello, \" + name)\n",
    );
    let output = run_with_stdin(&["run", &source.display().to_string()], "Neura\n");
    assert!(output.status.success(), "{}", combined_of(&output));
    let text = stdout_of(&output);
    assert!(text.lines().last() == Some("name: hello, Neura"), "{}", text);
    let _ = fs::remove_dir_all(source.parent().unwrap());
}

#[test]
fn typed_input_parses_int_float_and_bool() {
    let source = write_temp(
        "typed_input.neura",
        "fn main():\n    age = input(\"age: \", Int)\n    ratio = input(Float)\n    ok = input(Bool)\n    name = input(Str)\n    print(age + 1)\n    print(ratio * 2.0)\n    print(ok)\n    print(name)\n",
    );
    let output = run_with_stdin(
        &["run", &source.display().to_string()],
        "41\n2.5\nTRUE\nNeura\n",
    );
    assert!(output.status.success(), "{}", combined_of(&output));
    let text = stdout_of(&output);
    assert_eq!(
        text.lines().collect::<Vec<_>>(),
        vec!["age: 42", "5.0", "true", "Neura"],
        "{}",
        text
    );
    let _ = fs::remove_dir_all(source.parent().unwrap());
}

#[test]
fn typed_input_rejects_values_that_do_not_match_the_type() {
    let source = write_temp(
        "bad_input.neura",
        "fn main():\n    n = input(Int)\n    print(n)\n",
    );
    let output = run_with_stdin(&["run", &source.display().to_string()], "abc\n");
    assert_eq!(output.status.code(), Some(1), "{}", combined_of(&output));
    assert!(
        combined_of(&output).contains("expected an Int input"),
        "{}",
        combined_of(&output)
    );
    let _ = fs::remove_dir_all(source.parent().unwrap());
}

#[test]
fn math_operators_evaluate() {
    let source = write_temp(
        "math.neura",
        "fn main():\n    print(2 ** 10)\n    print(2 ** 3 ** 2)\n    print(-2 ** 2)\n    print(7 // 2)\n    print(-7 // 2)\n    print(7.5 // 2)\n    let mut x = 5\n    x += 3\n    x *= 2\n    x -= 1\n    x //= 3\n    x %= 4\n    x **= 2\n    print(x)\n    s = \"a\"\n    s += \"b\"\n    print(s)\n",
    );
    let output = run(&["run", &source.display().to_string()]);
    assert!(output.status.success(), "{}", combined_of(&output));
    let text = stdout_of(&output);
    assert_eq!(
        text.lines().collect::<Vec<_>>(),
        vec!["1024", "512", "-4", "3", "-4", "3.0", "1", "ab"],
        "{}",
        text
    );
    let _ = fs::remove_dir_all(source.parent().unwrap());
}

#[test]
fn add_ls_streams_sequential_models() {
    let source = write_temp(
        "stream.neura",
        "import neura.nn as nn\n\
         \n\
         add ls\n\
         \n\
         fn main():\n\
         \x20   model = nn.Sequential([nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 1)])\n\
         \x20   sample = Tensor([[1.0, 2.0]])\n\
         \x20   print(model.forward(sample))\n",
    );
    let run_output = run(&["run", &source.display().to_string()]);
    assert!(run_output.status.success(), "{}", combined_of(&run_output));
    assert!(
        stdout_of(&run_output).contains("Tensor[1, 1]"),
        "{}",
        stdout_of(&run_output)
    );

    let profile = run(&["profile", &source.display().to_string()]);
    assert!(profile.status.success(), "{}", combined_of(&profile));
    let text = stdout_of(&profile);
    let steps: Vec<&str> = text.lines().filter(|l| l.starts_with("[stream]")).collect();
    assert_eq!(steps.len(), 3, "expected one stream step per layer in:\n{}", text);
    assert!(steps[0].contains("Linear(2, 4"), "{}", text);
    assert!(steps[2].contains("Linear(4, 1"), "{}", text);
    let _ = fs::remove_dir_all(source.parent().unwrap());
}

#[test]
fn graph_emits_mlir_text_for_tensor_code() {
    let output = run(&["graph", "examples/tensors.neura"]);
    assert!(output.status.success(), "{}", combined_of(&output));
    let text = stdout_of(&output);
    assert!(text.contains("module attributes {neura.version = \"0.1.0\"}"), "{}", text);
    assert!(text.contains("func.func @main"), "{}", text);
    assert!(text.contains("\"neura_tir."), "{}", text);
}

#[test]
fn check_reports_type_errors_with_codes() {
    let source = write_temp("broken.neura", "fn main():\n    print(missing)\n");
    let output = run(&["check", &source.display().to_string()]);
    assert_eq!(output.status.code(), Some(1));
    let text = combined_of(&output);
    assert!(text.contains("error[E"), "{}", text);
    assert!(text.contains("missing"), "{}", text);
    let _ = fs::remove_dir_all(source.parent().unwrap());
}

#[test]
fn fmt_is_idempotent_and_detects_unformatted_files() {
    let source = write_temp(
        "layout.neura",
        "fn main():\n  x=1+2\n  print( x )\n",
    );
    let path = source.display().to_string();

    let check = run(&["fmt", "--check", &path]);
    assert_eq!(check.status.code(), Some(1), "{}", combined_of(&check));

    let format = run(&["fmt", &path]);
    assert!(format.status.success(), "{}", combined_of(&format));

    let check_again = run(&["fmt", "--check", &path]);
    assert!(check_again.status.success(), "{}", combined_of(&check_again));
    let formatted = fs::read_to_string(&source).unwrap();
    assert!(formatted.contains("    x = 1 + 2\n"), "{}", formatted);
    let _ = fs::remove_dir_all(source.parent().unwrap());
}

#[test]
fn optimize_report_is_measured_on_the_real_pipeline() {
    let output = run(&["optimize-report", "examples/train_classifier.neura"]);
    assert!(output.status.success(), "{}", combined_of(&output));
    let text = stdout_of(&output);
    assert!(text.contains("Neura automatic optimisation report"), "{}", text);
    assert!(text.contains("instructions: "), "{}", text);
    assert!(text.contains(" -> "), "{}", text);
}
