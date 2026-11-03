// Neura language support for Visual Studio Code.
//
// The extension drives the real `neura` compiler front end
// (`neura check FILE --json`) and maps the structured diagnostics it emits onto
// VS Code diagnostics, so parse / type / ownership errors appear as squiggles,
// in the Problems panel and in the editor gutter.
//
// It also exposes commands to run, build and format the active program by
// invoking the corresponding `neura` subcommands.

import * as vscode from "vscode";
import { spawn } from "child_process";
import * as crypto from "crypto";
import * as fs from "fs";
import * as os from "os";
import * as path from "path";

const LANGUAGE_ID = "neura";
const DIAGNOSTIC_SOURCE = "neura";
const OUTPUT_NAME = "Neura";

/** A single diagnostic as emitted by `neura check --json` (see compiler/src/diagnostics.rs). */
interface NeuraDiagnostic {
  severity: "error" | "warning" | "note";
  code: string;
  message: string;
  span?: { line: number; column: number; start: number; end: number };
  notes?: string[];
  help?: string;
}

interface NeuraCheckResult {
  status: "ok" | "error";
  file: string;
  diagnostics: NeuraDiagnostic[];
}

interface RunResult {
  code: number | null;
  stdout: string;
  stderr: string;
}

export function activate(context: vscode.ExtensionContext): void {
  const collection = vscode.languages.createDiagnosticCollection(DIAGNOSTIC_SOURCE);
  const output = vscode.window.createOutputChannel(OUTPUT_NAME);
  const timers = new Map<string, NodeJS.Timeout>();
  let missingCompilerReported = false;

  context.subscriptions.push(collection, output);

  const report = (line: string): void => {
    output.appendLine(line);
  };

  const warnMissingCompiler = (): void => {
    if (missingCompilerReported) {
      return;
    }
    missingCompilerReported = true;
    void vscode.window
      .showWarningMessage(
        `Neura: could not run the compiler "${compilerConfig()}". ` +
          "Build it with `cargo build -p neura-cli` or set `neura.compilerPath`.",
        "Open Settings"
      )
      .then((choice) => {
        if (choice === "Open Settings") {
          void vscode.commands.executeCommand(
            "workbench.action.openSettings",
            "neura.compilerPath"
          );
        }
      });
  };

  const scheduleCheck = (document: vscode.TextDocument, delay = 300): void => {
    if (document.languageId !== LANGUAGE_ID) {
      return;
    }
    if (!getConfig().get<boolean>("checkOnType", true) && delay > 0) {
      return;
    }
    const key = document.uri.toString();
    const pending = timers.get(key);
    if (pending) {
      clearTimeout(pending);
    }
    const timer = setTimeout(() => {
      timers.delete(key);
      void checkDocument(document, collection, report, warnMissingCompiler);
    }, delay);
    timers.set(key, timer);
  };

  context.subscriptions.push(
    vscode.workspace.onDidOpenTextDocument((document) => scheduleCheck(document, 0)),
    vscode.workspace.onDidSaveTextDocument((document) => {
      if (getConfig().get<boolean>("checkOnSave", true)) {
        scheduleCheck(document, 0);
      }
    }),
    vscode.workspace.onDidChangeTextDocument((event) => scheduleCheck(event.document)),
    vscode.workspace.onDidCloseTextDocument((document) => {
      const key = document.uri.toString();
      const pending = timers.get(key);
      if (pending) {
        clearTimeout(pending);
        timers.delete(key);
      }
      collection.delete(document.uri);
    }),
    vscode.workspace.onDidChangeConfiguration((event) => {
      if (event.affectsConfiguration("neura")) {
        for (const document of vscode.workspace.textDocuments) {
          scheduleCheck(document, 0);
        }
      }
    })
  );

  const activeNeuraDocument = (): vscode.TextDocument | undefined => {
    const document = vscode.window.activeTextEditor?.document;
    if (!document || document.languageId !== LANGUAGE_ID) {
      void vscode.window.showWarningMessage("Neura: open a .neura file first.");
      return undefined;
    }
    return document;
  };

  context.subscriptions.push(
    vscode.commands.registerCommand("neura.check", async () => {
      const document = activeNeuraDocument();
      if (!document) {
        return;
      }
      await checkDocument(document, collection, report, warnMissingCompiler);
      const count = collection.get(document.uri)?.length ?? 0;
      if (count === 0) {
        void vscode.window.showInformationMessage("Neura: no problems found.");
      } else {
        void vscode.window.showWarningMessage(`Neura: ${count} problem(s) found.`);
        output.show(true);
      }
    }),

    vscode.commands.registerCommand("neura.run", async () => {
      const document = activeNeuraDocument();
      if (!document) {
        return;
      }
      const file = await ensureSaved(document);
      if (!file) {
        return;
      }
      // Refresh diagnostics first so compile failures are visible as squiggles.
      await checkDocument(document, collection, report, warnMissingCompiler);
      const args = ["run", file, ...extraArgs()];
      report(`> neura ${args.join(" ")}`);
      const result = await runCompiler(args, cwdFor(document.uri), report, warnMissingCompiler);
      if (!result) {
        return;
      }
      if (result.stdout) {
        output.append(result.stdout.endsWith("\n") ? result.stdout : result.stdout + "\n");
      }
      if (result.stderr) {
        output.append(result.stderr.endsWith("\n") ? result.stderr : result.stderr + "\n");
      }
      report(`[run exited with code ${result.code ?? "?"}]`);
      output.show(true);
      if (result.code !== 0) {
        void vscode.window.showErrorMessage("Neura: run failed — see the Neura output channel.");
      }
    }),

    vscode.commands.registerCommand("neura.build", async () => {
      const document = activeNeuraDocument();
      if (!document) {
        return;
      }
      const file = await ensureSaved(document);
      if (!file) {
        return;
      }
      const args = ["build", file, ...extraArgs()];
      report(`> neura ${args.join(" ")}`);
      const result = await runCompiler(args, cwdFor(document.uri), report, warnMissingCompiler);
      if (!result) {
        return;
      }
      if (result.stdout) {
        output.append(result.stdout.endsWith("\n") ? result.stdout : result.stdout + "\n");
      }
      if (result.stderr) {
        output.append(result.stderr.endsWith("\n") ? result.stderr : result.stderr + "\n");
      }
      output.show(true);
      if (result.code !== 0) {
        void vscode.window.showErrorMessage("Neura: build failed — see the Neura output channel.");
      } else {
        void vscode.window.showInformationMessage(`Neura: built ${path.basename(file, ".neura")}.neurabc`);
      }
    }),

    vscode.commands.registerCommand("neura.format", async () => {
      const document = activeNeuraDocument();
      if (!document) {
        return;
      }
      const file = await ensureSaved(document);
      if (!file) {
        return;
      }
      const args = ["fmt", file];
      report(`> neura ${args.join(" ")}`);
      const result = await runCompiler(args, cwdFor(document.uri), report, warnMissingCompiler);
      if (!result) {
        return;
      }
      if (result.stdout) {
        output.append(result.stdout.endsWith("\n") ? result.stdout : result.stdout + "\n");
      }
      if (result.stderr) {
        output.append(result.stderr.endsWith("\n") ? result.stderr : result.stderr + "\n");
      }
      if (result.code !== 0) {
        output.show(true);
        void vscode.window.showErrorMessage("Neura: format failed — see the Neura output channel.");
        return;
      }
      // `neura fmt` rewrites the file on disk; pull the formatted text back in.
      if (!document.isDirty) {
        await vscode.commands.executeCommand("workbench.action.files.revert", document.uri);
        void vscode.window.showInformationMessage("Neura: file formatted.");
      }
    }),

    vscode.commands.registerCommand("neura.showOutput", () => output.show(true))
  );

  // Check documents that are already open when the extension activates.
  for (const document of vscode.workspace.textDocuments) {
    scheduleCheck(document, 0);
  }

  report("Neura extension active.");
}

export function deactivate(): void {
  // Diagnostics and timers are owned by the extension context and disposed with it.
}

// ---------------------------------------------------------------------------
// Checking
// ---------------------------------------------------------------------------

async function checkDocument(
  document: vscode.TextDocument,
  collection: vscode.DiagnosticCollection,
  report: (line: string) => void,
  onMissingCompiler: () => void
): Promise<void> {
  if (document.languageId !== LANGUAGE_ID) {
    return;
  }

  // `neura check` reads from disk, so unsaved buffers are written to a temp copy.
  let target = document.uri.fsPath;
  let temporary: string | undefined;
  try {
    if (document.isDirty || document.isUntitled) {
      temporary = writeTempSource(document);
      target = temporary;
    }

    const args = ["check", target, "--json", ...extraArgs()];
    report(`> neura ${args.join(" ")}`);
    const result = await runCompiler(args, cwdFor(document.uri), report, onMissingCompiler);
    if (!result) {
      return;
    }

    const parsed = parseJsonObject<NeuraCheckResult>(result.stdout);
    if (!parsed || !Array.isArray(parsed.diagnostics)) {
      collection.delete(document.uri);
      const detail = result.stderr.trim();
      report(
        `[check] no JSON diagnostics on stdout (exit ${result.code ?? "?"}).` +
          (detail ? ` stderr: ${detail}` : "")
      );
      return;
    }

    collection.set(document.uri, parsed.diagnostics.map((d) => toDiagnostic(d, document)));
  } catch (error) {
    collection.delete(document.uri);
    report(`[check] ${String(error)}`);
  } finally {
    if (temporary) {
      try {
        fs.unlinkSync(temporary);
      } catch {
        // Best effort cleanup only.
      }
    }
  }
}

function toDiagnostic(diagnostic: NeuraDiagnostic, document: vscode.TextDocument): vscode.Diagnostic {
  const severity =
    diagnostic.severity === "error"
      ? vscode.DiagnosticSeverity.Error
      : diagnostic.severity === "warning"
        ? vscode.DiagnosticSeverity.Warning
        : vscode.DiagnosticSeverity.Information;

  const lastLine = Math.max(0, document.lineCount - 1);
  const span = diagnostic.span;
  const line = Math.min(Math.max(0, (span?.line ?? 1) - 1), lastLine);
  const lineText = document.lineCount > 0 ? document.lineAt(line).text : "";
  const startChar = Math.min(Math.max(0, (span?.column ?? 1) - 1), lineText.length);
  const width = span && span.end > span.start ? span.end - span.start : 1;
  let endChar = Math.min(startChar + width, lineText.length);
  if (endChar <= startChar && lineText.length > 0) {
    endChar = Math.min(startChar + 1, lineText.length);
  }
  const range = new vscode.Range(line, startChar, line, endChar);

  let message = diagnostic.message;
  if (diagnostic.help) {
    message += `\nhelp: ${diagnostic.help}`;
  }
  if (diagnostic.notes && diagnostic.notes.length > 0) {
    message += "\n" + diagnostic.notes.map((note) => `note: ${note}`).join("\n");
  }

  const result = new vscode.Diagnostic(range, message, severity);
  result.code = diagnostic.code;
  result.source = DIAGNOSTIC_SOURCE;
  return result;
}

// ---------------------------------------------------------------------------
// Compiler process plumbing
// ---------------------------------------------------------------------------

function runCompiler(
  args: string[],
  cwd: string | undefined,
  report: (line: string) => void,
  onMissingCompiler: () => void
): Promise<RunResult | undefined> {
  const binary = resolveCompilerPath();
  return new Promise<RunResult | undefined>((resolve) => {
    let child;
    try {
      child = spawn(binary, args, { cwd, windowsHide: true });
    } catch (error) {
      report(`[compiler] failed to launch "${binary}": ${String(error)}`);
      onMissingCompiler();
      resolve(undefined);
      return;
    }

    let stdout = "";
    let stderr = "";
    let settled = false;
    const finish = (value: RunResult | undefined): void => {
      if (!settled) {
        settled = true;
        resolve(value);
      }
    };

    child.stdout.setEncoding("utf8");
    child.stdout.on("data", (chunk: string) => {
      stdout += chunk;
    });
    child.stderr.setEncoding("utf8");
    child.stderr.on("data", (chunk: string) => {
      stderr += chunk;
    });
    child.on("error", (error: NodeJS.ErrnoException) => {
      report(`[compiler] "${binary}" could not be executed: ${error.message}`);
      if (error.code === "ENOENT") {
        onMissingCompiler();
      }
      finish(undefined);
    });
    child.on("close", (code) => finish({ code, stdout, stderr }));
  });
}

function resolveCompilerPath(): string {
  const configured = compilerConfig();
  const candidates: string[] = [];
  const looksLikePath =
    path.isAbsolute(configured) || configured.includes("/") || configured.includes("\\");
  if (looksLikePath) {
    candidates.push(configured);
  }
  for (const folder of vscode.workspace.workspaceFolders ?? []) {
    candidates.push(path.join(folder.uri.fsPath, "target", "release", executableName()));
    candidates.push(path.join(folder.uri.fsPath, "target", "debug", executableName()));
  }
  candidates.push(configured);
  for (const candidate of candidates) {
    try {
      if (fs.existsSync(candidate) && fs.statSync(candidate).isFile()) {
        return candidate;
      }
    } catch {
      // Ignore probe failures and continue with the next candidate.
    }
  }
  // Fall back to the bare command; the OS resolves it on PATH at spawn time.
  return configured;
}

function executableName(): string {
  return process.platform === "win32" ? "neura.exe" : "neura";
}

function compilerConfig(): string {
  const value = getConfig().get<string>("compilerPath", "neura").trim();
  return value.length > 0 ? value : "neura";
}

function extraArgs(): string[] {
  const args = getConfig().get<string[]>("compilerArgs", []);
  return Array.isArray(args) ? args.filter((a) => typeof a === "string") : [];
}

function getConfig(): vscode.WorkspaceConfiguration {
  return vscode.workspace.getConfiguration("neura");
}

function cwdFor(uri: vscode.Uri): string | undefined {
  const folder = vscode.workspace.getWorkspaceFolder(uri);
  if (folder) {
    return folder.uri.fsPath;
  }
  return vscode.workspace.workspaceFolders?.[0]?.uri.fsPath ?? path.dirname(uri.fsPath);
}

function writeTempSource(document: vscode.TextDocument): string {
  const directory = path.join(os.tmpdir(), "neura-vscode");
  fs.mkdirSync(directory, { recursive: true });
  const digest = crypto.createHash("md5").update(document.uri.toString()).digest("hex").slice(0, 12);
  const base = document.isUntitled
    ? "untitled"
    : path.basename(document.uri.fsPath, path.extname(document.uri.fsPath));
  const file = path.join(directory, `${base || "untitled"}-${digest}.neura`);
  fs.writeFileSync(file, document.getText(), "utf8");
  return file;
}

async function ensureSaved(document: vscode.TextDocument): Promise<string | undefined> {
  if (document.isUntitled) {
    const saved = await document.save();
    if (!saved) {
      void vscode.window.showWarningMessage("Neura: save the file before running this command.");
      return undefined;
    }
  } else if (document.isDirty) {
    await document.save();
  }
  return document.uri.fsPath;
}

/** Extract the last JSON object found on its own line of `text`. */
function parseJsonObject<T>(text: string): T | undefined {
  const lines = text.split(/\r?\n/);
  for (let i = lines.length - 1; i >= 0; i--) {
    const line = lines[i].trim();
    if (!line.startsWith("{")) {
      continue;
    }
    try {
      return JSON.parse(line) as T;
    } catch {
      // Not valid JSON on this line; keep scanning upwards.
    }
  }
  return undefined;
}
