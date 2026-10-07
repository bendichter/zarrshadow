/**
 * The reference files the tests read are written by zarrshadow's Python
 * package (make_fixtures.py), so that the two implementations are held to
 * the same files. ZARRSHADOW_FIXTURES names a folder already written;
 * without it the script is run here, with the interpreter named by PYTHON.
 */
import { execFileSync } from "node:child_process";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

export default function setup(): (() => void) | undefined {
  if (process.env.ZARRSHADOW_FIXTURES) return undefined;
  const folder = mkdtempSync(join(tmpdir(), "zarrshadow-fixtures-"));
  const script = join(dirname(fileURLToPath(import.meta.url)), "make_fixtures.py");
  execFileSync(process.env.PYTHON ?? "python3", [script, folder], { stdio: "inherit" });
  process.env.ZARRSHADOW_FIXTURES = folder;
  return () => rmSync(folder, { recursive: true, force: true });
}
