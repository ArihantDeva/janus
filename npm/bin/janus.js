#!/usr/bin/env node
// janus guard: one tool call in, one decision out (exit 0 allow / 2 ask / 3 block).
// Spawns the bundled pure-python guard (precheck always; model only with --base-url).
import { spawnSync } from "node:child_process";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const args = process.argv.slice(2);
if (args.includes("--help") || args.includes("-h") || args.length === 0) {
  console.log(`janus — failure verdicts for agent tool calls

Usage:
  janus guard "<command>" [--no-model] [--base-url URL]   exit 0 allow / 2 ask / 3 block
  janus install [--project DIR]                           install the Muse plugin + project hook
  janus hook                                              Muse PostToolUseFailure hook (stdin JSON)

Backends: JANUS_BACKEND=keyword (offline rules) | janus (local encoder daemon) | glimmer (endpoint)`);
  process.exit(args.length === 0 ? 0 : 0);
}
const cmd = args[0];
if (cmd === "install") {
  const r = spawnSync(join(root, "bin", "janus-install.js"), args.slice(1), { stdio: "inherit" });
  process.exit(r.status ?? 1);
}
if (cmd === "hook") {
  const hook = join(root, "plugin", "hooks", "janus_hook.py");
  const r = spawnSync("python3", [hook], { stdio: "inherit", env: { ...process.env, JANUS_PKG_ROOT: root } });
  process.exit(r.status ?? 0);
}
if (cmd === "guard") {
  const rest = args.slice(1);
  const noModel = rest.includes("--no-model") ? ["--no-model"] : [];
  const cmdArgs = rest.filter((a) => a !== "--no-model");
  const py = join(root, "src", "muse_guard_cli.py");
  const r = spawnSync("python3", [py, ...cmdArgs, ...noModel], {
    stdio: "inherit",
    env: { ...process.env, PYTHONPATH: join(root, "src") },
  });
  process.exit(r.status ?? 4);
}
console.error(`janus: unknown command ${cmd} (try: guard | install | hook)`);
process.exit(64);
