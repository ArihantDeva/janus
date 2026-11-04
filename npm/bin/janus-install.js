#!/usr/bin/env node
// janus install: Muse plugin (validated bundle) + project hook fallback.
// 1. `muse plugins install <bundled plugin>` (user scope) when muse exists.
// 2. Writes <project>/.muse/hooks.json pointing at this package's hook, so the
//    verdict works even without the plugin runtime. Never overwrites an
//    existing hooks.json — prints the merge snippet instead.
import { spawnSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const argv = process.argv.slice(2);
const pi = argv.indexOf("--project");
const project = resolve(pi >= 0 ? argv[pi + 1] : process.cwd());
const pluginDir = join(root, "plugin");

let muse = spawnSync("muse", ["plugins", "validate", pluginDir, "--json"], { encoding: "utf8" });
if (muse.status === 0) {
  const v = JSON.parse(muse.stdout);
  console.log(`plugin valid: ${v.valid} (${v.plugin.id} ${v.plugin.version})`);
  const inst = spawnSync("muse", ["plugins", "install", pluginDir, "--scope", "user", "--json"], { encoding: "utf8" });
  console.log(inst.stdout?.trim() || inst.stderr?.trim() || "plugin install attempted");
  const en = spawnSync("muse", ["plugins", "enable", "janus-failure-lens", "--json"], { encoding: "utf8" });
  console.log(en.stdout?.trim() || "plugin enable attempted");
} else {
  console.log("muse CLI not found or plugin invalid — installing project hook only.");
}

const hookCmd = `JANUS_BACKEND=\${JANUS_BACKEND:-keyword} JANUS_PKG_ROOT="${root}" python3 "${join(pluginDir, "hooks", "janus_hook.py")}"`;
const snippet = {
  hooks: {
    PostToolUseFailure: [
      { matcher: "*", hooks: [{ type: "command", command: hookCmd, timeout: 300 }] },
    ],
  },
};
const destDir = join(project, ".muse");
const dest = join(destDir, "hooks.json");
if (existsSync(dest)) {
  console.log(`\n${dest} exists — merge this hook manually:\n${JSON.stringify(snippet, null, 2)}`);
} else {
  mkdirSync(destDir, { recursive: true });
  writeFileSync(dest, JSON.stringify(snippet, null, 2) + "\n");
  console.log(`wrote ${dest}`);
}
console.log("\nVerify:  echo '{\"tool_name\":\"Bash\",\"tool_input\":{\"command\":\"cat /nope\"},\"error\":\"No such file\"}' | janus hook");
console.log("Guard:   janus guard \"rm -rf /usr\" ; echo exit=$?   (expect block, exit 3)");
