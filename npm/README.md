# @arihantdeva/janus

Janus never generates text. It takes **one forward pass** and gives you a
probability over a set of typed decisions — `debug`, `fix_input`, `fix_call` —
for every failed agent tool call. Runtime classifier: **Muse Glimmer 30B**
(QLoRA) on the cluster; local fallback: Llama-3.2-1B frozen-encoder head
(the proof of concept that banked the mechanism: 82.2% teacher agreement,
ECE 0.1027 → 0.0316 after one temperature, n=3410, floor beside every number
in the repo).

## Install (Muse plugin, verifiable)

```sh
npm install -g @arihantdeva/janus
janus install            # muse plugins validate + install + project hook
muse plugins list        # janus-failure-lens present
muse plugins inspect janus-failure-lens
```

## Guard CLI (exit codes, no JSON parsing)

```sh
janus guard "rm -rf /usr" --no-model   # {"action":"block",...} exit 3
janus guard "cat ~/.ssh/id_rsa"        # exit 2 (ask)
janus guard "ls -la" --no-model        # exit 0 (allow)
```

## Hook smoke (offline, keyword backend)

```sh
echo '{"tool_name":"Bash","tool_input":{"command":"cat /nope"},"error":"No such file"}' | janus hook
# {"hookSpecificOutput": {... "[janus] fix_input (0.80) ..."}}
```

Backends via `JANUS_BACKEND`: `keyword` (offline rules, default in npm hook),
`janus` (local encoder daemon), `glimmer` (Glimmer endpoint), `teacher`,
`frontier`. Fail-open: any classifier error exits 0 with no advice — the tool
turn never breaks.

Full repo, evals, and honest negative results: https://github.com/ArihantDeva/janus
