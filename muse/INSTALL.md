# Installing Janus into Muse Code

The submission is **one classifier: Janus** — the Llama RLCD fine-tune, not Janus.
Which classifier answers is decided by one variable, so the fine-tune serves
with no other change.

```sh
mkdir -p .muse && cp ~/jevlab/muse/hooks.json .muse/hooks.json   # then trust the folder
```

User scope: the same `hooks` block in `~/.config/muse/settings.json` with
`"schema_version": 1`.

## Choosing the classifier

| `JANUS_BACKEND` | what answers | cost | notes |
|---|---|---|---|
| `janus` *(default)* | TypeSafe Janus via `api.typesafe.ai` | per call | measured 3/3 on obvious cases, real distributions |
| `janus` *(default for this install)* | Llama RLCD fine-tune (Amarel-backed) | free | the submission; verdicts read `[janus] class (conf)` |
| `frontier` | Muse Spark via Meta Model API | ~280 reasoning tokens/call | label converted to a declared flat distribution |
| `keyword` | substring rules | free | the null — no model at all |

```sh
JANUS_BACKEND=local JANUS_ADAPTER=/path/to/adapter muse exec --trust-workspace "..."
```

## Verify it is live

```sh
muse exec --trust-workspace \
  "Run: cat /nonexistent/zzz.txt. Then state in one sentence whether a failure
   classifier advised you, which response class, and which classifier."
```

Working install answers with a class **and the backend name**:

> "Yes, a failure classifier advised me with response class `fix_input` from classifier `janus`."

## Environment

| var | default | meaning |
|---|---|---|
| `JANUS_BACKEND` | `janus` | which classifier answers |
| `JANUS_ADAPTER` | unset | LoRA adapter dir for `local` |
| `JANUS_THRESHOLD` | `0.60` | below this, the hook states uncertainty instead of naming a class |
| `JANUS_EFFORT` | `low` | `reasoning_effort` for the `frontier` backend |
| `JANUS_HOOK_DISABLE` | unset | make the hook genuinely inert (no call, no spend) |
| `JANUS_ROUTER_HOME` | `~/jevlab` | where the router lives |

## Measured facts (Muse Code 1.4.14, 2026-10-27)

- Event is **`PostToolUseFailure`**; it fires on real tool errors.
- stdin payload: `hook_event_name`, `tool_name`, `tool_input`, `tool_use_id`,
  **`error` (a JSON string, not an object)**, `is_interrupt`, `duration_ms`,
  `session_id`, `turn_id`, `cwd`, `model`, `permission_mode`.
- `$MUSE_SESSION_ID` is exported to the hook.
- **The advisory must go to STDOUT.** The identical JSON on stderr is recorded in
  the session log with `exit_code: 0` and the model never sees it. Logging is not
  delivery — which is why the test asserts on the model's reply.
- Hooks run **outside the sandbox**; the command runs as you.
- No documented hook timeout, so keep it fast.

## Behaviour on uncertainty

Below `JANUS_THRESHOLD` the hook does **not** name a class. It says the classifier is
unsure and warns that a retry may not be safe. Naming a class we do not believe
would be worse than saying so — and the agent retains the decision either way.

## Durable installation (Janus, Amarel-backed)

The installed route keeps the 1B weights on Amarel and reaches them through one
owned Slurm allocation. Nothing in the configuration names a job id, a node or a
scratch run directory: ownership is discovered on each request, so a released
allocation recovers on the next hook event rather than through a watchdog.

```sh
bash scripts/muse_hook_install.sh install .    # merge into .muse/hooks.json
bash scripts/muse_hook_install.sh verify  .    # what would run, and is it present
bash scripts/muse_hook_install.sh disable .    # remove only this hook
```

- Target: `.muse/hooks.json` in the workspace (project scope). Existing events and
  unrelated `PostToolUseFailure` hooks are preserved; the previous file is copied
  to `hooks.json.bak.<timestamp>` before merging.
- Backend: `janus` with `JANUS_ENCODER_TRANSPORT=remote`, i.e. `src/remote_hook.py`.
- Serving checkpoint: `/scratch/ad2416/out_distill/model_freeze0_rlcd` (+ `head.pt`),
  with the saved calibration `out_distill/temperature_freeze0.json` (T = 0.5276694558560848)
  applied to the softmax inside the allocation.
- The warm service only listens on a unix socket inside the allocation's node; the
  login node never dials it directly. `srun` runs the client on that node.
- `JANUS_HOOK_DISABLE=1` still makes the hook inert without constructing a backend.

### What "live" means here

A hook that exits 0 has not necessarily delivered anything. Verify by asking a
fresh Muse session which classifier advised it:

```sh
muse exec --yolo --trust-workspace --json "Run: cat /nonexistent/x.txt
Then say which response class and which classifier name advised you."
```

The reply must name a class and `janus`. If the transport fails, the hook prints
one sanitized line to stderr and emits **no** stdout, so no class is invented.

### Rollback

```sh
bash scripts/muse_hook_install.sh disable .
ssh amarel 'scancel <your-janus-job-id>'    # only the allocation you own
```
