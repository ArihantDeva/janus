# JanusRouter inside Muse Code — integration report

## Status: WORKING, verified end to end

Command run inside a real Muse Code session (`muse exec`), given a tool call that
fails:

> **Model's own reply:** *"Yes, a failure classifier advised me and gave response
> class 'fix_input'."*

That is the whole integration: the hook fires on the failure, the classifier runs,
and the agent receives an actionable class instead of a bare stack of text.

## The hook

Event: **`PostToolUseFailure`** (fires when a tool call fails or crashes).

```json
{
  "hooks": {
    "PostToolUseFailure": [
      { "matcher": "*",
        "hooks": [{ "type": "command", "command": "uv run --project ~/jevlab python ~/jevlab/src/muse_hook.py" }] }
    ]
  }
}
```

Install: `mkdir -p .muse && cp ~/jevlab/muse/hooks.json .muse/hooks.json`, then
trust the folder. User-scope: same `hooks` block in `~/.config/muse/settings.json`
with `"schema_version": 1`.

## What we feed the classifier

Measured stdin payload (Muse Code 1.4.14):

```
hook_event_name  "PostToolUseFailure"       tool_name    "bash"
tool_input       {command, description, workdir}
tool_use_id      "call_..."                  error        JSON STRING (not object)
is_interrupt     false                       duration_ms  67
session_id, turn_id, cwd, transcript_path, model, permission_mode, model_provider
```

The hook builds `[tool: bash] [args: {...}] <error text>` and classifies that —
tool and args are needed for disambiguation (`cat` on a missing path is
`fix_input`; bad flags on a real path is `fix_call`).

## Three facts that cost time to learn

1. **The advisory must go to STDOUT.** Identical JSON on stderr is recorded in the
   session log with `exit_code: 0` — every diagnostic looks green — and the model
   never sees it. *Logging is not delivery.* The integration test therefore asserts
   on the model's reply, not the hook's exit code.
2. **`error` is a JSON string, not an object.** Code assuming a dict crashes and
   the agent silently loses the advisory.
3. **Hooks run outside the sandbox** and there is no documented hook timeout, so a
   slow hook stalls the turn. The frontier path is ~2.5 s; the local tier is much
   faster, which is the cost argument for tiering.

## Safety posture

- The hook **advises, never blocks**. A wrong advisory costs a sentence; blocking a
  turn on a misclassification costs the task.
- **Any internal failure exits 0 with no output**, so the agent behaves exactly as
  it would have without the hook. A router that crashes the turn is worse than none.
- Non-object JSON (`[]`, `5`, `"x"`) is rejected before use — parsing succeeding is
  not the same as a valid payload, and this crashed the hook until a test caught it.
- `JANUS_HOOK_DISABLE` makes it genuinely inert (no classifier call at all, not merely
  a suppressed print), so disabling it also stops the spend.
- User interrupts are not model failures and get no advisory.

## Cost

Frontier escalation burns ~280 reasoning tokens per classification even though the
answer is one word, because Muse Spark is a reasoning model. On a failure-heavy
session that dominates cost — which is precisely why the local 1B tier answers
first and the frontier is reserved for low-confidence rows.

## Tests

`tests/test_janus_route.py` — 21 tests, covering: stdout-only delivery, the
never-crash guarantee on malformed payloads, inertness when disabled, silence on
interrupts, and the error-as-string case. Each guard verified by reintroducing the
defect it prevents.
