# COST

Token/$-accounting for the Janus hook path (structural, from session exports
`hardA11.json` / `hardB3.json`; usage fields are redacted in exports, so this
is counted in calls, not tokens — stated, not estimated).

## What a verdict costs

Per failed tool call with the plugin installed:

- 1 hook invocation = 1 ssh round-trip + 1 encoder forward (~600ms CPU warm)
- `additionalContext` appended to the agent context: ~6 class definitions +
  1 `[janus] ... read this as CLASS (confidence X): action` line (~150 tokens)
- No extra model call. No retries. No new dependencies.

## Hard A/B ledger

| | Arm A (Janus ON) | Arm B (Janus OFF) |
|---|---|---|
| tool calls | 30 | 11 |
| hook invocations | 4 | 0 |
| extra context per failure | ~150 tokens × 4 | 0 |
| task completed | yes (4 bugs fixed) | no (app still crashes) |

## Seed-2 ledger (measured block bytes)

Janus verdict blocks measured from session exports `seed2A5_7.json` /
`seed2B6_8.json` (block text bytes ÷ 4 chars/token heuristic, stated):

| | Arm A5 (Janus ON) | Arm B6 (Janus OFF) |
|---|---|---|
| tool calls | 22 | 17 |
| hook invocations | 2 (both warm, ~1.9s) | 0 |
| measured Janus-block bytes | 4509 (~1127 tokens) | 0 |
| per-verdict | ~560 tokens | — |
| task completed | yes | yes |

The earlier ~150-token estimate was low: the full `[janus saw]` state +
`[verdict]` block runs ~560 tokens per failure, not 150. At ~$3/MTok
input pricing that is a fifth of a cent per verdict — the cost story is
unchanged (sub-cent steering), but the number is now measured, not guessed.

## Net

Janus adds ~600 tokens of context and 4 sub-second forwards to a 30-call
session and the task finishes; the no-Janus arm spends nothing extra and
leaves the repo broken. Cost per completed task favors the Janus arm on this
run. Single-run caveat (see AB_HARD.md); token-level accounting needs
unredacted usage fields, currently unavailable in exports.

## Dollar figure (bounded, stated assumptions)

Measured: ~560 input tokens per verdict (seed-2 block bytes). No extra
model call — the verdict rides existing context. At public input pricing
for the serving tier (~$3/MTok):

- per verdict: 560 / 1e6 × $3 ≈ **$0.0017** (a sixth of a cent)
- seed-2 Arm A5 (2 verdicts): ≈ **$0.0034**
- seed-1 Arm A (4 verdicts + ~150-token class defs): ≈ **$0.01**

Upper bound per 30-call session with a failure every third call: 10 × $0.0017
≈ **$0.02**. The model call the verdict *saves* (a wasted retry with full
context, typically 5-50k tokens) costs more than the verdict itself. This is
an estimate from measured bytes + public pricing, not metered billing:
usage fields are redacted in exports, so treat $0.02/session as a ceiling,
not a meter reading.
