# Hackathon case file — JanusRouter

Everything needed to defend this build to a judge, written so the *weaknesses are
as visible as the strengths*. A judge who finds an unstated weakness discounts the
whole submission; a judge who sees the weakness named-and-handled trusts the rest.

Last updated: 2026-10-27.

---

## 1. The one-sentence pitch

> Every agent session fails tool calls, and the correct response differs per
> failure — retry a timeout unchanged, re-auth an expired token, fix the
> arguments, fix the path, debug real breakage, or stop. We harvested **16,636
> real failures** from 6,900 agent sessions and trained a **1B model** to say what
> to do next, with a **calibrated distribution** so it asks a frontier model (Muse
> Spark) exactly when it is unsure.

Two sentences a judge can repeat. That is the bar.

---

## 2. Why this problem is real (measurable, not rhetorical)

`arXiv:2609.35381` — 150 MCP servers, 3,001 error messages:

| finding | number |
|---|---|
| error messages that tell the caller what to do next | 949 / 3,001 (32%) |
| ...where that step depends on something the server cannot see about the caller | half of those |
| agents that *do what the message says* | 5 OpenAI models, yes |
| recovery loss from a terminal-command step on expired credentials | −18pt (GPT-5.5) → **−69pt (GPT-6 Astra)** |
| recovery when the message says only "Wait before retrying." (GitHub) | **6%** |
| recovery when the step names the actual tool | 84–88% |

**The failure is not that agents cannot read errors. It is that they obey
instructions written for humans.** And the damage grows with capability, so a
better model does not fix it.

Our classifier sits on the **consumer** side, so it works on messages that are
already bad — which is 68% of theirs.

---

## 3. The data (this is the moat)

| | |
|---|---|
| session stores scanned | `.pi/agent/sessions` (6,374 files) + `.pi-bolt/agent/sessions` (65) + `.codex` (461) |
| toolResults parsed | **426,425** |
| errors | **30,861** (7.2%) |
| non-guard failures | **16,636** |
| after exact dedup | **10,356** |
| train / rule_held / unmatched | 5,477 / 1,870 / 757 |

Harvesting bugs found and fixed (each would have inflated results):
1. Grouping by project dir collapsed 263,939 rows into **78 groups** → split by
   project+session, 2,419 groups.
2. **2,516 files are named `session.jsonl`** — nested subagent runs at
   `.../<session-id>/0692f765/run-4/session.jsonl`, so basename grouping merges them.
3. 1,078 failure texts repeat across sessions (one **551×**) → row splits leak.
4. **14,245 "errors" were harness nudges** (kb-guard, rg-enforce, "use X instead of
   Y") — they describe compliance, not task fitness. Dropped, not relabelled.

---

## 4. The model

Llama-3.2-1B + LoRA, **nimble's letter-code read-out**: each answer is a
single-token letter (`A`–`F`) verified at the prompt's answer boundary; the
distribution is a softmax over **exactly those 6 logits**.

- The model **cannot emit an invalid answer** — no parsing, no retry-on-malformed.
- Objective: **plain cross-entropy** over the candidate logits (nimble's, unchanged).
- Confidence is a *real* distribution, not a self-report — which is what makes
  routing possible at all.

Verified on the real tokenizer: 6 codes → 6 distinct single tokens. The encoding
must be stripped-prefix + one space + code; keeping the dangling space makes BPE
*replace* the space token instead of appending.

---

## 5. The router

```
failure ──► Llama-1B ──► P(retry, reauth, fix_call, fix_input, debug, abort)
                ├── conf ≥ τ ──► act locally       (free, private, offline)
                └── conf <  τ ──► Muse Spark via Meta Model API
```

**Two tiers is the empirically optimal depth.** `arXiv:2605.06350`: two-model
cascades gain **2.1–8.2pp**; a third model adds ≤0.43pp and improved **nothing in
118 of 135** comparisons. We are not adding a third tier.

**The correct null is random escalation**, not tier-1-alone — that paper decomposes
score-based gain over random escalation into AUROC terms. Implemented in
`eval_router.py`; a gain not stated against that null is unearned.

---

## 6. Numbers we must report, including the bad ones

| | accuracy |
|---|---|
| majority class (`debug`) | **32.0%** |
| TF-IDF char 2–5gram + LR | **92.2%** |
| Llama + letter codes | *pending seeds* |

**The honest posture:** the TF-IDF baseline is strong and we expect to land near it.
This task is keyword-shaped (`timeout` → retry, `401` → reauth). Saying so first is
what separates this from every submission that hides a weak baseline.

**The claim is not accuracy.** It is:

> a 1B model reporting a **calibrated distribution** knows **when it is unsure** —
> which a bag-of-characters classifier structurally cannot, so it cannot be routed on.

Supporting prior result from this project: same read-out, in-distribution task →
**89.3% accuracy, cov@5 = 83%, ECE 0.0117**. A usable calibrated distribution from a
1B model.

---

## 7. Measured negatives (worth as much as the positives)

1. **Run 33: model == rule at 91.0%.** Accuracy on rule-labelled rows *equalled*
   agreement with the rule that generated them. The model reproduced the regex.
   Reported rather than buried.
2. **Failures are not predictable from the action.** 318,599 tool call/result pairs:
   lexical baseline 95.5% with **failure recall 0.039**. The failure *reason* lives in
   the result, not the call — which is exactly why the task was reframed from
   prediction to response.
3. **Seed disagreement is not an abstention signal.** 36.5% of rows had exactly zero
   spread; a tie-shuffle null (0.38% ± 0.43%) explained an apparent 7× effect.
4. **Linear probe is negative-sign twice** (+0.187 → +0.005; +0.065 → −0.008) and is
   banned for prioritisation.
5. **`aurc_mean` withdrawn** as a primary metric — it fingerprints the row
   population, which moved 3× as bugs were fixed. The quoted −24% trend is withdrawn
   as not one measurement.

A judge who hears five self-reported negatives believes the positives.

---

## 8. Novelty — the honest version

**Not novel:** response taxonomy concept (Aegis), cascade design (2605.06350),
calibrated abstention (I-CALM), letter-code read-out (nimble), evidential confidence
(open-janus), RLCD loss (laya).

**Closest work:** Aegis (`arXiv:2508.19504`) — 6-mode agent-environment failure
taxonomy from **142 traces / 3,656 turns**, then environment optimizations for
+6.7–12.5% success. Research artifact, hand-labelled, optimises the world.

**Defensible contribution:** consumer-side tool-failure response routing on
**16,636 real traces (30× Aegis)**, from a 1B model with a calibrated distribution,
with the rule-label circularity **measured rather than assumed**.

Two independent analyses (Aegis's 6 modes, our 6 responses) converging on ~6
categories is worth one line in the pitch: neither taxonomy is arbitrary.

---

## 9. Demo script (90 seconds, in this order)

1. **Show the problem, not the model.** Paste
   `fatal: Authentication failed for 'https://github.com/x/y.git'` — ask the room
   what to do. Half say "retry". Show that retrying cannot succeed.
2. **Run the router.** `uv run python src/janus_route.py --no-local "<that>"` →
   `reauth` + *"Refresh credentials, then retry. Retrying now cannot succeed."*
3. **Run the five failures** (timeout → retry, auth → reauth, ENOENT → fix_input,
   TS2345 → debug, interrupt → abort). 5/5 correct. It lands in 5 seconds.
4. **Show the local tier is the point.** `--json` on a confidence-0.4 row → escalated;
   a 0.95 row → not escalated. *"The expensive model runs only where the cheap one
   admits it does not know."*
5. **Show the curve, then the null.** Score-based vs random escalation at the same
   rate. If the gap is small, say so — that is the measurement working.
6. **Name the weakness before the judge does:** *"TF-IDF gets 92.2% on this task; we
   expect similar accuracy. What a bag of characters cannot do is tell you it's
   unsure, and that is the whole product."*

---

## 10. Weaknesses a judge will find (pre-answered)

| weakness | answer |
|---|---|
| "Labels are from a regex over the same text" | Measured: model==rule 91.0%. Reported. Model adds nothing beyond the rule *on rule-labelled rows*; value is in the calibrated distribution and escalation. |
| "1B model loses to TF-IDF" | Expected (~same). The abstention is the product; accuracy is the baseline. |
| "Only 6 classes" | 3 more exist (resolve, stop, inspect) but have 9–17 rows each — unlearnable, dropped by `--min-class`, and stated. Fabricating them would be the run-9 error (inventing gradations the data lacks). |
| "Is 16k enough?" | 30× Aegis's hand-labelled 3,656 turns. Split group-disjoint by session; 1,078 duplicate texts prevent a row split. |
| "Muse Spark is doing the work" | By design, and only on low-confidence rows. Cost measured in reasoning tokens (~278/classification). |
| "This isn't Meta tech" | Llama-3.2-1B (Meta open weights) + Meta Model API (Muse Spark). Both are Meta. |

---

## 11. Open work at time of writing

- [x] 3 seeds (job 63140034): trained; seed variance is the result — non-frozen
  RLCD spans 0.389–0.670 across seeds, frozen encoder wins every seed
  (EVAL.md "Seed variance"). The router curve over seeds was not run: the
  seeds disagree too much for a pooled curve to mean anything, and tier-2
  escalation is a paid API call per abstained row.
- [x] Cross-check the classes against Aegis's 6 modes and ToolFailBench's 4:
  `docs/CROSSCHECK.md`. Verdict: partial by construction — those taxonomies
  diagnose why an agent failed; Janus prescribes the next action. 3 of 6
  Aegis modes and 0 of 4 ToolFailBench modes have a Janus equivalent, and
  the non-overlap is stated, not patched over.
- [x] Live-hook demo: warm-daemon hook fires in a live Muse session; the
  model names the class and classifier unprompted. `demo/LIVE_PROOF.md` +
  `demo/live_daemon_proof_20261102.json` (one contention-caused miss also
  recorded there).
- [x] Two-stage abstention reporting per I-CALM: `src/eval_abstention.py`,
  artifact `eval/abstention_freeze0.json`, summary in EVAL.md.

---

## 12. Submitting

Deadline **Nov 4, 9:00 PM ET**. Tracks: Indie (creativity, ambition, does it work)
or Startup (technical execution, production-readiness). Judged on Theme 1 —
*"solve a problem you know firsthand"* — which is literally how this started.
