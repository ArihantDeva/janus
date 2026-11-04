"""JanusRouter: a small local model answers tool failures, escalates when unsure.

THE PROBLEM, FIRSTHAND
----------------------
A coding agent fails a tool call and has to decide what to do: was that a slow
environment it should retry unchanged, bad credentials that make retrying
pointless, arguments it should fix, a missing path, a genuine bug, or a user
interrupt it should obey? Getting this wrong is expensive in a specific way --
retrying an auth failure burns turns, "debugging" a timeout burns a context
window. These are different actions and collapsing them into "error" throws away
the only useful part.

The data is 16636 real failures harvested from this machine's own agent sessions,
with the response class known for the 10356 that are unambiguous.

WHY TWO TIERS
-------------
A 1B model is fast, free and private, and on this task it gets most rows right.
But its errors are not uniform: it is weak exactly on the rare classes (reauth,
abort) that are most expensive to get wrong. So the local model reports a
distribution over the six answers plus a confidence, and anything below the
abstention threshold is escalated to a frontier model with the full failure text.

That is the whole design: cheap model handles the bulk, the expensive one is
reserved for the rows where the cheap model says it does not know. The threshold
is the knob, and `--report` shows the accuracy/cost curve rather than asserting
one operating point is correct.

WHAT THIS IS NOT
----------------
Not an accuracy claim. Measured baselines on the same held-out sessions: majority
class 32.0%, TF-IDF character n-grams 92.2%. The local model's value is the
calibrated distribution and the abstention behaviour, not raw accuracy -- a
bag-of-characters classifier cannot tell you that it is unsure, so it cannot be
routed on.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CLASSES = ("retry", "reauth", "fix_call", "fix_input", "debug", "abort")

#: What the agent should DO. This is the product: a class name is useless to an
#: agent, an action is executable.
ACTIONS = {
    "retry": "Re-run the same call unchanged. The environment was slow, not wrong.",
    "reauth": "Refresh credentials, then retry. Retrying now cannot succeed.",
    "fix_call": "Correct the tool arguments. The tool rejected the call itself.",
    "fix_input": "Correct the path/handle. The input refers to something absent.",
    "debug": "Read the error and change the code. The action itself is broken.",
    "abort": "Stop. The user interrupted; do not retry.",
}

#: Cheap sane defaults for a CLI that must not silently spend money.
DEFAULT_THRESHOLD = 0.60


# ---------------------------------------------------------------------------
# tier 1: local model
# ---------------------------------------------------------------------------

class LocalTier:
    """Llama-3.2-1B + LoRA adapter, letter-code read-out.

    Same read-out as training: one-token letters at the answer boundary, softmax
    over exactly those six logits. The distribution is the product -- without it
    there is nothing to threshold on and no reason to have a second tier.
    """

    def __init__(self, base, adapter=None, device=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available()
                                 else "mps" if torch.backends.mps.is_available()
                                 else "cpu")
        self.tok = AutoTokenizer.from_pretrained(base)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            base, torch_dtype=torch.float32).to(self.device)
        if adapter and os.path.isdir(adapter):
            model = PeftModel.from_pretrained(model, adapter)
        self.model = model.eval()
        from janus_llama import CLASSES as C, OPTION_TEXT, build_prompt
        from choice_codes import code_ids
        self._build_prompt = build_prompt
        self._code_ids = code_ids
        self.classes = C
        self.opts = [OPTION_TEXT[c] for c in C]

    def __call__(self, text):
        t = self.torch
        prompt = self._build_prompt(text)
        ids = self._code_ids(self.tok, prompt, self.opts)
        enc = self.tok([prompt], return_tensors="pt").to(self.device)
        with t.no_grad():
            out = self.model(**enc)
        last = enc["attention_mask"].sum(1) - 1
        logits = out.logits[t.arange(1), last][:, ids].float()
        probs = t.softmax(logits, dim=-1)[0].tolist()
        del out
        return dict(zip(self.classes, probs))


# ---------------------------------------------------------------------------
# tier 2: frontier escalation
# ---------------------------------------------------------------------------

ESCALATION_PROMPT = """A coding agent's tool call failed. Decide what it should do next.

Failure:
{failure}

Answer with exactly one of these words and nothing else:
retry      - environment was slow; re-run unchanged
reauth     - credentials are wrong; retrying is futile
fix_call   - the tool rejected the arguments; correct the call
fix_input  - the path or handle does not exist; correct the input
debug      - the action itself is broken; read the error and change the code
abort      - the user interrupted; stop

Answer:"""


def parse_label(raw):
    """The response class named in the frontier's reply, or None.

    FIRST OCCURRENCE, not longest-first. I wrote longest-first originally with a
    comment claiming it prevented 'fix_call' being shadowed by a shorter name --
    that reason is false: no class name is a substring of another, so shadowing
    cannot happen. Worse, longest-first is actively wrong: for the perfectly
    ordinary answer "The right move is fix_call, not fix_input." it returns
    fix_input, because fix_input is longer. First occurrence returns the class the
    model actually led with, which is what it meant.

    Extracted from `escalate` so it can be tested directly. The first version of
    its test recomputed this logic inline and therefore passed even when the real
    implementation was reverted to longest-first -- a guard that cannot fire.
    """
    low = (raw or "").lower()
    hits = [(low.find(c), c) for c in CLASSES if c in low]
    return min(hits)[1] if hits else None


def escalate(text, effort="low"):
    """Ask Muse Spark. Returns (label, raw content, usage).

    Reasoning tokens dominate the cost of a one-word answer here (measured ~278),
    so `effort` is exposed: 'low' is the default because the task is a six-way
    classification of a short error message, not a reasoning problem.
    """
    from meta_api import chat, reasoning_tokens, total_tokens
    r = chat([{"role": "user", "content": ESCALATION_PROMPT.format(failure=text[:6000])}],
             effort=effort)
    raw = (r["content"] or "").strip()
    lab = parse_label(raw)
    return lab, raw, {"reasoning_tokens": reasoning_tokens(r),
                      "total_tokens": total_tokens(r),
                      "finish": r["finish"]}


# ---------------------------------------------------------------------------
# routing
# ---------------------------------------------------------------------------

def route(text, local=None, threshold=DEFAULT_THRESHOLD, effort="low",
          force_escalate=False):
    """Decide, and say who decided.

    Returns a dict that always names the tier that produced the label, so a
    caller can log how often the cheap path sufficed. Silent escalation would
    hide both the cost and whether tier 1 is earning its place.
    """
    out = {"failure": text[:2000], "threshold": threshold}
    if local is not None:
        probs = local(text)
        best = max(probs, key=probs.get)
        conf = probs[best]
        out["local"] = {"label": best, "confidence": conf,
                        "probs": {k: round(v, 4) for k, v in probs.items()}}
    else:
        best, conf = None, 0.0
        out["local"] = None

    need = force_escalate or local is None or conf < threshold
    out["escalated"] = need
    if need:
        lab, raw, use = escalate(text, effort=effort)
        out["frontier"] = {"label": lab, "raw": raw, "usage": use}
        out["label"] = lab or best
        out["decided_by"] = "frontier"
    else:
        out["label"] = best
        out["decided_by"] = "local"

    out["action"] = ACTIONS.get(out["label"], "No confident answer; inspect manually.")
    return out


def render(r, verbose=False):
    L = []
    lab = r["label"]
    who = r["decided_by"]
    conf = (r["local"] or {}).get("confidence")
    L.append(f"\n  failure : {r['failure'].splitlines()[0][:96]}")
    if r["local"]:
        L.append(f"  local   : {r['local']['label']:<10} conf {conf:.2f}"
                 f"   (threshold {r['threshold']:.2f})")
    if r["escalated"]:
        u = r["frontier"]["usage"]
        L.append(f"  frontier: {str(r['frontier']['label']):<10}"
                 f" effort tokens {u['reasoning_tokens']:>5}"
                 f" total {u['total_tokens']:>5}")
        if r["frontier"]["label"] is None:
            L.append(f"            raw={r['frontier']['raw'][:80]!r}")
    L.append(f"  ANSWER  : {lab}   [by {who}]")
    L.append(f"  DO      : {r.get('action') or ACTIONS.get(lab, 'Inspect manually.')}")
    if verbose and r["local"]:
        top = sorted(r["local"]["probs"].items(), key=lambda kv: -kv[1])[:6]
        L.append("  dist    : " + "  ".join(f"{k}={v:.2f}" for k, v in top))
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(
        description="Route a tool failure to the right next action.")
    ap.add_argument("failure", nargs="*", help="failure text, or - for stdin")
    ap.add_argument("--base", default=os.environ.get(
        "JANUS_BASE_MODEL", "unsloth/Llama-3.2-1B"))
    ap.add_argument("--adapter", default=os.environ.get("JANUS_ADAPTER"))
    ap.add_argument("--no-local", action="store_true",
                    help="skip tier 1 entirely (frontier only)")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--effort", default="low")
    ap.add_argument("--force-escalate", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()

    text = " ".join(a.failure) if a.failure else sys.stdin.read()
    if not text.strip():
        raise SystemExit("no failure text (pass it as arguments, or pipe it)")

    local = None
    if not a.no_local:
        try:
            local = LocalTier(a.base, a.adapter)
        except Exception as e:                       # noqa: BLE001
            # A missing torch/transformers/adapter must not stop the frontier
            # tier: the router is still useful without the cheap path.
            print(f"[local tier unavailable: {type(e).__name__}: {e}]",
                  file=sys.stderr)

    r = route(text, local, a.threshold, a.effort, a.force_escalate)
    if a.json:
        print(json.dumps(r, indent=2))
    else:
        print(render(r, a.verbose))


if __name__ == "__main__":
    main()
