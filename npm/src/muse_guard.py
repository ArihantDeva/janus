"""Muse Code integration: the guard as a tool-call interceptor.

The goal requires a *working* integration, not a description of one. Muse Code
supports `--provider local --base-url <URL>`, so the shape is:

    Muse Code tool call
        -> precheck (deterministic, wins)
        -> llama-server (one forward pass, two logits)
        -> allow / ask / block

Two things this module refuses to do, both for the same reason -- a guard that
silently degrades is worse than no guard:

  * If the scorer raises, the error propagates. It is NOT caught and turned into
    "allow". A crashed guard must not look like a clean bill of health. Callers
    that want a fallback must write it explicitly and say why.
  * If no scorer is supplied, `decide` still runs the precheck and reports
    source="precheck_only" for anything the patterns do not decide, rather than
    inventing a probability.

The scorer is injected, so every test here runs without a GPU, a server, or torch.
"""
import json
import os

import guard as guard_mod


class MuseGuard:
    """Wraps a scorer so one tool call becomes one decision."""

    def __init__(self, scorer=None, hi=0.7, lo=0.3, allow_roots=None):
        self.scorer = scorer
        self.hi = hi
        self.lo = lo
        self.allow_roots = allow_roots

    def inspect(self, command):
        """Return the decision dict for one shell command."""
        pre = guard_mod.precheck(command, self.allow_roots)
        if pre is not None:
            # Deterministic verdicts never reach the model. This ordering is the
            # point: a probability must not overrule a string comparison.
            return {
                "action": pre,
                "p_fail": None,
                "source": "precheck",
                "reason": ("matches a destructive-command pattern"
                           if pre == "block"
                           else "reaches outside the workspace or is ambiguous"),
            }
        if self.scorer is None:
            return {
                "action": "allow",
                "p_fail": None,
                "source": "precheck_only",
                "reason": "no patterns matched and no model configured",
            }
        p = self.scorer.score(command)
        return guard_mod.decide(command, p, hi=self.hi, lo=self.lo,
                                allow_roots=self.allow_roots)

    def to_json(self, decision):
        """One line, so a run log is greppable after the fact."""
        return json.dumps(decision, sort_keys=True)


def base_url_from_env(default="http://127.0.0.1:8080"):
    """Muse Code's --base-url is an OpenAI-compatible root; we append /v1."""
    return os.environ.get("MUSE_GUARD_BASE_URL", default)
