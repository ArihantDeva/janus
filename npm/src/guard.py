"""The guard: turn a tool call into an allow / ask / block decision.

This is the application layer, deliberately separate from the model. The model
produces p(fail); this module owns the policy. Keeping them apart is what lets
the same model serve different risk appetites, and it is why the reported
metric is coverage-at-a-risk-budget rather than accuracy alone.

Two rules that are code, not prose:

  * deterministic prechecks run BEFORE the model. A path outside the allowed
    roots, or an obviously destructive command, does not need a probability
    estimate -- and a small model should not be trusted to catch what a string
    comparison gets right every time.
  * the model is only consulted for what the prechecks cannot decide, so the
    measured latency covers the common path.

Nothing here imports torch. The scorer is injected, so the policy is testable
without a GPU and swappable between backends (transformers, llama.cpp HTTP).
"""
import re

import noul

#: Commands that are refused outright. Deliberately a small, explicit list:
#: this is a backstop, not the product.
BLOCK_PATTERNS = (
    # '/' itself, a top-level system directory, or the root glob. The original
    # pattern required '/' to be followed by whitespace or end-of-string, which
    # meant `rm -rf /home`, `rm -rf /usr` and `rm -rf /etc` all passed -- the
    # same destruction as '/' one level down. Verified by
    # test_recursive_delete_of_a_system_root_blocks.
    r"\brm\s+-[a-z]*[rf][a-z]*\s+/(?:\s|$|\*|\.\*)"
    r"|\brm\s+-[a-z]*[rf][a-z]*\s+/(?:bin|boot|dev|etc|home|lib|lib64|opt"
    r"|proc|root|sbin|srv|sys|usr|var(?!\s*/tmp)\b)\b",
    r"\bmkfs(\.\w+)?\b",
    r"\bdd\s+.*\bof=/dev/[sh]d",
    r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;",
    r"\bchmod\s+-R\s+777\s+/(?:\s|$)",
    r">\s*/dev/[sh]d[a-z]",
)

#: Substrings that suggest the command reaches outside the workspace.
ESCAPE_PATTERNS = (
    r"~/\.ssh\b",
    r"\bid_rsa\b",
    r"\bid_ed25519\b",
    r"\.aws/credentials",
    r"\.netrc\b",
)


def precheck(command, allow_roots=None):
    """Return 'block', 'ask', or None (undecided -> consult the model)."""
    cmd = command.strip()
    if not cmd:
        return "ask"
    for pat in BLOCK_PATTERNS:
        if re.search(pat, cmd):
            return "block"
    for pat in ESCAPE_PATTERNS:
        if re.search(pat, cmd):
            return "ask"
    if allow_roots is not None:
        for m in re.finditer(r"(?:^|\s)(/[\w./-]+)", cmd):
            path = m.group(1)
            if not any(path.startswith(r.rstrip("/")) for r in allow_roots):
                return "ask"
    return None


def decide(command, p_fail, hi=0.7, lo=0.3, allow_roots=None):
    """Full decision for one tool call.

    Returns a dict so a caller can log WHY, not just what. A guard that cannot
    explain itself gets disabled by the first person it blocks.
    """
    pre = precheck(command, allow_roots)
    if pre == "block":
        return {"action": "block", "p_fail": None, "source": "precheck",
                "reason": "matches a destructive-command pattern"}
    if pre == "ask":
        return {"action": "ask", "p_fail": None, "source": "precheck",
                "reason": "reaches outside the workspace or is ambiguous"}
    action = noul.decide(p_fail, hi=hi, lo=lo)
    return {"action": action, "p_fail": float(p_fail), "source": "model",
            "reason": f"p(fail)={p_fail:.3f} vs lo={lo} hi={hi}"}


def calibrated_band(p_fail, hi, lo):
    """The action without any precheck -- used for reporting metric parity."""
    return noul.decide(p_fail, hi=hi, lo=lo)
