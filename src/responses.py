"""Response taxonomy for real tool failures.

THE TASK THIS DEFINES
---------------------
Given the error text an agent just got back, what is the correct RESPONSE? Not
"will this fail" -- run 32 showed that is not learnable from the action at a 4.6%
base rate. This switch is the one an agent actually makes every time something
breaks, and the right answer differs per class:

    retry        the environment was slow. Re-run UNCHANGED. Don't touch anything.
    reauth       the credentials are wrong. Re-authenticate; retrying is futile.
    fix_input    the path/handle/offset does not exist. The INPUT is wrong.
    fix_call     the tool rejected the arguments. The CALL was wrong.
    debug        the action itself is broken. Read the traceback, change the action.
    resolve      a semantic conflict needs a decision (merge, non-fast-forward).
    abort        a human interrupted. Stop, do not retry.
    inspect      non-zero exit with no message. Something happened; go look.

THE CIRCULARITY PROBLEM, STATED PLAINLY
---------------------------------------
These labels are produced by a regex over the error text, and the error text is
also the model's input. So "the model learns the labels" is close to a tautology:
it could be learning my regex, and a high accuracy would prove nothing about the
world.

Three things make that honest instead of fatal:

1. The rule is LESS accurate than a model can be, and its failures are measured.
   Every row records which rule matched, or that none did. On the failure set,
   7377 of 16636 rows (44%) match no rule at all -- they land in `other_fail`.
   The rule is therefore an incomplete teacher, not an oracle.

2. The meaningful test is on the rows the RULE CANNOT LABEL. Training only on
   confidently-labelled rows and evaluating only on unmatched rows asks "did the
   model learn a general notion of response, or did it memorise my patterns?".
   That is a real question with a real chance of a negative answer.

3. A hand check of a sample is required before any number is reported. The rule
   is a hypothesis about what each error means; the sample is how the hypothesis
   gets tested by something other than the rule itself.

So the deliverable is not "a model that reproduces my regex". It is "a taxonomy
for real agent failures, made explicit and auditable, and a measurement of how far
a model generalises past the rule that generated it".
"""

from __future__ import annotations

import re

#: Ordered: first match wins. Order encodes specificity -- a timeout that also
#: mentions a file must be `retry`, because re-running unchanged is the fix.
RULES = [
    ("abort", re.compile(
        r"command aborted|interrupted|SIGINT|user cancel|cancell?ed by user", re.I)),

    ("retry", re.compile(
        r"\btimeout\b|timed out|time out|ETIMEDOUT|ECONNRESET|ECONNREFUSED|"
        r"socket hang up|rate.?limit|\b429\b|\b50[234]\b|will retry|retrying|"
        r"temporar|too many requests|overloaded|please try again|"
        r"network is unreachable|connection reset", re.I)),

    ("reauth", re.compile(
        r"permission denied|unauthori[sz]ed|forbidden|\b401\b|\b403\b|"
        r"not logged in|auth(entication)? (fail|required)|token (file|expired|invalid)|"
        r"invalid api key|access blocked|credential|please (log|sign) in", re.I)),

    ("fix_call", re.compile(
        r"must be of type|required propert|received undefined|validation failed|"
        r"invalid argument|argument must|missing required|unexpected keyword|"
        r"unknown (argument|parameter|option)|unsupported operand|"
        r"regex parse error|invalid regex|bad pattern", re.I)),

    ("fix_input", re.compile(
        r"no such file|ENOENT|does not exist|cannot find|command not found|"
        r"unknown (tool|command)|module not found|\b404\b|offset \d+ is beyond|"
        r"no matches found|not a git repository|unrecognized", re.I)),

    # Edit-tool failures: the CALL was wrong (its oldText did not match, was
    # ambiguous, overlapped, or was a no-op). Found by reading unm_dev only --
    # these were 44% of the unmatched rows and are not environment problems.
    ("fix_call", re.compile(
        r"could not find (the exact text|edits\[)|must be unique|"
        r"provide more context|edits\[\d+\] and edits\[\d+\] overlap|"
        r"replacement produced identical content|no changes made|"
        r"old ?text must match", re.I)),

    # A budget/limit stop is not a failure to retry -- the run must stop.
    ("stop", re.compile(
        r"tool budget hard limit|budget (hard )?limit reached|"
        r"hard limit reached|you have exceeded|max.?turns", re.I)),

    ("resolve", re.compile(
        r"merge conflict|CONFLICT \(|non-fast-forward|rejected.*(fetch first|non-fast)|"
        r"diverged|already exists|duplicate entry|would be overwritten", re.I)),

    ("debug", re.compile(
        r"Traceback|SyntaxError|IndentationError|TypeError|NameError|AttributeError|"
        r"ValueError|KeyError|IndexError|AssertionError|ZeroDivisionError|"
        r"\bException\b|FAILED|assertion|compilation failed|exit code [1-9]", re.I)),

    ("inspect", re.compile(r"^\s*\(no output\)\s*$", re.I | re.M)),
]

#: Matched nothing. Retained as a class rather than discarded, because the
#: unmatched rows are the held-out test set for generalisation.
UNMATCHED = "other_fail"

RESPONSES = tuple(n for n, _ in RULES) + (UNMATCHED,)


def classify_response(text):
    """Return (response, rule_name). rule_name is None when nothing matched.

    Returning the rule that fired is what makes a label auditable: a reader can
    see WHY a row was labelled `retry` without re-deriving it.
    """
    head = (text or "")[:1500]
    for name, pat in RULES:
        if pat.search(head):
            return name, name
    return UNMATCHED, None


def coverage_report(texts):
    """How much of the data the rule can actually label."""
    import collections
    c = collections.Counter(classify_response(t)[0] for t in texts)
    n = sum(c.values())
    matched = n - c.get(UNMATCHED, 0)
    return {"n": n, "matched": matched,
            "coverage": matched / max(n, 1), "counts": dict(c)}
