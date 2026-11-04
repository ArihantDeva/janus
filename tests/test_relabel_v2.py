"""Invariants of the v2 response dataset.

Each test pins one property of `src/relabel_v2.py`. Where the property is a GUARD
(the rule must refuse to label something), the test does not merely assert the
refusal -- it also reintroduces the exact defect the guard exists to catch and
asserts the guard trips. A guard nobody has seen fire is not a guard.

Reintroduced defects are derived from the v1 rule (`src/responses.py`) rather
than from a hand-written fake, because v1 is where each defect actually shipped.
"""

import json
import os
import re
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import relabel_v2 as R  # noqa: E402
from responses import classify_response as v1_classify  # noqa: E402

V2_DIR = os.path.join(ROOT, "data", "responses_v2")
SPLITS = ("train", "rule_held", "unmatched_held")
#: Every file `build()` returns. `unmatched` is the null-label bucket, not a
#: class, but it is still emitted and must satisfy the same hygiene rules.
BUILT_KEYS = SPLITS + ("unmatched",)
ROW_KEYS = {"x", "y", "rule", "span", "offset", "channel", "tool", "group"}


@pytest.fixture(scope="module")
def built():
    """Build once. `build()` reads ~8000 rows; ~15s once beats it per test."""
    return R.build()


def _row(text, tool="bash", group="g1", y="debug"):
    return {"x": text, "y": y, "tool": tool, "group": group}


def _guard_allows(text):
    return R.classify_row(_row(text))


def _v1_labels(text):
    label, rule = v1_classify(text[:1500])
    return label if rule else None


# ---------------------------------------------------------------------------
# GUARD 1: an injected harness annotation is not the tool's message
# ---------------------------------------------------------------------------

def test_injected_annotation_is_not_evidence():
    """v1's `retry` pattern contains `retrying`. This project injects
    `...retrying unchanged will not help.` into the tool result, so that
    sentence must not be able to label a row."""
    text = ("Traceback (most recent call last):\n"
            "  File \"x.py\", line 1, in <module>\n"
            "TypeError: unsupported operand type(s)\n"
            "[pi-janus] Janus read this as a code_bug failure (confidence 0.99): "
            "fix the code or types; retrying unchanged will not help.")
    got = _guard_allows(text)
    assert got is not None and got["y"] == "debug", got


def test_guard1_fires_reintroducing_the_defect():
    """DEFECT REINTRODUCED: the annotation is present and only it matches."""
    text = ("all good\n"
            "[pi-janus] Janus read this as a code_bug failure (confidence 0.99): "
            "fix the code or types; retrying unchanged will not help.")
    assert _v1_labels(text) == "retry"     # v1: labels off the word `retrying`
    assert _guard_allows(text) is None     # v2: annotation stripped, no evidence


def test_annotation_lines_are_actually_removed():
    text = "boom\n[pi-janus] anything at all\nmore"
    assert "[pi-janus]" not in R.strip_injected(text)
    assert R.strip_injected(text) == "boom\nmore"


# ---------------------------------------------------------------------------
# GUARD 2: generic prose tokens are not evidence
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "retrying", "retrying...", "will retry", "error", "failed", "fail",
    "failure", "exit code", "temporarily", "please try again", "overloaded",
    "500", "503", "429", "403", "401", "error during collection",
])
def test_generic_tokens_alone_do_not_label(text):
    assert _guard_allows(text) is None, text


def test_guard2_fires_reintroducing_the_defect():
    """DEFECT REINTRODUCED: v1 labelled a bare `502` and a bare `failed`.

    `502`, not `500`: v1's patterns were `\\b50[234]\\b` / `\\b429\\b`, so a
    literal `500` matched nothing. Getting this wrong would have made the guard
    look verified when it had never seen the defect.
    """
    for text in ("502", "503", "429", "failed"):
        assert _v1_labels(text) is not None, text
        assert _guard_allows(text) is None, text
    assert _v1_labels("500") is None     # v1 missed this one; noted, not used


def test_generic_span_is_refused_even_if_a_pattern_finds_it():
    """Belt-and-braces: GENERIC_SOLE is checked on the matched span, so a future
    pattern that widens onto a generic word still cannot label a row."""
    assert "retrying" in R.GENERIC_SOLE
    assert "conflict (" in R.GENERIC_SOLE


# ---------------------------------------------------------------------------
# GUARD 3: the positional rule -- evidence must be in the head
# ---------------------------------------------------------------------------

def test_deep_match_in_pasted_log_is_not_evidence():
    """A pasted body quoting an earlier `timed out` 3000 characters in says
    nothing about the failure named at the head."""
    text = ("Validation failed for tool \"read\":\n"
            "  - path: must have required properties path\n"
            + ("filler line\n" * 250)
            + "\n...probe timed out after 30s...\n")
    got = _guard_allows(text)
    assert got is not None and got["y"] == "fix_call", got


def test_guard3_fires_reintroducing_the_defect():
    """DEFECT REINTRODUCED: v1 searched the whole 1500 characters."""
    # 90 filler lines puts the match at offset 664 -- inside v1's 1500-char
    # window. At 300 lines (offset ~2134) v1 truncates to 1500 and matches
    # nothing, which would have made this test pass for the wrong reason.
    text = "plain output\n" + ("filler\n" * 90) + "connection timed out\n"
    assert len(text) < 1500
    assert _v1_labels(text) == "retry"     # v1: match at a deep offset
    assert _guard_allows(text) is None     # v2: nothing in the head


def test_evidence_offset_is_always_inside_the_head():
    got = _guard_allows("file.txt: no such file or directory\n" + "x" * 5000)
    assert got["offset"] < R.HEAD_CHARS


def test_head_sensitivity_is_monotone(built):
    """A bigger head can only label more rows. A non-monotone table would mean
    the rule is order-dependent in a way nobody measured."""
    ks = [k for k, _, _ in built["sensitivity"]]
    labelled = [n for _, n, _ in built["sensitivity"]]
    assert ks == sorted(ks) and labelled == sorted(labelled)


def test_head_chars_is_not_the_whole_row():
    """If HEAD_CHARS were 1500 the positional rule would be v1 again."""
    assert R.HEAD_CHARS < 1500


# ---------------------------------------------------------------------------
# GUARD 4: the status line is a declared channel with one admitted pattern
# ---------------------------------------------------------------------------

def test_status_timed_out_is_evidence():
    got = _guard_allows("some output\n\nCommand timed out after 60 seconds")
    assert got["y"] == "retry" and got["channel"] == "status", got


@pytest.mark.parametrize("status", [
    "Command exited with code 1", "Command exited with code 127",
    "Command aborted",
])
def test_non_timeout_status_is_not_evidence(status):
    assert _guard_allows("some output\n\n" + status) is None, status


def test_guard4_fires_reintroducing_the_defect():
    """DEFECT REINTRODUCED: v1's `exit code [1-9]` called a build log `debug`,
    and its `command aborted` called anything at all `abort`."""
    # `exit code 1`, not `Command exited with code 1`: v1's pattern was
    # `exit code [1-9]`, which the harness phrasing does not contain.
    assert _v1_labels("exit code 1") == "debug"
    assert _guard_allows("exit code 1") is None
    assert _v1_labels("x\n\nCommand exited with code 1") is None  # v1 missed it
    assert _v1_labels("x\n\nCommand aborted") == "abort"
    assert _guard_allows("x\n\nCommand aborted") is None


def test_head_evidence_beats_the_status_channel():
    got = _guard_allows("ENOENT: no such file or directory\n\n"
                        "Command timed out after 30 seconds")
    assert got["y"] == "fix_input" and got["channel"] == "head", got


def test_no_pattern_matches_a_bare_status_code():
    """v1 had `\\b50[234]\\b` and `\\b40[13]\\b`. Any pattern matching a bare
    digit string is a status code wearing a costume."""
    for _, name, pat in R.EVIDENCE:
        assert pat.search("429") is None, name
        assert pat.search("503") is None, name


# ---------------------------------------------------------------------------
# GUARD 5: abort / stop / inspect / resolve are not claimed
# ---------------------------------------------------------------------------

def test_unlabelable_classes_are_absent(built):
    called = {"abort", "stop", "inspect", "resolve"}
    for split in SPLITS:
        assert not called & {r["y"] for r in built[split]}, split


@pytest.mark.parametrize("text", [
    # 48 of the 149 v1 `abort` rows are this: a pytest COLLECTION error.
    "==================== ERRORS ====================\n"
    "ERROR collecting tests/test_acs.py\n"
    "!!!!!!!! Interrupted: 1 error during collection !!!!!!!!",
    # 17 are this: read(2) returning EINTR while walking ~/Library.
    "rg: /Users/x/Library/Saved Application State/foo: "
    "Interrupted system call (os error 4)",
    "Tool budget hard limit reached after 19 tool calls (hard 18).",
])
def test_abort_lookalikes_are_not_labelled(text):
    assert _guard_allows(text) is None


def test_guard5_fires_reintroducing_the_defect():
    """DEFECT REINTRODUCED: v1 called each of these `abort`."""
    for text in (
        "==================== ERRORS ====================\n"
        "ERROR collecting tests/test_acs.py\n"
        "!!!!!!!! Interrupted: 1 error during collection !!!!!!!!",
        "rg: /Users/x/Library/Saved Application State/foo: "
        "Interrupted system call (os error 4)",
    ):
        assert _v1_labels(text) == "abort"
        assert _guard_allows(text) is None


# ---------------------------------------------------------------------------
# GUARD 6: reauth needs a named auth phrase, never a bare status code
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,want", [
    ("EACCES: permission denied, open '/x'", "reauth"),
    ("fatal: Authentication failed for 'https://github.com/x'", "reauth"),
    ("invalid api key", "reauth"),
])
def test_named_auth_phrase_labels_reauth(text, want):
    got = _guard_allows(text)
    assert got is not None and got["y"] == want, got


@pytest.mark.parametrize("text", [
    # the v1 defect: the word `credential` inside a pytest test NAME
    "test_official_comment_generation_requires_channel_credentials\n"
    "monkeypatch = <_pytest.monkeypatch.MonkeyPatch object>",
    # the v1 defect: a bare 403 emitted by a webpage probe
    "PAGE: Deva | Substack\nPROBE: 403 Not authorized",
])
def test_reauth_lookalikes_are_not_labelled(text):
    got = _guard_allows(text)
    assert got is None or got["y"] != "reauth", got


def test_guard6_fires_reintroducing_the_defect():
    """DEFECT REINTRODUCED: v1 labelled a pytest test NAME `reauth` off the bare
    word `credential`."""
    text = ("___________ test_registry_entries_pending_credentials_by_default "
            "___________\n\n    def test_registry_entries_pending_:")
    assert _v1_labels(text) == "reauth"
    assert _guard_allows(text) is None


# ---------------------------------------------------------------------------
# GUARD 7: provenance is recorded, and it is true
# ---------------------------------------------------------------------------

def test_every_labelled_row_has_usable_provenance(built):
    for split in SPLITS:
        for r in built[split]:
            if r["y"] == R.UNMATCHED:
                assert r["rule"] is None and r["span"] is None
                continue
            assert r["rule"] in R._CLASS_OF, (split, r["rule"])
            assert r["span"], (split, r["rule"])
            assert set(r) == ROW_KEYS


def test_recorded_offset_reproduces_the_recorded_span(built):
    """A provenance claim that does not reproduce is worse than none."""
    checked = 0
    for split in SPLITS:
        for r in built[split]:
            if r["channel"] != "head":
                continue
            body = R.message_body(r["x"])
            off = r["offset"]
            assert body[off:off + len(r["span"])] == r["span"], (split, r)
            checked += 1
    assert checked > 100


def test_rule_name_matches_its_class(built):
    for split in SPLITS:
        for r in built[split]:
            if r["rule"]:
                assert R._CLASS_OF[r["rule"]] == r["y"], r


# ---------------------------------------------------------------------------
# GUARD 8: the split is group-disjoint and text-disjoint
# ---------------------------------------------------------------------------

def test_split_is_group_and_text_disjoint(built):
    a = R.audit_splits(built)
    assert a["group_overlap"] == {k: 0 for k in a["group_overlap"]}, a
    assert a["text_overlap"] == {k: 0 for k in a["text_overlap"]}, a


def test_guard8_fires_reintroducing_the_defect(built):
    """DEFECT REINTRODUCED: one train row copied into rule_held -- same group,
    same text. audit_splits must report both."""
    fake = dict(built)
    fake["rule_held"] = built["rule_held"] + [dict(built["train"][0])]
    a = R.audit_splits(fake)
    assert a["group_overlap"]["train^rule_held"] == 1
    assert a["text_overlap"]["train^rule_held"] == 1


def test_v1_split_defect_is_reproduced_by_the_guard(built):
    """Confirm the guard catches the shipped v1 split: rule_held and
    unmatched_held shared 87 group ids and 315 rows."""
    def load(name):
        p = os.path.join(ROOT, "data", "responses", f"{name}.jsonl")
        return [json.loads(l) for l in open(p)] if os.path.exists(p) else []

    rh, uh = load("rule_held"), load("unmatched_held")
    if not rh or not uh:
        pytest.skip("v1 splits absent")
    a = R.audit_splits({"train": [], "unmatched_held": uh,
                        "rule_held": [dict(r, rule=None) for r in rh]})
    assert a["group_overlap"]["rule_held^unmatched_held"] > 0
    assert R.audit_splits(built)["group_overlap"][
        "rule_held^unmatched_held"] == 0


def test_near_twin_removal_fires():
    """DEFECT REINTRODUCED: the same failure with different numbers. Exact-text
    dedup does not catch this; strip_near_twins must."""
    train = [_row("Offset 180 is beyond end of file (166 lines total)")]
    held = [_row("Offset 541 is beyond end of file (520 lines total)")]
    kept, removed = R.strip_near_twins(train, held)
    assert removed == 1 and kept == []


def test_no_unmatched_row_shares_a_group_or_text_with_train(built):
    tr_g = {r["group"] for r in built["train"]}
    tr_t = {R.norm_for_dedup(r["x"]) for r in built["train"]}
    for r in built["unmatched_held"]:
        assert r["group"] not in tr_g
        assert R.norm_for_dedup(r["x"]) not in tr_t
    assert {r["y"] for r in built["unmatched_held"]} == {R.UNMATCHED}


# ---------------------------------------------------------------------------
# GUARD 9: nothing invented, nothing silently relabelled
# ---------------------------------------------------------------------------

def test_no_row_is_invented(built):
    """Every emitted text exists in the v1 corpus, modulo redaction."""
    source = set()
    for name in ("train", "rule_held", "unmatched_held"):
        for r in R.load_v1(name):
            source.add(r["x"])
            source.add(R.redact(r["x"]))
    for split in BUILT_KEYS:
        for r in built[split]:
            assert r["x"] in source, (split, r["x"][:80])


def test_emitted_row_count_does_not_exceed_the_source(built):
    total = sum(len(built[s]) for s in BUILT_KEYS)
    n_src = sum(len(R.load_v1(n))
                for n in ("train", "rule_held", "unmatched_held"))
    assert 0 < total <= n_src


def test_unlabelable_rows_go_to_unmatched_not_to_a_class(built):
    assert R.audit_splits(built)["null_label_readable"]
    for split in ("train", "rule_held"):
        assert R.UNMATCHED not in {r["y"] for r in built[split]}


def test_no_text_carries_two_labels_in_one_split(built):
    for split in SPLITS:
        seen = {}
        for r in built[split]:
            assert seen.setdefault(r["x"], r["y"]) == r["y"]


# ---------------------------------------------------------------------------
# GUARD 10: no secrets, no benchmark content
# ---------------------------------------------------------------------------

def _secret_from_corpus():
    """Pull a real secret-shaped string out of the v1 corpus rather than
    hard-coding one. Measured: train.jsonl holds two live-looking keys of this
    shape, so the redactor has real input to prove itself on."""
    pat = re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}")
    for name in ("train", "rule_held"):
        for r in R.load_v1(name):
            m = pat.search(r["x"])
            if m:
                return m.group(0)
    return None


def test_secret_shaped_strings_are_redacted(built):
    for split in BUILT_KEYS:
        for r in built[split]:
            for pat in R.SECRET_RES:
                assert not pat.search(r["x"]), (split, pat.pattern)


def test_guard10_fires_on_the_shipped_corpus(built):
    """DEFECT REINTRODUCED / PRESENT IN v1: redaction is proven against a real
    key of the shape actually in the v1 corpus."""
    key = _secret_from_corpus()
    if key is None:
        pytest.skip("no secret-shaped string in the v1 corpus")
    assert any(key in r["x"] for r in R.load_v1("train")), \
        "corpus no longer holds the key this test exists for"
    assert key not in R.redact("KEY = " + key), "redact did not fire"
    for split in BUILT_KEYS:
        for r in built[split]:
            assert key not in r["x"], split


#: Markers that only a benchmark ITEM can produce. A bare task-argument mention
#: of a benchmark's name is not leakage; an item payload is.
BENCH_HARD = re.compile(
    r"test_patch|gold_patch|FAIL_TO_PASS|PASS_TO_PASS|\\binstance_id\\b|"
    r"(?:asb|rjudge|safety_(?:train|dev|test))\\.jsonl", re.I)
#: Names, which a user may legitimately type into a task argument.
BENCH_NAME = re.compile(r"swe-?bench|humaneval|mbpp|mmlu|gsm8k|pass@1", re.I)


def test_no_benchmark_item_content(built):
    for split in BUILT_KEYS:
        for r in built[split]:
            assert not BENCH_HARD.search(r["x"]), (split, r["x"][:120])


def test_benchmark_name_mentions_are_counted_not_hidden(built):
    """One row mentions `swe-bench`, inside an agent task argument the user
    typed ("(3) \\"swe-bench\\" or \\"SWE-bench\\""). No item payload, no
    split file, no gold patch. Pinned by COUNT so a second one shows up."""
    found = [(split, r["x"][:90])
             for split in BUILT_KEYS
             for r in built[split] if BENCH_NAME.search(r["x"])]
    assert len(found) <= 1, found


# ---------------------------------------------------------------------------
# Invariants that are measured, not assumed
# ---------------------------------------------------------------------------

def test_tool_only_oracle_is_reported(built):
    a = R.audit_splits(built)
    assert 0.0 < a["tool_only_oracle_train"] <= 1.0


def test_emitted_files_match_a_fresh_build(built):
    """The files on disk are the ones the code produces. A hand-edited file or a
    stale build would otherwise be invisible."""
    for split in BUILT_KEYS:
        p = os.path.join(V2_DIR, f"{split}.jsonl")
        if not os.path.exists(p):
            pytest.skip(f"{p} absent; run src/relabel_v2.py")
        assert [json.loads(l) for l in open(p)] == built[split], split
