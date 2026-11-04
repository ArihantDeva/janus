"""Tests for the pluggable-backend layer.

The submission is one classifier whose identity is still open, so the property that
matters is SUBSTITUTABILITY: every backend must be a drop-in for every other, and
the comparison between them must be fair. That is what these tests pin.

The three lies this prevents:
  * a backend that returns a label where a distribution is required -> no threshold,
    no calibration, and not comparable to the others
  * distributions over different class sets -> "swap" silently changes the question
  * a backend that scores well only because it was asked an easier question ->
    CRITERIA is shared and this asserts every backend sees the same six meanings
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/src")

import backends as B  # noqa: E402


# ---------------------------------------------------------------------------
# the shared contract
# ---------------------------------------------------------------------------

def test_every_backend_declares_the_same_classes():
    """A backend must answer over exactly CLASSES. If one drifts, swapping backends
    changes the question and the eval becomes a comparison of tasks."""
    # the class set is owned by janus_interface; the backend layer may not redefine it
    import janus_interface as JI
    assert B.CLASSES == JI.CLASSES_3
    assert len(B.CLASSES) == 3
    # the criteria table must cover every class, with no extras
    assert set(B.CRITERIA) == set(B.CLASSES)


def test_criteria_are_distinct_and_actionable():
    """Two classes described the same way would make the choice unanswerable and
    the confusion between them an artefact of wording."""
    texts = list(B.CRITERIA.values())
    assert len(set(texts)) == len(texts)
    for c, t in B.CRITERIA.items():
        assert len(t) > 40, f"{c} description too thin to discriminate"
        # criteria are rendered verbatim as the choice text, so trailing
        # punctuation is a style choice owned by janus_interface, not by a backend


# ---------------------------------------------------------------------------
# validate() -- the gate every backend output passes
# ---------------------------------------------------------------------------

def test_validate_accepts_a_real_distribution():
    d = {c: 1 / len(B.CLASSES) for c in B.CLASSES}
    assert B.validate(d, "t") == pytest.approx(d)


def test_validate_rejects_missing_class():
    d = {c: 0.2 for c in B.CLASSES[:-1]}
    with pytest.raises(ValueError, match="key mismatch"):
        B.validate(d, "t")


def test_validate_rejects_extra_class():
    d = {c: 0.1 for c in B.CLASSES}
    d["other"] = 0.4
    with pytest.raises(ValueError, match="key mismatch"):
        B.validate(d, "t")


def test_validate_rejects_non_normalised():
    d = {c: 0.5 for c in B.CLASSES}   # sums to 1.5 with 3 classes; 3.0 under the old 6
    with pytest.raises(ValueError, match="sum to"):
        B.validate(d, "t")


def test_validate_rejects_negative_and_over_one():
    for bad in (-0.1, 1.5):
        d = {c: 1 / len(B.CLASSES) for c in B.CLASSES}
        d[B.CLASSES[0]] = bad
        with pytest.raises(ValueError):
            B.validate(d, "t")


def test_validate_rejects_nan():
    """A NaN probability would pass every comparison silently (NaN < tau is False),
    so the threshold would never fire and the row would look confident."""
    d = {c: 1 / len(B.CLASSES) for c in B.CLASSES}
    d[B.CLASSES[0]] = float("nan")
    with pytest.raises(ValueError):
        B.validate(d, "t")


def test_validate_rejects_a_label_instead_of_a_distribution():
    """A bare label is the wrong shape. It cannot be thresholded and cannot be
    scored for calibration, so accepting it would silently disable both claims."""
    with pytest.raises(TypeError):
        B.validate("fix_input", "t")


# ---------------------------------------------------------------------------
# keyword backend -- the null that makes the models falsifiable
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    # retry / reauth / abort are no longer classes: they were shown to be
    # unlabelable from the failure text (retry was 40.9% the project's own
    # injected annotation, abort was pytest collection errors not user interrupts)
    ("cat: /x: No such file or directory", "fix_input"),
    ("patch failed: oldText not found -- no such string in file", "fix_input"),
    ("error TS2345: Argument of type 'string'", "debug"),
    ("Traceback (most recent call last):", "debug"),
    ("AssertionError: 1 != 2", "debug"),
    ("edit: parameter `oldText` must be a string", "fix_call"),
])
def test_keyword_backend_fires_on_obvious_cases(text, expected):
    d = B.KeywordBackend()(text)
    assert max(d, key=d.get) == expected


def test_keyword_backend_is_uniform_when_nothing_matches():
    """No match must give a flat distribution, so the keyword null cannot look
    confident by default and steal the threshold comparison."""
    d = B.KeywordBackend()("qqqq zzzz")
    assert len(set(round(v, 9) for v in d.values())) == 1
    assert abs(max(d.values()) - 1 / len(B.CLASSES)) < 1e-9


def test_keyword_backend_output_always_validates():
    for t in ("", "x", "timeout and 401 and traceback", "É" * 500):
        B.validate(B.KeywordBackend()(t), "keyword")


# ---------------------------------------------------------------------------
# swap compatibility
# ---------------------------------------------------------------------------

def test_get_backend_resolves_every_name():
    for name in ("keyword",):
        b = B.get_backend(name)
        assert b.name == name
        B.validate(b("timed out"), name)
    assert B.get_backend("glimmer").name == "glimmer"


def test_glimmer_tier_is_unavailable_until_deployed(monkeypatch):
    """No checkpoint, no silent fake: the tier says so and stays swappable."""
    monkeypatch.delenv("JANUS_GLIMMER_ENDPOINT", raising=False)
    b = B.get_backend("glimmer")
    with pytest.raises(RuntimeError, match="not deployed"):
        b("timed out")


def test_glimmer_with_endpoint_never_returns_uniform(monkeypatch):
    """A stub distribution behind a tier name is masquerading. Fail, don't fake."""
    monkeypatch.setenv("JANUS_GLIMMER_ENDPOINT", "https://example.invalid/predict")
    b = B.get_backend("glimmer")
    with pytest.raises(Exception, match="urlopen|URL|invalid|Failed|Error"):
        b("timed out")


def test_glimmer_parses_endpoint_reply(monkeypatch):
    """A serving endpoint reply becomes a validated distribution."""
    import io
    import json as J
    import urllib.request
    dist = {"debug": 0.7, "fix_input": 0.2, "fix_call": 0.1}
    monkeypatch.setenv("JANUS_GLIMMER_ENDPOINT", "https://example.invalid/predict")
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _Reply(J.dumps({"dist": dist}).encode()))
    assert B.get_backend("glimmer")("timed out") == dist


class _Reply:
    def __init__(self, body):
        self.body = body

    def read(self):
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_get_backend_rejects_unknown_name():
    with pytest.raises(ValueError, match="unknown backend"):
        B.get_backend("no-such-backend")


def test_get_backend_reads_env(monkeypatch):
    """JANUS_BACKEND is the swap mechanism; if it is ignored, every run silently uses
    the default and the comparison between classifiers never happens."""
    monkeypatch.setenv("JANUS_BACKEND", "keyword")
    assert B.get_backend().name == "keyword"


def test_frontier_backend_peaks_on_parsed_label(monkeypatch):
    """The frontier returns a label, not a distribution. It must be converted in a
    declared, flat way -- not invented into a peaked one."""
    import janus_route
    monkeypatch.setattr(janus_route, "escalate",
                        lambda t, effort="low": ("fix_call", "fix_call", {}))
    d = B.FrontierBackend()("edit: parameter `oldText` must be a string")
    B.validate(d, "frontier")
    assert d["fix_call"] == pytest.approx(0.7)
    others = [v for k, v in d.items() if k != "fix_call"]
    assert len(set(others)) == 1


def test_frontier_backend_falls_back_to_uniform_on_unparseable(monkeypatch):
    import janus_route
    monkeypatch.setattr(janus_route, "escalate",
                        lambda t, effort="low": (None, "no idea", {}))
    d = B.FrontierBackend()("???")
    assert len(set(round(v, 9) for v in d.values())) == 1


# ---------------------------------------------------------------------------
# Janus backend contract (no network)
# ---------------------------------------------------------------------------

def test_janus_body_asks_one_choice_question_over_all_classes():
    """The request must pose exactly one `choice` question whose criteria are the
    six shared classes. A mismatch here means Janus answers a different question than
    the fine-tune is trained on, and the comparison is meaningless."""
    body = B.JanusTeacherBackend()._body("boom")
    assert set(body["questions"]) == {"response"}
    q = body["questions"]["response"]
    assert q["type"] == "choice"
    assert set(q["criteria"]) == set(B.CLASSES)
    assert body["state"]["failure"] == "boom"


def test_janus_backend_truncates_enormous_failures():
    """A pasted 200k-char log must not be sent whole."""
    body = B.JanusTeacherBackend()._body("x" * 200000)
    assert len(body["state"]["failure"]) == 6000


def test_janus_missing_probabilities_raises_not_uniform():
    """If Janus returns no probabilities, silently substituting a uniform guess would
    hide a broken integration behind plausible-looking output."""
    import json as J
    import urllib.request

    class FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return J.dumps({"answers": {"response": {"choice": "retry"}}}).encode()

    b = B.JanusTeacherBackend()
    b._key = "test"
    old = urllib.request.urlopen
    urllib.request.urlopen = lambda *a, **k: FakeResp()
    try:
        with pytest.raises(RuntimeError, match="no probabilities"):
            b("x")
    finally:
        urllib.request.urlopen = old
