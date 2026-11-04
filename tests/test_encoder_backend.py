"""Contract tests for the encoder backend and the Muse hook path.

These use a deliberately tiny synthetic checkpoint (2 layers, 64 hidden) so they run
in seconds and test WIRING, not quality. A tiny checkpoint that answers close to
uniform is the correct test input: it proves the plumbing carries a distribution,
not that the distribution is good.
"""

import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def tiny_ckpt(tmp_path_factory):
    """Tiny offline checkpoint in the trainer's bare-head save layout."""
    from encoder_fixture import save_tiny_checkpoint

    d = tmp_path_factory.mktemp("ckpt")
    save_tiny_checkpoint(d)
    return str(d)


def test_backend_returns_a_valid_distribution(tiny_ckpt, monkeypatch):
    """The whole point of the swappable layer: same contract as janus."""
    monkeypatch.setenv("JANUS_ENCODER", tiny_ckpt)
    monkeypatch.setenv("JANUS_ENCODER_HEAD", os.path.join(tiny_ckpt, "head.pt"))
    import backends as B
    import janus_interface as JI

    b = B.get_backend("janus")
    d = b(JI.build_encoder_input(JI.build_state(tool="bash", error="boom")))
    B.validate(d, "janus")
    assert abs(sum(d.values()) - 1.0) < 1e-6
    assert set(d) == set(JI.CLASSES_3)


def test_backend_builds_from_the_checkpoint_not_a_hardcoded_base(tiny_ckpt,
                                                                monkeypatch):
    """A tiny checkpoint loaded into a 2048-hidden model raises
    `size mismatch for weight: [3,64] vs [3,2048]`. The checkpoint is the authority
    on its own architecture."""
    monkeypatch.setenv("JANUS_ENCODER", tiny_ckpt)
    monkeypatch.setenv("JANUS_ENCODER_HEAD", os.path.join(tiny_ckpt, "head.pt"))
    import backends as B

    import janus_interface as JI
    b = B.get_backend("janus")
    text = JI.build_encoder_input(JI.build_state(tool="bash", error="boom"))
    B.validate(b(text), "janus")


def test_backend_accepts_both_head_layouts(tiny_ckpt, monkeypatch, tmp_path):
    """The trainer writes a bare state_dict; an earlier hand-written checkpoint
    wrapped it as {'head': ...}. A loader that accepts one and crashes on the other is
    a format trap, not a model problem."""
    bare = torch.load(os.path.join(tiny_ckpt, "head.pt"), weights_only=True)
    wrapped = tmp_path / "wrapped.pt"
    torch.save({"head": bare, "classes": ["debug", "fix_input", "fix_call"]}, wrapped)
    monkeypatch.setenv("JANUS_ENCODER", tiny_ckpt)
    monkeypatch.setenv("JANUS_ENCODER_HEAD", str(wrapped))
    import backends as B

    B.get_backend("janus")          # must not raise KeyError('head')


def test_hook_emits_to_stdout_not_stderr(tiny_ckpt):
    """Logging is not delivery: identical JSON on stderr is recorded with exit 0 and
    never reaches the model. Learned the hard way."""
    env = dict(os.environ, JANUS_BACKEND="janus", JANUS_ENCODER=tiny_ckpt,
               JANUS_ENCODER_HEAD=os.path.join(tiny_ckpt, "head.pt"))
    payload = {"hook_event_name": "PostToolUseFailure", "tool_name": "bash",
               "tool_input": {"command": "pytest -q"},
               "error": json.dumps({"content": "E AssertionError: 1 != 2"})}
    r = subprocess.run(
        [sys.executable, os.path.join(ROOT, "src", "muse_hook.py")],
        input=json.dumps(payload), capture_output=True, text=True, env=env,
        cwd=ROOT, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    out = json.loads(r.stdout)
    ctx = out["hookSpecificOutput"]
    assert ctx["hookEventName"] == "PostToolUseFailure"
    assert ctx["additionalContext"]


def test_hook_survives_a_malformed_payload():
    """`error` is a JSON string, not an object, and valid JSON is not a valid
    payload: `[]` once crashed the hook via .get. A hook that raises breaks the
    user's tool call."""
    import muse_hook as M

    for bad in ([], {}, {"error": None}, {"error": "not json"},
                {"tool_name": None, "error": "{}"}):
        try:
            M.failure_text(bad)
        except AttributeError as e:      # .get on a list is the specific old bug
            pytest.fail(f"payload {bad!r} crashed the hook: {e}")


def test_saved_temperature_is_applied_to_the_softmax(tmp_path):
    """Unapplied calibration is a serving claim we cannot make."""
    import json
    import torch
    import encoder_backend as EB
    from encoder_fixture import save_tiny_checkpoint

    model, tok = save_tiny_checkpoint(tmp_path)
    text = "state\nboom"
    encoded = tok([text], return_tensors="pt")
    with torch.inference_mode():
        raw = model(**encoded).float()[0]
    saved = dict(zip(("debug", "fix_input", "fix_call"), raw.softmax(-1).tolist()))
    warm = dict(zip(("debug", "fix_input", "fix_call"),
                    (raw / 0.5).softmax(-1).tolist()))

    calibrated = EB.JanusBackend(encoder=tmp_path, head=tmp_path / "head.pt",
                                   temperature=0.5)
    assert calibrated(text) == pytest.approx(warm, abs=1e-5)
    assert calibrated(text) != pytest.approx(saved, abs=1e-3)


def test_calibration_file_is_read_from_the_saved_temperature(tmp_path):
    import json
    import encoder_backend as EB

    saved = tmp_path / "temperature.json"
    saved.write_text(json.dumps({"T": 0.5276694558560848,
                                 "classes": ["debug", "fix_input", "fix_call"]}))
    assert EB.load_temperature(str(saved)) == pytest.approx(0.5276694558560848)


def test_calibration_rejects_changed_class_order(tmp_path):
    import json
    import encoder_backend as EB

    saved = tmp_path / "temperature.json"
    saved.write_text(json.dumps({"T": 0.5, "classes": ["fix_call", "fix_input", "debug"]}))
    with pytest.raises(ValueError, match="class"):
        EB.load_temperature(str(saved))


def test_absent_calibration_means_no_scaling():
    import encoder_backend as EB

    assert EB.load_temperature(None) is None
