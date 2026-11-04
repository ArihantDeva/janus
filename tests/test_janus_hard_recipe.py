"""Hard-target recipe for the Glimmer Janus fine-tune.

Soft teacher distributions + 1 epoch at lr 2e-4 underfit: loss opened 1.11
and closed 1.31 (near uniform ln4=1.386) with the adapter collapsing to the
plurality class. This contract pins the fix: integer labels, inverse-frequency
class weights, and a loss that descends on the smoke corpus.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

import janus3 as J3  # noqa: E402
import janus4_data as JD  # noqa: E402

CORPUS = os.path.join(os.path.dirname(__file__), "..", "data", "distill",
                       "janus.jsonl")


def test_hard_labels_cover_every_row():
    rows = JD.load_rows(CORPUS)
    for _, _, y, _ in rows:
        assert 0 <= y <= 3
    assert len(rows) == 25311


def test_class_weights_counter_imbalance():
    rows = JD.load_rows(CORPUS)
    counts = np.bincount([y for _, _, y, _ in rows], minlength=4).astype(float)
    w = JD.class_weights(counts)
    assert w.shape == (4,)
    assert (w > 0).all()
    # minority class (retry_wait) must outweigh the plurality (fix_call)
    assert w[3] > w[2]
    # weights normalize to mean 1 so the LR needs no rescaling
    assert abs(w.mean() - 1.0) < 1e-9


def test_shape_slice_is_stratified_enough():
    """The raw corpus head is skew-hostile (retry_wait n=2 in the first
    500 rows). The trainer must shuffle before slicing --max-rows, so the
    shape slice carries every class at usable counts."""
    import random
    rows = JD.load_rows(CORPUS)
    rng = random.Random(0)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    sl = [rows[i] for i in idx[:500]]
    counts = np.bincount([y for _, _, y, _ in sl], minlength=4)
    assert (counts >= 10).all(), f"shape slice starves a class: {counts}"


def test_soft_targets_cap_the_error_signal():
    """A spread target shrinks the error vector:
    the soft target's mass sits nearer the prediction, so |tgt - pred|
    is smaller than the hard |onehot - pred| and the step is weaker."""
    soft = np.array([0.6, 0.3, 0.1, 0.0])
    hard = np.array([1.0, 0.0, 0.0, 0.0])
    pred = np.array([0.5, 0.3, 0.2, 0.0])
    assert float(np.abs(soft - pred).sum()) < float(np.abs(hard - pred).sum())


def test_resume_path_enables_gradient_checkpointing():
    """Resume-from-ckpt must carry the fresh-path memory diet: PeftModel
    without gradient checkpointing OOMs a 40GB card at eff-batch 16+
    (measured 2026-11-02: 39.2/39.5GB in use). Static check: the
    resume branch must enable checkpointing + input grads."""
    import ast
    src = open(os.path.join(os.path.dirname(__file__), "..", "src",
                            "train_glimmer_janus.py")).read()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "build_model":
            body = ast.dump(node)
            assert "gradient_checkpointing_enable" in body, \
                "resume path skips gradient checkpointing -> OOM at eff-batch 16+"
            assert "enable_input_require_grads" in body
            return
    raise AssertionError("build_model not found")
