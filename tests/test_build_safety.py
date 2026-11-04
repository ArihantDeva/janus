"""The safety dataset must not repeat the label-mismatch that wasted the last run.

Every test here guards one way a label can end up attached to a unit the source
never labelled.
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import build_safety  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "..", "data")


def _rows(name):
    path = os.path.join(DATA, f"safety_{name}.jsonl")
    if not os.path.exists(path):
        pytest.skip(f"{path} absent; run src/build_safety.py")
    return [json.loads(l) for l in open(path)]


# ------------------------------------------------- the granularity contract

def test_linuxarena_is_never_exploded_to_one_row_per_action():
    """Its safety label is side_task_success, a TRAJECTORY property.

    Expanding a trajectory to one row per action and giving every action that
    label marks `ls -R /app` unsafe because the trajectory later did something
    else. That is the original bug wearing a new dataset.
    """
    rows = _rows("train") + _rows("dev") + _rows("test")
    la = [r for r in rows if r["src"] == "linuxarena"]
    assert la, "linuxarena contributed nothing"
    for r in la:
        assert r["level"] == "trajectory", (
            "linuxarena rows must stay trajectory-level")


def test_every_row_declares_the_level_its_label_belongs_to():
    for split in ("train", "dev", "test"):
        for r in _rows(split):
            assert r["level"] in ("action", "trajectory"), r


def test_karanxa_rows_are_action_level():
    kx = [r for r in _rows("train") if r["src"] == "karanxa"]
    assert kx
    assert all(r["level"] == "action" for r in kx)


# ------------------------------------------------------- label plumbing

def test_labels_are_zero_or_one_and_one_means_unsafe():
    for split in ("train", "dev", "test"):
        for r in _rows(split):
            assert r["label"] in (0, 1), r["label"]


def test_unsafe_is_the_minority_or_balanced_not_degenerate():
    """A split that is 99% one class makes accuracy meaningless."""
    for split in ("train", "dev", "test"):
        rows = _rows(split)
        rate = sum(r["label"] for r in rows) / len(rows)
        assert 0.2 <= rate <= 0.8, f"{split} unsafe rate {rate:.2f}"


def test_karanxa_prompt_asks_the_safety_question():
    """The prompt must ask the question the eval asks. Asking "will it FAIL"
    against safety labels is the mismatch that put the first fine-tune below
    the majority-class floor on both benchmarks."""
    rec = {"action": "db.drop_all()", "constraints": [], "context": "",
           "is_safe": False, "domain": "d"}
    p = build_safety.karanxa_prompt(rec)
    assert "UNSAFE" in p
    assert "FAIL" not in p
    assert "db.drop_all()" in p


# ------------------------------------------------------- split integrity

def test_holdout_domains_never_appear_in_train_or_dev():
    """Domain-holdout is the point: a domain in train is a domain the model has
    seen, and arXiv 2610.22243 measured F1=.901 mixed vs .495 held out."""
    held = set(build_safety.HOLDOUT_DOMAINS)
    for split in ("train", "dev"):
        leaked = {r["domain"] for r in _rows(split)} & held
        assert not leaked, f"{split} contains held-out domains: {leaked}"


def test_test_split_is_made_only_of_held_out_domains():
    held = set(build_safety.HOLDOUT_DOMAINS)
    for r in _rows("test"):
        assert r["domain"] in held, r["domain"]


def test_prompts_do_not_cross_splits():
    tr = {r["prompt"] for r in _rows("train")}
    te = {r["prompt"] for r in _rows("test")}
    dv = {r["prompt"] for r in _rows("dev")}
    assert not (tr & te), f"{len(tr & te)} prompts leaked train->test"
    assert not (tr & dv), f"{len(tr & dv)} prompts leaked train->dev"


# ------------------------------------------------------- prompt shape

def test_karanxa_prompt_carries_the_constraints():
    """A constraint is what makes the label decidable. Dropping it invites the
    model to guess a policy instead of reading one."""
    rec = {"action": "db.drop_all()", "context": "ops agent",
           "constraints": ["Never drop tables in production"], "is_safe": False}
    p = build_safety.karanxa_prompt(rec)
    assert "Never drop tables in production" in p
    assert "db.drop_all()" in p


def test_karanxa_prompt_survives_missing_constraints():
    rec = {"action": "ls", "context": "", "constraints": None, "is_safe": True}
    p = build_safety.karanxa_prompt(rec)
    assert "(none stated)" in p


def test_linuxarena_prompt_truncates_long_trajectories_but_says_so():
    rec = {"main_task": "t", "environment": "e",
           "action_normalized": [f"bash:cmd{i}" for i in range(30)],
           "action_arguments_json": ["{}"] * 30}
    p = build_safety.linuxarena_prompt(rec)
    assert "18 more actions" in p


# ------------------------------------------------- corpus selection is explicit

def test_trainer_can_target_the_safety_corpus_without_editing_code():
    """The SWE corpus asks "will it error" and the safety corpus asks "is it
    unsafe". Mixing them silently is what made the first fine-tune score below
    its floor on both safety benchmarks, so the choice must be visible."""
    import pathlib
    src = (pathlib.Path(__file__).resolve().parent.parent
           / "src" / "train_lora.py").read_text()
    assert "JANUS_CORPUS" in src
    assert 'f"{D}/{CORPUS}_{split}.jsonl"' in src
