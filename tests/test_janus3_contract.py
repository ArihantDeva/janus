"""The 3+1 verdict contract must agree: prompt, target, and read name the same words.

Mirrors test_noul_contract.py for the binary guard. If the prompt binds
"retry_wait" to waiting but the target sends that class to the "debug" slot,
the model is trained to contradict its instructions with no error — so the
binding is pinned here, not trusted.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import janus3 as J3  # noqa: E402


def test_four_words_declared_in_order():
    assert J3.WORDS == ("debug", "fix_input", "fix_call", "retry_wait")


def test_meanings_cover_every_word():
    assert set(J3.MEANINGS) == set(J3.WORDS)
    for w, m in J3.MEANINGS.items():
        assert len(m) > 20, f"{w} meaning too thin to discriminate"


def test_instruction_binds_each_word_to_its_meaning():
    text = J3.INSTRUCTION
    for i, w in enumerate(J3.WORDS):
        assert f"{J3.CODES[i]}) {w}: {J3.MEANINGS[w]}" in text, \
            f"{w} not bound to letter {J3.CODES[i]} in prompt"
    assert "exactly one letter" in text


def test_target_sends_each_class_to_its_own_word():
    cids = {w: i + 100 for i, w in enumerate(J3.WORDS)}
    for i, w in enumerate(J3.WORDS):
        assert J3.target_index(w, cids) == i
    with pytest.raises(ValueError, match="unknown verdict"):
        J3.target_index("nope", cids)


def test_candidates_follow_words_order():
    cids = {c: i + 100 for i, c in enumerate(J3.CODES[:4])}
    assert J3.candidates(cids) == [100, 101, 102, 103]
