"""Deployed 3-class readers must key on CLASSES_3, never CLASSES.

30d5cc9 extended JI.CLASSES to 4 (retry_wait appended for the Glimmer
4-token port). The deployed encoder head is a 3-way softmax; every reader
that feeds it must use CLASSES_3. A reader over CLASSES silently scores
the wrong column (KeyError or 4-key dist into a 3-way head).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

import janus_interface as JI


def test_deployed_classes_are_three():
    assert JI.CLASSES_3 == ("debug", "fix_input", "fix_call")
    assert JI.CLASSES[:3] == JI.CLASSES_3


def test_distill_targets_follow_deployed_order():
    import train_distill as T

    r = {"teacher": {"fix_call": 0.6, "debug": 0.3, "fix_input": 0.1}}
    v = T.teacher_target(r)
    assert len(v) == len(JI.CLASSES_3)
    assert int(v.argmax()) == JI.CLASSES_3.index("fix_call")


def test_distill_corpus_loader_accepts_deployed_rows(tmp_path):
    import json

    import train_distill as T

    good = {"group": "g1", "tool": "bash", "error": "boom", "arguments": {},
            "teacher": {"debug": 0.9, "fix_input": 0.05, "fix_call": 0.05},
            "teacher_argmax": "debug", "teacher_conf": 0.9}
    bad = {"group": "g2", "tool": "bash", "error": "x",
           "teacher_error": "URLError"}
    p = tmp_path / "c.jsonl"
    p.write_text("\n".join(json.dumps(x) for x in (good, bad)) + "\n")
    rows = T.load_corpus(str(p))
    assert len(rows) == 1 and "teacher" in rows[0]
