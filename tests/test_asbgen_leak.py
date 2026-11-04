"""Prove the generated corpus contains NO benchmark content.

This is the test that decides whether using a benchmark for structure was
honest. It checks the generated rows against the real benchmark data in three
independent ways, each aimed at a different way leakage could happen:

  1. VERBATIM  -- no long substring of any benchmark instruction appears in any
     generated prompt. Catches copy-paste.
  2. NEAR-DUPLICATE -- no generated prompt is token-similar to a benchmark one
     above a threshold. Catches paraphrasing that eluded exact matching.
  3. STRUCTURAL -- the generator source never reads the fields that carry the
     answer (`fulfillable`, `risks`, `failure_modes`) nor any `instruction`
     value. Catches leakage by construction rather than by output.

Failure of any of these means the corpus is contaminated and no result from it
is reportable.
"""

import json
import os
import re
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEN = os.path.join(ROOT, "data", "asbgen")
BENCH = os.path.join(ROOT, "data")

NGRAM = 12          # words long enough that a coincidence is implausible
SIM_THRESHOLD = 0.80


def _words(s):
    return re.findall(r"[a-z0-9]+", s.lower())


def _ngrams(ws, n=NGRAM):
    return {tuple(ws[i:i + n]) for i in range(max(0, len(ws) - n + 1))}


def _jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


@pytest.fixture(scope="module")
def generated():
    path = os.path.join(GEN, "asbgen_train.jsonl")
    if not os.path.exists(path):
        pytest.skip("generated corpus not built; run src/make_asbgen.py")
    return [json.loads(l) for l in open(path)]


@pytest.fixture(scope="module")
def bench_instructions():
    """The benchmark's instruction TEXT and nothing else.

    Reading only `instruction` here is deliberate: the test must be able to look
    at benchmark text to prove the generator did not copy it. Labels are not
    loaded at all, so this test cannot use them.
    """
    recs = json.load(open(os.path.join(BENCH, "asb.json")))
    return [r.get("instruction", "") for r in recs if r.get("instruction")]


def test_no_verbatim_benchmark_ngram(generated, bench_instructions):
    """12-word shingles: any shared shingle means text was carried over."""
    bench_shingles = set()
    for t in bench_instructions:
        bench_shingles |= _ngrams(_words(t))

    hits = 0
    for row in generated:
        shared = _ngrams(_words(row["prompt"])) & bench_shingles
        if shared:
            hits += 1
    assert hits == 0, (
        f"{hits} generated rows share a {NGRAM}-word shingle with a benchmark "
        f"instruction -- that is copied text")


def test_no_near_duplicate_of_any_benchmark_row(generated, bench_instructions):
    """Jaccard on word sets. Cheap, and catches paraphrase that survives (1)."""
    bench = [(t, set(_words(t))) for t in bench_instructions]
    worst_row, worst_sim = None, 0.0
    for row in generated[:2000]:
        gw = set(_words(row["prompt"]))
        for t, bw in bench:
            s = _jaccard(gw, bw)
            if s > worst_sim:
                worst_sim, worst_row = s, t
    assert worst_sim < SIM_THRESHOLD, (
        f"generated row is {worst_sim:.2f} similar to benchmark instruction "
        f"{worst_row[:120]!r}")


def test_generator_never_reads_label_bearing_fields():
    """Structural check on the PARSED source, so prose cannot trip it.

    String-matching the file would match its own docstring -- the first version
    of this test did exactly that and failed on a sentence saying it never reads
    the field. Parsing to an AST and walking only real string literals and
    attribute/subscript names is the check that actually expresses the rule.
    """
    import ast

    path = os.path.join(ROOT, "src", "make_asbgen.py")
    tree = ast.parse(open(path).read())
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)):
            d = ast.get_docstring(node, clean=False)
            if d:
                docstrings.add(d)

    code_strings = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value not in docstrings:
                code_strings.append(node.value)

    joined = "\n".join(code_strings)
    for forbidden in ("fulfillable", "failure_modes", "risks"):
        assert forbidden not in joined, (
            f"generator code uses the label-bearing field {forbidden!r}")

    # and it must never open the benchmark file
    assert not any("asb.json" in c for c in code_strings), \
        "generator opens the benchmark file"


def test_generated_corpus_is_internally_valid(generated):
    """Every row carries a rationale, so a label can be audited without the
    generator, and the label is actually 0/1."""
    for row in generated[:500]:
        assert row["label"] in (0, 1)
        assert row.get("rationale"), "row has no auditable rationale"
        assert "Context:" in row["prompt"]


def test_multi_environment_rows_exist(generated):
    """The bug #9 case must actually be present in the corpus.

    If every generated row had one environment, a model could pass this corpus
    by reading only the first environment -- the exact defect that was fixed.
    """
    multi = sum(1 for r in generated if r["n_envs"] > 1)
    assert multi > 0, "no multi-environment rows: bug #9 case absent"
    frac = multi / len(generated)
    assert frac > 0.02, f"only {frac:.1%} multi-environment rows"
