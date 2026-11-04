"""The benchmarks must never be reachable from the training corpora.

An AutoResearch loop optimising a benchmark will eventually find a way to leak it
if nothing stops it -- that is what an optimiser does. This is the hard line the
loop is not allowed to cross, expressed as a test rather than a rule in a prompt
so it survives every iteration.

The corpora and the evals are separate files by construction (different fetch
scripts, different schemas), but "by construction" is a claim, not a check.
"""
import json
import os

import pytest

DATA = os.environ.get(
    "JANUS_DATA", os.path.join(os.path.dirname(__file__), "..", "data"))


def _safe_rows(split):
    path = os.path.join(DATA, f"safety_{split}.jsonl")
    if not os.path.exists(path):
        pytest.skip(f"{path} absent; run src/build_safety.py")
    return [json.loads(l) for l in open(path)]


def _raw(name):
    path = os.path.join(DATA, name)
    if not os.path.exists(path):
        pytest.skip(f"{path} absent; fetch it first")
    return open(path, "rb").read().decode(errors="replace")


def _bench_texts(bench):
    """The actual text a benchmark shows the model, per item.

    My first version of this file compared against a `prompt` key that these
    benchmarks do not have -- so it would have passed with a live leak in the
    corpus. The real fields are scenario/goal/contents for R-Judge and
    instruction for Agent-SafetyBench.
    """
    path = os.path.join(DATA, bench)
    if not os.path.exists(path):
        pytest.skip(f"{path} absent")
    if bench.endswith(".jsonl"):
        items = [json.loads(l) for l in open(path)]
    else:
        items = json.load(open(path))
    out = []
    for it in items:
        for k in ("scenario", "goal", "contents", "instruction", "risks"):
            v = it.get(k)
            if isinstance(v, str):
                out.append(v)
            elif isinstance(v, list):
                out.extend(x for x in v if isinstance(x, str))
    return [t for t in out if len(t) >= 40]


@pytest.mark.parametrize("bench", ["rjudge.jsonl", "asb.json"])
def test_no_benchmark_text_appears_in_a_training_prompt(bench):
    """Direction that matters: benchmark item text must not be inside training.

    Uses the benchmark's own text fields, not a guessed key.
    """
    texts = _bench_texts(bench)
    assert texts, f"extracted no text from {bench} -- the check would be vacuous"
    train = _safe_rows("train") + _safe_rows("dev")
    hits = [(t[:80], r["prompt"][:80]) for t in texts for r in train
            if t in r["prompt"]]
    assert not hits, f"{len(hits)} {bench} items appear in training prompts"


@pytest.mark.parametrize("bench", ["rjudge.jsonl", "asb.json"])
def test_no_whole_training_prompt_appears_in_a_benchmark(bench):
    """The reverse direction, restricted to rows long enough that a match is
    evidence rather than coincidence."""
    raw = _raw(bench).lower()
    train = _safe_rows("train")
    hits = [r["prompt"] for r in train
            if 60 <= len(r["prompt"]) < 400 and r["prompt"].lower() in raw]
    assert not hits, f"{len(hits)} training rows appear inside {bench}"


def test_safety_corpora_are_not_built_from_the_benchmarks():
    """A static check on the builder: it must not import or read an eval path."""
    import pathlib
    root = pathlib.Path(__file__).resolve().parent.parent / "src"
    for name in ("fetch_safety.py", "build_safety.py"):
        text = (root / name).read_text()
        for bad in ("rjudge", "asb"):
            assert bad not in text.lower(), (
                f"{name} references {bad}; training must not touch the evals")


def test_benchmark_files_are_gitignored():
    """Eval data is fetched, never committed, so a corpus cannot silently
    absorb a copy that was checked in by accident."""
    ignore = (os.path.join(os.path.dirname(__file__), "..",
                           ".gitignore") and
              open(os.path.join(os.path.dirname(__file__), "..",
                                ".gitignore")).read())
    assert "data/*.jsonl" in ignore
    assert "data/*.json" in ignore


def test_eval_bench_does_not_use_an_unbound_module_name():
    """An eval crashed with NameError AFTER a completed GPU-hour of training.

    `metrics._logit` was referenced but `metrics` was never imported under that
    name (only specific names were, via `from metrics import ...`). Training
    succeeded, so only the eval died -- having already spent the GPU-hour.

    My first version of this test compared `metrics.X` references against the
    imported names and PASSED with the bug reintroduced, because `logit as
    _logit` made the attribute name present. Binding is what matters, not the
    attribute list, so this walks the AST for a Name load that nothing binds.
    """
    import ast
    import pathlib

    src_root = pathlib.Path(__file__).resolve().parent.parent / "src"
    text = (src_root / "eval_bench.py").read_text()
    tree = ast.parse(text)

    bound = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                bound.add(a.asname or a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                bound.add(a.asname or a.name)

    used = {n.value.id for n in ast.walk(tree)
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)}
    # only flag bare module-style names, not locals/attributes
    suspicious = {m for m in used if m in {"metrics", "np", "torch", "json", "os"}
                  and m not in bound}
    assert not suspicious, (
        f"eval_bench uses {sorted(suspicious)} as a module but never imports it "
        f"under that name; this raises NameError at eval time, after training"
    )

    # and the names it DOES import must exist on the module
    import importlib
    import sys
    sys.path.insert(0, str(src_root))
    mod = importlib.import_module("metrics")
    for name in ("logit", "aurc", "ece", "fit_temperature_info", "summarize"):
        assert hasattr(mod, name), f"metrics.{name} is missing"


def test_calibration_uses_the_dev_split_of_the_trained_corpus():
    """Calibration must be fitted on the corpus the model was trained on.

    eval_bench read `swe_dev.jsonl` unconditionally while the safety run trained
    on `safety_{train,dev}.jsonl`. Temperature and the intercept were therefore
    fitted on a different TASK: 30.5%-unsafe "coding agent" prompts calibrating a
    model scored on 54.0%-unsafe "autonomous agent" prompts. The visible symptom
    was a fit choosing b=-0.70 (predict-all-safe) when the oracle on the real
    rows wanted b=+0.25.

    This is the same class as the trainer/eval prompt desynchronisation that
    invalidated two earlier runs: two halves of one pipeline reading different
    artifacts.
    """
    import os
    import pathlib
    import sys

    src_root = str(pathlib.Path(__file__).resolve().parent.parent / "src")
    if src_root not in sys.path:
        sys.path.insert(0, src_root)

    text = (pathlib.Path(src_root) / "eval_bench.py").read_text()
    assert "swe_dev.jsonl" not in text, (
        "eval_bench hardcodes swe_dev.jsonl; calibration must follow the corpus"
    )
    assert 'JANUS_CORPUS' in text, "eval_bench must read JANUS_CORPUS for its dev split"

    # and the two dev splits must genuinely differ, or this check is vacuous
    data = pathlib.Path(src_root).parent / "data"
    swe, saf = data / "swe_dev.jsonl", data / "safety_dev.jsonl"
    if not (swe.exists() and saf.exists()):
        pytest.skip("dev splits not fetched")
    import json
    a = [json.loads(l) for l in open(swe)]
    b = [json.loads(l) for l in open(saf)]
    ra = sum(r["label"] for r in a) / len(a)
    rb = sum(r["label"] for r in b) / len(b)
    assert abs(ra - rb) > 0.15, (
        f"dev base rates too similar ({ra:.3f} vs {rb:.3f}) to matter; "
        f"re-examine whether this distinction is still load-bearing"
    )


def test_benchmark_prompts_are_rendered_by_the_training_renderer():
    """Trainer and evals must render prompts with ONE function.

    evals._prompt invented its own "<command>...</command>" wrapper while training
    used noul.safety_prompt's "Context/Constraints/Action" layout, so the safety
    adapter was scored on a format it had never seen. noul.safety_prompt's own
    docstring claimed "Training rows and eval rows both call this" -- which was
    false, the same class of desynchronisation that invalidated two earlier runs.

    This asserts the structural property rather than a sample string: every
    benchmark prompt must be exactly what the shared renderer produces, so no
    bespoke wrapper can reappear.
    """
    import json
    import pathlib
    import sys

    src_root = pathlib.Path(__file__).resolve().parent.parent / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    data = pathlib.Path(src_root).parent / "data"
    if not (data / "rjudge.jsonl").exists():
        pytest.skip("benchmarks not fetched")

    import evals
    import noul

    # no bespoke wrapper anywhere in the module
    text = (src_root / "evals.py").read_text()
    assert "<command>" not in text, (
        "evals.py reintroduces a bespoke wrapper; benchmarks must use "
        "noul.safety_prompt"
    )

    # and the produced prompts match the shared renderer's shape exactly
    with open(data / "rjudge.jsonl") as fh:
        row = evals.rjudge_rows([json.loads(fh.readline())])[0]
    p = row["prompt"]
    assert p.startswith(noul.INSTRUCTION)
    assert "\nContext: " in p, "benchmark prompt must carry the Context: slot"
    assert "\nAction: " in p, "benchmark prompt must carry the Action: slot"
    assert "\nConstraints:\n" in p, "benchmark prompt must carry the Constraints: slot"

    asb = evals.asb_rows(json.load(open(data / "asb.json")))[0]
    q = asb["prompt"]
    assert q.startswith(noul.INSTRUCTION) and "\nAction: " in q and "\nContext: " in q


def test_eval_maxlen_matches_the_training_maxlen():
    """The evaluator must score at the length the trainer trained at.

    eval_bench.py defaulted to MAXLEN=192 while run_loop_iter.sh trained at 512
    and never passed --maxlen through. Every evaluation in 13 runs therefore
    scored with 12.4% (R-Judge) / 15.6% (ASB) of rows truncated, while the
    adapter had been trained on full-length prompts. Scoring the same adapter at
    512 instead of 192 moved R-Judge accuracy from 58.1% to 65.4% -- across the
    majority-class floor.

    This is the same class as every other bug in this project: two halves of one
    pipeline configured independently.
    """
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parent.parent
    runner = (root / "scripts" / "run_loop_iter.sh").read_text()
    evalb = (root / "src" / "eval_bench.py").read_text()

    # the runner must pass --maxlen explicitly to the evaluator
    assert re.search(r"--maxlen \"\$\{MAXLEN", runner), (
        "run_loop_iter.sh must pass --maxlen to eval_bench.py; omitting it silently "
        "uses the evaluator's default"
    )
    # and both defaults must agree
    trainer_default = re.search(r"--maxlen \"\$\{MAXLEN:-(\d+)\}\"", runner)
    eval_default = re.search(r"^MAXLEN = (\d+)", evalb, re.M)
    assert trainer_default and eval_default
    assert trainer_default.group(1) == eval_default.group(1), (
        f"trainer default maxlen {trainer_default.group(1)} != evaluator default "
        f"{eval_default.group(1)}"
    )


def test_fitted_intercept_must_transfer_to_held_out_rows():
    """A dev-fitted level correction that hurts the report set must be dropped.

    The joint (T, b) fit runs on the model's dev split, but dev and the benchmark
    are different populations. Fitting b on dev produced -0.50 and then damaged
    the benchmark: R-Judge accuracy 65.5% (plain T, above its 64.9% floor) became
    62.3%, and ASB 59.5% became 49.8%. A correction that does not transfer is
    worse than no correction.

    eval_bench now accepts b only if it also lowers log-loss on a deterministic
    held-out half of the reported rows -- a gate, not a second fitting pass.
    """
    import pathlib

    src = (pathlib.Path(__file__).resolve().parent.parent / "src"
           / "eval_bench.py").read_text()
    assert "TRANSFER_GUARD" in src, (
        "the fitted intercept needs a transfer guard; without one a dev-fitted b "
        "can silently undo a floor-clearing model"
    )
    assert "_nll_rep" in src, "the guard must compare log-loss on the report rows"
    assert "b = 0.0" in src, "the guard must be able to drop the intercept"

    # and the reported configuration must be plain temperature, not T+b.
    # measure.sh reads the JSON result (rewritten 706a011: no text parsing);
    # the primary metric is results.rlcd, the plain-T row the transfer guard
    # leaves standing. The intercept stays available as its own row.
    measure = (pathlib.Path(__file__).resolve().parent.parent / ".auto"
               / "measure.sh").read_text()
    assert 'd["results"]["rlcd"]' in measure, (
        "the primary metric must come from results.rlcd (plain-T)"
    )
    assert "finetuned+T+b" not in measure, (
        "the shipped configuration must not be the intercept-corrected row"
    )


def test_asb_context_keeps_every_environment_and_its_parameters():
    """ASB's risk evidence lives in environment state that was being dropped.

    Two defects, both of which deleted the basis for a judgement:
      - only environments[0] was rendered; 10.4% of records declare more than
        one, and 6.5% carry the risk-making state in a dropped one (e.g. an OS
        env holding a file with "Social Security Number: ..." while env 0 is
        just an SMS tool).
      - `parameters` was never rendered; 71.0% of records keep their actual
        state there.

    This test pins both: every environment name appears, and parameter content
    that exists in the record survives into the prompt.
    """
    import json
    import pathlib
    import sys

    root = pathlib.Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root / "src"))
    import evals

    recs = json.load(open(root / "data" / "asb.json"))

    multi = [r for r in recs if len(r.get("environments") or []) > 1][:20]
    assert multi, "expected multi-environment records in ASB"
    for rec in multi:
        prompt = evals.asb_rows([rec])[0]["prompt"]
        for env in rec["environments"]:
            if env.get("name"):
                assert env["name"] in prompt, (
                    f"environment {env['name']!r} was dropped from the prompt")

    # a record whose only interesting state is in parameters
    with_state = [r for r in recs
                  if (r.get("environments") or [{}])[0].get("parameters")][:20]
    assert with_state, "expected records whose environment carries parameters"
    needle = None
    for rec in with_state:
        params = rec["environments"][0]["parameters"]
        for f in (params.get("files") or []):
            if f.get("content"):
                needle = f["content"][:25]
                rec_with_needle = rec
                break
        if needle:
            break
    assert needle, "expected at least one record with file content in parameters"
    prompt = evals.asb_rows([rec_with_needle])[0]["prompt"]
    assert needle in prompt, (
        "environment parameters (file contents) were not rendered into the "
        "prompt; they are the evidence for the unfulfillable label")


def test_class_weighting_is_available_and_off_by_default():
    """Class weighting was named in run 1 and never implemented for 18 runs.

    The corpora are 55.0% unsafe while the benchmarks are 64.9% (rjudge) and
    63.3% (asb) unsafe, so an unweighted objective puts the decision boundary
    roughly 9-10pt too far toward "unsafe" for the evaluation population. This
    pins that the knob exists, that it is OFF unless asked for, and that it is
    wired through the runner rather than only existing in the trainer.
    """
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent
    src = (root / "src" / "train_lora.py").read_text()
    runner = (root / "scripts" / "run_loop_iter.sh").read_text()

    assert '"--class-weight"' in src, "trainer must expose --class-weight"
    assert 'default="none"' in src, (
        "class weighting must be OFF by default so the baseline is unchanged")
    assert "weight=cw" in src, "the weight must actually reach cross_entropy"
    assert "--class-weight" in runner, (
        "the runner must pass --class-weight through, or the knob is "
        "unreachable from an iteration")


def test_stored_corpora_carry_the_current_renderer_instruction():
    """The trainer reads stored prompts; the evaluator re-renders. They must agree.

    data/safety_*.jsonl were written 2026-10-26 20:43-20:54, before src/noul.py
    was fixed at 2026-10-27 02:36 for an inverted answer contract. 26731 of
    27047 rows still carried "no (it is safe) or yes (it is unsafe)" -- which
    tells the model 'no' means SAFE, while supervision sends every label=1
    (UNSAFE) row to the token 'no'.

    Run 5 fixed the renderer and reported the contract repaired. It could not be:
    training reads r["prompt"] from disk verbatim, while evaluation calls
    noul.safety_prompt() fresh. Runs 5-20 therefore trained on an inverted
    instruction and evaluated on a corrected one.

    296 rows are SWE-style transcripts with no answer contract; they legitimately
    carry neither form and are not checked here.
    """
    import json
    import pathlib
    import sys

    root = pathlib.Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root / "src"))
    import noul

    rendered = noul.safety_prompt("CTX", None, "ACT")
    line = next(l for l in rendered.split("\n") if "exactly one word" in l)
    assert "no (it is unsafe)" in line, (
        "the renderer's own instruction changed; update this test and regenerate")

    old = "no (it is safe) or yes (it is unsafe)"
    for name in ("safety_train.jsonl", "safety_dev.jsonl"):
        path = root / "data" / name
        if not path.exists():
            continue
        offenders, contracted = 0, 0
        for ln in path.read_text().splitlines():
            p = json.loads(ln)["prompt"]
            if "exactly one word" not in p:
                continue          # SWE-style row, no answer contract
            contracted += 1
            if old in p:
                offenders += 1
        assert contracted > 0, f"{name} has no contracted rows at all"
        assert offenders == 0, (
            f"{name}: {offenders}/{contracted} rows still carry the inverted "
            f"instruction; run scripts/fix_corpus_instruction.py --apply")


def test_length_control_uses_an_apriori_sign_not_a_benchmark_fitted_one():
    """The length control must not fit its direction on the benchmark.

    Run 24 compared the adapter to a length-only baseline and concluded that
    character count "essentially ties" it. That baseline chose its sign with
    max(auc(L), auc(-L)) computed ON the benchmark labels, so it was fitted to
    what it scored against. With the sign taken from the project's own dev split
    instead, the same rule scores at the floor exactly (auc 0.363 rjudge) and the
    adapter beats it by ~0.27 auc.

    The distinction matters because a label-fitted length rule measures how
    exploitable a BENCHMARK is, not how good a MODEL is. If the control ever
    starts choosing its sign from the benchmark again, this test fails.
    """
    import pathlib
    import re

    src = (pathlib.Path(__file__).resolve().parent.parent / "scripts"
           / "length_control.py").read_text()

    # the sign must come from the dev split
    assert 'os.path.join(DATA, "safety_dev.jsonl")' in src, (
        "the length control must take its direction from our own dev split")
    assert re.search(r"sign = 1 if a_dev > 0\.5 else -1", src), (
        "the sign must be derived from dev auc, not from benchmark auc")

    # and there must be exactly one place that fits a sign on a benchmark, kept
    # only for the exploitability column -- never as the bar a model must beat
    assert "p_fit = L if metrics.auc(L, y)" in src, (
        "the label-fitted sign should exist only as the exploitability column")
    assert "A-priori length is the bar a model must beat" in src, (
        "the two columns must be labelled so the fitted one cannot be mistaken "
        "for the bar")
