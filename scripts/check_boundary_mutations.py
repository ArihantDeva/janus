"""Reintroduce guarded defects only in disposable copies, never live sources."""

import pathlib
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
CASES = [
    ("stdout", "src/muse_hook.py", "with contextlib.redirect_stdout(sys.stderr):",
     "with contextlib.nullcontext():",
     "tests/test_hook_contract.py::test_backend_diagnostics_cannot_pollute_hook_stdout"),
    ("input", "src/muse_hook.py",
     'text = encoder_input(payload) if backend.name == "janus" else failure_text(payload)',
     "text = failure_text(payload)",
     "tests/test_hook_contract.py::test_janus_receives_state_question_choices_not_plain_failure_text"),
    ("threshold", "src/muse_hook.py",
     "if not math.isfinite(threshold) or not 0 <= threshold <= 1:", "if False:",
     "tests/test_hook_contract.py::test_invalid_threshold_is_inert_instead_of_crashing_or_advising"),
    ("tie", "src/muse_hook.py", "best, conf = JI.decode(dist)",
     "best = max(dist, key=dist.get)\n    conf = dist[best]",
     "tests/test_hook_contract.py::test_tied_distribution_never_invents_a_class_even_below_default_threshold"),
    ("failfast", "scripts/run_scopes.sh",
     'for scope in "$@"; do\n  bash "$runner" "$scope"\ndone',
     'set +e\nfor scope in "$@"; do\n  bash "$runner" "$scope"\ndone',
     "tests/test_validation_runner.py::test_failing_scope_stops_the_run_and_fails_the_launcher"),
    ("stale", "src/remote_slurm.py", "except Stale:\n            return self._ask(self._restart(), text)",
     "except Stale:\n            raise",
     "tests/test_remote_slurm.py::test_stale_allocation_is_replaced_by_a_new_owned_job"),
    ("validate", "src/remote_slurm.py", "checked = validate(dist, \"janus\")", "checked = dist",
     "tests/test_remote_slurm.py::test_malformed_payload_never_becomes_advice"),
    ("jobid", "src/install_muse_hook.py",
     "f'JANUS_BACKEND=janus JANUS_ENCODER_TRANSPORT=remote '",
     "f'JANUS_BACKEND=janus JANUS_SRUN_JOBID=1234 JANUS_ENCODER_TRANSPORT=remote '",
     "tests/test_remote_slurm.py::test_installed_configuration_carries_no_job_id"),
    ("weights", "src/encoder_backend.py",
     "return AutoModel.from_pretrained(path, torch_dtype=torch.float32,\n"
     "                                         local_files_only=True)",
     "from transformers import AutoConfig\n"
     "        return AutoModel.from_config(AutoConfig.from_pretrained(path))",
     "tests/test_encoder_roundtrip.py::test_full_checkpoint_restores_saved_predictions_not_random_base"),
    ("adapter", "src/encoder_backend.py",
     "return PeftModel.from_pretrained(base, path, local_files_only=True)", "return base",
     "tests/test_encoder_roundtrip.py::test_adapter_checkpoint_restores_base_adapter_and_head"),
    ("classes", "src/encoder_backend.py",
     'if "classes" in state and tuple(state["classes"]) != JI.CLASSES:', "if False:",
     "tests/test_encoder_roundtrip.py::test_wrapped_head_rejects_changed_class_order"),
    ("causal", "src/encoder_backend.py", "E.make_bidirectional(encoder)", "pass",
     "tests/test_encoder_roundtrip.py::test_full_checkpoint_restores_saved_predictions_not_random_base"),
]


def check_case(case):
    name, filename, old, new, target = case
    with tempfile.TemporaryDirectory(prefix="janus-mutation-") as directory:
        work = pathlib.Path(directory)
        for tree in ("src", "tests", "scripts"):
            shutil.copytree(ROOT / tree, work / tree,
                            ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
        path = work / filename
        text = path.read_text()
        if text.count(old) != 1:
            raise RuntimeError(f"{name}: mutation anchor is not unique")
        path.write_text(text.replace(old, new, 1))
        result = subprocess.run([sys.executable, "-m", "pytest", "-q", target],
                                cwd=work, capture_output=True, text=True, timeout=180)
        if result.returncode != 1 or "failed" not in result.stdout:
            raise RuntimeError(f"{name}: not caught\n{result.stdout}\n{result.stderr}")
        print(f"mutation {name}: caught", flush=True)


def main():
    for case in CASES:
        check_case(case)
    print(f"MUTATIONS CAUGHT: {len(CASES)}/{len(CASES)}")


if __name__ == "__main__":
    main()
