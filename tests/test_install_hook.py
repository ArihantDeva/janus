"""Installed Muse configuration: durable target, preserved hooks, reversible."""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import install_muse_hook as I

OTHER = {"PostToolUse": [{"matcher": "Edit", "hooks": [{"type": "command",
                                                       "command": "keep me"}]}]}


def read(path):
    return json.loads(open(path).read())


def test_install_into_an_empty_workspace_creates_the_project_target(tmp_path):
    path = I.install(str(tmp_path))
    assert path == str(tmp_path / ".muse" / "hooks.json")
    command = read(path)["hooks"]["PostToolUseFailure"][0]["hooks"][0]["command"]
    # local CPU default: self-contained demo, no cluster allocation needed
    assert "daemon_hook.py" in command
    assert "JANUS_ENCODER_TRANSPORT=local" in command


def test_install_remote_opt_in_uses_the_allocation_hook(tmp_path):
    path = I.install(str(tmp_path), local=False)
    command = read(path)["hooks"]["PostToolUseFailure"][0]["hooks"][0]["command"]
    assert "remote_hook.py" in command


def test_install_preserves_other_events_and_hooks(tmp_path):
    hooks = tmp_path / ".muse" / "hooks.json"
    hooks.parent.mkdir(parents=True)
    hooks.write_text(json.dumps({"hooks": OTHER}))
    I.install(str(tmp_path))
    assert read(hooks)["hooks"]["PostToolUse"] == OTHER["PostToolUse"]


def test_install_replaces_a_previous_classifier_hook(tmp_path):
    """Exactly one classifier hook survives, so a stale backend cannot win."""
    hooks = tmp_path / ".muse" / "hooks.json"
    hooks.parent.mkdir(parents=True)
    hooks.write_text(json.dumps({"hooks": {"PostToolUseFailure": [
        {"matcher": "*", "hooks": [{"type": "command",
                                    "command": "python /old/muse_hook.py"}]}]}}))
    I.install(str(tmp_path))
    entries = read(hooks)["hooks"]["PostToolUseFailure"]
    assert len(entries) == 1
    assert "daemon_hook.py" in entries[0]["hooks"][0]["command"]


def test_install_keeps_an_unrelated_failure_hook(tmp_path):
    hooks = tmp_path / ".muse" / "hooks.json"
    hooks.parent.mkdir(parents=True)
    hooks.write_text(json.dumps({"hooks": {"PostToolUseFailure": [
        {"matcher": "Edit", "hooks": [{"type": "command", "command": "notify"}]}]}}))
    I.install(str(tmp_path))
    commands = [h["command"] for entry in
                read(hooks)["hooks"]["PostToolUseFailure"] for h in entry["hooks"]]
    assert "notify" in commands and len(commands) == 2


def test_install_backs_up_the_existing_file(tmp_path):
    hooks = tmp_path / ".muse" / "hooks.json"
    hooks.parent.mkdir(parents=True)
    hooks.write_text(json.dumps({"hooks": OTHER}))
    I.install(str(tmp_path))
    backups = [name for name in os.listdir(hooks.parent) if ".bak" in name]
    assert backups, "existing configuration must be backed up before merging"
    assert read(hooks.parent / backups[0])["hooks"]["PostToolUse"] == OTHER["PostToolUse"]


def test_installed_configuration_has_no_expiring_job_id_or_node(tmp_path):
    """A durable install cannot embed an allocation id or a batch node name."""
    text = open(I.install(str(tmp_path))).read()
    assert "jobid" not in text.lower()
    assert "srun" not in text.lower()


def test_install_is_idempotent(tmp_path):
    first = open(I.install(str(tmp_path))).read()
    second = open(I.install(str(tmp_path))).read()
    assert first == second


def test_disable_removes_only_the_classifier_hook(tmp_path):
    hooks = tmp_path / ".muse" / "hooks.json"
    hooks.parent.mkdir(parents=True)
    hooks.write_text(json.dumps({"hooks": OTHER}))
    I.install(str(tmp_path))
    I.disable(str(tmp_path))
    assert read(hooks)["hooks"] == OTHER


def test_verify_reports_the_installed_command(tmp_path):
    I.install(str(tmp_path))
    report = I.verify(str(tmp_path))
    assert report["installed"] is True
    assert "janus" in report["command"] or "remote_hook" in report["command"]
