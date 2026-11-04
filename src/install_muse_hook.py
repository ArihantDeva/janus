"""Install, verify and disable the encoder hook in a Muse workspace.

The installed configuration names a stable command, never an allocation id or a
batch node. Ownership of the classifier is rediscovered on each request.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time

EVENT = "PostToolUseFailure"
HOOK = "remote_hook.py"
DAEMON_HOOK = "daemon_hook.py"
LOCAL_HOOK = "muse_hook.py"


def command_for(router_home, local=False):
    if local:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        weights = os.path.join(root, "weights", "model_freeze0_rlcd")
        # Plain python3: daemon_hook only talks to the warm daemon's socket;
        # the uv/torch spawn lives inside its cold fallback.
        return (f'JANUS_BACKEND=janus JANUS_ENCODER_TRANSPORT=local '
                f'JANUS_ENCODER="{weights}" JANUS_ENCODER_HEAD="{weights}/head.pt" '
                f'JANUS_THRESHOLD=${{JANUS_THRESHOLD:-0.60}} '
                f'JANUS_COMPACT=${{JANUS_COMPACT:-0}} '
                f'{sys.executable} "{os.path.join(router_home, "src", DAEMON_HOOK)}"')
    return (f'JANUS_BACKEND=janus JANUS_ENCODER_TRANSPORT=remote '
            f'JANUS_THRESHOLD=${{JANUS_THRESHOLD:-0.60}} '
            f'{sys.executable} "{os.path.join(router_home, "src", HOOK)}"')


def _entry(router_home, local=False):
    return {"matcher": "*", "hooks": [{"type": "command", "timeout": 300,
                                       "command": command_for(router_home, local=local)}]}


def _hooks_file(workspace):
    return os.path.join(os.path.abspath(workspace), ".muse", "hooks.json")


def _read(path):
    try:
        data = json.loads(open(path).read())
    except FileNotFoundError:
        return {"hooks": {}}
    if not isinstance(data, dict) or not isinstance(data.get("hooks", {}), dict):
        raise ValueError(f"{path} is not a Muse hooks file")
    data.setdefault("hooks", {})
    return data


def _write(path, data):
    with open(path, "w") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _ours(entry):
    commands = [hook.get("command", "") for hook in entry.get("hooks", [])
                if isinstance(hook, dict)]
    return any(HOOK in command or DAEMON_HOOK in command or
               LOCAL_HOOK in command for command in commands)


def install(workspace, router_home=None, local=True):
    """Merge our hook into the project target, keeping every other event."""
    path = _hooks_file(workspace)
    router_home = router_home or os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = _read(path)
    if os.path.exists(path):
        shutil.copy2(path, f"{path}.bak.{time.strftime('%Y%m%d%H%M%S')}")
    kept = [entry for entry in data["hooks"].get(EVENT, [])
            if not (isinstance(entry, dict) and _ours(entry))]
    data["hooks"][EVENT] = kept + [_entry(router_home, local=local)]
    _write(path, data)
    return path


def disable(workspace):
    """Remove only our hook; everything else is left byte-identical."""
    path = _hooks_file(workspace)
    data = _read(path)
    entries = [entry for entry in data["hooks"].get(EVENT, [])
               if not (isinstance(entry, dict) and _ours(entry))]
    if entries:
        data["hooks"][EVENT] = entries
    else:
        data["hooks"].pop(EVENT, None)
    _write(path, data)
    return path


def verify(workspace):
    path = _hooks_file(workspace)
    try:
        data = _read(path)
    except (OSError, ValueError) as error:
        return {"installed": False, "reason": type(error).__name__}
    entries = [e for e in data["hooks"].get(EVENT, []) if _ours(e)]
    if not entries:
        return {"installed": False, "reason": "no encoder hook"}
    command = entries[-1]["hooks"][0]["command"]
    target = next((part.strip('"') for part in command.split() if HOOK in part
                   or DAEMON_HOOK in part or LOCAL_HOOK in part), "")
    return {"installed": True, "path": path, "command": command,
            "hook_exists": os.path.isfile(target),
            "backend": "janus" if "JANUS_BACKEND=janus" in command else "unset"}
