"""Muse PostToolUseFailure boundary: JSON stdout, diagnostics stderr.

Muse supplies tool_name, tool_input, error (often a JSON string), and
is_interrupt. Only hookSpecificOutput.additionalContext reaches the model.
An exit-zero log record is not delivery proof; verify the model's own reply.
Classifier failures leave the tool turn running, with only the exception type
reported on stderr. JANUS_HOOK_DISABLE prevents even constructing a backend.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import janus_interface as JI

MAX_ERROR_CHARS = 4000


def error_text(payload):
    value = payload.get("error")
    if isinstance(value, (dict, list)):
        return json.dumps(value, default=str)
    return str(value).strip() if value is not None else ""


def failure_text(payload):
    """Legacy text contract for non-encoder backends."""
    if not isinstance(payload, dict):
        return "[no error text]"
    tool = payload.get("tool_name") or "?"
    arguments = payload.get("tool_input")
    args = json.dumps(arguments, sort_keys=True, default=str) if arguments else ""
    parts = [f"[tool: {tool}]"]
    if args:
        parts.append(f"[args: {args[:400]}]")
    parts.append(error_text(payload)[:MAX_ERROR_CHARS] or "[no error text]")
    return "\n".join(parts)


def encoder_input(payload):
    state = JI.build_state(tool=payload.get("tool_name"),
                           arguments=payload.get("tool_input"),
                           error=error_text(payload))
    return JI.build_encoder_input(state)


def classify(payload, backend_factory):
    """Keep model startup and inference chatter off the JSON transport.
    The canonical input is always returned: LOUD verdicts carry it."""
    from backends import validate

    with contextlib.redirect_stdout(sys.stderr):
        backend = backend_factory()
        use_encoder = getattr(backend, "uses_encoder_input", backend.name == "janus")
        text = encoder_input(payload) if use_encoder else failure_text(payload)
        dist = validate(backend(text), backend.name)
    shown = text
    return backend, dist, shown


def escalate_note(backend, shown, threshold):
    """Second opinion when the local classifier abstains. Fail-open."""
    try:
        from backends import FrontierBackend

        dist = FrontierBackend()(shown or "")
        best, conf = JI.decode(dist)
        if best is None:
            return "Frontier escalation also unsure; treat as unknown."
        return (f"Frontier second opinion ('{best}' at {conf:.2f}): "
                f"{__import__('janus_route').ACTIONS[best]}")
    except Exception as error:
        return f"Frontier escalation unavailable: {type(error).__name__}."


def inline_verdict(backend, dist, threshold):
    """One-line verdict for inline rendering (JANUS_COMPACT=1)."""
    from janus_route import ACTIONS
    best, conf = JI.decode(dist)
    if best is None:
        return f"[janus:{backend.name} unsure — tied classes]"
    if conf < threshold:
        return (f"[janus:{backend.name} unsure — best guess '{best}' "
                f"at {conf:.2f}, below {threshold:.2f}]")
    return (f"[janus:{backend.name} {best} ({conf:.2f}): "
            f"{ACTIONS.get(best, 'inspect manually')}]")


def advisory(backend, dist, threshold, shown=None):
    best, conf = JI.decode(dist)
    if best is None:
        return (f"Tool failure classifier '{backend.name}' is unsure (tied classes). "
                "Check the failure before repeating it.")
    if conf < threshold:
        base = (f"Tool failure classifier '{backend.name}' is unsure (best guess "
                f"'{best}' at {conf:.2f}, below the {threshold:.2f} threshold). "
                "Do not assume a retry is safe; check the failure before repeating it.")
        if os.environ.get("JANUS_ESCALATE"):
            return base + "\n[frontier]\n" + escalate_note(backend, shown, threshold)
        return base
    from janus_route import ACTIONS

    verdict = (f"[janus] {backend.name} read this as {best} "
               f"(confidence {conf:.2f}): {ACTIONS[best]}")
    if not shown:
        return verdict
    return (f"[janus saw]\n{shown}\n[verdict]\n{verdict}")


def emit_both(context, inline):
    """LOUD block for the model + compact line for the tool-output area.

    Two channels, one classification: additionalContext carries the full
    [janus saw] evidence (compliance-tested); systemMessage carries the
    one-liner the harness renders adjacent to the failed tool output."""
    out = {"hookSpecificOutput": {"hookEventName": "PostToolUseFailure",
                                    "additionalContext": context}}
    if inline:
        out["hookSpecificOutput"]["systemMessage"] = inline
    sys.stdout.write(json.dumps(out) + "\n")
    sys.stdout.flush()


def emit(context):
    """STDOUT is delivered; STDERR is only logged. Context always rides as
    hook JSON -- the transport requires JSON. LOUD is the default: the full
    [janus saw] block plus verdict."""
    sys.stdout.write(_advisory_json(context))
    sys.stdout.flush()


def _advisory_json(context):
    out = {"hookSpecificOutput": {"hookEventName": "PostToolUseFailure",
                                  "additionalContext": context}}
    return json.dumps(out)


def advisory_json(backend, dist, threshold, shown=None):
    """Full hook stdout for an already-classified failure (warm daemon)."""
    return _advisory_json(advisory(backend, dist, threshold, shown))


def threshold():
    """The one threshold parse, shared by the hook and the warm daemon."""
    value = float(os.environ.get("JANUS_THRESHOLD", "0.60"))
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("threshold must be finite and within [0,1]")
    return value


# Harness-internal tools: classifying them misfires on agent plumbing and
# lets the agent quote the wrong block. Silent exit, no stdout, no verdict.
SKIP_TOOLS = frozenset({"submit_reminder_decision"})


def main():
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return 0
    if not isinstance(payload, dict) or payload.get("is_interrupt"):
        return 0
    if os.environ.get("JANUS_HOOK_DISABLE") or not payload.get("error"):
        return 0
    if str(payload.get("tool_name")) in SKIP_TOOLS:
        return 0
    try:
        cutoff = threshold()
        from backends import get_backend

        backend, dist, shown = classify(payload, get_backend)
        if os.environ.get("JANUS_COMPACT") not in (None, "", "0"):
            context = inline_verdict(backend, dist, cutoff)
        else:
            context = advisory(backend, dist, cutoff, shown)
        inline = inline_verdict(backend, dist, cutoff)
    except Exception as error:
        print(f"Tool failure classifier unavailable: {type(error).__name__}", file=sys.stderr)
        return 0
    emit_both(context, inline)
    return 0


if __name__ == "__main__":
    sys.exit(main())
