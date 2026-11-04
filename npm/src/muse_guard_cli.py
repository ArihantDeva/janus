"""CLI: one tool call in, one decision out.

Muse Code has no plugin hook that runs arbitrary code per tool call today, so the
integration point is a pre-exec wrapper: Muse Code runs `muse-guard <command>`
and receives a decision on stdout with an exit code that a shell can act on.

    exit 0  allow
    exit 2  ask    (a shell wrapper should prompt)
    exit 3  block  (a shell wrapper must not run the command)

Exit codes rather than parsing JSON, because a shell wrapper that has to parse
JSON is a wrapper that will get it wrong. The JSON is still printed for logs.

    muse-guard "uv run pytest tests/"
    muse-guard --base-url http://host:8080 "npm install"
"""
import argparse
import sys

import guard as guard_mod
import llamacpp_client
import muse_guard

EXIT = {"allow": 0, "ask": 2, "block": 3}


def build_parser():
    ap = argparse.ArgumentParser(
        prog="muse-guard",
        description="Decide whether one tool call should run.")
    ap.add_argument("command", help="the shell command Muse Code wants to run")
    ap.add_argument("--base-url", default=None,
                    help="llama-server root (default: $MUSE_GUARD_BASE_URL or "
                         "http://127.0.0.1:8080)")
    ap.add_argument("--hi", type=float, default=0.7, help="block above this")
    ap.add_argument("--lo", type=float, default=0.3, help="ask above this")
    ap.add_argument("--allow-root", action="append", default=None,
                    help="workspace root; paths outside it are escalated")
    ap.add_argument("--no-model", action="store_true",
                    help="precheck only; never contact a server")
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    scorer = None
    if not a.no_model:
        url = a.base_url or muse_guard.base_url_from_env()
        scorer = llamacpp_client.LlamaCppScorer(base_url=url)
    g = muse_guard.MuseGuard(scorer, hi=a.hi, lo=a.lo,
                             allow_roots=a.allow_root)
    try:
        d = g.inspect(a.command)
    except Exception as e:                                   # noqa: BLE001
        # Fail closed AND loudly. A guard that cannot decide must not report
        # allow, and must not crash a build with a bare traceback either.
        print(f"muse-guard: cannot decide: {e}", file=sys.stderr)
        return 4
    print(g.to_json(d))
    return EXIT[d["action"]]


if __name__ == "__main__":
    sys.exit(main())
