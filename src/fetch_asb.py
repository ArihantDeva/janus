"""Fetch Agent-SafetyBench from the upstream release.

Why this exists: data/asb.json was committed to the repo as a 2.8 MB blob with no
provenance, while the sibling R-Judge file was fetched by script and gitignored.
Two datasets, two different policies, and the committed one carried no record of
where it came from or which release. It is also an EVALUATION set, so shipping a
vendored copy means a reader cannot tell whether the copy matches upstream.

Mirrors fetch_rjudge.py: fetch to data/, never commit, record the source URL.
"""
import json
import os
import urllib.request

REPO = "thu-coai/Agent-SafetyBench"
# The file lives at <repo>/data/released_data.json, not at the repo root --
# verified via the GitHub contents API rather than guessed.
CANDIDATES = (
    "https://raw.githubusercontent.com/thu-coai/Agent-SafetyBench/main/"
    "data/released_data.json",
    "https://raw.githubusercontent.com/thu-coai/Agent-SafetyBench/master/"
    "data/released_data.json",
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("JANUS_DATA", os.path.join(ROOT, "data"))
OUT = os.path.join(DATA, "asb.json")


def fetch(url):
    with urllib.request.urlopen(url, timeout=120) as r:
        return json.loads(r.read().decode())


def main():
    last = None
    for url in CANDIDATES:
        try:
            rows = fetch(url)
        except Exception as e:                      # noqa: BLE001
            last = f"{url}: {type(e).__name__}"
            print(f"  {last}")
            continue
        os.makedirs(DATA, exist_ok=True)
        with open(OUT, "w") as f:
            json.dump(rows, f)
        print(f"wrote {len(rows)} records -> {OUT}")
        print(f"source: {url}")
        print(f"repo:   https://github.com/{REPO}")
        return
    raise SystemExit(f"could not fetch Agent-SafetyBench (last: {last})")


if __name__ == "__main__":
    main()
