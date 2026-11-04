"""Fetch the SAFETY training corpora.

Deliberately separate from the eval sets. R-Judge and Agent-SafetyBench are
EVALS ONLY -- training on either makes the reported number meaningless, so they
are fetched by different scripts and never read here.

Two sources, two different granularities, and the difference matters:

  karanxa/agent-action-safety-dataset
      One row per TOOL CALL: an `action` string plus the `constraints` it may
      violate, labelled is_safe. This is the granularity the evals use.

  eval-aware/linuxarena-trajectories
      One row per TRAJECTORY. It has action_has_error per action but NO
      per-action safety flag -- the only safety label is side_task_success,
      which is a property of the whole trajectory. Expanding it to one row per
      action would label `ls -R /app` unsafe because the trajectory later
      completed a hidden side task. That is precisely the label-mismatch bug
      this rewrite exists to fix, so the trajectory is kept whole and the label
      is attached to the trajectory-level question it actually answers.
"""
import json
import os
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("JANUS_DATA", os.path.join(ROOT, "data"))

KARANXA = ("https://huggingface.co/datasets/karanxa/"
           "agent-action-safety-dataset/resolve/main/{split}.jsonl")
LINUXARENA = ("https://huggingface.co/datasets/eval-aware/"
              "linuxarena-trajectories/resolve/main/data/"
              "train-00000-of-00001.parquet")


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "januslab"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return r.read()


def fetch_karanxa():
    out = []
    for split in ("train", "val"):
        raw = get(KARANXA.format(split=split))
        rows = [json.loads(l) for l in raw.decode().splitlines() if l.strip()]
        for r in rows:
            r["_split"] = split
        out.extend(rows)
        print(f"  karanxa {split}: {len(rows)}")
    return out


def fetch_linuxarena():
    raw = get(LINUXARENA)
    path = os.path.join(DATA, "_linuxarena.parquet")
    os.makedirs(DATA, exist_ok=True)
    with open(path, "wb") as f:
        f.write(raw)
    print(f"  linuxarena parquet: {len(raw)} bytes -> {path}")
    return path


def main():
    os.makedirs(DATA, exist_ok=True)
    print("fetching karanxa ...")
    kx = fetch_karanxa()
    with open(os.path.join(DATA, "karanxa.jsonl"), "w") as f:
        for r in kx:
            f.write(json.dumps(r) + "\n")
    print(f"  -> data/karanxa.jsonl ({len(kx)} rows)")

    print("fetching linuxarena ...")
    fetch_linuxarena()
    print("done")


if __name__ == "__main__":
    main()
