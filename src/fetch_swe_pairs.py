"""Harvest (command, observation) pairs from SWE agent trajectories via the
HF datasets-server rows API. No full dataset download - we page through.

Each pair is labelled by labels.py deterministic rules, so the label is a
function of the observation text, not a human opinion.
"""
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(__file__))
from labels import pair_steps  # noqa: E402

DATASETS = [
    ("nebius/SWE-agent-trajectories", "default", "train"),
    ("SWE-bench/SWE-smith-trajectories", "default", "train"),
    ("nebius/SWE-rebench-openhands-trajectories", "default", "train"),
]
API = ("https://datasets-server.huggingface.co/rows"
       "?dataset={ds}&config={cfg}&split={sp}&offset={off}&length={n}")
DATA = os.environ.get("JANUS_DATA", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
OUT = os.path.join(DATA, "swe_pairs.jsonl")


def fetch(ds, cfg, sp, off, n):
    url = API.format(ds=urllib.parse.quote(ds, safe=""), cfg=cfg, sp=sp, off=off, n=n)
    with urllib.request.urlopen(url, timeout=90) as r:
        return json.loads(r.read().decode())


def main():
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    rows, seen, per = [], 0, 100
    import urllib.parse  # noqa: F401  (used in fetch via global)
    for ds, cfg, sp in DATASETS:
        off = 0
        while len(rows) < target:
            try:
                page = fetch(ds, cfg, sp, off, per)
            except Exception as e:
                print(f"  {ds} offset {off}: {type(e).__name__} - stop this dataset")
                break
            batch = page.get("rows", [])
            if not batch:
                break
            for item in batch:
                rec = item.get("row") or {}
                traj = rec.get("trajectory")
                if isinstance(traj, str):
                    try:
                        traj = json.loads(traj)
                    except json.JSONDecodeError:
                        continue
                if not isinstance(traj, list):
                    continue
                for p in pair_steps(traj):
                    p["source"] = ds
                    p["instance_id"] = rec.get("instance_id")
                    rows.append(p)
            seen += len(batch)
            off += per
            print(f"  {ds} rows={seen} pairs={len(rows)}", flush=True)
            if len(rows) >= target:
                break
            time.sleep(0.2)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    n_err = sum(1 for r in rows if r["is_error"])
    print(f"\nwrote {len(rows)} pairs -> {OUT}")
    print(f"  error {n_err} ({n_err/max(len(rows),1)*100:.1f}%)  ok {len(rows)-n_err}")
    from collections import Counter
    print("  classes:", dict(Counter(r["failure_class"] for r in rows).most_common()))


if __name__ == "__main__":
    main()
