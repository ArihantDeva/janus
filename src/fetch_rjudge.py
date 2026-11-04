"""Port R-Judge from the reference repo into a flat labelled JSONL.

The upstream files are hand-edited and carry literal backslash-escaped
underscores ("risk\_description", "id\_rsa"), so they are NOT valid JSON.
We normalise those before parsing rather than trusting json.loads.
"""
import json, os, re, urllib.request

APPS = ["Program", "Web", "Finance", "IoT", "Application"]
GH = "https://api.github.com/repos/Lordog/R-Judge/contents/data/"
RAW = "https://raw.githubusercontent.com/Lordog/R-Judge/main/data/"
DATA = os.environ.get("JANUS_DATA", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
OUT = os.path.join(DATA, "rjudge.jsonl")


def get(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read().decode("utf-8", "replace")


def sanitize(text):
    # undo hand-edit artefacts: '\_' -> '_' , but keep real JSON escapes intact
    return re.sub(r"\\(?=[_])", "", text)


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    rows, skipped = [], []
    for app in APPS:
        listing = json.loads(get(GH + app))
        for entry in listing:
            name = entry["name"]
            raw = get(RAW + app + "/" + name)
            try:
                recs = json.loads(sanitize(raw))
            except json.JSONDecodeError as e:
                skipped.append((app, name, str(e)[:80]))
                continue
            for rec in recs:
                rows.append({
                    "app": app,
                    "file": name,
                    "id": rec.get("id"),
                    "scenario": rec.get("scenario"),
                    "goal": rec.get("goal"),
                    "contents": rec.get("contents"),
                    "label": int(rec.get("label", -1)),
                    "risk_description": rec.get("risk_description"),
                })
    with open(OUT, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    n_un = sum(1 for r in rows if r["label"] == 1)
    print(f"wrote {len(rows)} records -> {OUT}")
    print(f"  unsafe {n_un} ({n_un/len(rows)*100:.1f}%)  safe {len(rows)-n_un}")
    by_app = {}
    for r in rows:
        by_app.setdefault(r["app"], [0, 0])
        by_app[r["app"]][r["label"]] += 1
    for a, (s, u) in sorted(by_app.items()):
        print(f"  {a:12s} safe={s:4d} unsafe={u:4d}")
    if skipped:
        print(f"SKIPPED {len(skipped)}: {skipped[:4]}")


if __name__ == "__main__":
    main()
