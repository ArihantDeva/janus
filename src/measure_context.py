"""How much context do the failures actually need, and what does 32K unlock?

The dataset caps `x` at 1500 CHARS at harvest time -- that cap, not maxlen, is the
real constraint. Re-measure the raw distribution from the session logs so the
decision is made on the data rather than on the cap I inherited.

Reports chars AND tokens (Llama-3.2 tokenizer if available, else 3.5 chars/token)
at several percentiles, and what fraction of real failures fit in a given context
once prompt overhead is subtracted.
"""
import glob, json, os, sys
sys.path.insert(0,"/Users/arihantdeva/jevlab/src")
from tool_outcomes import GUARD, _text_of

STORES = (".pi/agent/sessions", ".pi-bolt/agent/sessions")
files = []
for s in STORES:
    d = os.path.join(os.path.expanduser("~"), s)
    if os.path.isdir(d):
        files += glob.glob(os.path.join(d, "**", "*.jsonl"), recursive=True)

texts = []
for f in files:
    try: fh = open(f, errors="ignore")
    except Exception: continue
    for ln in fh:
        if "toolResult" not in ln: continue
        try: r = json.loads(ln)
        except Exception: continue
        m = r.get("message") or {}
        if m.get("role") != "toolResult" or not m.get("isError"): continue
        t = _text_of(m)
        if not t.strip() or GUARD.search(t[:400]): continue
        texts.append(t)
    fh.close()

print(f"raw non-guard failures harvested: {len(texts)}")

try:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B")
    def ntoks(s): return len(tok(s, add_special_tokens=False)["input_ids"])
    how = "llama-3.2 tokenizer"
except Exception as e:
    def ntoks(s): return int(len(s)/3.5)+1
    how = f"3.5 chars/token estimate ({type(e).__name__})"
print(f"tokeniser: {how}")

import random
random.seed(0)
samp = random.sample(texts, min(3000, len(texts)))
import numpy as np
L = np.array([len(t) for t in texts])
T = np.array([ntoks(t) for t in samp])
print(f"\nCHARS  p50 {np.percentile(L,50):7.0f}  p90 {np.percentile(L,90):7.0f}  "
      f"p99 {np.percentile(L,99):7.0f}  max {L.max():8.0f}")
print(f"TOKENS p50 {np.percentile(T,50):7.0f}  p90 {np.percentile(T,90):7.0f}  "
      f"p99 {np.percentile(T,99):7.0f}  max {T.max():8.0f}")

print("\n=== what fits, given ~700 tokens of prompt+options overhead ===")
OVER = 700
for ctx in (768, 2048, 4096, 8192, 16384, 32768):
    budget = ctx - OVER
    fit = float((T <= budget).mean())
    print(f"  ctx {ctx:6d}: text budget {budget:6d} tok -> fits {100*fit:5.1f}% of failures")

print("\n=== the staged training ladder this implies (Llama is 131072 max, no RoPE scaling) ===")
print("  stage 1  maxlen  2048  (~1400 tok of text)  ->  %d%% of rows uncut" %
      round(100*float((T <= 1348).mean())))
print("  stage 2  maxlen  8192  (~7500 tok of text)  ->  %d%% of rows uncut" %
      round(100*float((T <= 7492).mean())))
print("  stage 3  maxlen 32768  (~32000 tok of text) ->  %d%% of rows uncut" %
      round(100*float((T <= 32068).mean())))
