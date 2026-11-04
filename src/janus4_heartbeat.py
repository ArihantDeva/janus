"""Pre-load heartbeat: card, VRAM, cache. A silent stall must be
distinguishable from a slow load -- the 0-byte .out failure mode that cost
the first shape attempt 50 minutes of ambiguity.

The try/except is load-bearing: diagnostics must never kill the run.
"""

from __future__ import annotations


def _gpu_line(tag, d):
    import torch
    p = torch.cuda.get_device_properties(d)
    f = torch.cuda.mem_get_info(d)[0] / 1e9
    print(f"[{tag}] gpu{d}={p.name} total={p.total_memory/1e9:.0f}G "
          f"free={f:.1f}G", flush=True)


def heartbeat(tag):
    print(f"[{tag}] starting", flush=True)
    try:
        import torch
        if torch.cuda.is_available():
            for d in range(torch.cuda.device_count()):
                _gpu_line(tag, d)
    except Exception as e:  # heartbeat must never kill the run
        print(f"[{tag}] heartbeat failed: {e}", flush=True)


def progress(tag, done, total, t0, every=50, step=None):
    """One line per `every` batches plus the final batch. Silence between
    progress lines is bounded and expected; silence past the bound is a
    stall -- the failure mode that cost shape 63395500 four silent hours."""
    import time
    n = step if step is not None else done
    if n % every == 0 or done >= total:
        print(f"  {tag} {min(done, total)}/{total} "
              f"({time.time()-t0:.0f}s)", flush=True)
