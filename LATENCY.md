# LATENCY

CPU forward on hal0357 (main partition, `cuda: False`), Llama-3.2-1B RLCD encoder, n=60 timed forwards, batch-1, max_length=256.

| stat | run 1 (ms) | run 2 (ms) |
|---|---|---|
| min | 594 | 583 |
| p50 | 598 | 596 |
| p95 | 1210 | 1211 |
| p99 | 3011 | 2752 |
| max | 3011 | 2752 |

Two independent n=60 runs agree: steady state ~600ms, a ~2% tail above 1.2s
(p95), worst observed ~3s. Histogram (run 2, ms bins):

```
 500-700  | ################################################## 52
 700-900  | #### 4
 900-1300 | ## 2
1300-2800 | ## 2
```

Tail cause: first-forward warmup + occasional node jitter on shared hal0357,
not model variance — medians are identical across runs.

End-to-end hook timings (live Muse sessions, remote transport):

| condition | ms |
|---|---|
| cold (fresh allocation warming up) | 121846 |
| warm | 8824 / 3016 / 2878 |
| warm daemon (launchd, local socket) | ~1700 |

## Reading

Forward pass is the wall (~600ms steady CPU). Cold allocation spin-up dominates
first verdict; warm daemon kills the fixed costs.

## GPU forward (fp16, landed)

L40S, Llama-3.2-1B RLCD encoder, `.cuda().half()` + autocast, batch-1,
max_length=256, n=55 timed forwards (5 warmup discarded). Slurm job 63392006
COMPLETED exit 0. Raw timings in `~/slurm-63392006.out` (head node).

| stat | ms |
|---|---|
| min | 9.3 |
| p50 | 9.4 |
| p95 | 9.6 |
| max | 9.6 |

Sub-200ms path confirmed with 20x headroom: 9.4ms vs the 200ms bar.
Warm-hook end-to-end (1.6-8.8s) is dominated by ssh round-trip + daemon
framing, not the forward — the model itself is never the bottleneck on GPU.

Caveat: the `janus_gpu.py` bench needed a float-head + autocast workaround
for a Float-vs-Half dtype mismatch in `encoder_llama.py:383`. Production
`muse_hook` runs fp32 on CPU (no mismatch there); a GPU production path
should carry the same autocast guard.
