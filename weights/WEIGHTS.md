# WEIGHTS (reproducible checkpoint, 26KB in-repo)

The deployed classifier is a frozen base + a trained head. Only the head
is learned; the base is byte-identical to its public checkpoint.

## What is committed

| file | bytes | sha256 |
|---|---|---|
| `weights/freeze0_head.pt` | 25990 | `b893e1a0f534014e253ffd53491b2e2a816dd3adc6190b9b4e5299c49a770d16` |
| `weights/base_config.json` | config | base architecture record (see below) |
| `EVAL.md` / `eval/result_freeze0_artifact.json` | metrics | acc 0.8220, ECE 0.1027, bins |

## Base (not committed — pull from HuggingFace)

- model: `meta-llama/Llama-3.2-1B` (gated; accept the Llama 3.2 Community
  License on the model page first)
- architecture: `LlamaModel`, hidden 2048, 16 layers (`weights/base_config.json`)
- verification: after download, `config.json` must match
  `weights/base_config.json` field-for-field on
  `architectures`/`hidden_size`/`num_hidden_layers`/`model_type`.

## Reassemble

1. Download the base into `model_freeze0_rlcd/` (same layout as the training
   output: `config.json`, `model.safetensors`, tokenizer files).
2. Copy `weights/freeze0_head.pt` to `model_freeze0_rlcd/head.pt`.
3. Set `JEV_ENCODER=model_freeze0_rlcd JEV_ENCODER_HEAD=model_freeze0_rlcd/head.pt`
   (`T=0.5276694558560848`, recorded in `EVAL.md`).
4. Verify: `sha256sum` the head against the table above, then run
   `src/jevroute.py --report` — agreement must reproduce 82.20% ± sampling noise.

## Why not commit 4.7GB

`model.safetensors` (4.7GB) is byte-identical to the public checkpoint — the
encoder is never unfrozen (that IS the finding, see `ABLATION.md`). Committing
Meta's bytes duplicates the public artifact and complicates the Llama
Community License (§1.b.i) without adding one bit of reproducibility: the
pin + 26KB head reproduce the deployed checkpoint exactly.

## Licence

Base weights: Llama 3.2 Community License (accept on the model page; 700M-MAU
clause applies). This repo's code is MIT. The combination ships "Llama"-prefixed
per §1.b.i (see `MODEL_CARD.md`).
