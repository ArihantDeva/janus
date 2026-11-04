# Submission checklist — Meta Global AI Developer Hackathon

Deadline: **Wed Nov 4, 9:00 PM ET** via the CV platform (one submission).
Track: **Indie Developer**. Problem statement: **Push the Frontier**.

## Representative
- [ ] Name the Representative (acts for the project; solo → yourself)

## Links (verify each opens in a logged-out window)
- [ ] Public repo: https://github.com/ArihantDeva/janus (PUBLIC ✓ 2026-11-01; renamed from glimmer-guard 2026-11-04)
- [ ] 3-minute demo video link (to record; must show ONLY work built Oct 27–Nov 4)
- [ ] Every contributor listed on the submission page

## Video provenance (new-work-only rule)
- [ ] Final video is newly cut for this submission (not a reuse)
- [ ] demo/demo_real.mp4, demo_with_without.mp4, demo_live_contributor.mp4
      (84–183KB, committed pre-freeze): confirm all footage postdates Oct 27,
      or re-shoot. DO NOT claim pre-Oct-27 work as new.

## Stage-One viability (pass/fail)
- [x] Incorporates Meta models: Janus-1B verdict tier (Llama-3.2-1B, deployed),
      Glimmer 30B tier (QLoRA complete 2026-11-01: job 63397362, 2738 steps,
      EVAL n=3410 zero-shot 33.3%/0.3347 vs qlora 52.7%/0.2718 — artifact
      eval/glimmer_janus_seed0.json; negative result: adapter collapsed to
      majority class, reported honestly in README), Meta Model API
      escalation route (key present, measured), MuseSpark eval harness.
- [x] Not an anti-project (no medical/RAG-chatbot/screener content — scanned).
- [x] No secrets in tree (scanned; only false-positive prose hits).

## Results banked (2026-11-02)
- [x] Glimmer QLoRA run complete + EVAL artifact in repo
- [x] `make check` green (541 passed, 11 skipped; floor + leak + guard green)
- [x] 4-vs-3 class contract fault fixed (train_distill/eval_distill/remote_slurm
      -> CLASSES_3; RED test tests/test_deployed_classes.py; 602 torch-suite green)
- [ ] Demo video (blocked: shoot only, no code dependency)
- [ ] Representative name + CV submission page (operator action)

## Theme line (must read identically everywhere)
- [x] README.md: Push the Frontier ✓
- [x] README_HACKATHON.md: Push the Frontier ✓ (fixed 2026-11-01)
