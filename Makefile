# Thin wrapper so `make check` works without mise. Python deps come from uv so
# a clean checkout needs no virtualenv management.
PYRUN := uv run --with numpy --with pytest

.PHONY: check test sync train probe distill eval-distill aggregate
check:
	@$(PYRUN) bash scripts/check.sh
test:
	@$(PYRUN) python -m pytest tests/ -q
sync:
	@bash scripts/sync.sh
train:
	@bash scripts/train.sh
probe:
	@bash scripts/probe.sh
# The full pipeline in one command, from raw session stores to a scored checkpoint.
# Every stage is idempotent: harvesting is deterministic, labelling resumes, and a
# rerun costs nothing for rows already labelled.
distill:
	@$(PYRUN) python src/build_distill.py
	@$(PYRUN) python src/label_janus.py
eval-distill:
	@$(PYRUN) --with torch --with transformers python src/eval_distill.py \
		--model $$(ls -d out_distill/model_seed0_distill 2>/dev/null || echo out_distill/model_seed0)
# mean+spread across seeds; excludes any run that did not finish or went non-finite
aggregate:
	@$(PYRUN) python src/aggregate.py --dir out_distill --out out_distill/aggregate.json
