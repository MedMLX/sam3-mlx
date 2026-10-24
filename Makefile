# Development and qualification targets require the public Git checkout.
# Runtime sdists intentionally omit this Makefile, tools, tests and evidence.
.DEFAULT_GOAL := qa
.PHONY: check-env loc clean
TESTS ?= tests
PYTEST_ARGS ?=
SOURCE_DIRS ?= sam3_mlx scripts tests experiments
GENERATED_PATHS ?= .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
export PYTHONNOUSERSITE := 1
PYTHON ?= .venv/bin/python

.PHONY: env lint typecheck test torch-test qa parity-qa benchmark benchmark-fast-preprocessing benchmark-regression-check benchmark-roi-align preprocess-parity-check roi-align-parity-check roi-align-oracle-check artifact-check release-evidence-audit release-check

BENCHMARK_CHECKPOINT ?= checkpoints/mlx-community/sam3-image/model.safetensors
BENCHMARK_BASELINE ?= benchmarks/baselines/apple-silicon-m1-max-20260829.json
ROI_ALIGN_ORACLE_RUN ?= $(PYTHON)
ROI_ALIGN_ORACLE_OUT_DIR ?= benchmarks/results

env:
	uv lock --check
	uv sync --locked

check-env:
	@command -v "$(PYTHON)" >/dev/null || { echo "Missing project Python: run make env explicitly, or set PYTHON to its existing interpreter" >&2; exit 1; }

lint: check-env
	$(PYTHON) -m ruff check .

typecheck: check-env
	@command -v node >/dev/null || { echo "Maintained host Node is required for Pyright" >&2; exit 1; }
	PYRIGHT_PYTHON_NODEJS_WHEEL=0 PYRIGHT_PYTHON_GLOBAL_NODE=1 $(PYTHON) -m pyright --project . --warnings --pythonpath "$(PYTHON)"

test: check-env
	$(PYTHON) -m pytest -q $(TESTS) $(PYTEST_ARGS)

torch-test:
	SAM3_MLX_REQUIRE_TORCHVISION=1 SAM3_MLX_REQUIRE_OFFICIAL_CHECKOUT=1 $(ROI_ALIGN_ORACLE_RUN) -m pytest -q

qa: lint typecheck

parity-qa:
	$(MAKE) torch-test
	$(MAKE) roi-align-oracle-check

benchmark:
	$(PYTHON) scripts/benchmark_image_runtime.py \
		--checkpoint $(BENCHMARK_CHECKPOINT) \
		--out benchmarks/results/current.json

benchmark-fast-preprocessing:
	$(PYTHON) scripts/benchmark_image_runtime.py \
		--checkpoint $(BENCHMARK_CHECKPOINT) \
		--preprocessing fast \
		--out benchmarks/results/fast-preprocessing-current.json

benchmark-roi-align:
	$(PYTHON) scripts/benchmark_roi_align.py \
		--out benchmarks/results/roi-align-current.json

benchmark-regression-check:
	$(PYTHON) scripts/benchmark_image_runtime.py \
		--checkpoint $(BENCHMARK_CHECKPOINT) \
		--resolution 504 \
		--resolution 672 \
		--resolution 1008 \
		--out benchmarks/results/current.json \
		--baseline $(BENCHMARK_BASELINE) \
		--comparison-out benchmarks/results/comparison.json \
		--metric resolutions.504.full_image_text.median_s \
		--metric resolutions.504.full_cached_text.median_s \
		--metric resolutions.504.set_image.median_s \
		--metric resolutions.672.full_image_text.median_s \
		--metric resolutions.672.full_cached_text.median_s \
		--metric resolutions.672.set_image.median_s \
		--metric resolutions.1008.full_image_text.median_s \
		--metric resolutions.1008.full_cached_text.median_s \
		--metric resolutions.1008.set_image.median_s \
		--metric peak_active_memory_bytes

preprocess-parity-check:
	SAM3_MLX_REQUIRE_TORCHVISION=1 $(ROI_ALIGN_ORACLE_RUN) -m pytest -q tests/image/test_sam3_image_processor.py::test_transform_matches_torchvision_on_synthetic_aspect_ratios_when_available

roi-align-parity-check:
	SAM3_MLX_REQUIRE_TORCHVISION=1 $(ROI_ALIGN_ORACLE_RUN) -m pytest -q tests/image/test_roi_align.py -k torchvision

roi-align-oracle-check: roi-align-parity-check
	$(ROI_ALIGN_ORACLE_RUN) scripts/fuzz_roi_align.py --dtype fp32 --require-full-matrix --out $(ROI_ALIGN_ORACLE_OUT_DIR)/roi-align-fuzz-fp32.json
	$(ROI_ALIGN_ORACLE_RUN) scripts/fuzz_roi_align.py --dtype fp16 --require-full-matrix --out $(ROI_ALIGN_ORACLE_OUT_DIR)/roi-align-fuzz-fp16.json
	$(ROI_ALIGN_ORACLE_RUN) scripts/fuzz_roi_align.py --dtype bf16 --require-full-matrix --out $(ROI_ALIGN_ORACLE_OUT_DIR)/roi-align-fuzz-bf16.json

artifact-check:
	$(PYTHON) scripts/validate_release.py

release-evidence-audit:
	$(PYTHON) scripts/audit_release_evidence.py

release-check:
	$(MAKE) parity-qa
	$(MAKE) artifact-check
	$(MAKE) release-evidence-audit

loc:
	@set -eu; . ./devtools/repo-paths.sh; \
	validate_repo_paths SOURCE_DIRS $(SOURCE_DIRS); \
	list=$$(mktemp); trap 'rm -f "$$list"' EXIT; \
	for dir in $(SOURCE_DIRS); do \
		if [ -e "$$dir" ]; then find "./$$dir" -type f \
			\( -name '*.py' -o -name '*.pyi' -o -name '*.sh' -o -name '*.ts' -o -name '*.tsx' -o -name '*.js' -o -name '*.jsx' -o -name '*.rs' \) -print0 >> "$$list"; fi; \
	done; test -s "$$list" || { echo "No source files found"; exit 1; }; xargs -0 wc -l < "$$list"

clean:
	@set -eu; . ./devtools/repo-paths.sh; \
	validate_repo_paths GENERATED_PATHS $(GENERATED_PATHS); \
	validate_repo_paths SOURCE_DIRS $(SOURCE_DIRS); \
	for path in $(GENERATED_PATHS); do \
		rm -rf -- "$$path"; \
	done; for dir in $(SOURCE_DIRS); do \
		if [ -d "$$dir" ]; then find "./$$dir" -type d -name __pycache__ -prune -exec rm -rf {} +; fi; \
	done
