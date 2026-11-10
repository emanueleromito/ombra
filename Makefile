# Ombra development commands. Run `make help` for the list.

PYTHON      ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
BUILD_DIR   ?= build
CUDA_ARCHS  ?= 80;86
TASK        ?= all

.DEFAULT_GOAL := help
.PHONY: help venv build build-cpu test test-gpu test-openfhe bench gpu-info results-pull fmt lint clean

help: ## List commands
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-14s %s\n", $$1, $$2}'

venv: ## Create .venv with pytest and ruff
	python3 -m venv .venv
	.venv/bin/python -m pip install --quiet pytest ruff grpcio-tools

build: ## Build C++/CUDA locally (needs CUDA)
	cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Debug -DOMBRA_ENABLE_CUDA=ON "-DCMAKE_CUDA_ARCHITECTURES=$(CUDA_ARCHS)"
	cmake --build $(BUILD_DIR) -j

build-cpu: ## Build C++ code without CUDA (for CPU-only tests)
	cmake -S . -B $(BUILD_DIR)-cpu -DCMAKE_BUILD_TYPE=Debug -DOMBRA_ENABLE_CUDA=OFF
	cmake --build $(BUILD_DIR)-cpu -j

test: build-cpu ## Run CPU tests locally (C++ and Python)
	ctest --test-dir $(BUILD_DIR)-cpu --output-on-failure --no-tests=ignore
	$(PYTHON) -m pytest -m "not gpu and not openfhe"; code=$$?; [ $$code -eq 0 ] || [ $$code -eq 5 ]

test-gpu: ## Run GPU tests on Modal (A10)
	modal run modal_app.py::test_gpu

test-openfhe: ## Run tests marked openfhe on Modal (CPU only, pinned image)
	modal run modal_app.py::test_openfhe

bench: ## Run benchmarks on Modal (A100); TASK=t0.4 to pick one
	modal run modal_app.py::bench --task $(TASK)

gpu-info: ## Check GPU, CUDA and OpenFHE in the Modal image (A10)
	modal run modal_app.py::gpu_info

results-pull: ## Copy benchmark results from the Modal volume into docs/plan/results/raw/
	mkdir -p docs/plan/results/raw
	modal volume get --force ombra-data results docs/plan/results/raw

fmt: ## Format C++/CUDA (clang-format) and Python (ruff)
	@files=$$(git ls-files '*.cpp' '*.hpp' '*.h' '*.cu' '*.cuh'); [ -z "$$files" ] || clang-format -i $$files
	$(PYTHON) -m ruff format .

lint: ## Lint Python (ruff) and check C++ formatting
	$(PYTHON) -m ruff check .
	@files=$$(git ls-files '*.cpp' '*.hpp' '*.h' '*.cu' '*.cuh'); [ -z "$$files" ] || clang-format --dry-run --Werror $$files

clean: ## Remove local build directories
	rm -rf $(BUILD_DIR) $(BUILD_DIR)-cpu
