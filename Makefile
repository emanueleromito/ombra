# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
# Ombra development commands. Run `make help` for the list.

PYTHON      ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
BUILD_DIR   ?= build
CUDA_ARCHS  ?= 75;80;86;89;90
TASK        ?= all
RESULTS_DIR ?= results-raw
BUILD_TYPE  ?= Release
GPU         ?= a10
CMD         ?= nvidia-smi

.DEFAULT_GOAL := help
.PHONY: help venv build build-cpu test test-gpu test-openfhe test-openfhe-local bench gpu-info results-pull fmt lint clean prebuild on-gpu

help: ## List commands
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-14s %s\n", $$1, $$2}'

venv: ## Create .venv with pytest, ruff and the docs packages
	python3 -m venv .venv
	.venv/bin/python -m pip install --quiet pytest ruff grpcio-tools cryptography numpy
	.venv/bin/python -m pip install --quiet -r docs/requirements.txt

build: ## Build C++/CUDA locally (needs CUDA)
	cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Debug -DOMBRA_ENABLE_CUDA=ON "-DCMAKE_CUDA_ARCHITECTURES=$(CUDA_ARCHS)"
	cmake --build $(BUILD_DIR) -j

build-cpu: ## Build C++ code without CUDA, with a CPU-only ombra_server (tests and development)
	cmake -S . -B $(BUILD_DIR)-cpu -DCMAKE_BUILD_TYPE=Debug -DOMBRA_ENABLE_CUDA=OFF
	cmake --build $(BUILD_DIR)-cpu -j

test: build-cpu ## Run CPU tests locally (C++ and Python)
	ctest --test-dir $(BUILD_DIR)-cpu --output-on-failure --no-tests=ignore
	$(PYTHON) -m pytest -m "not gpu and not openfhe"; code=$$?; [ $$code -eq 0 ] || [ $$code -eq 5 ]

test-gpu: ## Run GPU tests on Modal (A10)
	modal run modal_app.py::test_gpu

test-openfhe: ## Run tests marked openfhe on Modal (CPU only, pinned image)
	modal run modal_app.py::test_openfhe

test-openfhe-local: build-cpu ## Run tests marked openfhe here, no Modal (needs OpenFHE 1.6.1; the CI image has it)
	cmake -S ci/drivers -B $(BUILD_DIR)-openfhe -DCMAKE_BUILD_TYPE=Release -DOMBRA_DRIVER_OUT=$(CURDIR)/$(BUILD_DIR)
	cmake --build $(BUILD_DIR)-openfhe -j
	OMBRA_REQUIRE_NATIVE=1 cmake -S client -B $(BUILD_DIR)-native -DCMAKE_BUILD_TYPE=Release \
		-Dpybind11_DIR=$$($(PYTHON) -m pybind11 --cmakedir) \
		-DPython_EXECUTABLE=$$($(PYTHON) -c 'import sys; print(sys.executable)')
	cmake --build $(BUILD_DIR)-native -j
	cp $(BUILD_DIR)-native/_native*.so client/ombra/
	OMBRA_SERVER_BIN=$(CURDIR)/$(BUILD_DIR)-cpu/server/grpc/ombra_server $(PYTHON) -m pytest -m openfhe -v

bench: ## Run benchmarks on Modal (A100); TASK=<name> to pick one
	modal run modal_app.py::bench --task $(TASK)

prebuild: ## Build once on Modal CPUs for every GPU family (no GPU billed); BUILD_TYPE=Release|Debug
	modal run modal_app.py::prebuild --build-type $(BUILD_TYPE) --commit $$(git rev-parse --short HEAD)

on-gpu: ## Run CMD on one GPU family from the prebuilt binaries; GPU=t4|a10|a100|l4|l40s|h100
	modal run modal_app.py::on_gpu --gpu $(GPU) --build-type $(BUILD_TYPE) --cmd "$(CMD)"

gpu-info: ## Check GPU, CUDA and OpenFHE in the Modal image (A10)
	modal run modal_app.py::gpu_info

results-pull: ## Copy benchmark results from the Modal volume into $(RESULTS_DIR)/
	mkdir -p $(RESULTS_DIR)
	modal volume get --force ombra-data results $(RESULTS_DIR)

fmt: ## Format C++/CUDA (clang-format) and Python (ruff)
	@files=$$(git ls-files '*.cpp' '*.hpp' '*.h' '*.cu' '*.cuh'); [ -z "$$files" ] || clang-format -i $$files
	$(PYTHON) -m ruff format .

lint: ## Lint Python (ruff) and check C++ formatting
	$(PYTHON) -m ruff check .
	@files=$$(git ls-files '*.cpp' '*.hpp' '*.h' '*.cu' '*.cuh'); [ -z "$$files" ] || clang-format --dry-run --Werror $$files

clean: ## Remove local build directories
	rm -rf $(BUILD_DIR) $(BUILD_DIR)-cpu $(BUILD_DIR)-openfhe $(BUILD_DIR)-native
