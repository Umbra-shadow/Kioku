# Kioku v1 — make dev / test / eval / demo / deploy

PY ?= $(shell test -x .venv/bin/python && echo .venv/bin/python || echo python3)
RUSTC ?= rustc --edition=2021

.PHONY: dev test test-rust test-py eval demo deploy kiokud clean

kiokud: substrate/kiokud  ## build the substrate daemon

substrate/kiokud: substrate/kiokud.rs substrate/cadran_vram.rs substrate/cadran_storage.rs substrate/space.rs
	cd substrate && $(RUSTC) -C opt-level=3 kiokud.rs -o kiokud

test: test-rust test-py  ## all tests

test-rust:
	cd substrate && $(RUSTC) --test kiokud.rs -o kiokud_tests && ./kiokud_tests
	cd substrate && $(RUSTC) --test cadran_vgpu.rs -o cadran_tests && ./cadran_tests

test-py:
	$(PY) -m pytest

dev: kiokud  ## run the FastAPI engine (Qwen Cloud brain via .env)
	$(PY) -m uvicorn engine.main:get_app --factory --reload --port 8000

eval:  ## run the eval harness (arrives with build step 7)
	@echo "make eval arrives with the eval harness (build step 7)"

demo: dev

deploy:
	@echo "see deploy/alibaba/deploy.sh (arrives with build step 8)"

clean:
	rm -f substrate/kiokud substrate/kiokud_tests substrate/cadran_tests substrate/cadran_engine
	rm -rf .pytest_cache engine/__pycache__ engine/tests/__pycache__
