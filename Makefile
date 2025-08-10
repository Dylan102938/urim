PYTHON ?= python3
UV ?= uv

.PHONY: help sync lint fmt test build run

help:
	@echo "Targets:"
	@echo "  sync   - install dev dependencies via uv"
	@echo "  lint   - run ruff lints"
	@echo "  fmt    - run ruff format"
	@echo "  test   - run pytest"
	@echo "  build  - build wheel"
	@echo "  run    - run CLI (urim --help)"

sync:
	$(UV) sync --dev

lint:
	$(UV) run ruff check .

fmt:
	$(UV) run ruff format .

test:
	$(UV) run pytest -q

build:
	$(UV) build

run:
	$(UV) run urim --help
