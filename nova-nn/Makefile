
.PHONY: setup test lint format train

setup:
	python -m pip install -U pip
	pip install -e ".[dev]"
	pre-commit install || true

test:
	pytest -q

lint:
	ruff check .
	black --check .

format:
	ruff check --fix .
	ruff format .
	black .

train:
	python -m nova_nn.train --config configs/default.yaml
