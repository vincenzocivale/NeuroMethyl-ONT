.PHONY: install test lint registry data-bootstrap

install:
	pip install -e '.[dev]'

test:
	pytest

lint:
	ruff check .

registry:
	python scripts/data/validate_registry.py

data-bootstrap:
	bash scripts/data/bootstrap_external_data.sh
