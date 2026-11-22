.PHONY: install lint typecheck test reproduce verify-numbers check

install:
	pip install -e ".[ml,dev]"

lint:
	ruff check .
	ruff format --check .

typecheck:
	mypy

test:
	pytest --cov=bearingbench

reproduce:
	python scripts/reproduce.py

verify-numbers:
	python scripts/verify_report_numbers.py

check: lint typecheck test verify-numbers
