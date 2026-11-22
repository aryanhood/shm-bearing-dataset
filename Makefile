.PHONY: install check lint typecheck test demo

install:
	pip install -e ".[dev]"

lint:
	ruff check .
	ruff format --check .

typecheck:
	mypy

test:
	pytest --cov=vibcheck

check: lint typecheck test

demo:
	vibcheck check examples/motor_inner_fault.csv --rpm 1750 --bearing 6205 --report report.html
