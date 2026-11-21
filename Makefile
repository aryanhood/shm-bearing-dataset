.PHONY: install check lint typecheck test download train benchmark replay plots rebuild

install:
	pip install -e ".[train,dev]"

lint:
	ruff check .
	ruff format --check .

typecheck:
	mypy

test:
	pytest --cov=orderguard

check: lint typecheck test

download:
	orderguard download

train:
	orderguard train

benchmark:
	orderguard benchmark

replay:
	orderguard replay

plots:
	orderguard plots

rebuild: download train benchmark replay plots
