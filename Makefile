.PHONY: install check lint typecheck test web download evaluate train

install:
	pip install -e ".[dev]"
	cd web && npm install

lint:
	ruff check .
	ruff format --check .

typecheck:
	mypy

test:
	pytest --cov=bearingscope

web:
	cd web && npm run check

check: lint typecheck test web

download:
	bearingscope download

evaluate:
	bearingscope evaluate

train:
	bearingscope train
