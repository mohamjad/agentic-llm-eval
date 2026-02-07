.PHONY: install test lint format type-check clean docs

install:
	pip install -r requirements.txt
	pip install -e .

test:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

test-fast:
	pytest tests/ -v

lint:
	flake8 src tests --max-line-length=127 --max-complexity=10

format:
	black src tests --line-length=127

format-check:
	black --check src tests --line-length=127

type-check:
	mypy src --ignore-missing-imports

clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -r {} +
	find . -type d -name ".pytest_cache" -exec rm -r {} +
	find . -type d -name ".mypy_cache" -exec rm -r {} +
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf dist/
	rm -rf build/

docs:
	@echo "Documentation is in docs/ directory"

quality: format-check lint type-check

all: clean install quality test
