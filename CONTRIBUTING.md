# Contributing to agentic-llm-eval

Thanks for your interest in contributing. This document provides guidelines and instructions.

For development setup, fork and clone the repository, create a virtual environment with python -m venv venv and activate it, install dependencies with pip install -r requirements.txt and pip install -e ".[dev]", then run tests with pytest tests/ -v.

Code style uses black for formatting with line length 127, follows flake8 guidelines for linting, uses type hints for all functions, and uses Google-style docstrings. Format code with black src tests, check linting with flake8 src tests, and type check with mypy src.

To make changes, create a branch with git checkout -b feature/your-feature-name, make your changes with clear documented code, add tests for new functionality, update documentation as needed, run tests and checks with make quality and pytest tests/ -v, commit with clear descriptive messages following conventional commit format when possible, then push and create a pull request.

Testing guidelines include writing tests for all new features, ensuring existing tests pass, aiming for high test coverage, and using descriptive test names.

Documentation should be updated in README.md for user-facing changes, API.md for API changes, and include docstrings for all public functions and classes with examples when helpful.

The pull request process involves ensuring all tests pass, updating documentation, adding changelog entry if needed, requesting review, and addressing feedback.

Open an issue for questions or discussions about contributions.
