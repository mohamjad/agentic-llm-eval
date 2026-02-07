# Contributing to agentic-llm-eval

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing.

## Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/mohamjad/agentic-llm-eval.git
   cd agentic-llm-eval
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -e ".[dev]"
   ```

4. **Run tests**
   ```bash
   pytest tests/ -v
   ```

## Code Style

- **Formatting**: Use `black` with line length 127
- **Linting**: Follow `flake8` guidelines
- **Type hints**: Use type hints for all functions
- **Docstrings**: Use Google-style docstrings

```bash
# Format code
black src tests

# Check linting
flake8 src tests

# Type checking
mypy src
```

## Making Changes

1. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write clear, documented code
   - Add tests for new functionality
   - Update documentation as needed

3. **Run tests and checks**
   ```bash
   make quality  # Runs all quality checks
   pytest tests/ -v  # Run tests
   ```

4. **Commit your changes**
   - Use clear, descriptive commit messages
   - Follow conventional commit format when possible

5. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

## Testing Guidelines

- Write tests for all new features
- Ensure existing tests pass
- Aim for high test coverage
- Use descriptive test names

## Documentation

- Update README.md for user-facing changes
- Update API.md for API changes
- Add docstrings to all public functions/classes
- Include examples in docstrings when helpful

## Pull Request Process

1. Ensure all tests pass
2. Update documentation
3. Add changelog entry if needed
4. Request review
5. Address feedback

## Questions?

Open an issue for questions or discussions about contributions.
