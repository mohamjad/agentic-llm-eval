# Changelog

## [0.3.1] - 2026-03-11

Clarified repository scope and public documentation so the project is described
as a lightweight evaluation toolkit with optional experimental components.

Removed tracked git-history rewrite utilities from version control and added
ignore rules to keep commit-date manipulation artifacts out of future changes.

## [0.3.0] - 2026-02-07

Added optional neural policy components, semantic metrics, parameter-search
helpers, and statistics utilities.

Expanded the test suite, validation logic, and configuration handling.

## [0.2.0] - 2026-02-07

Added comprehensive test coverage with tests for all metric modules in test_metrics.py, tests for agent base classes and tracing in test_agents.py, tests for RL components in test_rl.py, tests for benchmark functionality in test_benchmarks.py, and improved existing evaluator tests.

Enhanced RL implementation with momentum-based parameter updates, experience replay buffer, adaptive learning rate decay, multi-objective optimization support, and performance tracking and statistics.

Improved safety metric with multi-level keyword detection for severe, moderate, and mild violations, security pattern detection for password leaks, code injection, and XSS, context-aware analysis to reduce false positives, and configurable severity thresholds.

Added centralized configuration system with new Config class supporting file and environment variable loading, JSON and YAML configuration file support, environment variable loading with prefix support, configuration validation, and default configuration values.

Added input validation with new validation.py module including comprehensive validators, parameter validation for AgentParameters, metric value validation, task input security checks, and configuration validation.

Added documentation including complete API documentation in docs/API.md, architecture documentation in docs/ARCHITECTURE.md, enhanced README with testing and quality checks, and improved code docstrings.

Enhanced CI/CD with improved GitHub Actions workflow, multi-platform testing for Linux, Windows, and macOS, multiple Python version support for 3.10, 3.11, and 3.12, code quality checks using flake8, black, and mypy, and coverage reporting with Codecov integration.

Added developer tools including pre-commit hooks configuration, Makefile for common tasks, and improved code formatting standards.

Changed RL Policy Network with complete rewrite and enhanced features, Safety Metric with major improvements in detection accuracy, Agent Parameters with automatic validation on initialization, and Configuration migrated to centralized Config system.

Fixed type hints compatibility issues, missing numpy dependency handling, and configuration loading edge cases.

## [0.1.0] - Initial Release

Added basic evaluation framework, core metrics including accuracy, efficiency, safety, coherence, and adaptability, RL training infrastructure, task benchmarks, and execution tracing.
