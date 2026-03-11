# Project Structure

Overview of the repository layout.

## Top-level directories

- `.github/`: CI and GitHub templates
- `configs/`: configuration files
- `docs/`: API, architecture, and limitations docs
- `examples/`: runnable usage examples
- `scripts/`: local helper scripts that are not part of the package API
- `src/`: package source
- `tests/`: automated test suite

## Package layout

- `src/agents/`: agent interfaces and tracing helpers
- `src/benchmarks/`: task and benchmark loaders
- `src/evaluators/`: evaluation orchestration and result objects
- `src/metrics/`: scoring implementations
- `src/rl/`: parameter-tuning helpers and optional experimental modules
- `src/utils/`: config, validation, logging, and statistics utilities

## Key docs

- `README.md`: project overview and quick start
- `docs/API.md`: public interfaces
- `docs/ARCHITECTURE.md`: package design
- `docs/METHODS_AND_LIMITATIONS.md`: scope and tradeoffs
- `METRICS.md`: metric behavior
- `RL_GUIDE.md`: parameter tuning workflow

## Extending the repo

- Add a metric by implementing a metric class and wiring it into the evaluator
- Add a benchmark by extending the benchmark abstraction and loading new tasks
- Add an agent by subclassing `BaseAgent` and implementing `execute`
