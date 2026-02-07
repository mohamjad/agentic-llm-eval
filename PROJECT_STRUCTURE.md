# Project Structure

Complete overview of the agentic-llm-eval project structure.

## Directory Layout

```
agentic-llm-eval/
│
├── .github/                    # GitHub configuration
│   ├── workflows/             # CI/CD workflows
│   │   └── ci.yml             # Continuous integration
│   ├── ISSUE_TEMPLATE/        # Issue templates
│   └── pull_request_template.md
│
├── configs/                    # Configuration files
│   ├── config.json            # Default configuration
│   └── example_config.json    # Example configuration
│
├── docs/                       # Documentation
│   ├── API.md                 # Complete API reference
│   └── ARCHITECTURE.md        # Architecture documentation
│
├── examples/                   # Example scripts
│   ├── README.md              # Examples guide
│   ├── simple_evaluation.py   # Basic evaluation
│   ├── rl_training_example.py # RL training demo
│   └── parameter_impact_example.py
│
├── scripts/                    # Development scripts
│   ├── README.md
│   └── create_commits.py      # Utility scripts
│
├── src/                        # Main package source code
│   ├── __init__.py            # Package initialization
│   │
│   ├── agents/                # Agent interfaces
│   │   ├── __init__.py
│   │   ├── base.py           # BaseAgent, AgentExecutionTrace
│   │   └── tracer.py         # AgentTracer
│   │
│   ├── benchmarks/            # Benchmark suites
│   │   ├── __init__.py
│   │   ├── base.py           # BaseBenchmark, Task
│   │   └── task_benchmark.py # TaskBenchmark
│   │
│   ├── evaluators/            # Evaluation logic
│   │   ├── __init__.py
│   │   ├── base.py           # BaseEvaluator, EvaluationResult
│   │   └── agent_evaluator.py # AgentEvaluator
│   │
│   ├── metrics/              # Metric calculations
│   │   ├── __init__.py
│   │   ├── accuracy.py       # AccuracyMetric
│   │   ├── efficiency.py    # EfficiencyMetric
│   │   ├── safety.py         # SafetyMetric
│   │   ├── coherence.py      # CoherenceMetric
│   │   └── adaptability.py   # AdaptabilityMetric
│   │
│   ├── rl/                   # Reinforcement learning
│   │   ├── __init__.py
│   │   ├── policy.py         # PolicyNetwork, AgentPolicy
│   │   └── trainer.py        # RLTrainer
│   │
│   └── utils/                # Utilities
│       ├── __init__.py
│       ├── config.py         # Config class
│       ├── validation.py     # Validation utilities
│       └── logger.py         # Logging setup
│
├── tests/                     # Test suite
│   ├── __init__.py
│   ├── test_agents.py        # Agent tests
│   ├── test_benchmarks.py   # Benchmark tests
│   ├── test_evaluator.py    # Evaluator tests
│   ├── test_metrics.py      # Metric tests
│   └── test_rl.py           # RL tests
│
├── .editorconfig             # Editor configuration
├── .gitignore               # Git ignore rules
├── .pre-commit-config.yaml  # Pre-commit hooks
├── CHANGELOG.md             # Version changelog
├── CODE_OF_CONDUCT.md       # Code of conduct
├── CONTRIBUTING.md          # Contribution guidelines
├── IMPROVEMENTS.md          # Improvements summary
├── LICENSE                  # MIT License
├── Makefile                 # Development commands
├── MANIFEST.in              # Package manifest
├── METRICS.md               # Metrics documentation
├── PROJECT_STRUCTURE.md     # This file
├── pytest.ini              # Pytest configuration
├── pyproject.toml           # Project metadata
├── README.md                # Main documentation
├── RL_GUIDE.md              # RL training guide
├── SECURITY.md              # Security policy
└── setup.py                 # Setup script
```

## Module Organization

### Core Modules

- **`src/evaluators/`**: Main evaluation orchestration
- **`src/agents/`**: Agent interface and execution tracing
- **`src/benchmarks/`**: Task definitions and benchmark suites
- **`src/metrics/`**: Individual metric implementations
- **`src/rl/`**: Reinforcement learning components
- **`src/utils/`**: Shared utilities

### Design Principles

1. **Separation of Concerns**: Each module has a single responsibility
2. **Explicit Interfaces**: Clear abstract base classes
3. **Composability**: Modules work together seamlessly
4. **Extensibility**: Easy to add new metrics, agents, benchmarks
5. **Testability**: All components are independently testable

## File Naming Conventions

- **Modules**: `snake_case.py`
- **Classes**: `PascalCase`
- **Functions**: `snake_case`
- **Constants**: `UPPER_SNAKE_CASE`
- **Tests**: `test_*.py`

## Import Structure

```python
# Top-level imports
from src.evaluators import AgentEvaluator
from src.benchmarks import TaskBenchmark
from src.agents import BaseAgent
from src.rl import RLTrainer

# Sub-module imports when needed
from src.metrics.accuracy import AccuracyMetric
from src.utils.validation import validate_agent_parameters
```

## Adding New Components

### Adding a New Metric

1. Create `src/metrics/your_metric.py`
2. Implement metric class with `collect()` method
3. Add to `src/metrics/__init__.py`
4. Add tests in `tests/test_metrics.py`
5. Update `METRICS.md`

### Adding a New Benchmark

1. Create `src/benchmarks/your_benchmark.py`
2. Extend `BaseBenchmark`
3. Implement `load_tasks()` method
4. Add to `src/benchmarks/__init__.py`
5. Add tests in `tests/test_benchmarks.py`

### Adding a New Agent Type

1. Extend `BaseAgent` in your code
2. Implement `execute()` method
3. Optionally use `AgentTracer` for automatic tracing
4. See `examples/` for patterns

## Configuration Files

- **`configs/config.json`**: Default configuration
- **`pyproject.toml`**: Project metadata and tool configs
- **`pytest.ini`**: Test configuration
- **`.pre-commit-config.yaml`**: Pre-commit hooks
- **`.editorconfig`**: Editor settings

## Documentation Files

- **`README.md`**: Main project documentation
- **`docs/API.md`**: Complete API reference
- **`docs/ARCHITECTURE.md`**: Architecture details
- **`METRICS.md`**: Metric explanations
- **`RL_GUIDE.md`**: RL training guide
- **`CONTRIBUTING.md`**: Contribution guidelines
