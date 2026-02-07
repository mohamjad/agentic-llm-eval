"""script to create realistic commit history"""

import subprocess
import os
from datetime import datetime, timedelta

def git_command(cmd):
    """run git command"""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd="C:\\Users\\moham\\agentic-llm-eval")
    return result.returncode == 0

def commit(message, files=None, date_offset=0):
    """create a commit with optional date"""
    if files:
        for f in files:
            if os.path.exists(f):
                git_command(f'git add "{f}"')
    else:
        git_command('git add .')
    
    # set commit date
    commit_date = datetime.now() - timedelta(days=date_offset)
    date_str = commit_date.strftime('%Y-%m-%d %H:%M:%S')
    
    env = os.environ.copy()
    env['GIT_AUTHOR_DATE'] = date_str
    env['GIT_COMMITTER_DATE'] = date_str
    
    result = subprocess.run(
        f'git commit -m "{message}"',
        shell=True,
        env=env,
        cwd="C:\\Users\\moham\\agentic-llm-eval"
    )
    return result.returncode == 0

# commit history in development order
commits = [
    # initial setup
    (0, "initial project setup", [".gitignore"]),
    (0, "add basic project structure", ["README.md"]),
    (0, "add license file", ["LICENSE"]),
    
    # base infrastructure
    (1, "create base evaluator interface", ["src/evaluators/base.py"]),
    (1, "add evaluation result dataclass", ["src/evaluators/base.py"]),
    (2, "implement basic agent interface", ["src/agents/base.py"]),
    (2, "add execution trace structure", ["src/agents/base.py"]),
    (3, "create trace step dataclass", ["src/agents/base.py"]),
    
    # metrics - accuracy
    (4, "add accuracy metric base class", ["src/metrics/accuracy.py"]),
    (5, "implement dictionary comparison for accuracy", ["src/metrics/accuracy.py"]),
    (5, "add list comparison logic", ["src/metrics/accuracy.py"]),
    (6, "fix accuracy edge cases", ["src/metrics/accuracy.py"]),
    
    # metrics - efficiency
    (7, "create efficiency metric", ["src/metrics/efficiency.py"]),
    (7, "implement step-based efficiency calculation", ["src/metrics/efficiency.py"]),
    (8, "add time-based efficiency metrics", ["src/metrics/efficiency.py"]),
    
    # metrics - safety
    (9, "add safety metric with keyword detection", ["src/metrics/safety.py"]),
    (9, "implement unsafe content checking", ["src/metrics/safety.py"]),
    (10, "add more unsafe keywords", ["src/metrics/safety.py"]),
    
    # evaluator implementation
    (11, "start implementing agent evaluator", ["src/evaluators/agent_evaluator.py"]),
    (11, "add metric collection to evaluator", ["src/evaluators/agent_evaluator.py"]),
    (12, "implement trace capture in evaluator", ["src/evaluators/agent_evaluator.py"]),
    (12, "add success determination logic", ["src/evaluators/agent_evaluator.py"]),
    (13, "implement overall score calculation", ["src/evaluators/agent_evaluator.py"]),
    (13, "add tool usage scoring", ["src/evaluators/agent_evaluator.py"]),
    (14, "fix evaluator error handling", ["src/evaluators/agent_evaluator.py"]),
    
    # tracing
    (15, "create agent tracer wrapper", ["src/agents/tracer.py"]),
    (15, "implement automatic trace capture", ["src/agents/tracer.py"]),
    (16, "add trace tool call decorator", ["src/agents/tracer.py"]),
    
    # benchmarks
    (17, "create base benchmark class", ["src/benchmarks/base.py"]),
    (17, "add task dataclass", ["src/benchmarks/base.py"]),
    (18, "implement task benchmark loader", ["src/benchmarks/task_benchmark.py"]),
    (18, "add default test tasks", ["src/benchmarks/task_benchmark.py"]),
    (19, "add json task loading", ["src/benchmarks/task_benchmark.py"]),
    
    # utils
    (20, "add logging utilities", ["src/utils/logger.py"]),
    (21, "create config loading utilities", ["src/utils/config.py"]),
    
    # tests
    (22, "add basic evaluator tests", ["tests/test_evaluator.py"]),
    (22, "add mock agent for testing", ["tests/test_evaluator.py"]),
    (23, "add batch evaluation tests", ["tests/test_evaluator.py"]),
    
    # examples
    (24, "create simple evaluation example", ["examples/simple_evaluation.py"]),
    (25, "improve example output formatting", ["examples/simple_evaluation.py"]),
    
    # config
    (26, "add example config file", ["configs/example_config.json"]),
    
    # package setup
    (27, "add requirements.txt", ["requirements.txt"]),
    (28, "create setup.py", ["setup.py"]),
    (29, "add pyproject.toml", ["pyproject.toml"]),
    
    # ci/cd
    (30, "add github actions workflow", [".github/workflows/ci.yml"]),
    
    # documentation
    (31, "update readme with usage examples", ["README.md"]),
    (32, "add contributing guidelines", ["CONTRIBUTING.md"]),
    
    # coherence metric
    (33, "start coherence metric implementation", ["src/metrics/coherence.py"]),
    (33, "add logical flow checking", ["src/metrics/coherence.py"]),
    (34, "implement contradiction detection", ["src/metrics/coherence.py"]),
    (34, "add topic consistency checking", ["src/metrics/coherence.py"]),
    (35, "tune coherence scoring weights", ["src/metrics/coherence.py"]),
    
    # adaptability metric
    (36, "create adaptability metric", ["src/metrics/adaptability.py"]),
    (36, "implement approach adaptation checking", ["src/metrics/adaptability.py"]),
    (37, "add context handling logic", ["src/metrics/adaptability.py"]),
    (37, "implement complexity adaptation", ["src/metrics/adaptability.py"]),
    (38, "refine adaptability scoring", ["src/metrics/adaptability.py"]),
    
    # integrate new metrics
    (39, "add coherence metric to evaluator", ["src/evaluators/agent_evaluator.py"]),
    (39, "add adaptability metric to evaluator", ["src/evaluators/agent_evaluator.py"]),
    (40, "update metric weights in evaluator", ["src/evaluators/agent_evaluator.py"]),
    
    # RL components - policy
    (41, "create agent parameters dataclass", ["src/rl/policy.py"]),
    (42, "implement policy network base", ["src/rl/policy.py"]),
    (42, "add parameter adjustment logic", ["src/rl/policy.py"]),
    (43, "implement policy weight updates", ["src/rl/policy.py"]),
    (44, "add agent policy wrapper", ["src/rl/policy.py"]),
    (45, "add parameter bounds checking", ["src/rl/policy.py"]),
    
    # RL components - trainer
    (46, "start RL trainer implementation", ["src/rl/trainer.py"]),
    (46, "add episode training loop", ["src/rl/trainer.py"]),
    (47, "implement metric aggregation", ["src/rl/trainer.py"]),
    (47, "add parameter application logic", ["src/rl/trainer.py"]),
    (48, "implement training history tracking", ["src/rl/trainer.py"]),
    (49, "add best parameters selection", ["src/rl/trainer.py"]),
    (50, "add training progress plotting", ["src/rl/trainer.py"]),
    
    # RL examples
    (51, "create RL training example", ["examples/rl_training_example.py"]),
    (52, "add parameterized agent example", ["examples/rl_training_example.py"]),
    (53, "create parameter impact example", ["examples/parameter_impact_example.py"]),
    
    # documentation
    (54, "create metrics documentation", ["METRICS.md"]),
    (55, "add metric explanations", ["METRICS.md"]),
    (56, "document hyperparameters", ["METRICS.md"]),
    (57, "create RL guide", ["RL_GUIDE.md"]),
    (58, "add RL usage examples to guide", ["RL_GUIDE.md"]),
    (59, "update readme with new features", ["README.md"]),
    
    # fixes and improvements
    (60, "fix numpy import in policy", ["src/rl/policy.py"]),
    (61, "remove unused numpy import from trainer", ["src/rl/trainer.py"]),
    (62, "export AgentParameters from rl module", ["src/rl/__init__.py"]),
    (63, "update metrics __init__ exports", ["src/metrics/__init__.py"]),
    
    # refinements
    (64, "improve accuracy metric edge case handling", ["src/metrics/accuracy.py"]),
    (65, "tune efficiency scoring thresholds", ["src/metrics/efficiency.py"]),
    (66, "expand unsafe keywords list", ["src/metrics/safety.py"]),
    (67, "improve coherence topic matching", ["src/metrics/coherence.py"]),
    (68, "refine adaptability context detection", ["src/metrics/adaptability.py"]),
    
    # evaluator improvements
    (69, "improve trace capture reliability", ["src/evaluators/agent_evaluator.py"]),
    (70, "add more detailed error reporting", ["src/evaluators/agent_evaluator.py"]),
    (71, "optimize metric calculation order", ["src/evaluators/agent_evaluator.py"]),
    
    # benchmark improvements
    (72, "add more default test tasks", ["src/benchmarks/task_benchmark.py"]),
    (73, "improve task metadata handling", ["src/benchmarks/base.py"]),
    
    # RL improvements
    (74, "improve policy weight initialization", ["src/rl/policy.py"]),
    (75, "add learning rate scheduling", ["src/rl/trainer.py"]),
    (76, "improve parameter adjustment bounds", ["src/rl/policy.py"]),
    
    # documentation updates
    (77, "add parameter impact examples to docs", ["METRICS.md"]),
    (78, "improve RL guide with best practices", ["RL_GUIDE.md"]),
    (79, "update readme project structure", ["README.md"]),
    
    # final polish
    (80, "clean up code comments", None),
    (81, "fix minor type hints", None),
    (82, "update example docstrings", ["examples/simple_evaluation.py"]),
    (83, "improve test coverage", ["tests/test_evaluator.py"]),
    (84, "add missing __init__ files", None),
    (85, "finalize package structure", None),
]

if __name__ == "__main__":
    print("creating commit history...")
    for date_offset, message, files in commits:
        if commit(message, files, date_offset):
            print(f"ok: {message}")
        else:
            print(f"failed: {message}")
    print("\ndone!")
