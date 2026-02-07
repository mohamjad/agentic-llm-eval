"""Bayesian optimization for hyperparameter tuning

Uses Gaussian Process (GP) regression to model the performance landscape
and optimize agent parameters efficiently.
"""

from typing import Dict, Any, List, Optional, Tuple, Callable
import numpy as np

try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer
    from skopt.acquisition import gaussian_ei
    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False
    gp_minimize = None
    Real = None
    Integer = None


class BayesianParameterOptimizer:
    """Bayesian optimization for agent parameter tuning
    
    Uses Gaussian Process regression to model the relationship between
    parameters and performance, then optimizes using Expected Improvement (EI).
    """
    
    def __init__(
        self,
        n_calls: int = 50,
        n_initial_points: int = 10,
        acq_func: str = "EI",
        random_state: Optional[int] = None
    ):
        """
        Initialize Bayesian optimizer
        
        Args:
            n_calls: Total number of optimization iterations
            n_initial_points: Number of random initial points
            acq_func: Acquisition function ('EI', 'LCB', 'PI')
            random_state: Random seed for reproducibility
        """
        if not BAYESIAN_AVAILABLE:
            raise ImportError(
                "scikit-optimize is required for BayesianParameterOptimizer. "
                "Install with: pip install scikit-optimize>=0.9.0"
            )
        
        self.n_calls = n_calls
        self.n_initial_points = n_initial_points
        self.acq_func = acq_func
        self.random_state = random_state
        
        # Parameter bounds
        self.parameter_space = [
            Integer(100, 5000, name="context_length"),
            Real(0.1, 2.0, name="temperature"),
            Integer(1, 50, name="max_steps"),
            Real(0.0, 1.0, name="tool_usage_threshold"),
            Integer(1, 10, name="reasoning_depth")
        ]
        
        # Optimization history
        self.optimization_history: List[Dict[str, Any]] = []
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_score: float = float('-inf')
    
    def objective_function(
        self,
        params: Tuple[float, ...],
        evaluator: Any,
        agent: Any,
        benchmark: Any
    ) -> float:
        """
        Objective function to minimize (negative of performance score)
        
        Args:
            params: Parameter tuple (context_length, temperature, max_steps, ...)
            evaluator: AgentEvaluator instance
            agent: Agent instance
            benchmark: TaskBenchmark instance
            
        Returns:
            Negative performance score (to minimize)
        """
        # Convert params to dict
        param_dict = {
            "context_length": int(params[0]),
            "temperature": float(params[1]),
            "max_steps": int(params[2]),
            "tool_usage_threshold": float(params[3]),
            "reasoning_depth": int(params[4])
        }
        
        # Apply parameters to agent
        if hasattr(agent, "set_parameters"):
            agent.set_parameters(param_dict)
        
        # Evaluate on benchmark
        results = evaluator.evaluate_batch(agent, benchmark.tasks[:10])  # Sample tasks
        
        # Compute average score
        scores = [r.score for r in results]
        avg_score = np.mean(scores) if scores else 0.0
        
        # Store in history
        self.optimization_history.append({
            "params": param_dict,
            "score": avg_score
        })
        
        # Update best
        if avg_score > self.best_score:
            self.best_score = avg_score
            self.best_params = param_dict.copy()
        
        # Return negative (to minimize)
        return -avg_score
    
    def optimize(
        self,
        evaluator: Any,
        agent: Any,
        benchmark: Any,
        callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Run Bayesian optimization
        
        Args:
            evaluator: AgentEvaluator instance
            agent: Agent instance
            benchmark: TaskBenchmark instance
            callback: Optional callback function called after each iteration
            
        Returns:
            Dictionary with best parameters and optimization results
        """
        # Define objective
        def objective(params):
            return self.objective_function(params, evaluator, agent, benchmark)
        
        # Run optimization
        result = gp_minimize(
            func=objective,
            dimensions=self.parameter_space,
            n_calls=self.n_calls,
            n_initial_points=self.n_initial_points,
            acq_func=self.acq_func,
            random_state=self.random_state,
            callback=callback
        )
        
        # Extract best parameters
        best_params_tuple = result.x
        best_params = {
            "context_length": int(best_params_tuple[0]),
            "temperature": float(best_params_tuple[1]),
            "max_steps": int(best_params_tuple[2]),
            "tool_usage_threshold": float(best_params_tuple[3]),
            "reasoning_depth": int(best_params_tuple[4])
        }
        
        return {
            "best_params": best_params,
            "best_score": -result.fun,  # Convert back to positive
            "optimization_history": self.optimization_history,
            "n_iterations": len(self.optimization_history),
            "convergence": result.func_vals.tolist()
        }
    
    def get_uncertainty_estimate(
        self,
        params: Dict[str, Any]
    ) -> float:
        """
        Estimate uncertainty in performance prediction for given parameters
        
        Note: This requires access to the GP model, which is internal to gp_minimize.
        For a full implementation, you'd need to refit the GP model.
        
        Args:
            params: Parameter dictionary
            
        Returns:
            Estimated uncertainty (standard deviation)
        """
        # Simplified uncertainty estimate based on distance to explored points
        if not self.optimization_history:
            return 1.0
        
        # Compute distance to nearest explored point
        param_vec = np.array([
            params["context_length"] / 5000,
            params["temperature"] / 2.0,
            params["max_steps"] / 50,
            params["tool_usage_threshold"],
            params["reasoning_depth"] / 10
        ])
        
        min_distance = float('inf')
        for hist in self.optimization_history:
            hist_vec = np.array([
                hist["params"]["context_length"] / 5000,
                hist["params"]["temperature"] / 2.0,
                hist["params"]["max_steps"] / 50,
                hist["params"]["tool_usage_threshold"],
                hist["params"]["reasoning_depth"] / 10
            ])
            distance = np.linalg.norm(param_vec - hist_vec)
            min_distance = min(min_distance, distance)
        
        # Uncertainty increases with distance
        return min(min_distance * 2.0, 1.0)
