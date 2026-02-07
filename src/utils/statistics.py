"""Statistical analysis and significance testing

Provides statistical tests for evaluating agent performance:
- T-tests for comparing agents
- Confidence intervals
- Effect size calculations
- Bootstrap sampling
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from scipy import stats


class StatisticalAnalyzer:
    """Statistical analysis for evaluation results"""
    
    @staticmethod
    def confidence_interval(
        data: List[float],
        confidence: float = 0.95
    ) -> Tuple[float, float, float]:
        """
        Compute confidence interval for sample mean
        
        Args:
            data: Sample data
            confidence: Confidence level (default: 0.95)
            
        Returns:
            (mean, lower_bound, upper_bound)
        """
        if not data:
            return 0.0, 0.0, 0.0
        
        data_array = np.array(data)
        n = len(data_array)
        mean = np.mean(data_array)
        std = np.std(data_array, ddof=1)  # Sample standard deviation
        
        # Standard error
        se = std / np.sqrt(n)
        
        # T-distribution critical value
        alpha = 1 - confidence
        t_critical = stats.t.ppf(1 - alpha / 2, df=n - 1)
        
        # Confidence interval
        margin = t_critical * se
        lower = mean - margin
        upper = mean + margin
        
        return mean, lower, upper
    
    @staticmethod
    def t_test(
        sample1: List[float],
        sample2: List[float],
        alternative: str = "two-sided"
    ) -> Dict[str, float]:
        """
        Perform independent samples t-test
        
        Args:
            sample1: First sample
            sample2: Second sample
            alternative: 'two-sided', 'less', or 'greater'
            
        Returns:
            Dictionary with test statistics
        """
        if not sample1 or not sample2:
            return {
                "statistic": 0.0,
                "pvalue": 1.0,
                "significant": False
            }
        
        # Perform t-test
        statistic, pvalue = stats.ttest_ind(sample1, sample2, alternative=alternative)
        
        # Determine significance (p < 0.05)
        significant = pvalue < 0.05
        
        return {
            "statistic": float(statistic),
            "pvalue": float(pvalue),
            "significant": significant,
            "mean1": float(np.mean(sample1)),
            "mean2": float(np.mean(sample2)),
            "std1": float(np.std(sample1, ddof=1)),
            "std2": float(np.std(sample2, ddof=1))
        }
    
    @staticmethod
    def effect_size(
        sample1: List[float],
        sample2: List[float]
    ) -> Dict[str, float]:
        """
        Compute Cohen's d effect size
        
        Args:
            sample1: First sample
            sample2: Second sample
            
        Returns:
            Dictionary with effect size metrics
        """
        if not sample1 or not sample2:
            return {"cohens_d": 0.0, "magnitude": "negligible"}
        
        mean1 = np.mean(sample1)
        mean2 = np.mean(sample2)
        std1 = np.std(sample1, ddof=1)
        std2 = np.std(sample2, ddof=1)
        
        # Pooled standard deviation
        n1, n2 = len(sample1), len(sample2)
        pooled_std = np.sqrt(
            ((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2)
        )
        
        # Cohen's d
        cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0.0
        
        # Interpret magnitude
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            magnitude = "negligible"
        elif abs_d < 0.5:
            magnitude = "small"
        elif abs_d < 0.8:
            magnitude = "medium"
        else:
            magnitude = "large"
        
        return {
            "cohens_d": float(cohens_d),
            "magnitude": magnitude,
            "mean_diff": float(mean1 - mean2)
        }
    
    @staticmethod
    def bootstrap_confidence_interval(
        data: List[float],
        n_bootstrap: int = 1000,
        confidence: float = 0.95
    ) -> Tuple[float, float, float]:
        """
        Compute confidence interval using bootstrap sampling
        
        Args:
            data: Sample data
            n_bootstrap: Number of bootstrap samples
            confidence: Confidence level
            
        Returns:
            (mean, lower_bound, upper_bound)
        """
        if not data:
            return 0.0, 0.0, 0.0
        
        data_array = np.array(data)
        n = len(data_array)
        
        # Generate bootstrap samples
        bootstrap_means = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(data_array, size=n, replace=True)
            bootstrap_means.append(np.mean(sample))
        
        bootstrap_means = np.array(bootstrap_means)
        
        # Compute percentiles
        alpha = 1 - confidence
        lower = np.percentile(bootstrap_means, 100 * alpha / 2)
        upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
        mean = np.mean(bootstrap_means)
        
        return float(mean), float(lower), float(upper)
    
    @staticmethod
    def compare_agents(
        results1: List[Dict[str, Any]],
        results2: List[Dict[str, Any]],
        metric_name: str = "score"
    ) -> Dict[str, Any]:
        """
        Compare two agents statistically
        
        Args:
            results1: Evaluation results for agent 1
            results2: Evaluation results for agent 2
            metric_name: Metric to compare
            
        Returns:
            Dictionary with comparison statistics
        """
        # Extract metric values
        scores1 = [r.get(metric_name, 0.0) for r in results1]
        scores2 = [r.get(metric_name, 0.0) for r in results2]
        
        # T-test
        t_test_result = StatisticalAnalyzer.t_test(scores1, scores2)
        
        # Effect size
        effect_size_result = StatisticalAnalyzer.effect_size(scores1, scores2)
        
        # Confidence intervals
        ci1 = StatisticalAnalyzer.confidence_interval(scores1)
        ci2 = StatisticalAnalyzer.confidence_interval(scores2)
        
        return {
            "t_test": t_test_result,
            "effect_size": effect_size_result,
            "agent1_ci": {
                "mean": ci1[0],
                "lower": ci1[1],
                "upper": ci1[2]
            },
            "agent2_ci": {
                "mean": ci2[0],
                "lower": ci2[1],
                "upper": ci2[2]
            },
            "n1": len(scores1),
            "n2": len(scores2)
        }
