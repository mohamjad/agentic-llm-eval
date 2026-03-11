"""
Utility functions for the evaluation framework
"""

from .logger import setup_logger
from .config import load_config

# Optional statistical analysis
try:
    from .statistics import StatisticalAnalyzer
    __all__ = ["setup_logger", "load_config", "StatisticalAnalyzer"]
except Exception:
    __all__ = ["setup_logger", "load_config"]
