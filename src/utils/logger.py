"""logging utils"""

import logging
from typing import Optional


def setup_logger(name: str = "agentic_llm_eval", level: Optional[int] = None) -> logging.Logger:
    """setup logger"""
    logger = logging.getLogger(name)
    
    if level is None:
        level = logging.INFO
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
    
    return logger
