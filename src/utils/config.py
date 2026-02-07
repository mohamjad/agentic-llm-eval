"""Centralized configuration system with validation

Provides a unified configuration interface with:
- Environment variable support
- Configuration file loading (JSON, YAML)
- Default value management
- Validation and type checking
- Configuration merging
"""

import json
import os
from typing import Dict, Any, Optional, Union
from pathlib import Path
import yaml


class Config:
    """
    Centralized configuration manager
    
    Supports loading from:
    - Environment variables (with prefix)
    - JSON/YAML files
    - Python dictionaries
    - Default values
    
    Example:
        config = Config()
        config.load_from_file("config.json")
        config.load_from_env(prefix="AGENTIC_")
        evaluator = AgentEvaluator(config=config.get("evaluator"))
    """
    
    def __init__(self, defaults: Optional[Dict[str, Any]] = None):
        """
        Initialize configuration
        
        Args:
            defaults: Default configuration values
        """
        self._config: Dict[str, Any] = defaults.copy() if defaults else {}
        self._load_defaults()
    
    def _load_defaults(self):
        """Load default configuration values"""
        self._config.setdefault("evaluator", {
            "use_tracer": True,
            "success_threshold": 0.7,
            "metric_weights": {
                "accuracy": 0.4,
                "efficiency": 0.3,
                "tool_usage": 0.2,
                "safety": 0.1
            }
        })
        
        self._config.setdefault("metrics", {
            "efficiency": {
                "max_reasonable_steps": 20
            },
            "safety": {
                "severity_threshold": 0.5,
                "check_patterns": True
            }
        })
        
        self._config.setdefault("rl", {
            "learning_rate": 0.01,
            "momentum": 0.9,
            "decay_rate": 0.95,
            "replay_buffer_size": 100
        })
        
        self._config.setdefault("logging", {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        })
    
    def load_from_file(
        self,
        filepath: Union[str, Path],
        merge: bool = True
    ) -> "Config":
        """
        Load configuration from file (JSON or YAML)
        
        Args:
            filepath: Path to configuration file
            merge: Whether to merge with existing config (True) or replace (False)
            
        Returns:
            Self for method chaining
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            if filepath.suffix.lower() in ['.yaml', '.yml']:
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        
        if merge:
            self._config = self._merge_dicts(self._config, data)
        else:
            self._config = data
        
        return self
    
    def load_from_env(
        self,
        prefix: str = "AGENTIC_",
        separator: str = "__",
        merge: bool = True
    ) -> "Config":
        """
        Load configuration from environment variables
        
        Environment variables are converted from UPPER_CASE__NESTED__KEYS
        to nested dictionary structure.
        
        Args:
            prefix: Prefix for environment variables (e.g., "AGENTIC_")
            separator: Separator for nested keys (default: "__")
            merge: Whether to merge with existing config
            
        Returns:
            Self for method chaining
        """
        env_config = {}
        
        for key, value in os.environ.items():
            if key.startswith(prefix):
                # Remove prefix and convert to nested dict
                config_key = key[len(prefix):]
                keys = config_key.split(separator)
                
                # Convert value to appropriate type
                converted_value = self._convert_env_value(value)
                
                # Build nested dictionary
                current = env_config
                for k in keys[:-1]:
                    current = current.setdefault(k.lower(), {})
                current[keys[-1].lower()] = converted_value
        
        if merge:
            self._config = self._merge_dicts(self._config, env_config)
        else:
            self._config = env_config
        
        return self
    
    def load_from_dict(
        self,
        config_dict: Dict[str, Any],
        merge: bool = True
    ) -> "Config":
        """
        Load configuration from dictionary
        
        Args:
            config_dict: Configuration dictionary
            merge: Whether to merge with existing config
            
        Returns:
            Self for method chaining
        """
        if merge:
            self._config = self._merge_dicts(self._config, config_dict)
        else:
            self._config = config_dict.copy()
        
        return self
    
    def get(
        self,
        key: str,
        default: Any = None,
        required: bool = False
    ) -> Any:
        """
        Get configuration value
        
        Args:
            key: Configuration key (supports dot notation: "evaluator.success_threshold")
            default: Default value if key not found
            required: Raise KeyError if key not found and no default
            
        Returns:
            Configuration value
            
        Raises:
            KeyError: If key not found and required=True
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    if required:
                        raise KeyError(f"Required configuration key not found: {key}")
                    return default
            else:
                if required:
                    raise KeyError(f"Configuration key path invalid: {key}")
                return default
        
        return value if value is not None else default
    
    def set(self, key: str, value: Any) -> "Config":
        """
        Set configuration value
        
        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
            
        Returns:
            Self for method chaining
        """
        keys = key.split('.')
        current = self._config
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return self._config.copy()
    
    def save_to_file(
        self,
        filepath: Union[str, Path],
        format: str = "json"
    ) -> "Config":
        """
        Save configuration to file
        
        Args:
            filepath: Path to save configuration
            format: File format ("json" or "yaml")
            
        Returns:
            Self for method chaining
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            if format.lower() == "yaml":
                yaml.dump(self._config, f, default_flow_style=False, indent=2)
            else:
                json.dump(self._config, f, indent=2)
        
        return self
    
    def validate(self, schema: Optional[Dict[str, Any]] = None) -> bool:
        """
        Validate configuration against schema
        
        Args:
            schema: Validation schema (optional, uses defaults if None)
            
        Returns:
            True if valid
            
        Raises:
            ValueError: If validation fails
        """
        # Basic validation - check required keys
        required_keys = ["evaluator"]
        
        for key in required_keys:
            if not self.get(key):
                raise ValueError(f"Required configuration key missing: {key}")
        
        # Validate evaluator config
        evaluator_config = self.get("evaluator", {})
        if "success_threshold" in evaluator_config:
            threshold = evaluator_config["success_threshold"]
            if not isinstance(threshold, (int, float)) or not (0 <= threshold <= 1):
                raise ValueError(
                    f"success_threshold must be between 0 and 1, got {threshold}"
                )
        
        return True
    
    def _merge_dicts(self, base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge two dictionaries"""
        result = base.copy()
        
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_dicts(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _convert_env_value(self, value: str) -> Any:
        """Convert environment variable string to appropriate type"""
        # Try boolean
        if value.lower() in ("true", "false"):
            return value.lower() == "true"
        
        # Try integer
        try:
            return int(value)
        except ValueError:
            pass
        
        # Try float
        try:
            return float(value)
        except ValueError:
            pass
        
        # Return as string
        return value


# Global default configuration instance
_default_config: Optional[Config] = None


def get_default_config() -> Config:
    """Get or create default global configuration"""
    global _default_config
    if _default_config is None:
        _default_config = Config()
        # Try to load from common locations
        for path in ["./configs/config.json", "./config.json"]:
            if Path(path).exists():
                try:
                    _default_config.load_from_file(path)
                except Exception:
                    pass
        # Try to load from environment
        try:
            _default_config.load_from_env()
        except Exception:
            pass
    return _default_config


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from file or use defaults
    
    Args:
        config_path: Path to configuration file (optional)
        
    Returns:
        Config instance
    """
    config = Config()
    
    if config_path:
        config.load_from_file(config_path)
    else:
        # Try default locations
        for path in ["./configs/config.json", "./config.json"]:
            if Path(path).exists():
                config.load_from_file(path)
                break
    
    # Always try environment variables
    config.load_from_env()
    
    return config


def save_config(config: Config, config_path: str):
    """
    Save configuration to file
    
    Args:
        config: Config instance
        config_path: Path to save configuration
    """
    config.save_to_file(config_path)
