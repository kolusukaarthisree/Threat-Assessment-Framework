"""
Centralized Configuration Management Module

This module serves as the single entry point for loading, validating, and
accessing all configuration files used throughout the Threat Assessment Framework.

It sits at the infrastructure layer of the dependency graph:
    config.py
        ▲
        │
    Every framework module depends on it.

Design Philosophy:
    - Instance-based: No global state. Each experiment can hold its own config.
    - Fail Fast: Invalid YAML or missing required files halt execution immediately.
    - Immutability: Configuration is frozen after loading.
    - Explicit Ownership: No Singletons. Loader instances own their cached config.

Future Enhancements (Not yet implemented):
    - Schema Validation: Ensure 'confidence' keys are floats, 'paths' exist, etc.
    - Default Values: Merge loaded configs with a base defaults.yaml to fill gaps.

Dependencies:
    - Python Standard Library (pathlib, typing)
    - PyYAML (for parsing YAML files)
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import warnings


# -----------------------------------------------------------------------------
# Custom Exceptions
# -----------------------------------------------------------------------------

class ConfigError(Exception):
    """Base exception for all configuration-related errors."""
    pass


class ConfigNotFoundError(ConfigError):
    """Raised when the configuration directory or a required file is missing."""
    pass


class ConfigSyntaxError(ConfigError):
    """Raised when a YAML file contains invalid syntax or malformed structure."""
    pass


class ConfigValidationError(ConfigError):
    """Raised when a required configuration key is missing or has an invalid type."""
    pass


# -----------------------------------------------------------------------------
# Internal Helpers
# -----------------------------------------------------------------------------

def _resolve_config_path(config_dir: Union[str, Path]) -> Path:
    """
    Resolves the configuration directory path.

    If a relative path is provided, it is resolved against the current
    working directory. For absolute paths, it is used as-is.
    """
    path = Path(config_dir)
    if not path.is_absolute():
        # Resolve relative to the current working directory.
        # This makes `python project/main.py` work predictably if run from root.
        path = Path.cwd() / path
    return path.resolve()


def _validate_required_files(config_dir: Path) -> None:
    """
    Ensures the core configuration files exist.

    Raises ConfigNotFoundError if the directory or any core file is missing.
    """
    if not config_dir.exists():
        raise ConfigNotFoundError(
            f"Configuration directory not found: {config_dir}. "
            "Ensure the path is correct and the directory exists."
        )

    # Core required files (without the '_config' suffix in their stem)
    required_stems = {"system", "model", "thresholds", "dataset", "logging"}
    existing_files = {f.stem.replace("_config", "") for f in config_dir.glob("*.yaml")}

    missing = required_stems - existing_files
    if missing:
        raise ConfigNotFoundError(
            f"Missing required configuration file(s): {', '.join(missing)}. "
            f"Ensure they exist in {config_dir}."
        )


def _load_yaml_file(file_path: Path) -> Dict[str, Any]:
    """Loads a single YAML file, raising a ConfigSyntaxError on failure."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            if data is None:
                return {}
            if not isinstance(data, dict):
                raise ConfigSyntaxError(
                    f"Configuration file {file_path.name} must contain a root dictionary, "
                    f"found {type(data).__name__}."
                )
            return data
    except yaml.YAMLError as e:
        raise ConfigSyntaxError(
            f"YAML syntax error in {file_path.name}: {e}"
        ) from e
    except OSError as e:
        raise ConfigNotFoundError(
            f"Unable to read configuration file {file_path.name}: {e}"
        ) from e


# -----------------------------------------------------------------------------
# Config Tree Builder
# -----------------------------------------------------------------------------

def _build_config_tree(data: Any) -> Any:
    """
    Recursively converts raw dicts/lists into ConfigNodes or lists thereof.

    This separates the tree-building logic from the ConfigNode class,
    keeping the data container lean.
    """
    if isinstance(data, dict):
        # Primitives (strings, ints, floats, bools) are stored as-is.
        # Nested dicts become ConfigNodes.
        processed = {}
        for key, value in data.items():
            if not isinstance(key, str):
                raise ConfigValidationError(
                    f"Configuration keys must be strings, found {type(key).__name__}."
                )
            processed[key] = _build_config_tree(value)
        return ConfigNode(processed, _root=False)
    elif isinstance(data, list):
        return [_build_config_tree(item) for item in data]
    else:
        # Base case: int, float, str, bool, None
        return data


# -----------------------------------------------------------------------------
# Immutable Configuration Node
# -----------------------------------------------------------------------------

class ConfigNode:
    """
    A recursive, immutable configuration container supporting attribute-style access.

    This class represents a frozen node in the configuration tree. Once created,
    it cannot be modified, ensuring that runtime behavior is deterministic.

    Example:
        node = ConfigNode({"model": {"name": "YOLO11"}})
        print(node.model.name)  # "YOLO11"
        node.model.name = "New"  # Raises AttributeError
    """

    def __init__(self, data: Dict[str, Any], _root: bool = True):
        """
        Initializes the ConfigNode with pre-validated data.

        Note: This expects `data` to already be processed by `_build_config_tree`.
        Do not pass raw lists or primitives directly to this constructor.
        """
        object.__setattr__(self, '_data', data)
        object.__setattr__(self, '_frozen', True)

    def __getattr__(self, key: str) -> Any:
        """Provides attribute-style access to dictionary keys."""
        if key in ['_data', '_frozen']:
            return object.__getattribute__(self, key)

        data = object.__getattribute__(self, '_data')
        if isinstance(data, dict) and key in data:
            return data[key]

        if isinstance(data, dict):
            raise AttributeError(
                f"'{self.__class__.__name__}' has no attribute '{key}'. "
                f"Available keys: {', '.join(data.keys())}"
            )
        raise AttributeError(
            f"'{self.__class__.__name__}' has no attribute '{key}'."
        )

    def __setattr__(self, key: str, value: Any) -> None:
        """Prevents modification, enforcing immutability."""
        raise AttributeError(
            f"Configuration is immutable. Cannot assign '{key}' after initialization."
        )

    def __getitem__(self, key: str) -> Any:
        """Supports dictionary-style access (e.g., config['model']) as a fallback."""
        return self.__getattr__(key)

    def __setitem__(self, key: str, value: Any) -> None:
        """Explicitly prevents item assignment for immutability."""
        raise AttributeError(
            f"Configuration is immutable. Cannot assign '{key}' via item access."
        )

    def __contains__(self, key: str) -> bool:
        """Supports 'in' operator for checking nested keys."""
        data = object.__getattribute__(self, '_data')
        if isinstance(data, dict):
            return key in data
        return False

    def __repr__(self) -> str:
        data = object.__getattribute__(self, '_data')
        return f"ConfigNode({data})"

    def to_dict(self) -> Dict[str, Any]:
        """
        Recursively converts the ConfigNode tree back into a pure Python dict.

        Useful for serializing the loaded configuration for experiment logs.
        """
        data = object.__getattribute__(self, '_data')
        if isinstance(data, dict):
            return {
                k: v.to_dict() if isinstance(v, ConfigNode) else v
                for k, v in data.items()
            }
        if isinstance(data, list):
            return [
                item.to_dict() if isinstance(item, ConfigNode) else item
                for item in data
            ]
        return data


# -----------------------------------------------------------------------------
# Configuration Loader (Instance-based, No Singleton)
# -----------------------------------------------------------------------------

class ConfigLoader:
    """
    Instance-based configuration loader.

    Each instance manages its own cache. This allows multiple experiments
    running in the same process to hold distinct configurations.

    Usage:
        # Instantiate with the path to the config directory
        loader = ConfigLoader("configs")
        
        # Load once (caches the result)
        config = loader.load()
        
        # Access values
        print(config.model.person_detector.name)
        
        # For experiments, you can create separate loaders:
        exp1_config = ConfigLoader("configs/exp1").load()
        exp2_config = ConfigLoader("configs/exp2").load()
    """

    def __init__(self, config_dir: Union[str, Path] = "configs"):
        """
        Initializes the loader with a configuration directory path.

        Args:
            config_dir: Path to the root configuration directory.
                        If relative, it resolves against the current working directory.
        """
        self._config_dir = _resolve_config_path(config_dir)
        self._config: Optional[ConfigNode] = None
        self._loaded = False

    def load(self) -> ConfigNode:
        """
        Loads and caches the complete configuration tree.

        This method performs disk I/O only on the first call. Subsequent calls
        return the cached tree instantly.

        Returns:
            A frozen ConfigNode representing the entire framework configuration.

        Raises:
            ConfigNotFoundError: If the config directory or required files are missing.
            ConfigSyntaxError: If a YAML file contains invalid syntax.
            ConfigValidationError: If a configuration file has an invalid structure.
        """
        if self._loaded and self._config is not None:
            return self._config

        config_path = self._config_dir

        # 1. Validate core file existence
        _validate_required_files(config_path)

        # 2. Load root-level YAML files
        root_data = {}
        for yaml_file in config_path.glob("*.yaml"):
            namespace = yaml_file.stem
            if namespace.endswith("_config"):
                namespace = namespace[:-7]  # Remove '_config'

            raw_content = _load_yaml_file(yaml_file)

            if namespace in root_data:
                raise ConfigValidationError(
                    f"Namespace conflict: '{namespace}' is defined in "
                    f"'{yaml_file.name}' but already loaded from another file."
                )
            root_data[namespace] = raw_content

        # 3. Load the 'reasoning/' subdirectory (optional)
        reasoning_path = config_path / "reasoning"
        if reasoning_path.exists() and reasoning_path.is_dir():
            reasoning_data = {}
            for yaml_file in reasoning_path.glob("*.yaml"):
                namespace = yaml_file.stem
                if namespace.endswith("_config"):
                    namespace = namespace[:-7]

                raw_content = _load_yaml_file(yaml_file)
                if namespace in reasoning_data:
                    raise ConfigValidationError(
                        f"Namespace conflict in reasoning subdir: '{namespace}' "
                        f"defined in '{yaml_file.name}'."
                    )
                reasoning_data[namespace] = raw_content

            if "reasoning" in root_data:
                raise ConfigValidationError(
                    "Namespace conflict: 'reasoning' is already defined in the root "
                    "config directory, but a 'reasoning' subdirectory also exists."
                )
            root_data["reasoning"] = reasoning_data
        else:
            # Ensure an empty reasoning namespace exists to prevent downstream crashes.
            root_data["reasoning"] = {}

        # 4. Convert the raw dictionary tree into an immutable ConfigNode tree
        #    using the specialized builder.
        self._config = _build_config_tree(root_data)
        self._loaded = True

        return self._config

    def get_config(self) -> ConfigNode:
        """
        Returns the cached configuration.

        Raises:
            RuntimeError: If load() hasn't been called successfully before.
        """
        if not self._loaded or self._config is None:
            raise RuntimeError(
                f"Configuration not loaded for loader (dir: {self._config_dir}). "
                "Call .load() before attempting to access configuration values."
            )
        return self._config


# -----------------------------------------------------------------------------
# Explicit Exports
# -----------------------------------------------------------------------------

__all__ = [
    "ConfigError",
    "ConfigNotFoundError",
    "ConfigSyntaxError",
    "ConfigValidationError",
    "ConfigNode",
    "ConfigLoader",
]