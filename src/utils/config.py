"""
Centralized Configuration Management Module

This module serves as the single entry point for loading, validating, and
accessing all configuration files used throughout the Threat Assessment Framework.

Architecture Position:
    schemas.py → config.py → logger.py → interfaces.py → base_module.py

Design Philosophy:
    - Immutable: Configuration is frozen after loading; no runtime mutation.
    - Fail Fast: Invalid YAML or missing required files halt execution immediately.
    - Instance-based: No global state. Each experiment can hold its own config.
    - Strongly Typed: Public API is strongly typed; internal parser uses `Any`.
    - Explicit Ownership: Loader instances own their cached configuration.
    - Mapping Protocol: ConfigNode behaves like a read-only Mapping.

Dependencies:
    - Python Standard Library (pathlib, typing, types, collections.abc)
    - PyYAML (for parsing YAML files)
"""

from __future__ import annotations

import yaml
from pathlib import Path
from typing import Any
from types import MappingProxyType
from collections.abc import Iterator, Mapping


# -----------------------------------------------------------------------------
# Type Aliases (Simple, Practical)
# -----------------------------------------------------------------------------

# Internal parser values can be any valid YAML type.
# The strong type guarantees come from ConfigNode and the public API.
ConfigValue = Any
JSONValue = Any


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
# Internal Helper Functions
# -----------------------------------------------------------------------------

def _resolve_config_path(config_dir: str | Path) -> Path:
    """
    Resolves the configuration directory path.

    If a relative path is provided, it is resolved against the current
    working directory. Absolute paths are returned as-is.

    Args:
        config_dir: The configuration directory path.

    Returns:
        An absolute, resolved Path object.
    """
    path = Path(config_dir)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def _validate_required_files(config_dir: Path) -> None:
    """
    Ensures the core configuration files exist.

    Raises ConfigNotFoundError if the directory or any core file is missing.

    Args:
        config_dir: The resolved configuration directory path.

    Raises:
        ConfigNotFoundError: If the directory or required files are missing.
    """
    if not config_dir.exists():
        raise ConfigNotFoundError(
            f"Configuration directory not found: {config_dir}. "
            "Ensure the path is correct and the directory exists."
        )

    required_stems = {"system", "model", "thresholds", "dataset", "logging"}
    existing_files = {f.stem.replace("_config", "") for f in config_dir.glob("*.yaml")}

    missing = required_stems - existing_files
    if missing:
        raise ConfigNotFoundError(
            f"Missing required configuration file(s): {', '.join(missing)}. "
            f"Ensure they exist in {config_dir}."
        )


def _load_yaml_file(file_path: Path) -> dict[str, Any]:
    """
    Loads a single YAML file.

    Args:
        file_path: Path to the YAML file.

    Returns:
        The parsed YAML content as a dictionary.

    Raises:
        ConfigSyntaxError: If the YAML is malformed.
        ConfigNotFoundError: If the file cannot be read.
    """
    try:
        with file_path.open("r", encoding="utf-8") as f:
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
# Configuration Tree Builders
# -----------------------------------------------------------------------------

def _build_config_value(data: Any) -> Any:
    """
    Recursively converts arbitrary data into a ConfigValue.

    This is the recursive helper for building configuration trees.
    It handles:
        - dictionaries → ConfigNode
        - lists → lists of ConfigValue
        - primitives → pass-through

    Args:
        data: The raw data to convert.

    Returns:
        A ConfigValue (ConfigNode, list, or primitive).
    """
    if isinstance(data, dict):
        processed: dict[str, Any] = {}
        for key, value in data.items():
            if not isinstance(key, str):
                raise ConfigValidationError(
                    f"Configuration keys must be strings, found {type(key).__name__}."
                )
            processed[key] = _build_config_value(value)
        return ConfigNode(processed)
    if isinstance(data, list):
        return [_build_config_value(item) for item in data]
    return data


def _build_config_node(data: Mapping[str, Any]) -> ConfigNode:
    """
    Builds a root ConfigNode from a dictionary.

    This is the top-level entry point for configuration construction.
    It guarantees that the result is a ConfigNode, not a list or primitive.

    Args:
        data: A dictionary mapping keys to raw values.

    Returns:
        A ConfigNode representing the entire configuration tree.

    Raises:
        ConfigValidationError: If the input is not a dictionary.
    """
    if not isinstance(data, dict):
        raise ConfigValidationError(
            f"Root configuration must be a dictionary, got {type(data).__name__}."
        )
    # Use the recursive builder to process the dictionary
    result = _build_config_value(data)
    if not isinstance(result, ConfigNode):
        # This should never happen because _build_config_value returns a ConfigNode for dicts
        raise ConfigValidationError(
            "Internal error: root configuration is not a ConfigNode."
        )
    return result


# -----------------------------------------------------------------------------
# Immutable Configuration Node (Implements Mapping Protocol)
# -----------------------------------------------------------------------------

class ConfigNode(Mapping[str, Any]):
    """
    A recursive, immutable configuration container implementing Mapping.

    This class represents a node in the configuration tree. It provides
    attribute-style and dictionary-style access to nested values.

    Implements the full Mapping protocol:
        - __getitem__
        - __iter__
        - __len__
        - __contains__ (inherited from Mapping)

    Immutability is enforced both statically (frozen dataclass semantics)
    and at runtime using MappingProxyType for the internal storage.

    Example:
        node = ConfigNode({"model": {"name": "YOLO11"}})
        print(node.model.name)       # "YOLO11"
        print(node["model"]["name"]) # "YOLO11"
        print(len(node))             # 1
        print(list(node))            # ["model"]
        node.model.name = "New"      # Raises AttributeError

    Attributes:
        __data: A read-only mapping of keys to ConfigValue objects (name-mangled).
    """

    __slots__ = ("__data",)

    def __init__(self, data: Mapping[str, Any]) -> None:
        """
        Initializes a ConfigNode with a read-only mapping.

        The provided mapping is copied and wrapped in MappingProxyType to
        enforce immutability at runtime and ensure ownership.

        Args:
            data: A mapping of configuration keys to values.
        """
        # Copy the data first to ensure we own the state, then wrap in MappingProxyType
        # This prevents external mutations to the original dict from affecting this node.
        object.__setattr__(self, "__data", MappingProxyType(dict(data)))

    def __getattr__(self, key: str) -> Any:
        """
        Provides attribute-style access to configuration keys.

        Args:
            key: The attribute name.

        Returns:
            The configuration value for the given key.

        Raises:
            AttributeError: If the key does not exist.
        """
        if key in ("__data",):
            return object.__getattribute__(self, key)

        data = object.__getattribute__(self, "__data")
        if key in data:
            return data[key]

        raise AttributeError(
            f"'{self.__class__.__name__}' has no attribute '{key}'. "
            f"Available keys: {', '.join(data.keys())}"
        )

    def __setattr__(self, key: str, value: Any) -> None:
        """
        Prevents attribute assignment, enforcing immutability.

        Raises:
            AttributeError: Always raised.
        """
        raise AttributeError(
            f"Configuration is immutable. Cannot assign '{key}' after initialization."
        )

    def __getitem__(self, key: str) -> Any:
        """
        Provides dictionary-style access to configuration keys.

        Args:
            key: The key to look up.

        Returns:
            The configuration value for the given key.

        Raises:
            KeyError: If the key does not exist.
        """
        data = object.__getattribute__(self, "__data")
        if key in data:
            return data[key]
        raise KeyError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        """
        Prevents item assignment, enforcing immutability.

        Raises:
            TypeError: Always raised.
        """
        raise TypeError(
            f"Configuration is immutable. Cannot assign '{key}' via item access."
        )

    def __iter__(self) -> Iterator[str]:
        """
        Returns an iterator over the configuration keys.

        This enables:
            for key in config:
                print(key)
        """
        data = object.__getattribute__(self, "__data")
        return iter(data)

    def __len__(self) -> int:
        """
        Returns the number of keys in this configuration node.

        This enables:
            len(config)
        """
        data = object.__getattribute__(self, "__data")
        return len(data)

    def __repr__(self) -> str:
        """Returns a compact string representation of the configuration node."""
        data = object.__getattribute__(self, "__data")
        keys = list(data.keys())
        if len(keys) <= 3:
            return f"ConfigNode(keys={keys})"
        return f"ConfigNode(keys=[{', '.join(keys[:3])}, ...], size={len(keys)})"

    def _convert_to_json(self, value: Any) -> Any:
        """
        Recursively converts a ConfigValue to a JSON-serializable value.

        This internal helper ensures type safety when converting nested structures.

        Args:
            value: The ConfigValue to convert.

        Returns:
            A JSONValue representation.
        """
        if isinstance(value, ConfigNode):
            return value.to_dict()
        if isinstance(value, list):
            return [self._convert_to_json(item) for item in value]
        return value

    def to_dict(self) -> dict[str, Any]:
        """
        Recursively converts the ConfigNode tree to a pure Python dictionary.

        This method produces a JSON-serializable structure. Nested ConfigNode
        objects are recursively converted, and lists are fully expanded.

        Returns:
            A dictionary representing the complete configuration tree.
        """
        data = object.__getattribute__(self, "__data")
        result: dict[str, Any] = {}
        for key, value in data.items():
            result[key] = self._convert_to_json(value)
        return result


# -----------------------------------------------------------------------------
# Configuration Loader (Instance-based, No Singleton)
# -----------------------------------------------------------------------------

class ConfigLoader:
    """
    Instance-based configuration loader.

    Each instance manages its own cache. This allows multiple experiments
    running in the same process to hold distinct configurations.

    Usage:
        loader = ConfigLoader("configs")
        config = loader.load()
        print(config.model.person_detector.name)

        # Multiple independent loaders
        exp1 = ConfigLoader("configs/experiment_a").load()
        exp2 = ConfigLoader("configs/experiment_b").load()

    Attributes:
        _config_dir: The resolved configuration directory path.
        _config: Cached ConfigNode (or None if not loaded).
        _loaded: Whether the configuration has been successfully loaded.
    """

    def __init__(self, config_dir: str | Path = "configs") -> None:
        """
        Initializes the loader with a configuration directory path.

        Args:
            config_dir: Path to the root configuration directory.
                        If relative, it resolves against the current working directory.
        """
        self._config_dir = _resolve_config_path(config_dir)
        self._config: ConfigNode | None = None
        self._loaded: bool = False

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
        # Return cached config if available
        if self._loaded and self._config is not None:
            return self._config

        config_path = self._config_dir

        # 1. Validate core file existence
        _validate_required_files(config_path)

        # 2. Load root-level YAML files
        root_data: dict[str, Any] = {}
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
            reasoning_data: dict[str, Any] = {}
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
            # Ensure an empty reasoning namespace exists to prevent downstream crashes
            root_data["reasoning"] = {}

        # 4. Convert the raw dictionary tree into an immutable ConfigNode tree
        config = _build_config_node(root_data)
        self._config = config
        self._loaded = True
        return config

    def reload(self) -> ConfigNode:
        """
        Forces a reload of the configuration from disk.

        This is useful for development scenarios where configuration files
        may change at runtime. It clears the cache and reloads everything.

        Returns:
            A fresh ConfigNode representing the updated configuration.

        Raises:
            Same exceptions as load().
        """
        # Clear the cache and reload
        self._config = None
        self._loaded = False
        return self.load()

    def get_config(self) -> ConfigNode:
        """
        Returns the cached configuration.

        This is a convenience method that raises a clear error if the
        configuration has not been loaded yet.

        Returns:
            The cached ConfigNode.

        Raises:
            RuntimeError: If load() hasn't been called successfully before.
        """
        config = self._config
        if not self._loaded or config is None:
            raise RuntimeError(
                f"Configuration not loaded for loader (dir: {self._config_dir}). "
                "Call .load() before attempting to access configuration values."
            )
        # Assert to help the type checker narrow the type
        assert config is not None
        return config


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