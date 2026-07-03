"""
Centralized Logging System for the Threat Assessment Framework

This module serves as the sole logging interface for the entire framework.
It provides a thread-safe, configurable logging system that supports both
console and rotating file output.

Design Philosophy:
    - Dedicated Namespace: Uses "ThreatAssessmentFramework" as the root logger.
      This prevents conflicts with third-party libraries (we never touch root).
    - Fail Fast: Invalid or missing configuration keys raise explicit errors
      during initialization, preventing silent misconfiguration.
    - Dependency Injection: Configuration is passed in via the manager.
    - Good Citizen: Does not clear or modify existing handlers on the global root.

Dependencies:
    - Python Standard Library only (logging, pathlib, datetime, typing, os)
"""

import logging
import logging.handlers
from pathlib import Path
from typing import Optional, Union, Dict, Any
import sys


# -----------------------------------------------------------------------------
# Custom Exceptions
# -----------------------------------------------------------------------------

class LoggerError(Exception):
    """Base exception for all logger-related errors."""
    pass


class LoggerConfigError(LoggerError):
    """Raised when the logging configuration is invalid or missing required keys."""
    pass


# -----------------------------------------------------------------------------
# Logger Manager (The Public Interface)
# -----------------------------------------------------------------------------

class LoggerManager:
    """
    Centralized logger manager using a dedicated framework namespace.

    This manager creates a top-level logger named "ThreatAssessmentFramework".
    All child loggers requested via `get_logger(name)` become children of this
    namespace (e.g., "ThreatAssessmentFramework.PersonDetector").

    This approach ensures that:
        - Our logs never interfere with third-party library logs.
        - We never accidentally clear handlers configured by other code.
        - We can independently control our log levels and outputs.

    Usage:
        # During framework initialization
        config = ConfigLoader("configs").load()
        logger_mgr = LoggerManager(config.logging, base_path=Path.cwd())
        
        # In any module
        logger = logger_mgr.get_logger(__name__)
        logger.info("Module initialized successfully.")
        
        # During shutdown
        logger_mgr.shutdown()
    """

    def __init__(
        self,
        logging_config: Optional[Union[Dict[str, Any], Any]] = None,
        framework_namespace: str = "ThreatAssessmentFramework",
        base_path: Optional[Path] = None,
    ) -> None:
        """
        Initializes the LoggerManager.

        Args:
            logging_config: The logging configuration section (typically `config.logging`).
                            If None, falls back to console-only logging with default format.
            framework_namespace: The root logger name for the framework.
            base_path: The base path for resolving relative log directories.
                       Defaults to the current working directory.
        """
        self._framework_namespace = framework_namespace
        self._base_path = Path.cwd() if base_path is None else base_path.resolve()
        self._logging_config = logging_config
        self._initialized = False
        self._framework_logger: Optional[logging.Logger] = None

        if self._logging_config is not None:
            self._initialize()

    def _resolve_required_attribute(self, config: Any, path: str) -> Any:
        """
        Resolves a dot-separated attribute path from a configuration object.

        This method strictly enforces the Fail Fast philosophy. If any segment
        of the path is missing, it raises a descriptive LoggerConfigError.

        Supports:
            - ConfigNode objects and any object supporting __getattr__.
            - Standard dictionaries (via __getitem__).
            - Lists/Tuples accessed via integer indices (e.g., "handlers.0.type").

        Args:
            config: The configuration object to traverse.
            path: Dot-separated path (e.g., "format.standard" or "file.directory").

        Returns:
            The resolved attribute value.

        Raises:
            LoggerConfigError: If the path cannot be fully resolved.
        """
        keys = path.split(".")
        current = config
        current_path = []

        for key in keys:
            current_path.append(key)
            full_path = ".".join(current_path)

            if current is None:
                raise LoggerConfigError(
                    f"Configuration path '{full_path}' resolved to None. "
                    f"Failed at '{key}' in path '{path}'."
                )

            # 1. Try attribute-style access (EAFP - Pythonic approach)
            try:
                current = getattr(current, key)
                continue
            except AttributeError:
                pass

            # 2. Try dictionary-style access
            if isinstance(current, dict):
                if key in current:
                    current = current[key]
                    continue
                raise LoggerConfigError(
                    f"Configuration path '{full_path}' not found. "
                    f"Available keys in '{'.'.join(current_path[:-1])}': {', '.join(current.keys())}"
                )

            # 3. Try list/tuple indexing (if key is an integer)
            if isinstance(current, (list, tuple)):
                if key.isdigit():
                    idx = int(key)
                    if 0 <= idx < len(current):
                        current = current[idx]
                        continue
                    raise LoggerConfigError(
                        f"Index {idx} out of range for path '{full_path}'. "
                        f"List has length {len(current)}."
                    )
                raise LoggerConfigError(
                    f"Attempted to access non-integer key '{key}' on a list at path '{full_path}'."
                )

            # 4. If we reached here, the traversal failed
            raise LoggerConfigError(
                f"Configuration path '{path}' failed at '{key}'. "
                f"Current object type: {type(current).__name__}."
            )

        return current

    def _resolve_path(self, path_str: str) -> Path:
        """Resolves a path against the configured base path."""
        path = Path(path_str)
        if not path.is_absolute():
            path = self._base_path / path
        return path.resolve()

    def _initialize(self) -> None:
        """
        Sets up the dedicated framework logger with console and file handlers.

        This method is idempotent. If the framework logger already has handlers,
        it returns without making changes.
        """
        if self._initialized:
            return

        config = self._logging_config
        if config is None:
            self._setup_default_logging()
            return

        # 1. Get the dedicated framework logger (NEVER the root logger)
        self._framework_logger = logging.getLogger(self._framework_namespace)

        # If handlers already exist on this logger, we assume it's already configured.
        # This prevents duplicate handlers if the manager is accidentally re-initialized.
        if self._framework_logger.handlers:
            self._initialized = True
            return

        # 2. Resolve required configuration values (Fail Fast)
        try:
            level_str = self._resolve_required_attribute(config, "level").upper()
            level = getattr(logging, level_str, logging.INFO)

            format_str = self._resolve_required_attribute(
                config, "format.standard"
            )
            date_format = self._resolve_required_attribute(
                config, "format.date"
            )
            formatter = logging.Formatter(format_str, datefmt=date_format)

            console_enabled = self._resolve_required_attribute(
                config, "console.enabled"
            )
            file_enabled = self._resolve_required_attribute(
                config, "file.enabled"
            )

        except LoggerConfigError as e:
            # Re-raise with context about the logging configuration
            raise LoggerConfigError(
                f"Failed to initialize logging system: {e}"
            ) from e

        # 3. Configure Console Handler
        if console_enabled:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            console_handler.setFormatter(formatter)
            self._framework_logger.addHandler(console_handler)

        # 4. Configure File Handler (Rotating)
        if file_enabled:
            try:
                log_dir_str = self._resolve_required_attribute(
                    config, "file.directory"
                )
                log_filename = self._resolve_required_attribute(
                    config, "file.filename"
                )
                max_bytes = self._resolve_required_attribute(
                    config, "file.max_bytes"
                )
                backup_count = self._resolve_required_attribute(
                    config, "file.backup_count"
                )
            except LoggerConfigError as e:
                raise LoggerConfigError(
                    f"File logging is enabled but file configuration is incomplete: {e}"
                ) from e

            log_dir = self._resolve_path(log_dir_str)

            try:
                log_dir.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                raise LoggerConfigError(
                    f"Failed to create log directory '{log_dir}': {e}"
                ) from e

            log_file_path = log_dir / log_filename
            try:
                file_handler = logging.handlers.RotatingFileHandler(
                    str(log_file_path),
                    maxBytes=max_bytes,
                    backupCount=backup_count,
                    encoding="utf-8",
                )
                file_handler.setLevel(level)
                file_handler.setFormatter(formatter)
                self._framework_logger.addHandler(file_handler)
            except OSError as e:
                raise LoggerConfigError(
                    f"Failed to create log file '{log_file_path}': {e}"
                ) from e

        # 5. Set the logger level
        self._framework_logger.setLevel(level)

        # 6. Ensure at least one handler exists (fallback to console)
        if not self._framework_logger.handlers:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            console_handler.setFormatter(formatter)
            self._framework_logger.addHandler(console_handler)

        # 7. Prevent propagation to the global root logger.
        #    We want our dedicated framework logger to stand alone.
        self._framework_logger.propagate = False

        self._initialized = True

    def _setup_default_logging(self) -> None:
        """Sets up a default console-only logger for the framework namespace."""
        self._framework_logger = logging.getLogger(self._framework_namespace)

        if self._framework_logger.handlers:
            self._initialized = True
            return

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        self._framework_logger.addHandler(console_handler)
        self._framework_logger.setLevel(logging.INFO)
        self._framework_logger.propagate = False
        self._initialized = True

    def get_logger(self, name: str) -> logging.Logger:
        """
        Returns a named logger as a child of the framework namespace.

        Repeated calls with the same name return the same logger instance,
        as Python's `logging.getLogger` caches loggers by name.

        Args:
            name: The name of the logger (typically `__name__` of the calling module).

        Returns:
            A configured `logging.Logger` instance.

        Example:
            logger = logger_manager.get_logger("ImageValidator")
            # Actual logger name: "ThreatAssessmentFramework.ImageValidator"
        """
        if not self._initialized:
            self._setup_default_logging()

        # Automatically nest under the framework namespace
        if name.startswith(self._framework_namespace):
            full_name = name
        else:
            full_name = f"{self._framework_namespace}.{name}"

        return logging.getLogger(full_name)

    def get_root_logger(self) -> logging.Logger:
        """
        Returns the root logger of the framework namespace.

        This is useful for framework-level events (e.g., "Framework started").
        """
        if not self._initialized:
            self._setup_default_logging()
        return self._framework_logger

    def shutdown(self) -> None:
        """
        Flushes and closes all log handlers.

        IMPORTANT: Call this during graceful framework shutdown (e.g., in a
        `finally` block in your `main.py`) to ensure all log messages are
        flushed to disk before the process exits.

        Example:
            try:
                run_framework()
            finally:
                logger_manager.shutdown()
        """
        if self._framework_logger:
            for handler in self._framework_logger.handlers[:]:
                handler.flush()
                handler.close()
                self._framework_logger.removeHandler(handler)
        logging.shutdown()


# -----------------------------------------------------------------------------
# Convenience Function (For quick prototyping or legacy code)
# -----------------------------------------------------------------------------

def get_logger(name: str, config: Optional[Any] = None) -> logging.Logger:
    """
    Quick convenience function to get a logger.

    This creates a new LoggerManager instance each time. For production use
    in the framework, it is recommended to create a single LoggerManager
    and inject it where needed.

    Args:
        name: The name of the logger.
        config: Optional logging configuration (defaults to None -> console only).

    Returns:
        A configured logger instance.
    """
    return LoggerManager(config).get_logger(name)


# -----------------------------------------------------------------------------
# Explicit Exports
# -----------------------------------------------------------------------------

__all__ = [
    "LoggerError",
    "LoggerConfigError",
    "LoggerManager",
    "get_logger",
]