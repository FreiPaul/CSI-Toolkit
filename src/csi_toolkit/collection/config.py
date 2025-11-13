"""Configuration management for data collection."""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

from ..core.constants import (
    DEFAULT_BAUDRATE,
    DEFAULT_SERIAL_PORT,
    DEFAULT_FLUSH_INTERVAL,
)
from ..core.exceptions import ConfigurationError


class CollectorConfig:
    """Configuration for serial data collector."""

    def __init__(
        self,
        serial_port: Optional[str] = None,
        baudrate: Optional[int] = None,
        flush_interval: Optional[int] = None,
        output_dir: str = "data",
        env_file: Optional[str] = None,
    ):
        """
        Initialize collector configuration.

        Args:
            serial_port: Serial port path (overrides env)
            baudrate: Baud rate (overrides env)
            flush_interval: Flush interval (overrides env)
            output_dir: Output directory for CSV files
            env_file: Path to .env file
        """
        # Load environment variables
        self._env_file_loaded = None
        if env_file:
            env_path = Path(env_file)
            if env_path.exists():
                load_dotenv(env_path, override=True)
                self._env_file_loaded = str(env_path)
            else:
                raise ConfigurationError(f"Specified .env file not found: {env_file}")
        else:
            # Try to load .env from current working directory first
            cwd_env = Path.cwd() / '.env'
            if cwd_env.exists():
                load_dotenv(cwd_env, override=True)
                self._env_file_loaded = str(cwd_env)
            else:
                # Fall back to search in parent directories
                result = load_dotenv(override=True)
                if result:
                    self._env_file_loaded = "found in parent directory"

        # Set configuration with fallback to env vars and defaults
        self.serial_port = (
            serial_port or
            os.getenv('SERIAL_PORT') or
            DEFAULT_SERIAL_PORT
        )

        self.baudrate = (
            baudrate or
            int(os.getenv('BAUDRATE', DEFAULT_BAUDRATE))
        )

        self.flush_interval = (
            flush_interval or
            int(os.getenv('FLUSH_INTERVAL', DEFAULT_FLUSH_INTERVAL))
        )

        self.output_dir = output_dir

        # Validate configuration
        self._validate()

    def _validate(self):
        """Validate configuration parameters."""
        if not self.serial_port:
            raise ConfigurationError("Serial port not specified")

        if self.baudrate <= 0:
            raise ConfigurationError(f"Invalid baudrate: {self.baudrate}")

        if self.flush_interval <= 0:
            raise ConfigurationError(f"Invalid flush interval: {self.flush_interval}")

    @classmethod
    def from_env_file(cls, env_file: str) -> 'CollectorConfig':
        """
        Create configuration from environment file.

        Args:
            env_file: Path to .env file

        Returns:
            CollectorConfig instance
        """
        return cls(env_file=env_file)

    def to_dict(self) -> dict:
        """
        Convert configuration to dictionary.

        Returns:
            Configuration dictionary
        """
        return {
            'serial_port': self.serial_port,
            'baudrate': self.baudrate,
            'flush_interval': self.flush_interval,
            'output_dir': self.output_dir,
        }

    def __str__(self) -> str:
        """String representation of configuration."""
        env_info = f"\n  env_file={self._env_file_loaded}" if self._env_file_loaded else "\n  env_file=not loaded"
        return (
            f"CollectorConfig(\n"
            f"  serial_port={self.serial_port},\n"
            f"  baudrate={self.baudrate},\n"
            f"  flush_interval={self.flush_interval},\n"
            f"  output_dir={self.output_dir},{env_info}\n"
            f")"
        )