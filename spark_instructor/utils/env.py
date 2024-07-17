"""Module for environment utilities."""

import os


def assert_env_is_set(name: str):
    """Assert that an environment variable is set."""
    assert name in os.environ, f"``{name}`` is not set!"


def get_env_variable(name: str) -> str:
    """Get environment variable with the given name."""
    assert_env_is_set(name)
    return os.environ[name]
