import os
import subprocess
from dotenv import load_dotenv

# Get environment variables
load_dotenv()

SAFETY_API_KEY = os.getenv("SAFETY_API_KEY")

# Check if we are running in a CI environment
IS_CI = os.getenv("CI") == "true"

if not SAFETY_API_KEY:
    if IS_CI:
        raise EnvironmentError("`SAFETY_API_KEY` must be set. Please add it to your GitHub environment.")
    print("WARNING: `SAFETY_API_KEY` is not set!")

# The commands to run in order
key_command = f" --key {SAFETY_API_KEY}" if SAFETY_API_KEY else ""
COMMANDS = [
    # Check for package vulnerabilities
    "safety check --full-report -i 70612" + key_command,
    # Sort package imports
    "isort spark_instructor tests",
    # Clean code formatting
    "black spark_instructor tests",
    # Lint the code
    "flake8 spark_instructor tests",
    # Run static typing
    "mypy spark_instructor tests",
    # Check for docstring issues
    "pydocstyle spark_instructor",
    # Run full coverage test suite
    f"pytest --cov=spark_instructor --cov-report{'=term-missing' if not IS_CI else '=json'} .",
]


def main():
    """Run each command."""
    for command in COMMANDS:
        subprocess.run(command.split(), check=IS_CI)


if __name__ == "__main__":
    main()
