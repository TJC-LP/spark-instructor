# Contributing

We're excited that you're interested in contributing to Spark Instructor! This document outlines the process for contributing to this project.

## Getting Started
1. Fork [this repository](https://github.com/TJC-LP/spark-instructor) and clone locally 
2. Install [poetry](https://python-poetry.org/)
3. Run `poetry install`
4. Run `poetry run pre-commit install`
5. Set up your IDE environment using the created poetry environment
6. Run `poetry run lint` to run a full linting suite as well as tests
7. Run `poetry run mkdocs build` to build documentation locally

## Guidelines
- Create meaningful branches for all PRs (e.g. `update-openai-schema` or `add-email-response-model`)
- Make sure to add relevant tests for code changes
- Add sufficient documentation with Google-style docstrings
- Keep it simple