# Spark Instructor

Spark Instructor is a powerful library that combines the capabilities of Apache Spark and the `instructor` library to enable AI-powered structured data generation within Spark SQL DataFrames.

## Overview

This project aims to bridge the gap between large-scale data processing with Apache Spark and AI-driven content generation. By leveraging the `instructor` library's ability to work with various AI models (such as OpenAI, Anthropic, and Databricks), Spark Instructor allows users to create User-Defined Functions (UDFs) that generate structured, AI-powered columns in Spark SQL DataFrames.

## Key Features

- **AI-Powered UDFs**: Create Spark UDFs that utilize AI models to generate structured data.
- **Multi-Provider Support**: Work with various AI providers including OpenAI, Anthropic, and Databricks.
- **Type-Safe Responses**: Utilize Pydantic models to ensure type safety and data validation for AI-generated content.
- **Seamless Integration**: Easily incorporate AI-generated columns into your existing Spark SQL workflows.
- **Scalable Processing**: Leverage Spark's distributed computing capabilities for processing large datasets with AI augmentation.

## Use Cases

- Enhance datasets with AI-generated insights or summaries.
- Perform large-scale text classification or entity extraction.
- Generate structured metadata for unstructured text data.
- Create synthetic datasets for testing or machine learning purposes.

## Getting Started

1. Install [poetry](https://python-poetry.org/docs/)
2. Run `poetry install`
3. Run `poetry build`

## Project Structure

- `spark_instructor/`: Main package directory
  - `completions/`: Subpackage for completion object models
    - `base.py`: Base classes for completion models
    - `anthropic.py`: Anthropic-specific completion models
    - `openai.py`: OpenAI-specific completion models
    - `databricks.py`: Databricks-specific completion models
  - `client.py`: Submodule for routing API calls
  - `udf.py`: Submodule for generating Spark UDFs


## Contributing

We welcome contributions to Spark Instructor! Please see our contributing guidelines (TODO) for more information on how to get involved.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## Acknowledgements

This project builds upon the excellent work of the Apache Spark community and the creators of the `instructor` library.
