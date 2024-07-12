[![codecov](https://codecov.io/github/TJC-LP/spark-instructor/graph/badge.svg?token=19SPU3HC0L)](https://codecov.io/github/TJC-LP/spark-instructor)
[![PyPI version](https://badge.fury.io/py/spark-instructor.svg)](https://badge.fury.io/py/spark-instructor)
---

# Spark Instructor

`spark-instructor` combines the capabilities of Apache Spark and the [`instructor`](https://github.com/jxnl/instructor) library to enable AI-powered structured data generation within Spark SQL DataFrames.

## Overview

This project aims to bridge the gap between large-scale data processing with Apache Spark and AI-driven content generation. By leveraging the [`instructor`](https://github.com/jxnl/instructor) library's ability to work with various AI models (such as OpenAI, Anthropic, and Databricks), Spark Instructor allows users to create User-Defined Functions (UDFs) that generate structured, AI-powered columns in Spark SQL DataFrames.

## Key Features

- **AI-Powered UDFs**: Create Spark UDFs that utilize AI models to generate structured data.
- **Multi-Provider Support**: Work with various AI providers including OpenAI, Anthropic, and Databricks. Support for other providers will be added on an as-needed basis. Contributions for added provider support are highly encouraged.
- **Type-Safe Responses**: Utilize Pydantic models to ensure type safety and data validation for AI-generated content.
- **Seamless Integration**: Easily incorporate AI-generated columns into your existing Spark SQL workflows.
- **Scalable Processing**: Leverage Spark's distributed computing capabilities for processing large datasets with AI augmentation.

## Use Cases

- Enhance datasets with AI-generated insights or summaries.
- Perform large-scale text classification or entity extraction.
- Generate structured metadata for unstructured text data.
- Create synthetic datasets for testing or machine learning purposes.

## Getting Started

1. Run `pip install spark-instructor`, or `pip install spark-instructor[anthropic]` for Anthropic SDK support.
   1. `spark-instructor` must be installed on the Spark driver and workers to generate working UDFs.
2. Add the necessary environment variables and config to your spark environment (recommended). See the [Spark documentation](https://spark.apache.org/docs/latest/configuration.html#environment-variables) for more details.
   1. For OpenAI support, add `OPENAI_API_KEY=<openai-api-key>`
   2. For Databricks support, add `DATABRICKS_HOST=<databricks-host>` and `DATABRICKS_TOKEN=<databricks-token>`
   3. For Anthropic support, make sure `spark-instructor[anthropic]` is installed and add `ANTHROPIC_API_KEY=<anthropic-api-key>`
   4. For Ollama support, run [init-ollama](init/init-ollama.sh) as an init script to install `ollama` on all nodes

## Examples
The following example demonstrates a sample run using the provided `User` model. The `User` model is defined below:
```python
from pydantic import BaseModel


class User(BaseModel):
    """A user."""

    name: str
    age: int
```
For proper Spark serialization, we import our Pydantic model so that it is accessible on the Spark workers. See [Limitations](#limitations) for more details.
```python 
import pyspark.sql.functions as f
from pyspark.sql import SparkSession

from spark_instructor.response_models import User
from spark_instructor.udf import MessageRouter
from spark_instructor.utils import zero_shot_prompt

# Create spark session 
spark = SparkSession.builder.getOrCreate()

# Run an example using OpenAI's gpt-4o
model = "gpt-4o"
message_router = MessageRouter(model=model, response_model_type=User)
parser_udf = message_router.create_object_and_completion_from_messages_udf(max_tokens=400)

# Create a sample DataFrame
data = [
    ("Extract Jason is 25 years old.",),
    ("Extract Emma is 30 years old.",),
    ("Extract Michael is 42 years old.",),
]

# Format content as chat messages
df = (
   spark.createDataFrame(data, ["content"])
   .withColumn(
      "messages", 
      zero_shot_prompt(
         user_message_column="content", 
         system_message_column=f.lit("You are a helpful assistant.")
      )
   )
   .withColumn("response", parser_udf(f.col("messages")))
)

# Run the parser
result_df = df.select(
    "content",
    "messages",
    f.col("response.*")
)
result_df.show(truncate=False)
```
```markdown
+--------------------------------+------------------------------------------------------------------------------------------------------------------------+-------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|content                         |messages                                                                                                                |user         |chat_completion                                                                                                                                                                                                                                         |
+--------------------------------+------------------------------------------------------------------------------------------------------------------------+-------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|Extract Jason is 25 years old.  |[{role -> system, content -> You are a helpful assistant.}, {role -> user, content -> Extract Jason is 25 years old.}]  |{Jason, 25}  |{chatcmpl-9jx75u6lErs6H2SzQ9P5Ig9h93KnQ, [{stop, 0, NULL, {NULL, assistant, NULL, [{call_amzxOZAFnzCIjwECYV2DeR3v, {{"name":"Jason","age":25}, User}, function}]}}], 1720739019, gpt-4o-2024-05-13, chat.completion, NULL, fp_d33f7b429e, {9, 70, 79}}  |
|Extract Emma is 30 years old.   |[{role -> system, content -> You are a helpful assistant.}, {role -> user, content -> Extract Emma is 30 years old.}]   |{Emma, 30}   |{chatcmpl-9jx76p2yElY4NuFBvbxAYWMr9BhAW, [{stop, 0, NULL, {NULL, assistant, NULL, [{call_Wtmv95JbNcQ2nRQCZBoOfcJy, {{"name":"Emma","age":30}, User}, function}]}}], 1720739020, gpt-4o-2024-05-13, chat.completion, NULL, fp_d33f7b429e, {9, 70, 79}}   |
|Extract Michael is 42 years old.|[{role -> system, content -> You are a helpful assistant.}, {role -> user, content -> Extract Michael is 42 years old.}]|{Michael, 42}|{chatcmpl-9jx76z9S5P0sEp7lINe2RADAvAz2T, [{stop, 0, NULL, {NULL, assistant, NULL, [{call_NOXuYfkZ1XLQq5L3eUwdjmSY, {{"name":"Michael","age":42}, User}, function}]}}], 1720739020, gpt-4o-2024-05-13, chat.completion, NULL, fp_d33f7b429e, {9, 70, 79}}|
+--------------------------------+------------------------------------------------------------------------------------------------------------------------+-------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
```
## Limitations
Serialization of raw Python code in Spark is tricky. Spark UDFs attempt to pickle everything associated with functions, and it is known that Pydantic objects do not serialize well when defined on the driver level. However, when Pydantic classes are imported from another source, these seem to serialize perfectly fine. See this [example](https://learn.microsoft.com/en-us/answers/questions/1178741/can-not-use-pydantic-objects-in-udf) for more details.

Even if a Pydantic model is installed at the cluster level, it may not have a well-defined Spark schema. Try to avoid using `typing.Any` or `typing.Union` fields. Very complex types are not guaranteed to work.

We plan to continue adding generalizable examples to the `response_models.py` module, but users should consider building private Pydantic libraries where necessary, and install those at the cluster level. We will continue to investigate workarounds in the meantime.

## Contributing

We welcome contributions to Spark Instructor! Please see our [contributing guidelines](docs/contributing.md) for more information on how to get involved.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## Acknowledgements

This project builds upon the excellent work of the Apache Spark community, the creators of the [`instructor`](https://github.com/jxnl/instructor) library, and the creators of the [`sparkdantic`](https://github.com/mitchelllisle/sparkdantic) library.
