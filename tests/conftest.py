from typing import Generator

import pytest
from pyspark.sql import SparkSession


@pytest.fixture(scope="session")
def spark() -> Generator[SparkSession, None, None]:
    try:
        # If `databricks-connect` is installed, we will route to Databricks serverless
        from databricks.connect import DatabricksSession  # type: ignore

        session = DatabricksSession.builder.serverless().getOrCreate()
    except ImportError:
        # This won't work if `databricks-connect` is installed
        session = SparkSession.builder.master("local[*]").appName("test").getOrCreate()

    yield session
    session.stop()
