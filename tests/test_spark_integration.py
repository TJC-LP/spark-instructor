import pyspark.sql.functions as f
import pytest

from spark_instructor.response_models import User
from spark_instructor.udf import MessageRouter


@pytest.fixture
def message_router() -> MessageRouter:
    return MessageRouter(model="gpt-4o", response_model_type=User)


def test_serialize(message_router):
    schema = message_router.spark_schema

    @f.udf(returnType=schema)
    def serialize(messages):
        return message_router.create_object_from_messages(messages)

    serialize(f.col("text"))
