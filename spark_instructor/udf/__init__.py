"""Package for defining ``spark-instructor`` user-defined functions in Spark."""

from spark_instructor.udf.instruct import instruct
from spark_instructor.udf.message_router import MessageRouter

__all__ = ["MessageRouter", "instruct"]
