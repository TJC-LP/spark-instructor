from typing import Type, Union

from pydantic import BaseModel
from pydantic.json_schema import JsonSchemaMode
from pyspark.sql.types import StructType

BaseModelOrSparkModel = Union[BaseModel, "SparkModel"]

def create_spark_schema(
    model: Type[BaseModelOrSparkModel],
    safe_casting: bool = False,
    by_alias: bool = True,
    mode: JsonSchemaMode = "validation",
) -> StructType: ...

class SparkModel(BaseModel): ...