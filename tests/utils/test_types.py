from typing import Dict, List, Literal, Optional, Union

import pytest
from pydantic import BaseModel
from pyspark.sql.types import (
    ArrayType,
    BooleanType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)
from typing_extensions import NotRequired, Required, TypedDict

from spark_instructor.utils.types import (
    make_spark_schema_nullable,
    typeddict_to_pydantic,
)


def test_make_nullable_simple_types():
    # Test with simple types
    assert make_spark_schema_nullable(StringType()) == StringType()
    assert make_spark_schema_nullable(IntegerType()) == IntegerType()
    assert make_spark_schema_nullable(BooleanType()) == BooleanType()


def test_make_nullable_struct_field():
    # Test with StructField
    field = StructField("test", StringType(), False)
    nullable_field = make_spark_schema_nullable(field)
    assert nullable_field.nullable
    assert isinstance(nullable_field.dataType, StringType)


def test_make_nullable_array():
    # Test with ArrayType
    array = ArrayType(StringType(), False)
    nullable_array = make_spark_schema_nullable(array)
    assert nullable_array.containsNull
    assert isinstance(nullable_array.elementType, StringType)


def test_make_nullable_struct():
    # Test with StructType
    struct = StructType([StructField("field1", StringType(), False), StructField("field2", IntegerType(), True)])
    nullable_struct = make_spark_schema_nullable(struct)
    assert all(field.nullable for field in nullable_struct.fields)
    assert isinstance(nullable_struct.fields[0].dataType, StringType)
    assert isinstance(nullable_struct.fields[1].dataType, IntegerType)


def test_make_nullable_nested_struct():
    # Test with nested StructType
    nested_struct = StructType(
        [
            StructField(
                "outer",
                StructType(
                    [
                        StructField("inner1", StringType(), False),
                        StructField("inner2", ArrayType(IntegerType(), False), False),
                    ]
                ),
                False,
            )
        ]
    )
    nullable_nested_struct = make_spark_schema_nullable(nested_struct)

    assert nullable_nested_struct.fields[0].nullable
    inner_struct = nullable_nested_struct.fields[0].dataType
    assert isinstance(inner_struct, StructType)
    assert inner_struct.fields[0].nullable
    assert isinstance(inner_struct.fields[0].dataType, StringType)
    assert inner_struct.fields[1].nullable
    assert isinstance(inner_struct.fields[1].dataType, ArrayType)
    assert inner_struct.fields[1].dataType.containsNull


def test_make_nullable_complex_schema():
    # Test with a complex schema
    complex_schema = StructType(
        [
            StructField("id", IntegerType(), False),
            StructField("name", StringType(), True),
            StructField("tags", ArrayType(StringType(), False), False),
            StructField(
                "nested",
                StructType(
                    [
                        StructField("inner1", BooleanType(), False),
                        StructField(
                            "inner2",
                            ArrayType(
                                StructType(
                                    [
                                        StructField("deep1", StringType(), False),
                                        StructField("deep2", IntegerType(), True),
                                    ]
                                ),
                                False,
                            ),
                            False,
                        ),
                    ]
                ),
                False,
            ),
        ]
    )

    nullable_complex_schema = make_spark_schema_nullable(complex_schema)

    # Assertions for top-level fields
    assert all(field.nullable for field in nullable_complex_schema.fields)
    assert isinstance(nullable_complex_schema.fields[0].dataType, IntegerType)
    assert isinstance(nullable_complex_schema.fields[1].dataType, StringType)
    assert isinstance(nullable_complex_schema.fields[2].dataType, ArrayType)
    assert nullable_complex_schema.fields[2].dataType.containsNull

    # Assertions for nested struct
    nested = nullable_complex_schema.fields[3].dataType
    assert isinstance(nested, StructType)
    assert all(field.nullable for field in nested.fields)
    assert isinstance(nested.fields[0].dataType, BooleanType)
    assert isinstance(nested.fields[1].dataType, ArrayType)
    assert nested.fields[1].dataType.containsNull

    # Assertions for deeply nested struct
    deep_struct = nested.fields[1].dataType.elementType
    assert isinstance(deep_struct, StructType)
    assert all(field.nullable for field in deep_struct.fields)
    assert isinstance(deep_struct.fields[0].dataType, StringType)
    assert isinstance(deep_struct.fields[1].dataType, IntegerType)


class SimpleTypedDict(TypedDict):
    name: str
    age: int


@pytest.fixture
def simple_typeddict():
    return SimpleTypedDict


class NestedTypedDict(TypedDict):
    user: SimpleTypedDict
    active: bool


@pytest.fixture
def nested_typeddict(simple_typeddict):
    return NestedTypedDict


class ComplexTypedDict(TypedDict):
    id: int
    data: List[SimpleTypedDict]
    metadata: Dict[str, str]


@pytest.fixture
def complex_typeddict(simple_typeddict):
    return ComplexTypedDict


class TypedDictWithLiteral(TypedDict):
    status: Literal["active", "inactive"]


@pytest.fixture
def typeddict_with_literal():
    return TypedDictWithLiteral


class TypedDictWithNotRequired(TypedDict):
    name: str
    nickname: NotRequired[str]


@pytest.fixture
def typeddict_with_not_required():
    return TypedDictWithNotRequired


class TypedDictWithRequired(TypedDict):
    id: Required[int]
    name: str


@pytest.fixture
def typeddict_with_required():
    return TypedDictWithRequired


class TypedDictWithTotalFalse(TypedDict, total=False):
    name: str
    age: int


@pytest.fixture
def typeddict_with_total_false():
    return TypedDictWithTotalFalse


class TypedDictWithTotalFalseRequired(TypedDict, total=False):
    name: Required[str]
    age: int


@pytest.fixture
def typeddict_with_total_false_and_required_field():
    return TypedDictWithTotalFalseRequired


def test_simple_typeddict(simple_typeddict):
    PydanticModel = typeddict_to_pydantic(simple_typeddict)
    assert issubclass(PydanticModel, BaseModel)
    assert set(PydanticModel.__annotations__.keys()) == {"name", "age"}
    assert PydanticModel.__annotations__["name"] == str
    assert PydanticModel.__annotations__["age"] == int


def test_nested_typeddict(nested_typeddict):
    PydanticModel = typeddict_to_pydantic(nested_typeddict)
    assert issubclass(PydanticModel, BaseModel)
    assert set(PydanticModel.__annotations__.keys()) == {"user", "active"}
    assert issubclass(PydanticModel.__annotations__["user"], BaseModel)
    assert PydanticModel.__annotations__["active"] == bool


def test_complex_typeddict(complex_typeddict):
    PydanticModel = typeddict_to_pydantic(complex_typeddict)
    assert issubclass(PydanticModel, BaseModel)
    assert set(PydanticModel.__annotations__.keys()) == {"id", "data", "metadata"}
    assert PydanticModel.__annotations__["id"] == int
    assert PydanticModel.__annotations__["metadata"] == Dict[str, str]

    DataModel = PydanticModel.__annotations__["data"]
    assert DataModel.__origin__ == list
    assert issubclass(DataModel.__args__[0], BaseModel)


def test_typeddict_with_literal(typeddict_with_literal):
    PydanticModel = typeddict_to_pydantic(typeddict_with_literal)
    assert issubclass(PydanticModel, BaseModel)
    assert PydanticModel.__annotations__["status"] == Literal["active", "inactive"]


def test_typeddict_with_not_required(typeddict_with_not_required):
    PydanticModel = typeddict_to_pydantic(typeddict_with_not_required)
    assert issubclass(PydanticModel, BaseModel)
    assert PydanticModel.__annotations__["name"] == str
    assert PydanticModel.__annotations__["nickname"] == Optional[str]


def test_typeddict_with_required(typeddict_with_required):
    PydanticModel = typeddict_to_pydantic(typeddict_with_required)
    assert issubclass(PydanticModel, BaseModel)
    assert PydanticModel.model_fields["id"].is_required()
    assert PydanticModel.model_fields["name"].is_required()


def test_typeddict_with_total_false(typeddict_with_total_false):
    PydanticModel = typeddict_to_pydantic(typeddict_with_total_false)
    assert issubclass(PydanticModel, BaseModel)
    assert set(PydanticModel.__annotations__.keys()) == {"name", "age"}
    assert PydanticModel.__annotations__["name"].__origin__ == Union
    assert PydanticModel.__annotations__["age"].__origin__ == Union
    assert str in PydanticModel.__annotations__["name"].__args__
    assert int in PydanticModel.__annotations__["age"].__args__
    assert type(None) in PydanticModel.__annotations__["name"].__args__
    assert type(None) in PydanticModel.__annotations__["age"].__args__

    # Test that all fields are optional
    instance = PydanticModel()
    assert instance.model_dump() == {"name": None, "age": None}

    # Test that we can still set values
    instance = PydanticModel(name="John", age=30)
    assert instance.model_dump() == {"name": "John", "age": 30}


def test_typeddict_with_total_false_and_required_field(typeddict_with_total_false_and_required_field):
    PydanticModel = typeddict_to_pydantic(typeddict_with_total_false_and_required_field)
    assert issubclass(PydanticModel, BaseModel)
    assert set(PydanticModel.__annotations__.keys()) == {"name", "age"}
    assert PydanticModel.__annotations__["name"] == str
    assert PydanticModel.__annotations__["age"].__origin__ == Union
    assert int in PydanticModel.__annotations__["age"].__args__
    assert type(None) in PydanticModel.__annotations__["age"].__args__

    # Test that we can still set values
    instance = PydanticModel(name="John")
    assert instance.model_dump() == {"name": "John", "age": None}
