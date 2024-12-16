"""Module for handling type conversion."""

from typing import Any, Dict, Mapping, Optional, Type, TypeVar

from pydantic import BaseModel, Field, TypeAdapter, create_model
from pydantic.fields import FieldInfo
from pyspark.sql.types import ArrayType, DataType, StructField, StructType
from typing_extensions import TypedDict

__all__ = ["typeddict_to_pydantic", "pydantic_to_typeddict", "make_spark_schema_nullable"]


class BaseTypedDict(TypedDict):
    pass


T = TypeVar("T", bound=BaseTypedDict)
T_Spark = TypeVar("T_Spark", StructType, ArrayType, StructField, DataType)


def make_nullable_struct(schema: StructType) -> StructType:
    """Make a struct nullable."""
    return StructType([make_spark_schema_nullable(field) for field in schema.fields])


def make_nullable_field(schema: StructField) -> StructField:
    """Make a field nullable."""
    return StructField(schema.name, make_spark_schema_nullable(schema.dataType), nullable=True)


def make_nullable_array(schema: ArrayType) -> ArrayType:
    """Make an array nullable."""
    return ArrayType(make_spark_schema_nullable(schema.elementType), True)


def make_spark_schema_nullable(schema: T_Spark) -> T_Spark:
    """Make a spark type nullable."""
    if isinstance(schema, StructType):
        return make_nullable_struct(schema)
    elif isinstance(schema, StructField):
        return make_nullable_field(schema)
    elif isinstance(schema, ArrayType):
        return make_nullable_array(schema)
    else:
        return schema


def pydantic_to_typeddict(pydantic_model: BaseModel, return_type: Type[T], all_required: bool = False) -> T:
    """Convert a pydantic model to a typed dict."""
    result = pydantic_model.model_dump()
    if not all_required:
        result = _remove_none_from_dict(result)

    return return_type(**result)


def _remove_none_from_dict(d: Any) -> Any:
    """Remove null values from a dictionary."""
    if not isinstance(d, dict):
        return d
    return {key: _remove_none_from_dict(val) for key, val in d.items() if val is not None}


def typeddict_to_pydantic(typeddict_class: Type[Any]) -> Type[BaseModel]:
    """Convert a TypedDict to a Pydantic model using TypeAdapter and core_schema.

    Args:
        typeddict_class (Type[Any]): The TypedDict class to convert.

    Returns:
        Type[BaseModel]: A Pydantic model class equivalent to the input TypedDict.

    Example:
        ```python

        >>> from pydantic import BaseModel
        >>> from typing import TypedDict
        >>> class MyTypedDict(TypedDict):
        ...     name: str
        ...     age: int
        >>> PydanticModel = typeddict_to_pydantic(MyTypedDict)
        >>> isinstance(PydanticModel(name="John", age=30), BaseModel)
        True

        ```
    """
    adapter = TypeAdapter(typeddict_class)
    core_schema = adapter.core_schema

    fields: Dict[str, tuple[Any, FieldInfo]] = {}
    _process_schema(core_schema, fields)

    return create_model(f"{typeddict_class.__name__}PD", __doc__=typeddict_class.__doc__, **fields)  # type: ignore


def _process_schema(schema: Mapping[str, Any], fields: Dict[str, tuple[Any, FieldInfo]]):
    """Process the core schema of a TypedDict and populate the fields dictionary.

    Args:
        schema (Mapping[str, Any]): The core schema of a TypedDict.
        fields (Dict[str, tuple[Any, FieldInfo]]): A dictionary to store processed fields.

    Note:
        This function modifies the `fields` dictionary in-place.
    """
    if schema["type"] == "typed-dict":
        for field_name, field_schema in schema["fields"].items():
            _process_field(field_name, field_schema, fields)
    elif schema["type"] == "dict":
        key_schema = schema.get("keys_schema", {})
        value_schema = schema.get("values_schema", {})
        key_type = _get_field_type(key_schema)
        value_type = _get_field_type(value_schema)
        fields[schema.get("ref", "DictField")] = (Dict[key_type, value_type], Field())  # type: ignore


def _process_field(field_name: str, field_schema: dict, fields: Dict[str, tuple[Any, FieldInfo]]):
    """Process a single field from the TypedDict schema and add it to the fields dictionary.

    Args:
        field_name (str): The name of the field.
        field_schema (dict): The schema of the field.
        fields (Dict[str, tuple[Any, FieldInfo]]): A dictionary to store processed fields.

    Note:
        This function modifies the `fields` dictionary in-place.
    """
    if field_schema["type"] == "typed-dict-field":
        field_type = _get_field_type(field_schema["schema"])
        is_required = field_schema.get("required", True)
        if not is_required:
            field_type = Optional[field_type]
        default = ... if is_required else None
        fields[field_name] = (field_type, Field(default=default))  # type: ignore
    elif field_schema["type"] in ["list", "dict", "typed-dict", "generator"]:
        nested_fields: Dict[str, tuple[Any, FieldInfo]] = {}
        _process_schema(field_schema, nested_fields)
        nested_model = create_model(f"Nested{field_name.capitalize()}", **nested_fields)  # type: ignore
        fields[field_name] = (nested_model, Field())


def _get_field_type(schema: dict) -> Any:
    """Determine the Python type based on the given schema.

    Args:
        schema (dict): The schema of a field.

    Returns:
        Any: The corresponding Python type for the schema.

    Note:
        This function handles various schema types including primitive types,
        lists, dicts, nested TypedDicts, literals, and unions.
    """
    schema_type = schema.get("type")
    if schema_type == "str":
        return str
    elif schema_type == "int":
        return int
    elif schema_type == "float":
        return float
    elif schema_type == "bool":
        return bool
    elif schema_type == "list":
        item_schema = schema.get("items_schema", {})
        return list[_get_field_type(item_schema)]  # type: ignore
    elif schema_type == "dict":
        key_schema = schema.get("keys_schema", {})
        value_schema = schema.get("values_schema", {})
        return Dict[_get_field_type(key_schema), _get_field_type(value_schema)]  # type: ignore
    elif schema_type == "typed-dict":
        nested_fields: Dict[str, tuple[Any, FieldInfo]] = {}
        _process_schema(schema, nested_fields)
        return create_model(f"Nested{schema.get('ref', 'Model')}", **nested_fields)  # type: ignore
    elif schema_type == "literal":
        from typing import Literal

        return Literal[tuple(schema.get("expected", []))]  # type: ignore
    elif schema_type == "union":
        from typing import Union

        choices = "choices"
        return Union[tuple(_get_field_type(s) for s in schema.get(choices, []))]  # type: ignore
    elif schema_type == "nullable":
        inner_schema = schema.get("schema", {})
        return Optional[_get_field_type(inner_schema)]
    elif schema_type == "generator":
        item_schema = schema.get("items_schema", {})
        return list[_get_field_type(item_schema)]  # type: ignore
    elif schema_type == "any":
        return Any
    else:
        return Any
