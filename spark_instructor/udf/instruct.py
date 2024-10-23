"""Module for defining spark instruct functions."""

import asyncio
import logging
import sys
from typing import Any, Callable, Optional, Type, TypedDict, TypeVar

import instructor
import pandas as pd
from pydantic import BaseModel
from pyspark.sql import Column
from pyspark.sql.functions import lit, pandas_udf, to_json
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from spark_instructor.completions import OpenAICompletion
from spark_instructor.registry import ClientRegistry
from spark_instructor.types.base import SparkChatCompletionMessages
from spark_instructor.udf.message_router import ModelSerializer

if sys.version_info >= (3, 11):
    from asyncio import timeout
else:
    from async_timeout import timeout

T = TypeVar("T", bound=BaseModel)

# Configure logging
logging.basicConfig(level=logging.INFO)
logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
default_logger = logging.getLogger(__name__)


def get_or_create_event_loop() -> asyncio.AbstractEventLoop:
    """Get the current event loop or create a new one."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


class AsyncRetryingConfig(TypedDict, total=False):
    """Config for creating an ``tenacity.AsyncRetrying`` class."""

    max_attempts: int
    wait_multiplier: float
    wait_min: float
    wait_max: float
    retry_exception: Type[Exception]
    reraise: bool


def create_async_retrying(config: AsyncRetryingConfig) -> AsyncRetrying:
    """Create a ``tenacity.AsyncRetrying`` object from a configuration dictionary.

    pyspark has trouble serializing ``tenacity.AsyncRetrying`` objects, so we can create one within the UDF instead.

    Args:
        config (AsyncRetryingConfig): A dictionary containing retry configuration parameters
    Returns
        An AsyncRetrying object
    """
    stop_strategy = stop_after_attempt(config.get("max_attempts", 5))

    wait_strategy = wait_exponential(
        multiplier=config.get("wait_multiplier", 1), min=config.get("wait_min", 1), max=config.get("wait_max", 60)
    )

    retry_exception = config.get("retry_exception", Exception)
    if not isinstance(retry_exception, type):
        raise ValueError("retry_exception must be a type")

    retry_strategy = retry_if_exception_type(retry_exception)

    return AsyncRetrying(
        stop=stop_strategy, wait=wait_strategy, retry=retry_strategy, reraise=config.get("reraise", True)
    )


def instruct(
    response_model: Optional[Type[T]] = None,
    default_model: Optional[str] = None,
    default_model_class: Optional[str] = None,
    default_mode: Optional[instructor.Mode] = None,
    default_max_tokens: Optional[int] = None,
    default_temperature: Optional[float] = None,
    default_max_retries: Optional[int | AsyncRetryingConfig] = 1,
    registry: ClientRegistry = ClientRegistry(),
    concurrency_limit: int = 16,
    task_timeout: float = 600,
    logger: logging.Logger = default_logger,
    safe_mode: bool = False,
    enable_caching: bool = False,
    **kwargs,
) -> Callable:
    """Create a pandas UDF for serving model responses in a Spark environment.

    This function generates a UDF that can process conversations and return model responses,
    optionally structured according to a Pydantic model. It supports various configuration
    options and can work with different model types and completion modes.

    Args:
        response_model (Optional[Type[T]]): The Pydantic model type for the response.
            If None, the UDF will return a standard chat completion message as a string.
        default_model (Optional[str]): The default model to use if not specified in the UDF call.
        default_model_class (Optional[str]): The default model class to use if not specified.
        default_mode (Optional[instructor.Mode]): The default instructor mode to use.
        default_max_tokens (Optional[int]): The default maximum number of tokens for the response.
        default_temperature (Optional[float]): The default temperature for the model's output.
        default_max_retries (Optional[int | AsyncRetryingConfig): The default maximum number of retries for failed requests.
            Can be a dictionary for creating an exponential backoff using ``create_async_retrying``.
        registry (ClientRegistry): The client registry for routing requests to appropriate model factories.
        concurrency_limit (int): The concurrency limit for the Sephamore.
        task_timeout (float): The timeout to use before raising an exception.
            This represents timeout on the task level.
        logger (Logger): Logger to use.
        safe_mode (bool): If True, return a null row instead of raising an exception when an error occurs.
            Recommended for large batches.
        enable_caching (bool): If True, enable Anthropic prompt caching.
        **kwargs: Additional keyword arguments to pass to the model creation function.

    Returns:
        Callable: A pandas UDF that can be used in Spark DataFrame operations.

    The returned UDF accepts the following parameters:
        - conversation (Column): Column containing conversation data as SparkChatCompletionMessages.
        - model (Optional[Column]): Column containing model names.
        - model_class (Optional[Column]): Column containing model classes.
        - mode (Optional[Column]): Column containing instructor modes.
        - max_tokens (Optional[Column]): Column containing maximum token values.
        - temperature (Optional[Column]): Column containing temperature values.
        - max_retries (Optional[Column]): Column containing maximum retry values.

    If any of the optional parameters are not provided in the UDF call, they will use the default
    values specified in the `instruct` function arguments.

    The UDF processes each row asynchronously, allowing for efficient parallel processing of
    multiple conversations.

    Example:
        ```python

        >>> from databricks.connect import DatabricksSession
        >>> from pydantic import BaseModel
        >>> from pyspark.sql.functions import lit
        >>> from spark_instructor.utils.prompt import create_chat_completion_messages
        >>> from spark_instructor.response_models import TextResponse
        >>> import json
        >>>
        >>> spark = DatabricksSession.builder.serverless().getOrCreate()
        >>> df = spark.createDataFrame([("What is the capital of France?",)], ["content"])
        >>> df = df.withColumn("conversation", create_chat_completion_messages([{"role": lit("user"), "content": "content"}]))
        >>> instruct_udf = instruct(TextResponse, default_model="gpt-4o-mini")
        >>> result_df = df.withColumn("response", instruct_udf("conversation"))
        >>> result_df.schema.jsonValue()
        {'type': 'struct', 'fields': [{'name': 'content', 'type': 'string', 'nullable': True, 'metadata': {}}, {'name': 'conversation', 'type': {'type': 'array', 'elementType': {'type': 'struct', 'fields': [{'name': 'role', 'type': 'string', 'nullable': False, 'metadata': {}}, {'name': 'content', 'type': 'string', 'nullable': True, 'metadata': {}}, {'name': 'image_urls', 'type': {'type': 'array', 'elementType': {'type': 'struct', 'fields': [{'name': 'url', 'type': 'string', 'nullable': False, 'metadata': {}}, {'name': 'detail', 'type': 'string', 'nullable': True, 'metadata': {}}]}, 'containsNull': True}, 'nullable': True, 'metadata': {}}, {'name': 'name', 'type': 'string', 'nullable': True, 'metadata': {}}, {'name': 'tool_calls', 'type': {'type': 'array', 'elementType': {'type': 'struct', 'fields': [{'name': 'id', 'type': 'string', 'nullable': False, 'metadata': {}}, {'name': 'function', 'type': {'type': 'struct', 'fields': [{'name': 'arguments', 'type': 'string', 'nullable': False, 'metadata': {}}, {'name': 'name', 'type': 'string', 'nullable': False, 'metadata': {}}]}, 'nullable': False, 'metadata': {}}, {'name': 'type', 'type': 'string', 'nullable': False, 'metadata': {}}]}, 'containsNull': True}, 'nullable': True, 'metadata': {}}, {'name': 'tool_call_id', 'type': 'string', 'nullable': True, 'metadata': {}}]}, 'containsNull': False}, 'nullable': False, 'metadata': {}}, {'name': 'response', 'type': {'type': 'struct', 'fields': [{'name': 'text_response', 'type': {'type': 'struct', 'fields': [{'name': 'text', 'type': 'string', 'nullable': False, 'metadata': {}}]}, 'nullable': True, 'metadata': {}}, {'name': 'chat_completion', 'type': {'type': 'struct', 'fields': [{'name': 'id', 'type': 'string', 'nullable': False, 'metadata': {}}, {'name': 'choices', 'type': {'type': 'array', 'elementType': {'type': 'struct', 'fields': [{'name': 'finish_reason', 'type': 'string', 'nullable': False, 'metadata': {}}, {'name': 'index', 'type': 'integer', 'nullable': False, 'metadata': {}}, {'name': 'logprobs', 'type': {'type': 'struct', 'fields': [{'name': 'content', 'type': {'type': 'array', 'elementType': {'type': 'struct', 'fields': [{'name': 'token', 'type': 'string', 'nullable': False, 'metadata': {}}, {'name': 'bytes', 'type': {'type': 'array', 'elementType': 'integer', 'containsNull': False}, 'nullable': True, 'metadata': {}}, {'name': 'logprob', 'type': 'double', 'nullable': False, 'metadata': {}}, {'name': 'top_logprobs', 'type': {'type': 'array', 'elementType': {'type': 'struct', 'fields': [{'name': 'token', 'type': 'string', 'nullable': False, 'metadata': {}}, {'name': 'bytes', 'type': {'type': 'array', 'elementType': 'integer', 'containsNull': False}, 'nullable': True, 'metadata': {}}, {'name': 'logprob', 'type': 'double', 'nullable': False, 'metadata': {}}]}, 'containsNull': False}, 'nullable': False, 'metadata': {}}]}, 'containsNull': True}, 'nullable': True, 'metadata': {}}]}, 'nullable': True, 'metadata': {}}, {'name': 'message', 'type': {'type': 'struct', 'fields': [{'name': 'content', 'type': 'string', 'nullable': True, 'metadata': {}}, {'name': 'role', 'type': 'string', 'nullable': False, 'metadata': {}}, {'name': 'function_call', 'type': {'type': 'struct', 'fields': [{'name': 'arguments', 'type': 'string', 'nullable': False, 'metadata': {}}, {'name': 'name', 'type': 'string', 'nullable': False, 'metadata': {}}]}, 'nullable': True, 'metadata': {}}, {'name': 'tool_calls', 'type': {'type': 'array', 'elementType': {'type': 'struct', 'fields': [{'name': 'id', 'type': 'string', 'nullable': False, 'metadata': {}}, {'name': 'function', 'type': {'type': 'struct', 'fields': [{'name': 'arguments', 'type': 'string', 'nullable': False, 'metadata': {}}, {'name': 'name', 'type': 'string', 'nullable': False, 'metadata': {}}]}, 'nullable': False, 'metadata': {}}, {'name': 'type', 'type': 'string', 'nullable': False, 'metadata': {}}]}, 'containsNull': True}, 'nullable': True, 'metadata': {}}]}, 'nullable': False, 'metadata': {}}]}, 'containsNull': False}, 'nullable': False, 'metadata': {}}, {'name': 'created', 'type': 'integer', 'nullable': False, 'metadata': {}}, {'name': 'model', 'type': 'string', 'nullable': False, 'metadata': {}}, {'name': 'object', 'type': 'string', 'nullable': False, 'metadata': {}}, {'name': 'service_tier', 'type': 'string', 'nullable': True, 'metadata': {}}, {'name': 'system_fingerprint', 'type': 'string', 'nullable': True, 'metadata': {}}, {'name': 'usage', 'type': {'type': 'struct', 'fields': [{'name': 'completion_tokens', 'type': 'integer', 'nullable': False, 'metadata': {}}, {'name': 'prompt_tokens', 'type': 'integer', 'nullable': False, 'metadata': {}}, {'name': 'total_tokens', 'type': 'integer', 'nullable': False, 'metadata': {}}]}, 'nullable': True, 'metadata': {}}]}, 'nullable': True, 'metadata': {}}]}, 'nullable': True, 'metadata': {}}]}

        ```
    Note:
        - The UDF uses the provided ClientRegistry to determine which model factory to use.
        - If `response_model` is None, the UDF will return the raw completion message as a string.
        - The function supports both structured (Pydantic model) and unstructured (string) responses.
    """  # noqa: E501
    logger.info(
        f"Initializing instruct function with concurrency limit: "
        f"{concurrency_limit}, timeout: {task_timeout}, safe_mode: {safe_mode}"
    )
    model_serializer = ModelSerializer(response_model, OpenAICompletion)
    response_model_name = model_serializer.response_model_name
    completion_model_name = model_serializer.completion_model_name

    @pandas_udf(returnType=model_serializer.spark_schema)  # type: ignore
    def _pandas_udf(
        conversation: pd.Series,
        model: pd.Series,
        model_class: pd.Series,
        mode: pd.Series,
        max_tokens: pd.Series,
        temperature: pd.Series,
        max_retries: pd.Series,
    ) -> pd.DataFrame:
        """Pandas UDF for processing conversations and generating model responses.

        Args:
            conversation (pd.Series): DataFrame containing conversation data.
            model (pd.Series): Series containing model names.
            model_class (pd.Series): Series containing model classes.
            mode (pd.Series): Series containing modes.
            max_tokens (pd.Series): Series containing maximum token values.
            temperature (pd.Series): Series containing temperature values.
            max_retries (pd.Series): Series containing maximum retry values.

        Returns:
            pd.DataFrame: DataFrame containing the processed responses.
        """
        # Convert dataframe rows to list of Conversation objects
        logger.info(f"Processing batch of {len(conversation)} conversations")
        semaphore = asyncio.Semaphore(concurrency_limit)

        async def process_row(
            conversation_: str,
            model_: str,
            model_class_: str,
            mode_: str,
            max_tokens_: int,
            temperature_: float,
            max_retries_: int | AsyncRetryingConfig,
        ) -> Optional[Any]:
            logger.debug(f"Processing row with model: {model_}, model_class: {model_class_}")
            factory_type = (
                registry.get_factory(model_class_)
                if model_class_ is not None
                else registry.get_factory_from_model(model_)
            )
            factory = factory_type.from_config(
                instructor.Mode(mode_) if mode_ is not None else mode_, enable_caching=enable_caching
            )
            create_fn = factory.create_with_completion if response_model else factory.create
            try:
                async with timeout(task_timeout):
                    result = await create_fn(
                        messages=SparkChatCompletionMessages.model_validate_json(conversation_),
                        response_model=response_model,  # type: ignore
                        model=model_,
                        max_tokens=max_tokens_,
                        temperature=temperature_,
                        max_retries=(
                            create_async_retrying(max_retries_)
                            if isinstance(max_retries_, dict)
                            else max_retries_  # type: ignore
                        ),
                        **kwargs,
                    )
                logger.debug("Row processed successfully")
                return result
            except asyncio.TimeoutError:
                logger.error(f"Timeout occurred while processing row with model: {model_}")
                if safe_mode:
                    return None
                raise
            except Exception as e:
                logger.error(f"Error processing row with model {model_}: {str(e)}")
                if safe_mode:
                    return None
                raise

        async def process_row_with_semaphore(
            conversation_,
            model_,
            model_class_,
            mode_,
            max_tokens_,
            temperature_,
            max_retries_,
        ) -> dict[str, Any]:
            async with semaphore:
                result = await process_row(
                    conversation_,
                    model_,
                    model_class_,
                    mode_,
                    max_tokens_,
                    temperature_,
                    max_retries_,
                )
                if result is None:
                    # Ensure failed result is serializable
                    return (
                        {completion_model_name: None}
                        if not response_model_name
                        else {response_model_name: None, completion_model_name: None}
                    )
                if not response_model_name:
                    # Basic text response
                    return {completion_model_name: result.model_dump()}
                # Structured response
                return {
                    response_model_name: result[0].model_dump(),
                    completion_model_name: result[1].model_dump(),
                }

        async def process_all_rows():
            tasks = [
                process_row_with_semaphore(
                    conv,
                    mdl or default_model,
                    mdl_cls,
                    md,
                    max_tkns or default_max_tokens,
                    temp or default_temperature,
                    max_rtr or default_max_retries,
                )
                for conv, mdl, mdl_cls, md, max_tkns, temp, max_rtr in zip(
                    conversation,
                    model,
                    model_class,
                    mode,
                    max_tokens,
                    temperature,
                    max_retries,
                )
            ]
            return await asyncio.gather(*tasks)

        loop = get_or_create_event_loop()
        try:
            results = loop.run_until_complete(process_all_rows())
            logger.info(f"Successfully processed {len(results)} rows")
        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}")
            raise

        # Convert results to DataFrame
        return pd.DataFrame(results)

    def pandas_udf_wrapped(
        conversation: Column,
        model: Optional[Column] = None,
        model_class: Optional[Column] = None,
        mode: Optional[Column] = None,
        max_tokens: Optional[Column] = None,
        temperature: Optional[Column] = None,
        max_retries: Optional[Column] = None,
    ) -> Column:
        """Create a pandas UDF that wraps the model inference function.

        Column arguments which are not passed will be set to their defaults provided by the constructor.

        Args:
            conversation (Column): Column containing conversation data.
            model (Optional[Column]): Column containing model names.
            model_class (Optional[Column]): Column containing model classes.
            mode (Optional[Column]): Column containing modes.
            max_tokens (Optional[Column]): Column containing maximum token values.
            temperature (Optional[Column]): Column containing temperature values.
            max_retries (Optional[Column]): Column containing maximum retry values.

        Returns:
            Column: Column containing the processed responses.
        """
        if model is None:
            model = lit(default_model)
        if max_tokens is None:
            max_tokens = lit(default_max_tokens)
        if temperature is None:
            temperature = lit(default_temperature)
        if max_retries is None:
            max_retries = lit(default_max_retries) if isinstance(default_max_retries, int) else lit(None)
        if model_class is None:
            model_class = lit(default_model_class) if default_model_class else lit(None)
        if mode is None:
            mode = lit(default_mode.value) if default_mode else lit(None)

        return _pandas_udf(to_json(conversation), model, model_class, mode, max_tokens, temperature, max_retries)

    return pandas_udf_wrapped
