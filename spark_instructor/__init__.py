"""Init for ``spark_instructor``."""

__version__ = "0.2.5"
__author__ = "Richie Caputo"
__email__ = "rcaputo3@tjclp.com"

from dotenv import load_dotenv

from spark_instructor.completions import is_anthropic_available
from spark_instructor.udf.instruct import instruct
from spark_instructor.utils.prompt import create_chat_completion_messages

__all__ = ["create_chat_completion_messages", "instruct", "is_anthropic_available"]

load_dotenv()
