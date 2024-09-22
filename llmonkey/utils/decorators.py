import json
from functools import wraps

from pydantic import ValidationError


def validate_llm_output(model, retries=3):
    """
    Decorator to validate output of a function against a given Pydantic model.

    The decorator will call the function and attempt to parse the result as JSON.
    Then it will try to validate the parsed data against the given Pydantic model.
    If validation fails, it will retry the function call up to a certain number
    of times, specified by the `retries` parameter. Original result of the decorated
    function will be returned as the second

    If all retries fail, it will raise a ValueError with a message indicating
    how many retries were attempted.

    :param model: The Pydantic model to validate against
    :param retries: The number of times to retry the function call if validation fails
    :return: A decorator function
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(retries):
                result = func(*args, **kwargs)
                try:
                    # Try to parse string as JSON, assumind last element of conversation is the output of LLM
                    data = json.loads(s := result.conversation[-1].content)
                    return model(**data), result  # Validate against Pydantic model
                except (json.JSONDecodeError, ValidationError) as e:
                    if attempt == retries - 1:
                        raise ValueError(
                            f"Validation failed after {retries} attempts: {e}. str: {s}"
                        )
            raise ValueError(f"Failed after {retries} retries, last str: {s}")

        return wrapper

    return decorator
