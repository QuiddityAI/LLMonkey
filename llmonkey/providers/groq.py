import logging
import os
import time
from collections import defaultdict
from enum import Enum

from ratelimit import RateLimitException, limits, sleep_and_retry

from ..models import ChatRequest, ChatResponse
from .openai_like import OpenAILikeProvider


class GROQ_MODELS(str, Enum):
    LLAMA_3_8B = "llama3-8b-8192"
    LLAMA_3_70B = "llama3-70b-8192"
    MIXTRAL_8_7B = "mixtral-8x7b-32768"
    GEMMA_7B = "gemma-7b-it"
    OTHER = "other"


timestamps_and_num_tokens_per_model = defaultdict(dict)
max_tokens_per_minute_per_model = {
    GROQ_MODELS.LLAMA_3_8B: 30000,
    GROQ_MODELS.LLAMA_3_70B: 6000,
    GROQ_MODELS.MIXTRAL_8_7B: 5000,
    GROQ_MODELS.GEMMA_7B: 15000,
    GROQ_MODELS.OTHER: 5000,
}


class GroqProvider(OpenAILikeProvider):
    def __init__(self, api_key: str = ""):
        if not api_key:
            api_key = os.environ.get("LLMONKEY_GROQ_API_KEY")
        super().__init__(api_key, "https://api.groq.com/openai/v1")

    @sleep_and_retry
    @limits(calls=28, period=30)
    def generate_chat_response(self, request: ChatRequest) -> ChatResponse:
        """
        Handle chat responses using Groq's chat API.

        The Groq API has a token limit per minute. This function will raise a
        `RateLimitException` if the token limit is reached. The exception's
        `remaining` attribute will contain the number of seconds until the limit
        is reset.

        The token limit is as follows:

        - LLaMA 3.8B: 30000 tokens per minute
        - LLaMA 3.70B: 6000 tokens per minute
        - MixTRAL 8x7B: 5000 tokens per minute
        - Gemma 7B: 15000 tokens per minute
        - Other models: 5000 tokens per minute

        The rate limiting is done on a per-model basis.

        :param request: The chat request
        :return: The chat response
        """
        model = request.model_name
        for timestamp in timestamps_and_num_tokens_per_model[model].copy():
            if timestamp < time.time() - 60:
                del timestamps_and_num_tokens_per_model[model][timestamp]
        num_tokens = sum(timestamps_and_num_tokens_per_model[model].values())
        limit_model_name = (
            model if model in GROQ_MODELS._value2member_map_ else GROQ_MODELS.OTHER
        )
        if num_tokens > max_tokens_per_minute_per_model[limit_model_name]:
            remaining_time = (
                max(timestamps_and_num_tokens_per_model[model].keys())
                + 60
                - time.time()
            )
            logging.warning(
                f"LLMonkey: Token rate limit reached for Groq model {model}. Remaining time: {remaining_time}"
            )
            raise RateLimitException("too many calls", remaining_time)

        try:
            response = super().generate_chat_response(request)
        except Exception as e:
            # Check if the error is a rate limit error
            if "Error 429" in str(e):
                logging.warning(f"Token rate limit reached for Groq model {model}.")
                raise RateLimitException("too many calls", 10)
            else:
                # If the error is not a rate limit error, raise it
                raise e
        return response
