import json
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Type

import requests
from pydantic import BaseModel, ValidationError

from ..models import (
    ChatRequest,
    ChatResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    PromptMessage,
)


class BaseModelProvider(ABC):
    """Base class for all model providers using OpenAI-like APIs with Pydantic models."""

    def __init__(self, api_key: str, base_url: str):
        """
        Initialize with common parameters such as API key, base URL, and model name.
        """
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _post(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Helper method to make POST requests to the provider's API.
        """
        url = f"{self.base_url}/{endpoint}"
        try:
            response = requests.post(url, json=payload, headers=self.headers)
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            raise ValueError(f"LLMonkey: Error {response.status_code}: {response.text}")
        return response.json()

    def generate_structured_response(
        self, request: ChatRequest, data_model: Type[BaseModel], retries=3
    ) -> Tuple[BaseModel, ChatResponse]:
        """
        Generate a structured response using a Pydantic model.

        This method will call `generate_prompt_response` and attempt to
        parse the response as JSON. The parsed data will be validated
        against the given Pydantic model.

        If validation fails, it will retry the function call up to a
        certain number of times, specified by the `retries` parameter.
        Original result of the decorated function will be returned as the
        second element of the tuple.

        If all retries fail, it will raise a ValueError with a message
        indicating how many retries were attempted.
        """
        for attempt in range(retries):
            result = self.generate_prompt_response(request)
            try:
                # Try to parse string as JSON, assumind last element of conversation is the output of LLM
                data = json.loads(s := result.conversation[-1].content)
                return data_model(**data), result  # Validate against Pydantic model
            except (json.JSONDecodeError, ValidationError) as e:
                if attempt == retries - 1:
                    raise ValueError(
                        f"Validation failed after {retries} attempts: {e}. str: {s}"
                    )
        raise ValueError(f"Failed after {retries} retries, last str: {s}")

    @abstractmethod
    def generate_prompt_response(self, request: ChatRequest) -> ChatResponse:
        """
        Abstract method to generate a response from a single prompt.
        Derived classes must implement this.
        """
        pass

    @abstractmethod
    def generate_chat_response(self, request: ChatRequest) -> ChatResponse:
        """
        Abstract method to handle multi-turn chat (sequence of messages).
        Derived classes must implement this.
        """
        pass

    @abstractmethod
    def get_embedding(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """
        Abstract method to get text embeddings.
        Derived classes must implement this.
        """
        pass
