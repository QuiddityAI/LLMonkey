from abc import ABC, abstractmethod
from typing import Any, Dict

import requests

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
        response = requests.post(url, json=payload, headers=self.headers)
        response.raise_for_status()
        return response.json()

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
