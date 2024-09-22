import os

from ratelimit import limits, sleep_and_retry

from ..models import (
    ChatRequest,
    ChatResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    PromptMessage,
    TokenUsage,
)
from .base import BaseModelProvider


class OpenAILikeProvider(BaseModelProvider):
    def __init__(self, api_key: str, base_url: str):
        super().__init__(api_key, base_url)

    def generate_prompt_response(self, request: ChatRequest) -> ChatResponse:
        """
        Handle a single prompt response using OpenAI's completion API (treated as chat with length 1).
        """
        return self.generate_chat_response(request)

    def generate_chat_response(self, request: ChatRequest) -> ChatResponse:
        """
        Handle multi-turn chat responses using OpenAI's chat API.
        """
        endpoint = "chat/completions"
        payload = {
            "model": request.model_name,
            "messages": [
                {"role": msg.role, "content": msg.content}
                for msg in request.conversation
            ],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }

        # Send the request to OpenAI API
        response_data = self._post(endpoint, payload)
        msg = response_data["choices"][0]["message"]
        conversation = request.conversation + [
            PromptMessage(role=msg["role"], content=msg["content"])
        ]
        token_usage = TokenUsage(
            prompt_tokens=response_data["usage"]["prompt_tokens"],
            completion_tokens=response_data["usage"]["completion_tokens"],
            total_tokens=response_data["usage"]["total_tokens"],
        )
        return ChatResponse(
            conversation=conversation,
            model_used=request.model_provider,
            token_usage=token_usage,
        )

    def get_embedding(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """
        Get text embeddings using OpenAI's embedding API.
        """
        endpoint = "embeddings"
        payload = {"model": request.model_name, "input": request.text}

        # Send the request to OpenAI API
        response_data = self._post(endpoint, payload)
        embedding = response_data["data"][0]["embedding"]

        return EmbeddingResponse(embedding=embedding, model_used=request.model_provider)


class OpenAIProvider(OpenAILikeProvider):
    def __init__(self, api_key: str = ""):
        if not api_key:
            api_key = os.environ.get("LLMONKEY_OPENAI_API_KEY")
        super().__init__(api_key, "https://api.openai.com/v1")


class DeepInfraProvider(OpenAILikeProvider):
    def __init__(self, api_key: str = ""):
        if not api_key:
            api_key = os.environ.get("LLMONKEY_DEEPINFRA_API_KEY")
        super().__init__(api_key, "https://api.deepinfra.com/v1/openai")
