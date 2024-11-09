import base64
import os

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

        messages = []
        for msg in request.conversation:
            if msg.role in ["system", "assistant"]:
                json_message = {"role": msg.role, "content": msg.content}
                messages.append(json_message)
                continue
            # different types of content is only supported for user
            json_message = {
                "role": msg.role,
                "content": [{"type": "text", "text": msg.content}],
            }

            if msg.image:
                image_url = self._prepare_image(msg)
                if image_url:
                    json_message["content"].append(
                        {"type": "image_url", "image_url": image_url}
                    )

            messages.append(json_message)

        payload = {
            "model": request.model_name,
            "messages": messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": False,
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
            provider_used=request.model_provider,
            model_used=request.model_name,
            token_usage=token_usage,
        )

    def _prepare_image(self, msg: PromptMessage):
        if not msg.image:
            return None
        image = msg.image
        if isinstance(image, str):
            image_url = {"url": image}
        elif isinstance(image, bytes):
            image_url = {
                "url": "data:image/jpeg;base64,"
                + base64.b64encode(image).decode("utf-8")
            }
        else:
            raise ValueError(f"Image must be a URL or bytes, provided {type(image)}")
        return image_url

    def generate_embedding(self, request: EmbeddingRequest) -> EmbeddingResponse:
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


class IonosProvider(OpenAILikeProvider):
    def __init__(self, api_key: str = ""):
        if not api_key:
            api_key = os.environ.get("LLMONKEY_IONOS_API_KEY")
        super().__init__(api_key, "https://openai.inference.de-txl.ionos.com/v1")


class MistralProvider(OpenAILikeProvider):
    def __init__(self, api_key: str = ""):
        if not api_key:
            api_key = os.environ.get("LLMONKEY_MISTRAL_API_KEY")
        super().__init__(api_key, "https://api.mistral.ai/v1")


class NebiusProvider(OpenAILikeProvider):
    def __init__(self, api_key: str = ""):
        if not api_key:
            api_key = os.environ.get("LLMONKEY_NEBIUS_API_KEY")
        super().__init__(api_key, "https://api.studio.nebius.ai/v1")
