import base64
import os

from openai import AzureOpenAI

from ..models import (
    ChatRequest,
    ChatResponse,
    PromptMessage,
    TokenUsage,
)
from .base import BaseModelProvider


class AzureOpenAIProvider(BaseModelProvider):
    def __init__(self, api_key: str):
        if not api_key:
            api_key = os.environ.get("LLMONKEY_AZURE_API_KEY")
        if not os.environ.get("LLMONKEY_AZURE_OPENAI_URL"):
            raise ValueError("LLMONKEY_AZURE_OPENAI_URL environment variable is required")
        super().__init__(api_key, os.environ.get("LLMONKEY_AZURE_OPENAI_URL"))

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

        from llmonkey.llms.openai import OpenAI_o1, OpenAI_o1_Mini
        if request.model_name in [OpenAI_o1.config.identifier, OpenAI_o1_Mini.config.identifier]:
            del payload["max_tokens"]
            del payload["temperature"]
            payload["max_completion_tokens"] = request.max_tokens
            for message in messages:
                # only roles user and assistant are allowed
                if message["role"] == "system":
                    message["role"] = "user"

        from llmonkey.llms.deepinfra import Deepinfra_Qwen_QwQ_32B
        if request.model_name == Deepinfra_Qwen_QwQ_32B.config.identifier:
            for message in messages:
                # only roles user and assistant are allowed / there needs to be at least one user message (?)
                if message["role"] == "system":
                    message["role"] = "user"

        # Send the request to OpenAI API
        response_data = self._post(endpoint, payload)

        client = AzureOpenAI(
            azure_endpoint=self.base_url,
            api_key=self.api_key,
            api_version="2024-08-01-preview",
        )
        deployment = request.model_name  # is this always the same?
        completion = client.chat.completions.create(
            model=deployment,
            messages=messages,
            max_tokens=800,
            temperature=0.7,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            stream=False
        )
        msg = completion.choices[0].message
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
