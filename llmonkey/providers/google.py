import logging
import os

from ratelimit import RateLimitException, limits, sleep_and_retry

import vertexai
from vertexai.generative_models import GenerativeModel, Content, Image, Part, HarmCategory, HarmBlockThreshold
from google.api_core.exceptions import ResourceExhausted

from ..models import (
    ChatRequest,
    ChatResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    PromptMessage,
    TokenUsage,
)
from .base import BaseModelProvider

VERTEX_PROJECT_ID = os.environ.get("LLMONKEY_GOOGLE_PROJECT_ID")
VERTEX_LOCATION = os.environ.get("LLMONKEY_GOOGLE_LOCATION", "europe-west3")


class GoogleProvider(BaseModelProvider):
    def __init__(self, api_key: str):
        # note: the api_key is not used for the Google provider, instead an ADC (Application Default Credentials) is used
        vertexai.init(project=VERTEX_PROJECT_ID, location=VERTEX_LOCATION)
        super().__init__(api_key, None)

    def generate_prompt_response(self, request: ChatRequest) -> ChatResponse:
        """
        Handle a single prompt response using OpenAI's completion API (treated as chat with length 1).
        """
        return self.generate_chat_response(request)

    @sleep_and_retry
    def generate_chat_response(self, request: ChatRequest) -> ChatResponse:
        """
        Handle multi-turn chat responses.
        """

        model = GenerativeModel(request.model_name)

        contents = []
        for msg in request.conversation:
            if msg.role == "system":
                # Google models can't work with only system messages,
                # so we'll use user messages instead for everything
                contents.append(Content(role="user", parts=[Part.from_text(msg.content)]))
            elif msg.role == "user":
                parts = []
                if msg.image:
                    if isinstance(msg.image, str):
                        # treat as Google Cloud Storage URI
                        parts.append(Part.from_uri(msg.image, mime_type="image/jpeg"))
                    elif isinstance(msg.image, bytes):
                        # treat as raw image data
                        parts.append(Part.from_image(Image.from_bytes(msg.image)))
                if msg.content:
                    parts.append(Part.from_text(msg.content))
                contents.append(Content(role="user", parts=parts))
            else:
                raise ValueError(f"Unknown role: {msg.role}")

        try:
            response = model.generate_content(
                contents,
                generation_config={"temperature": request.temperature, "max_output_tokens": request.max_tokens},
                safety_settings={
                    HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY: HarmBlockThreshold.OFF,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.OFF,
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.OFF,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.OFF,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.OFF,
                },  # disable safety settings (filtering of unsafe content)
            )
        except ResourceExhausted as e:
            logging.error(f"ResourceExhausted: {e}, waiting 10 seconds")
            raise RateLimitException("ResourceExhausted", 10)

        conversation = request.conversation + [PromptMessage(role="assistant", content=response.text)]
        token_usage = TokenUsage(
            prompt_tokens=response.usage_metadata.prompt_token_count,
            completion_tokens=response.usage_metadata.candidates_token_count,
            total_tokens=response.usage_metadata.total_token_count,
        )
        return ChatResponse(
            conversation=conversation,
            provider_used=request.model_provider,
            model_used=request.model_name,
            token_usage=token_usage,
        )

    def generate_embedding(self, request: EmbeddingRequest) -> EmbeddingResponse:
        raise NotImplementedError("Embedding generation is not implemented for Google models.")

    def to_litellm(self):
        conf = dict(vertex_project=VERTEX_PROJECT_ID, vertex_location=VERTEX_LOCATION, prefix="vertex_ai")
        return conf
