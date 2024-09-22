from typing import Any

from pydantic import BaseModel, validator

from .models import ChatRequest, ChatResponse, ModelProvider, PromptMessage
from .providers.base import BaseModelProvider
from .providers.groq import GroqProvider
from .providers.openai_like import DeepInfraProvider, OpenAIProvider


class ProviderConfig(BaseModel):
    provider: ModelProvider
    implementation: Any

    @validator("implementation")
    def validate_implementation(cls, implementation):
        if not issubclass(implementation, BaseModelProvider):
            raise ValueError("implementation must be a subclass of BaseModelProvider")
        return implementation


providers = {
    "openai": ProviderConfig(
        provider=ModelProvider.openai, implementation=OpenAIProvider
    ),
    "groq": ProviderConfig(provider=ModelProvider.groq, implementation=GroqProvider),
    "deepinfra": ProviderConfig(
        provider=ModelProvider.deepinfra, implementation=DeepInfraProvider
    ),
}


class LLMonkey(object):
    providers = providers

    def __init__(self):
        pass

    def generate_chat_response(
        self,
        provider: str,
        model_name: str,
        user_prompt: str = "",
        system_prompt: str = "",
        temperature=0.7,
        max_tokens=150,
        api_key: str = "",
    ) -> ChatResponse:
        conversation = []
        if system_prompt:
            conversation.append(PromptMessage(role="system", content=system_prompt))
        if user_prompt:
            conversation.append(PromptMessage(role="user", content=user_prompt))
        chat_request = ChatRequest(
            model_provider=provider,
            model_name=model_name,
            conversation=conversation,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return (
            providers[provider]
            .implementation(api_key=api_key)
            .generate_chat_response(chat_request)
        )
