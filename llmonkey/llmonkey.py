from typing import Any, Dict, List, Type

from pydantic import BaseModel, field_validator

from .models import (
    ChatRequest,
    ChatResponse,
    ModelProvider,
    PromptMessage,
    RerankRequest,
    RerankResponse,
)
from .providers.base import BaseModelProvider
from .providers.cohere import CohereProvider
from .providers.groq import GroqProvider
from .providers.openai_like import DeepInfraProvider, OpenAIProvider


class ProviderConfig(BaseModel):
    provider: ModelProvider
    implementation: Any

    @field_validator("implementation")
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
    "cohere": ProviderConfig(
        provider=ModelProvider.cohere, implementation=CohereProvider
    ),
}


class LLMonkey(object):
    providers = providers

    def __init__(self):
        pass

    def generate_structured_response(
        self,
        provider: str,
        model_name: str,
        data_model: Type[BaseModel],
        user_prompt: str = "",
        system_prompt: str = "",
        temperature=0.7,
        max_tokens=150,
        api_key: str = "",
    ):
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
            .generate_structured_response(chat_request, data_model=data_model)
        )

    def generate_prompt_response(
        self,
        provider: str,
        model_name: str,
        user_prompt: str = "",
        system_prompt: str = "",
        temperature=0.7,
        max_tokens=150,
        api_key: str = "",
    ) -> ChatResponse:
        """
        Generate a response to a single prompt.

        Args:
        provider: The name of the LLM provider to use.
        model_name: The name of the model to use.
        user_prompt: The user's prompt. Defaults to an empty string.
        system_prompt: The system's prompt. Defaults to an empty string.
        temperature: The temperature of the model. Defaults to 0.7.
        max_tokens: The maximum number of tokens to generate. Defaults to 150.
        api_key: The API key to use for the provider. Defaults to an empty string.

        Returns:
        A ChatResponse containing the response.
        """
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

    def generate_chat_response(
        self,
        provider: str,
        model_name: str,
        conversation: List[PromptMessage] = [],
        temperature=0.7,
        max_tokens=150,
        api_key: str = "",
    ) -> ChatResponse:
        """
        Generate a response to a multi-turn chat prompt.

        Args:
        provider: The name of the LLM provider to use.
        model_name: The name of the model to use.
        conversation: A list of previous messages in the conversation,
            where each message is a PromptMessage.
        temperature: The temperature of the model. Defaults to 0.7.
        max_tokens: The maximum number of tokens to generate. Defaults to 150.
        api_key: The API key for the provider. Defaults to None.

        Returns:
        A ChatResponse with the generated response and the model used.
        """
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

    def rerank(
        self,
        provider: str,
        model_name: str,
        query: str = "",
        documents: List[str] | Dict[str, str] = [],
        top_n: int = 3,
        rank_fields: List[str] = [],
        api_key: str = "",
    ) -> RerankResponse:
        """
        Rerank a list of documents using a given model.

        Args:
        provider: The name of the LLM provider to use.
        model_name: The name of the model to use.
        query: The query to use for the reranking. Defaults to an empty string.
        documents: A list of documents to rerank. Defaults to an empty list.
        top_n: The number of most relevant documents to return. Defaults to 3.
        rank_fields: The fields to rank documents on. Defaults to an empty list.
        api_key: The API key for the provider. Defaults to None.

        Returns:
        A RerankResponse with the reranked documents and the model used.
        """
        rerank_request = RerankRequest(
            model_provider=provider,
            model_name=model_name,
            query=query,
            documents=documents,
            top_n=top_n,
            rank_fields=rank_fields,
        )
        return (
            providers[provider].implementation(api_key=api_key).rerank(rerank_request)
        )
