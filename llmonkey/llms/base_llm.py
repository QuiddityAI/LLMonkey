import re
from abc import ABCMeta, abstractmethod
from typing import Dict, List, Self, Type

from pydantic import BaseModel

from ..models import (
    ChatRequest,
    ChatResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    ModelConfig,
    PromptMessage,
    RerankRequest,
    RerankResponse,
    TokenUsage,
)
from ..providers import providers
from ..providers.base import BaseModelProvider


def count_tokens_rough(text):
    # Split the text based on whitespace and common code symbols
    tokens = re.split(r"\s+|[()\[\]{}.,:;+=*/\\\"\'<>-]", text)
    # Filter out any empty strings resulting from the split
    tokens = [token for token in tokens if token]
    return len(tokens)


class BaseLLMModel(metaclass=ABCMeta):
    def __init__(self, api_key: str = ""):
        self.provider_instance: BaseModelProvider = providers[
            self.provider
        ].implementation(api_key=api_key)

    @classmethod
    def _get_subclasses(cls):
        return cls.__subclasses__()

    @classmethod
    def _build_models_dict(cls):
        return {model.__name__: model for model in cls._get_subclasses()}

    @classmethod
    def available_models(cls) -> Dict[str, type[Self]]:
        """
        Get a dictionary of available model classes.

        Returns:
        A dictionary of model identifiers to their classes.
        """
        if getattr(cls, "_models", None) is None:
            cls._models = cls._build_models_dict()
        return cls._models

    @classmethod
    def available_model_configs(cls) -> List[str]:
        """
        Get a dictionary of available model configs,
        combining the config of the model and of the provider.
        """
        if getattr(cls, "_models", None) is None:
            cls._models = cls._build_models_dict()
        result = {}
        for model_name, model_class in cls._models.items():
            # this will serialize all the fields of the model config
            # including enums and other nested models
            result[model_name] = model_class.config.model_dump(mode="json")
            # add the provider to the result
            result[model_name]["provider"] = model_class.provider.value
        return result

    @classmethod
    def load(self, model_class_name: str, **kwargs) -> Self:
        """
        Load a model by its class name. Use `available_models` to get the list of available models.

        Args:
        model_class_name: The identifier of the model to load.
        """
        if getattr(self, "_models", None) is None:
            self._models = self._build_models_dict()
        if model_class_name not in self._models:
            raise ValueError(f"Model {model_class_name} not found.")
        return self._models[model_class_name](**kwargs)

    @property
    @abstractmethod
    def config(self) -> ModelConfig:
        pass

    @property
    @abstractmethod
    def provider(self) -> BaseModelProvider:
        pass

    def generate_structured_response(
        self,
        data_model: Type[BaseModel],
        user_prompt: str = "",
        system_prompt: str = "",
        image=None,
        temperature=0.7,
        max_tokens=None,
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
        provider = self.provider
        model_name = self.config.identifier
        conversation = []
        if system_prompt:
            conversation.append(PromptMessage(role="system", content=system_prompt))
        if user_prompt:
            conversation.append(
                PromptMessage(role="user", content=user_prompt, image=image)
            )
        self._validate_input(conversation)
        chat_request = ChatRequest(
            model_provider=provider,
            model_name=model_name,
            conversation=conversation,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        data_instance, resp = self.provider_instance.generate_structured_response(
            chat_request, data_model=data_model
        )
        resp.token_usage.total_cost = self._calculate_cost(resp.token_usage)
        return data_instance, resp

    def _calculate_cost(self, token_usage: TokenUsage) -> float:
        """
        Calculate the cost of the request based on the token usage.

        Args:
        token_usage: The token usage details.

        Returns:
        The total cost of the request.
        """
        cost = 0.0
        if token_usage.prompt_tokens and token_usage.completion_tokens:
            cost += (
                token_usage.prompt_tokens / 1e6 * self.config.euro_per_1M_input_tokens
            )
            cost += (
                token_usage.completion_tokens
                / 1e6
                * self.config.euro_per_1M_output_tokens
            )
            return cost
        if token_usage.total_tokens:
            av_price = (
                self.config.euro_per_1M_input_tokens
                + self.config.euro_per_1M_output_tokens
            ) / 2
            cost += token_usage.total_tokens / 1e6 * av_price
        return None

    def _validate_input(self, conversation: List[PromptMessage]) -> None:
        total_tokens = sum([count_tokens_rough(msg.content) for msg in conversation])
        if total_tokens > self.config.max_input_tokens:
            raise ValueError(
                f"Input tokens exceed the maximum limit of {self.config.max_input_tokens}."
            )

    def generate_prompt_response(
        self,
        user_prompt: str = "",
        system_prompt: str = "",
        image=None,
        temperature=0.7,
        max_tokens=None,
        api_key: str = "",
    ) -> ChatResponse:
        """
        Generate a response to a single prompt.

        Args:
        user_prompt: The user's prompt. Defaults to an empty string.
        system_prompt: The system's prompt. Defaults to an empty string.
        temperature: The temperature of the model. Defaults to 0.7.
        max_tokens: The maximum number of tokens to generate. Defaults to 150.
        api_key: The API key to use for the provider. Defaults to an empty string.

        Returns:
        A ChatResponse containing the response.
        """
        provider = self.provider
        model_name = self.config.identifier
        conversation = []
        if system_prompt:
            conversation.append(PromptMessage(role="system", content=system_prompt))
        if user_prompt:
            conversation.append(
                PromptMessage(role="user", content=user_prompt, image=image)
            )
        self._validate_input(conversation)
        chat_request = ChatRequest(
            model_provider=provider,
            model_name=model_name,
            conversation=conversation,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        resp = self.provider_instance.generate_chat_response(chat_request)
        resp.token_usage.total_cost = self._calculate_cost(resp.token_usage)
        return resp

    def generate_chat_response(
        self,
        conversation: List[PromptMessage] = [],
        temperature=0.7,
        max_tokens=None,
        api_key: str = "",
    ) -> ChatResponse:
        """
        Generate a response to a multi-turn chat prompt.

        Args:
        conversation: A list of previous messages in the conversation,
            where each message is a PromptMessage.
        temperature: The temperature of the model. Defaults to 0.7.
        max_tokens: The maximum number of tokens to generate. Defaults to 150.
        api_key: The API key for the provider. Defaults to None.

        Returns:
        A ChatResponse with the generated response and the model used.
        """
        provider = self.provider
        model_name = self.config.identifier
        self._validate_input(conversation)
        chat_request = ChatRequest(
            model_provider=provider,
            model_name=model_name,
            conversation=conversation,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        resp = self.provider_instance.generate_chat_response(chat_request)
        resp.token_usage.total_cost = self._calculate_cost(resp.token_usage)
        return resp

    def rerank(
        self,
        query: str = "",
        documents: List[str] | Dict[str, str] = [],
        top_n: int = 3,
        rank_fields: List[str] = [],
        api_key: str = "",
    ) -> RerankResponse:
        """
        Rerank a list of documents using a given model.

        Args:
        query: The query to use for the reranking. Defaults to an empty string.
        documents: A list of documents to rerank. Defaults to an empty list.
        top_n: The number of most relevant documents to return. Defaults to 3.
        rank_fields: The fields to rank documents on. Defaults to an empty list.
        api_key: The API key for the provider. Defaults to None.

        Returns:
        A RerankResponse with the reranked documents and the model used.
        """
        provider = self.provider
        model_name = self.config.identifier
        rerank_request = RerankRequest(
            model_provider=provider,
            model_name=model_name,
            query=query,
            documents=documents,
            top_n=top_n,
            rank_fields=rank_fields,
        )
        resp = self.provider_instance.rerank(rerank_request)
        resp.token_usage.total_cost = self._calculate_cost(resp.token_usage)
        return resp

    def generate_embeddings(
        self,
        text: str = "",
    ) -> EmbeddingResponse:
        """
        Generate embeddings for a given text.

        Args:
        text: The text to generate embeddings for. Defaults to an empty string.

        Returns:
        An EmbeddingResponse with the text embeddings and the model used.
        """
        provider = self.provider
        model_name = self.config.identifier
        tokens = count_tokens_rough(text)
        if tokens > self.config.max_input_tokens:
            raise ValueError(
                f"Input tokens exceed the maximum limit of {self.config.max_input_tokens}."
            )
        embedding_request = EmbeddingRequest(
            model_provider=provider,
            model_name=model_name,
            text=text,
        )
        resp = self.provider_instance.generate_embeddings(embedding_request)
        resp.token_usage.total_cost = self._calculate_cost(resp.token_usage)
        return resp
