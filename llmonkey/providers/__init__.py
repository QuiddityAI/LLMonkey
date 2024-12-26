from typing import Any

from pydantic import BaseModel, field_validator

from ..models import ModelProvider
from .base import BaseModelProvider

# from .cohere import CohereProvider
from .groq import GroqProvider
from .openai_like import (
    DeepInfraProvider,
    IonosProvider,
    MistralProvider,
    NebiusProvider,
    OpenAIProvider,
)
from .google import GoogleProvider
from .azure import AzureOpenAIProvider


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
    # "cohere": ProviderConfig(
    #     provider=ModelProvider.cohere, implementation=CohereProvider
    # ),
    "ionos": ProviderConfig(provider=ModelProvider.ionos, implementation=IonosProvider),
    "mistral": ProviderConfig(
        provider=ModelProvider.mistral, implementation=MistralProvider
    ),
    "nebius": ProviderConfig(
        provider=ModelProvider.nebius, implementation=NebiusProvider
    ),
    "google": ProviderConfig(provider=ModelProvider.google, implementation=GoogleProvider),
    "azure_openai": ProviderConfig(
        provider=ModelProvider.azure_openai, implementation=AzureOpenAIProvider
    ),
    # "azure_non_openai": ProviderConfig(
    #     provider=ModelProvider.azure_non_openai, implementation=AzureNonOpenAIProvider
    # ),
}
