from ..models import ModelCapabilities, ModelConfig, ModelProvider
from .base_llm import BaseLLMModel


class Mistral_Ministral3b(BaseLLMModel):
    config = ModelConfig(
        identifier="ministral-3b-latest",
        verbose_name="Ministral 3B",
        description="Considered the world’s best edge model for various tasks.",
        max_input_tokens=128000,
        euro_per_1M_input_tokens=0.04,
        euro_per_1M_output_tokens=0.04,
        capabilities=[ModelCapabilities.chat],
    )
    provider = ModelProvider.mistral


class Mistral_Ministral8b(BaseLLMModel):
    config = ModelConfig(
        identifier="ministral-8b-latest",
        verbose_name="Ministral 8B",
        description="High-performance model with a great performance/price ratio.",
        max_input_tokens=128000,
        euro_per_1M_input_tokens=0.1,
        euro_per_1M_output_tokens=0.1,
        capabilities=[ModelCapabilities.chat],
    )
    provider = ModelProvider.mistral


class Mistral_Mistral_Large(BaseLLMModel):
    config = ModelConfig(
        identifier="mistral-large-latest",
        verbose_name="Mistral Large",
        description="Top-tier reasoning model for high-complexity tasks, with the latest v2 version released in July 2024.",
        max_input_tokens=128000,
        euro_per_1M_input_tokens=2,
        euro_per_1M_output_tokens=6,
        capabilities=[ModelCapabilities.chat],
    )
    provider = ModelProvider.mistral


class Mistral_Mistral_Small(BaseLLMModel):
    config = ModelConfig(
        identifier="mistral-small-latest",
        verbose_name="Mistral Small",
        description="Enterprise-grade model for small-scale tasks, with the latest v2 version released in September 2024.",
        max_input_tokens=32000,
        euro_per_1M_input_tokens=0.2,
        euro_per_1M_output_tokens=0.6,
        capabilities=[ModelCapabilities.chat],
    )
    provider = ModelProvider.mistral


class Mistral_Mistral_Embed(BaseLLMModel):
    config = ModelConfig(
        identifier="mistral-embed",
        verbose_name="Mistral Embed",
        description="Designed for extracting semantic representations from text.",
        max_input_tokens=8000,
        euro_per_1M_input_tokens=0.1,
        euro_per_1M_output_tokens=0,
        capabilities=[ModelCapabilities.embeddings],
    )
    provider = ModelProvider.mistral


class Mistral_Mistral_Nemo(BaseLLMModel):
    config = ModelConfig(
        identifier="open-mistral-nemo",
        verbose_name="Mistral Nemo",
        description="A multilingual model released in July 2024.",
        max_input_tokens=128000,
        euro_per_1M_input_tokens=0.15,
        euro_per_1M_output_tokens=0.15,
        capabilities=[ModelCapabilities.chat],
    )
    provider = ModelProvider.mistral


class Mistral_Codestral(BaseLLMModel):
    config = ModelConfig(
        identifier="codestral-latest",
        verbose_name="Codestral",
        description="A language model optimized for coding tasks, released in May 2024.",
        max_input_tokens=32000,
        euro_per_1M_input_tokens=0.2,
        euro_per_1M_output_tokens=0.6,
        capabilities=[ModelCapabilities.chat],
    )
    provider = ModelProvider.mistral


class Mistral_Pixtral(BaseLLMModel):
    config = ModelConfig(
        identifier="pixtral-12b-2409",
        verbose_name="Pixtral",
        description="A 12B model combining image understanding with text processing, released in September 2024.",
        max_input_tokens=128000,
        euro_per_1M_input_tokens=0.15,
        euro_per_1M_output_tokens=0.15,
        capabilities=[ModelCapabilities.vision],
    )
    provider = ModelProvider.mistral
