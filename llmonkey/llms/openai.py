from ..models import ModelCapabilities, ModelConfig, ModelProvider
from .base_llm import BaseLLMModel


class OpenAI_GPT4o(BaseLLMModel):
    config = ModelConfig(
        identifier="gpt-4o",
        verbose_name="GPT-4o",
        description="Advanced multimodal model with vision capabilities, 128K context length, faster and cheaper than GPT-4 Turbo. Knowledge cutoff is October 2023.",
        max_input_tokens=128000,
        euro_per_1M_input_tokens=2.50,
        euro_per_1M_output_tokens=10.00,
        capabilities=[ModelCapabilities.chat, ModelCapabilities.vision],
    )
    provider = ModelProvider.openai


class OpenAI_GPT4o_Mini(BaseLLMModel):
    config = ModelConfig(
        identifier="gpt-4o-mini",
        verbose_name="GPT-4o Mini",
        description="Cost-efficient small model, smarter and cheaper than GPT-3.5 Turbo, with vision capabilities. Also has a 128K context and an October 2023 knowledge cutoff.",
        max_input_tokens=128000,
        euro_per_1M_input_tokens=0.150,
        euro_per_1M_output_tokens=0.600,
        capabilities=[ModelCapabilities.chat, ModelCapabilities.vision],
    )
    provider = ModelProvider.openai


class OpenAI_o1(BaseLLMModel):
    config = ModelConfig(
        identifier="o1-preview",
        verbose_name="o1",
        description="New reasoning model designed for complex tasks with 128K context and an October 2023 knowledge cutoff.",
        max_input_tokens=128000,
        euro_per_1M_input_tokens=15.00,
        euro_per_1M_output_tokens=60.00,
        capabilities=[ModelCapabilities.chat],
    )
    provider = ModelProvider.openai


class OpenAI_o1_Mini(BaseLLMModel):
    config = ModelConfig(
        identifier="o1-mini",
        verbose_name="o1 Mini",
        description="Fast, cost-effective reasoning model aimed at coding, math, and science. Features 128K context and October 2023 knowledge cutoff.",
        max_input_tokens=128000,
        euro_per_1M_input_tokens=3.00,
        euro_per_1M_output_tokens=12.00,
        capabilities=[ModelCapabilities.chat],
    )
    provider = ModelProvider.openai


class OpenAI_TextEmbedding_3_small(BaseLLMModel):
    config = ModelConfig(
        identifier="text-embedding-3-small",
        verbose_name="Text Embedding 3 Small",
        description="Small-scale text embedding model for various NLP tasks.",
        max_input_tokens=8191,
        euro_per_1M_input_tokens=0.020,
        euro_per_1M_output_tokens=0.0,
        capabilities=[ModelCapabilities.embeddings],
    )
    provider = ModelProvider.openai


class OpenAI_TextEmbedding_3_large(BaseLLMModel):
    config = ModelConfig(
        identifier="text-embedding-3-large",
        verbose_name="Text Embedding 3 Large",
        description="Large-scale text embedding model for various NLP tasks.",
        max_input_tokens=8191,
        euro_per_1M_input_tokens=0.130,
        euro_per_1M_output_tokens=0.0,
        capabilities=[ModelCapabilities.embeddings],
    )
    provider = ModelProvider.openai


class OpenAI_Ada_v2(BaseLLMModel):
    config = ModelConfig(
        identifier="ada-v2",
        verbose_name="Ada v2",
        description="Older OpenAI embedding model.",
        max_input_tokens=8191,
        euro_per_1M_input_tokens=0.100,
        euro_per_1M_output_tokens=0.0,
        capabilities=[ModelCapabilities.embeddings],
    )
    provider = ModelProvider.openai
