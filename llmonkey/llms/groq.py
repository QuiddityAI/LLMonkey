from ..models import ModelCapabilities, ModelConfig, ModelProvider
from .base_llm import BaseLLMModel


class GroqLlama3_3_70bVersatile(BaseLLMModel):
    config = ModelConfig(
        identifier="llama-3.3-70b-versatile",
        verbose_name="Llama 3.3 70B Versatile (Groq)",
        description="Llama 3.3 70B Versatile is a versatile model that can be used for various tasks.",
        max_input_tokens=128 * 1024,
        euro_per_1M_input_tokens=0.59,  # USD
        euro_per_1M_output_tokens=0.79,
        capabilities=[ModelCapabilities.chat],
    )
    provider = ModelProvider.groq


class GroqLlama3_1_70bVersatile(BaseLLMModel):
    config = ModelConfig(
        identifier="llama-3.1-70b-versatile",
        verbose_name="Llama 3.1 70B Versatile (Groq, deprecated)",
        description="Llama 3.1 70B Versatile is a versatile model that can be used for various tasks.",
        max_input_tokens=128 * 1024,
        euro_per_1M_input_tokens=0.59,
        euro_per_1M_output_tokens=0.79,
        capabilities=[ModelCapabilities.chat],
    )
    provider = ModelProvider.groq


class GroqLlama3_1_8bInstant(BaseLLMModel):
    config = ModelConfig(
        identifier="llama-3.1-8b-instant",
        verbose_name="Llama 3.1 8B Instant (Groq)",
        description="Llama 3.1 8B Instant is an optimized for speed versoin of the Llama 3.1 8B model.",
        max_input_tokens=128 * 1024,
        euro_per_1M_input_tokens=0.05,
        euro_per_1M_output_tokens=0.08,
        capabilities=[ModelCapabilities.chat],
    )
    provider = ModelProvider.groq


class GroqLlama3_70BToolUsePreview(BaseLLMModel):
    config = ModelConfig(
        identifier="llama3-groq-70b-8192-tool-use-preview",
        verbose_name="Llama 3 Groq 70B Tool Use Preview (Groq)",
        description="Llama 3 Groq 70B Tool Use Preview is a model that is optimized for tool use.",
        max_input_tokens=8192,
        euro_per_1M_input_tokens=0.89,
        euro_per_1M_output_tokens=0.89,
        capabilities=[ModelCapabilities.chat],
    )
    provider = ModelProvider.groq


class GroqLlama3_8BToolUsePreview(BaseLLMModel):
    config = ModelConfig(
        identifier="llama3-groq-8b-8192-tool-use-preview",
        verbose_name="Llama 3 Groq 8B Tool Use Preview (Groq)",
        description="Llama 3 Groq 8B Tool Use Preview is a model that is optimized for tool use.",
        max_input_tokens=8192,
        euro_per_1M_input_tokens=0.19,
        euro_per_1M_output_tokens=0.19,
        capabilities=[ModelCapabilities.chat],
    )
    provider = ModelProvider.groq


class GroqLlamaGuard38B(BaseLLMModel):
    config = ModelConfig(
        identifier="llama-guard-3-8b",
        verbose_name="Llama Guard 3 8B (Groq)",
        description="Llama Guard 3 8B is a model that is optimized for guarding.",
        max_input_tokens=8192,
        euro_per_1M_input_tokens=0.20,
        euro_per_1M_output_tokens=0.20,
        capabilities=[ModelCapabilities.chat],
    )
    provider = ModelProvider.groq


class GroqLlama3_2_1BPreview(BaseLLMModel):
    config = ModelConfig(
        identifier="llama-3.2-1b-preview",
        verbose_name="Llama 3.2 1B (Preview) (Groq)",
        description="Llama 3.2 1B (Preview) very small, cheap and quick model.",
        max_input_tokens=8192,
        euro_per_1M_input_tokens=0.04,
        euro_per_1M_output_tokens=0.04,
        capabilities=[ModelCapabilities.chat],
    )
    provider = ModelProvider.groq


class GroqLlama3_2_3BPreview(BaseLLMModel):
    config = ModelConfig(
        identifier="llama-3.2-3b-preview",
        verbose_name="Llama 3.2 3B (Preview) (Groq)",
        description="Llama 3.2 3B (Preview) is a small, cheap and quick model.",
        max_input_tokens=8192,
        euro_per_1M_input_tokens=0.06,
        euro_per_1M_output_tokens=0.06,
        capabilities=[ModelCapabilities.chat],
    )
    provider = ModelProvider.groq


class GroqLlama3_2_11BVisionPreview(BaseLLMModel):
    config = ModelConfig(
        identifier="llama-3.2-11b-vision-preview",
        verbose_name="Llama 3.2 11B Vision (Groq)",
        description="Llama 3.2 11B Vision is a multi-modal model that is optimized for vision tasks.",
        max_input_tokens=8192,
        euro_per_1M_input_tokens=0.04,
        euro_per_1M_output_tokens=0.04,
        capabilities=[ModelCapabilities.chat, ModelCapabilities.vision],
    )
    provider = ModelProvider.groq


class GroqLlama3_2_90BPreview(BaseLLMModel):
    config = ModelConfig(
        identifier="llama-3.2-90b-vision-preview",
        verbose_name="Llama 3.2 90B (Groq)",
        description="Llama 3.2 90B is a multi-modal model that is optimized for vision tasks.",
        max_input_tokens=8192,
        euro_per_1M_input_tokens=0.04,
        euro_per_1M_output_tokens=0.04,
        capabilities=[ModelCapabilities.chat, ModelCapabilities.vision],
    )
    provider = ModelProvider.groq


class GroqLlama3_70B8k(BaseLLMModel):
    config = ModelConfig(
        identifier="llama3-70b-8192",
        verbose_name="Llama 3 70B (Groq)",
        description="Llama 3 70B is a versatile model that can be used for various tasks.",
        max_input_tokens=8192,
        euro_per_1M_input_tokens=0.59,
        euro_per_1M_output_tokens=0.79,
        capabilities=[ModelCapabilities.chat],
    )
    provider = ModelProvider.groq


class GroqLlama3_8B8k(BaseLLMModel):
    config = ModelConfig(
        identifier="llama3-8b-8192",
        verbose_name="Llama 3 8B (Groq)",
        description="Llama 3 8B is a versatile model that can be used for various tasks.",
        max_input_tokens=8192,
        euro_per_1M_input_tokens=0.05,
        euro_per_1M_output_tokens=0.08,
        capabilities=[ModelCapabilities.chat],
    )
    provider = ModelProvider.groq


class GroqMixtral8x7B(BaseLLMModel):
    config = ModelConfig(
        identifier="mixtral-8x7b-32768",
        verbose_name="Mixtral 8x7B (Groq)",
        description="Mixtral 8x7B is a mixture of experts model.",
        max_input_tokens=32768,
        euro_per_1M_input_tokens=0.24,
        euro_per_1M_output_tokens=0.24,
        capabilities=[ModelCapabilities.chat],
    )
    provider = ModelProvider.groq


class GroqGemma7B8kInstruct(BaseLLMModel):
    config = ModelConfig(
        identifier="gemma-7b-it",
        verbose_name="Gemma 7B 8k Instruct (Groq)",
        description="Gemma 7B 8k Instruct is a model that is optimized for instruction following.",
        max_input_tokens=8192,
        euro_per_1M_input_tokens=0.07,
        euro_per_1M_output_tokens=0.07,
        capabilities=[ModelCapabilities.chat],
    )
    provider = ModelProvider.groq


class GroqGemma2_9B8k(BaseLLMModel):
    config = ModelConfig(
        identifier="gemma2-9b-it",
        verbose_name="Gemma 2 9B 8k (Groq)",
        description="Gemma 2 9B 8k is a model that is optimized for instruction following.",
        max_input_tokens=8192,
        euro_per_1M_input_tokens=0.20,
        euro_per_1M_output_tokens=0.20,
        capabilities=[ModelCapabilities.chat],
    )
    provider = ModelProvider.groq
