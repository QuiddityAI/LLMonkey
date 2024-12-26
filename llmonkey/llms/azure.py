from ..models import ModelCapabilities, ModelConfig, ModelLocation, ModelProvider
from .base_llm import BaseLLMModel


class Azure_GPT4o(BaseLLMModel):
    config = ModelConfig(
        identifier="gpt-4o",
        verbose_name="Azure GPT-4o",
        description="Advanced multimodal model with vision capabilities, 128K context length, faster and cheaper than GPT-4 Turbo. Knowledge cutoff is October 2023.",
        max_input_tokens=128000,
        euro_per_1M_input_tokens=2.50,
        euro_per_1M_output_tokens=10.00,
        capabilities=[ModelCapabilities.chat, ModelCapabilities.vision],
        location=ModelLocation.EU,
        parameters="1800B",  # Add the number of parameters here
    )
    provider = ModelProvider.azure_openai
