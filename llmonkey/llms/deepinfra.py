from ..models import ModelCapabilities, ModelConfig, ModelLocation, ModelProvider
from .base_llm import BaseLLMModel

class Deepinfra_Qwen_QwQ_32B(BaseLLMModel):
    config = ModelConfig(
        identifier="Qwen/QwQ-32B-Preview",
        verbose_name="Deepinfra Qwen QwQ 32B",
        description="Deepinfra Qwen QwQ 32B is a model that uses chain-of-thought reasoning by default.",
        max_input_tokens=32_000,
        euro_per_1M_input_tokens=0.15,
        euro_per_1M_output_tokens=0.6,
        capabilities=[ModelCapabilities.chat],
        location=ModelLocation.US,
        parameters="405B",
    )
    provider = ModelProvider.deepinfra
