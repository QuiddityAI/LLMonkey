from ..models import ModelCapabilities, ModelConfig, ModelLocation, ModelProvider
from .base_llm import BaseLLMModel

class Ionos_Llama_3_1_405B(BaseLLMModel):
    config = ModelConfig(
        identifier="meta-llama/Meta-Llama-3.1-405B-Instruct-FP8",
        verbose_name="Nebius Llama 3.1 405B",
        description="Nebius Llama 3.1 405B (FP8) is a large-scale model with 128K context and a 405B parameter size.",
        max_input_tokens=131072,
        euro_per_1M_input_tokens=1.50,
        euro_per_1M_output_tokens=1.75,
        capabilities=[ModelCapabilities.chat],
        location=ModelLocation.EU,
        parameters="405B",
    )
    provider = ModelProvider.ionos

