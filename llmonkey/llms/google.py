from ..models import ModelCapabilities, ModelConfig, ModelLocation, ModelProvider
from .base_llm import BaseLLMModel


class Google_Gemini_Flash_1_5_v1(BaseLLMModel):
    config = ModelConfig(
        identifier="gemini-1.5-flash-001",
        verbose_name="Google Gemini Flash 1.5 v1",
        description="Google's Gemini Flash 1.5 model, version 1",
        max_input_tokens=1_048_576,
        euro_per_1M_input_tokens=0.017,
        euro_per_1M_output_tokens=0.07,
        capabilities=[ModelCapabilities.chat, ModelCapabilities.vision],
        location=ModelLocation.EU,
        parameters="~16B",
    )
    provider = ModelProvider.google


class Google_Gemini_Flash_1_5_v2(BaseLLMModel):
    config = ModelConfig(
        identifier="gemini-1.5-flash-002",
        verbose_name="Google Gemini Flash 1.5 v2",
        description="Google's Gemini Flash 1.5 model, version 2",
        max_input_tokens=1_048_576,
        euro_per_1M_input_tokens=0.017,
        euro_per_1M_output_tokens=0.07,
        capabilities=[ModelCapabilities.chat, ModelCapabilities.vision],
        location=ModelLocation.EU,
        parameters="~16B",
    )
    provider = ModelProvider.google


class Google_Gemini_Flash_1_5_latest(BaseLLMModel):
    config = ModelConfig(
        identifier="gemini-1.5-flash-latest",
        verbose_name="Google Gemini Flash 1.5 (latest)",
        description="Google's Gemini Flash 1.5 model (latest)",
        max_input_tokens=1_048_576,
        euro_per_1M_input_tokens=0.017,
        euro_per_1M_output_tokens=0.07,
        capabilities=[ModelCapabilities.chat, ModelCapabilities.vision],
        location=ModelLocation.EU,
        parameters="~16B",
    )
    provider = ModelProvider.google


class Google_Gemini_Flash_1_5_8B(BaseLLMModel):
    config = ModelConfig(
        identifier="gemini-1.5-flash-8b",
        verbose_name="Google Gemini Flash 1.5 8B",
        description="Google's Gemini Flash 1.5 model, 8B version",
        max_input_tokens=1_048_576,
        euro_per_1M_input_tokens=0.017,
        euro_per_1M_output_tokens=0.07,
        capabilities=[ModelCapabilities.chat, ModelCapabilities.vision],
        location=ModelLocation.EU,
        parameters="8B",
    )
    provider = ModelProvider.google