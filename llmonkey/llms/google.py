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


class Google_Gemini_Flash_1_5(BaseLLMModel):
    config = ModelConfig(
        identifier="gemini-1.5-flash",
        verbose_name="Google Gemini Flash 1.5 (latest stable)",
        description="Google's Gemini Flash 1.5 model",
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


class Google_Gemini_Flash_2_0_Exp(BaseLLMModel):
    config = ModelConfig(
        identifier="gemini-2.0-flash-exp",
        verbose_name="Google Gemini Flash 2.0 Exp",
        description="Google's Gemini Flash 2.0 Expermiental model",
        max_input_tokens=1_048_576,
        euro_per_1M_input_tokens=0.017,
        euro_per_1M_output_tokens=0.07,
        capabilities=[ModelCapabilities.chat, ModelCapabilities.vision],
        location=ModelLocation.EU,
        parameters="16B (?)",
    )
    provider = ModelProvider.google


class Google_Gemini_Pro_1_5(BaseLLMModel):
    config = ModelConfig(
        identifier="gemini-1.5-pro",  # latest stable, 002 as of 2024-12-12
        verbose_name="Google Gemini Pro 1.5",
        description="Google's Gemini Pro 1.5 model",
        max_input_tokens=2_097_152,
        euro_per_1M_input_tokens=0.47,
        euro_per_1M_output_tokens=1.42,
        capabilities=[ModelCapabilities.chat, ModelCapabilities.vision],
        location=ModelLocation.EU,
        parameters="120B MOE (?)",
    )
    provider = ModelProvider.google