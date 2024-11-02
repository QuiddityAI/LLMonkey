from ..models import ModelCapabilities, ModelConfig, ModelLocation, ModelProvider
from .base_llm import BaseLLMModel


class Nebius_Llama_3_1_70B_fast(BaseLLMModel):
    config = ModelConfig(
        identifier="meta-llama/Meta-Llama-3.1-70B-Instruct-fast",
        verbose_name="Nebius Llama 3.1 70B",
        description="Nebius Llama 3.1 70B is a large-scale model with 128K context and a 70B parameter size.",
        max_input_tokens=131072,
        euro_per_1M_input_tokens=0.25,
        euro_per_1M_output_tokens=0.75,
        capabilities=[ModelCapabilities.chat],
        location=ModelLocation.EU,
        parameters="70B",
    )
    provider = ModelProvider.nebius


class Nebius_Llama_3_1_70B_cheap(BaseLLMModel):
    config = ModelConfig(
        identifier="meta-llama/Meta-Llama-3.1-70B-Instruct",
        verbose_name="Nebius Llama 3.1 70B",
        description="Nebius Llama 3.1 70B is a large-scale model with 128K context and a 70B parameter size.",
        max_input_tokens=131072,
        euro_per_1M_input_tokens=0.13,
        euro_per_1M_output_tokens=0.4,
        capabilities=[ModelCapabilities.chat],
        location=ModelLocation.EU,
        parameters="70B",
    )
    provider = ModelProvider.nebius


class Nebius_Llama_3_1_8B_fast(BaseLLMModel):
    config = ModelConfig(
        identifier="meta-llama/Meta-Llama-3.1-8B-Instruct-fast",
        verbose_name="Nebius Llama 3.1 8B",
        description="Nebius Llama 3.1 8B is a medium-scale model with 128K context and an 8B parameter size.",
        max_input_tokens=131072,
        euro_per_1M_input_tokens=0.03,
        euro_per_1M_output_tokens=0.09,
        capabilities=[ModelCapabilities.chat],
        location=ModelLocation.EU,
        parameters="8B",
    )
    provider = ModelProvider.nebius


class Nebius_Llama_3_1_8B_cheap(BaseLLMModel):
    config = ModelConfig(
        identifier="meta-llama/Meta-Llama-3.1-8B-Instruct",
        verbose_name="Nebius Llama 3.1 8B",
        description="Nebius Llama 3.1 8B is a medium-scale model with 128K context and an 8B parameter size.",
        max_input_tokens=131072,
        euro_per_1M_input_tokens=0.02,
        euro_per_1M_output_tokens=0.06,
        capabilities=[ModelCapabilities.chat],
        location=ModelLocation.EU,
        parameters="8B",
    )
    provider = ModelProvider.nebius


class Nebius_Llama_3_1_405B_cheap(BaseLLMModel):
    config = ModelConfig(
        identifier="meta-llama/Meta-Llama-3.1-405B-Instruct",
        verbose_name="Nebius Llama 3.1 405B",
        description="Nebius Llama 3.1 405B is a large-scale model with 128K context and a 405B parameter size.",
        max_input_tokens=131072,
        euro_per_1M_input_tokens=1.0,
        euro_per_1M_output_tokens=3.0,
        capabilities=[ModelCapabilities.chat],
        location=ModelLocation.EU,
        parameters="405B",
    )
    provider = ModelProvider.nebius


class Nebius_Mistral_Nemo_Instruct_2407_fast(BaseLLMModel):
    config = ModelConfig(
        identifier="mistralai/Mistral-Nemo-Instruct-2407-fast",
        verbose_name="Nebius Mistral Nemo Instruct 2407",
        description="Nebius Mistral Nemo Instruct 2407 is a medium-scale model with 128K context and a 12.2B parameter size.",
        max_input_tokens=128000,
        euro_per_1M_input_tokens=0.08,
        euro_per_1M_output_tokens=0.24,
        capabilities=[ModelCapabilities.chat],
        location=ModelLocation.EU,
        parameters="12.2B",
    )
    provider = ModelProvider.nebius


class Nebius_Mistral_Nemo_Instruct_2407_cheap(BaseLLMModel):
    config = ModelConfig(
        identifier="mistralai/Mistral-Nemo-Instruct-2407",
        verbose_name="Nebius Mistral Nemo Instruct 2407",
        description="Nebius Mistral Nemo Instruct 2407 is a medium-scale model with 128K context and a 12.2B parameter size.",
        max_input_tokens=128000,
        euro_per_1M_input_tokens=0.04,
        euro_per_1M_output_tokens=0.12,
        capabilities=[ModelCapabilities.chat],
        location=ModelLocation.EU,
        parameters="12.2B",
    )
    provider = ModelProvider.nebius


class Nebius_Mixtral_8x7B_Instruct_v0_1_fast(BaseLLMModel):
    config = ModelConfig(
        identifier="mistralai/Mixtral-8x7B-Instruct-v0.1-fast",
        verbose_name="Nebius Mixtral 8x7B",
        description="Nebius Mixtral 8x7B is a medium-scale model with 33K context and a 46.7B parameter size.",
        max_input_tokens=32768,
        euro_per_1M_input_tokens=0.15,
        euro_per_1M_output_tokens=0.45,
        capabilities=[ModelCapabilities.chat],
        location=ModelLocation.EU,
        parameters="46.7B",
    )
    provider = ModelProvider.nebius


class Nebius_Mixtral_8x7B_Instruct_v0_1_cheap(BaseLLMModel):
    config = ModelConfig(
        identifier="mistralai/Mixtral-8x7B-Instruct-v0.1",
        verbose_name="Nebius Mixtral 8x7B",
        description="Nebius Mixtral 8x7B is a medium-scale model with 33K context and a 46.7B parameter size.",
        max_input_tokens=32768,
        euro_per_1M_input_tokens=0.08,
        euro_per_1M_output_tokens=0.24,
        capabilities=[ModelCapabilities.chat],
        location=ModelLocation.EU,
        parameters="46.7B",
    )
    provider = ModelProvider.nebius


class Nebius_Mixtral_8x22B_Instruct_v0_1_fast(BaseLLMModel):
    config = ModelConfig(
        identifier="mistralai/Mixtral-8x22B-Instruct-v0.1-fast",
        verbose_name="Nebius Mixtral 8x22B",
        description="Nebius Mixtral 8x22B is a medium-scale model with 65K context and a 141B parameter size.",
        max_input_tokens=65536,
        euro_per_1M_input_tokens=0.7,
        euro_per_1M_output_tokens=2.1,
        capabilities=[ModelCapabilities.chat],
        location=ModelLocation.EU,
        parameters="141B",
    )
    provider = ModelProvider.nebius


class Nebius_Mixtral_8x22B_Instruct_v0_1_cheap(BaseLLMModel):
    config = ModelConfig(
        identifier="mistralai/Mixtral-8x22B-Instruct-v0.1",
        verbose_name="Nebius Mixtral 8x22B",
        description="Nebius Mixtral 8x22B is a medium-scale model with 65K context and a 141B parameter size.",
        max_input_tokens=65536,
        euro_per_1M_input_tokens=0.4,
        euro_per_1M_output_tokens=1.2,
        capabilities=[ModelCapabilities.chat],
        location=ModelLocation.EU,
        parameters="141B",
    )
    provider = ModelProvider.nebius


class Nebius_Qwen2_5_Coder_7B_fast(BaseLLMModel):
    config = ModelConfig(
        identifier="Qwen/Qwen2.5-Coder-7B-fast",
        verbose_name="Nebius Qwen2.5 Coder 7B",
        description="Nebius Qwen2.5 Coder 7B is a medium-scale model with 32K context and a 7B parameter size.",
        max_input_tokens=32768,
        euro_per_1M_input_tokens=0.03,
        euro_per_1M_output_tokens=0.09,
        capabilities=[ModelCapabilities.chat],
        location=ModelLocation.EU,
        parameters="7B",
    )
    provider = ModelProvider.nebius


class Nebius_Qwen2_5_Coder_7B_cheap(BaseLLMModel):
    config = ModelConfig(
        identifier="Qwen/Qwen2.5-Coder-7B",
        verbose_name="Nebius Qwen2.5 Coder 7B",
        description="Nebius Qwen2.5 Coder 7B is a medium-scale model with 32K context and a 7B parameter size.",
        max_input_tokens=32768,
        euro_per_1M_input_tokens=0.01,
        euro_per_1M_output_tokens=0.03,
        capabilities=[ModelCapabilities.chat],
        location=ModelLocation.EU,
        parameters="7B",
    )
    provider = ModelProvider.nebius


class Nebius_Qwen2_5_Coder_7B_Instruct_fast(BaseLLMModel):
    config = ModelConfig(
        identifier="Qwen/Qwen2.5-Coder-7B-Instruct-fast",
        verbose_name="Nebius Qwen2.5 Coder 7B Instruct",
        description="Nebius Qwen2.5 Coder 7B Instruct is a medium-scale model with 32K context and a 7B parameter size.",
        max_input_tokens=32768,
        euro_per_1M_input_tokens=0.03,
        euro_per_1M_output_tokens=0.09,
        capabilities=[ModelCapabilities.chat],
        location=ModelLocation.EU,
        parameters="7B",
    )
    provider = ModelProvider.nebius


class Nebius_Qwen2_5_Coder_7B_Instruct_cheap(BaseLLMModel):
    config = ModelConfig(
        identifier="Qwen/Qwen2.5-Coder-7B-Instruct",
        verbose_name="Nebius Qwen2.5 Coder 7B Instruct",
        description="Nebius Qwen2.5 Coder 7B Instruct is a medium-scale model with 32K context and a 7B parameter size.",
        max_input_tokens=32768,
        euro_per_1M_input_tokens=0.01,
        euro_per_1M_output_tokens=0.03,
        capabilities=[ModelCapabilities.chat],
        location=ModelLocation.EU,
        parameters="7B",
    )
    provider = ModelProvider.nebius


class Nebius_DeepSeek_Coder_V2_Lite_Instruct_fast(BaseLLMModel):
    config = ModelConfig(
        identifier="deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct-fast",
        verbose_name="Nebius DeepSeek Coder V2 Lite Instruct",
        description="Nebius DeepSeek Coder V2 Lite Instruct is a medium-scale model with 128K context and a 15.7B parameter size.",
        max_input_tokens=128000,
        euro_per_1M_input_tokens=0.08,
        euro_per_1M_output_tokens=0.24,
        capabilities=[ModelCapabilities.chat],
        location=ModelLocation.EU,
        parameters="15.7B",
    )
    provider = ModelProvider.nebius


class Nebius_DeepSeek_Coder_V2_Lite_Instruct_cheap(BaseLLMModel):
    config = ModelConfig(
        identifier="deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
        verbose_name="Nebius DeepSeek Coder V2 Lite Instruct",
        description="Nebius DeepSeek Coder V2 Lite Instruct is a medium-scale model with 128K context and a 15.7B parameter size.",
        max_input_tokens=128000,
        euro_per_1M_input_tokens=0.04,
        euro_per_1M_output_tokens=0.12,
        capabilities=[ModelCapabilities.chat],
        location=ModelLocation.EU,
        parameters="15.7B",
    )
    provider = ModelProvider.nebius


class Nebius_Phi_3_mini_4k_instruct_fast(BaseLLMModel):
    config = ModelConfig(
        identifier="microsoft/Phi-3-mini-4k-instruct-fast",
        verbose_name="Nebius Phi 3 mini 4k instruct",
        description="Nebius Phi 3 mini 4k instruct is a medium-scale model with 4K context and a 3.82B parameter size.",
        max_input_tokens=4096,
        euro_per_1M_input_tokens=0.13,
        euro_per_1M_output_tokens=0.4,
        capabilities=[ModelCapabilities.chat],
        location=ModelLocation.EU,
        parameters="3.82B",
    )
    provider = ModelProvider.nebius


class Nebius_Phi_3_mini_4k_instruct_cheap(BaseLLMModel):
    config = ModelConfig(
        identifier="microsoft/Phi-3-mini-4k-instruct",
        verbose_name="Nebius Phi 3 mini 4k instruct",
        description="Nebius Phi 3 mini 4k instruct is a medium-scale model with 4K context and a 3.82B parameter size.",
        max_input_tokens=4096,
        euro_per_1M_input_tokens=0.04,
        euro_per_1M_output_tokens=0.13,
        capabilities=[ModelCapabilities.chat],
        location=ModelLocation.EU,
        parameters="3.82B",
    )
    provider = ModelProvider.nebius


class Nebius_Phi_3_medium_128k_instruct_fast(BaseLLMModel):
    config = ModelConfig(
        identifier="microsoft/Phi-3-medium-128k-instruct-fast",
        verbose_name="Nebius Phi 3 medium 128k instruct",
        description="Nebius Phi 3 medium 128k instruct is a medium-scale model with 128K context and a 3.82B parameter size.",
        max_input_tokens=131072,
        euro_per_1M_input_tokens=0.15,
        euro_per_1M_output_tokens=0.45,
        capabilities=[ModelCapabilities.chat],
        location=ModelLocation.EU,
        parameters="3.82B",
    )
    provider = ModelProvider.nebius


class Nebius_Phi_3_medium_128k_instruct_cheap(BaseLLMModel):
    config = ModelConfig(
        identifier="microsoft/Phi-3-medium-128k-instruct",
        verbose_name="Nebius Phi 3 medium 128k instruct",
        description="Nebius Phi 3 medium 128k instruct is a medium-scale model with 128K context and a 3.82B parameter size.",
        max_input_tokens=131072,
        euro_per_1M_input_tokens=0.1,
        euro_per_1M_output_tokens=0.3,
        capabilities=[ModelCapabilities.chat],
        location=ModelLocation.EU,
        parameters="3.82B",
    )
    provider = ModelProvider.nebius


class Nebius_OLMo_7B_Instruct(BaseLLMModel):
    config = ModelConfig(
        identifier="allenai/OLMo-7B-Instruct-hf",
        verbose_name="Nebius OLMo 7B Instruct",
        description="Nebius OLMo 7B Instruct is a medium-scale model with 2K context and a 6.89B parameter size.",
        max_input_tokens=2048,
        euro_per_1M_input_tokens=0.08,
        euro_per_1M_output_tokens=0.24,
        capabilities=[ModelCapabilities.chat],
        location=ModelLocation.EU,
        parameters="6.89B",
    )
    provider = ModelProvider.nebius


class Nebius_Gemma_2_9b_it_fast(BaseLLMModel):
    config = ModelConfig(
        identifier="google/gemma-2-9b-it-fast",
        verbose_name="Nebius Gemma 2 9b it",
        description="Nebius Gemma 2 9b it is a medium-scale model with 8K context and a 9.24B parameter size.",
        max_input_tokens=4096,
        euro_per_1M_input_tokens=0.03,
        euro_per_1M_output_tokens=0.09,
        capabilities=[ModelCapabilities.chat],
        location=ModelLocation.EU,
        parameters="9.24B",
    )
    provider = ModelProvider.nebius


class Nebius_Gemma_2_9b_it_cheap(BaseLLMModel):
    config = ModelConfig(
        identifier="google/gemma-2-9b-it",
        verbose_name="Nebius Gemma 2 9b it",
        description="Nebius Gemma 2 9b it is a medium-scale model with 8K context and a 9.24B parameter size.",
        max_input_tokens=4096,
        euro_per_1M_input_tokens=0.02,
        euro_per_1M_output_tokens=0.06,
        capabilities=[ModelCapabilities.chat],
        location=ModelLocation.EU,
        parameters="9.24B",
    )
    provider = ModelProvider.nebius
