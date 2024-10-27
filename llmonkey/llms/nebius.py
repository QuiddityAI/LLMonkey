from ..models import ModelCapabilities, ModelConfig, ModelProvider
from .base_llm import BaseLLMModel

info = [
    {
        "type": "text2text",
        "name": "Meta-Llama-3.1-70B-Instruct",
        "status": "active",
        "logo_url": "",
        "huggingface_url": "https://huggingface.co/meta-llama/Meta-Llama-3.1-70B-Instruct",
        "flavors": [
            {
                "model_id": "meta-llama/Meta-Llama-3.1-70B-Instruct-fast",
                "label": "fast",
                "input_price_per_million_tokens": 0.25,
                "output_price_per_million_tokens": 0.75,
                "tokens_per_second": 60,
                "limits": {
                    "tokens_per_minute": 1000000.0,
                    "requests_per_minute": 600.0,
                    "burst_ratio": 1.0,
                },
                "max_model_len": 131072,
            },
            {
                "model_id": "meta-llama/Meta-Llama-3.1-70B-Instruct",
                "label": "cheap",
                "input_price_per_million_tokens": 0.13,
                "output_price_per_million_tokens": 0.4,
                "tokens_per_second": 25,
                "limits": {
                    "tokens_per_minute": 1000000.0,
                    "requests_per_minute": 600.0,
                    "burst_ratio": 1.0,
                },
                "max_model_len": 131072,
            },
        ],
        "vendor": "meta",
        "quality": 86,
        "context_window_k": 128,
        "size_b": 70.6,
        "tags": ["128K context", "Llama 3.1 License"],
        "use_cases": [
            "summarization",
            "context_and_rag",
            "code",
            "tool_and_function_calling",
        ],
        "policy_url": "https://llama.meta.com/llama3_1/use-policy/",
        "license": {
            "url": "https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/LICENSE",
            "type": "",
            "name": "Llama 3.1 License",
        },
    },
    {
        "type": "text2text",
        "name": "Meta-Llama-3.1-8B-Instruct",
        "status": "active",
        "logo_url": "",
        "huggingface_url": "https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct",
        "flavors": [
            {
                "model_id": "meta-llama/Meta-Llama-3.1-8B-Instruct-fast",
                "label": "fast",
                "input_price_per_million_tokens": 0.03,
                "output_price_per_million_tokens": 0.09,
                "tokens_per_second": 155,
                "limits": {
                    "tokens_per_minute": 400000.0,
                    "requests_per_minute": 600.0,
                    "burst_ratio": 1.0,
                },
                "max_model_len": 131072,
            },
            {
                "model_id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
                "label": "cheap",
                "input_price_per_million_tokens": 0.02,
                "output_price_per_million_tokens": 0.06,
                "tokens_per_second": 30,
                "limits": {
                    "tokens_per_minute": 400000.0,
                    "requests_per_minute": 600.0,
                    "burst_ratio": 1.0,
                },
                "max_model_len": 131072,
            },
        ],
        "vendor": "meta",
        "quality": 73,
        "context_window_k": 128,
        "size_b": 8.03,
        "tags": ["128K context", "small", "Llama 3.1 License"],
        "use_cases": ["summarization", "context_and_rag", "fast_and_cost_efficient"],
        "policy_url": "https://llama.meta.com/llama3_1/use-policy/",
        "license": {
            "url": "https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/LICENSE",
            "type": "",
            "name": "Llama 3.1 License",
        },
    },
    {
        "type": "text2text",
        "name": "Meta-Llama-3.1-405B-Instruct",
        "status": "active",
        "logo_url": "",
        "huggingface_url": "https://huggingface.co/meta-llama/Meta-Llama-3.1-405B-Instruct",
        "flavors": [
            {
                "model_id": "meta-llama/Meta-Llama-3.1-405B-Instruct",
                "label": "cheap",
                "input_price_per_million_tokens": 1.0,
                "output_price_per_million_tokens": 3.0,
                "tokens_per_second": 20,
                "limits": {
                    "tokens_per_minute": 131072.0,
                    "requests_per_minute": 120.0,
                    "burst_ratio": 1.0,
                },
                "max_model_len": 131072,
            }
        ],
        "vendor": "meta",
        "quality": 89,
        "context_window_k": 128,
        "size_b": 406,
        "tags": ["128K context", "Llama 3.1 License"],
        "use_cases": [
            "summarization",
            "context_and_rag",
            "code",
            "tool_and_function_calling",
            "complex_writing",
        ],
        "policy_url": "https://llama.meta.com/llama3_1/use-policy/",
        "license": {
            "url": "https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/LICENSE",
            "type": "",
            "name": "Llama 3.1 License",
        },
    },
    {
        "type": "text2text",
        "name": "Mistral-Nemo-Instruct-2407",
        "status": "active",
        "logo_url": "",
        "huggingface_url": "https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407",
        "flavors": [
            {
                "model_id": "mistralai/Mistral-Nemo-Instruct-2407-fast",
                "label": "fast",
                "input_price_per_million_tokens": 0.08,
                "output_price_per_million_tokens": 0.24,
                "tokens_per_second": 100,
                "limits": {
                    "tokens_per_minute": 400000.0,
                    "requests_per_minute": 600.0,
                    "burst_ratio": 1.0,
                },
                "max_model_len": 128000,
            },
            {
                "model_id": "mistralai/Mistral-Nemo-Instruct-2407",
                "label": "cheap",
                "input_price_per_million_tokens": 0.04,
                "output_price_per_million_tokens": 0.12,
                "tokens_per_second": 30,
                "limits": {
                    "tokens_per_minute": 400000.0,
                    "requests_per_minute": 600.0,
                    "burst_ratio": 1.0,
                },
                "max_model_len": 128000,
            },
        ],
        "vendor": "mistralai",
        "quality": 68,
        "context_window_k": 128,
        "size_b": 12.2,
        "tags": ["128K context", "small", "Apache 2.0 License"],
        "use_cases": ["summarization", "context_and_rag", "fast_and_cost_efficient"],
        "license": {
            "url": "https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md",
            "type": "",
            "name": "Apache 2.0 License",
        },
    },
    {
        "type": "text2text",
        "name": "Mixtral-8x7B-Instruct-v0.1",
        "status": "active",
        "logo_url": "",
        "huggingface_url": "https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1",
        "flavors": [
            {
                "model_id": "mistralai/Mixtral-8x7B-Instruct-v0.1-fast",
                "label": "fast",
                "input_price_per_million_tokens": 0.15,
                "output_price_per_million_tokens": 0.45,
                "tokens_per_second": 43,
                "limits": {
                    "tokens_per_minute": 400000.0,
                    "requests_per_minute": 600.0,
                    "burst_ratio": 1.0,
                },
                "max_model_len": 32768,
            },
            {
                "model_id": "mistralai/Mixtral-8x7B-Instruct-v0.1",
                "label": "cheap",
                "input_price_per_million_tokens": 0.08,
                "output_price_per_million_tokens": 0.24,
                "tokens_per_second": 25,
                "limits": {
                    "tokens_per_minute": 400000.0,
                    "requests_per_minute": 600.0,
                    "burst_ratio": 1.0,
                },
                "max_model_len": 32768,
            },
        ],
        "vendor": "mistralai",
        "quality": 71,
        "context_window_k": 33,
        "size_b": 46.7,
        "tags": ["33K context", "Apache 2.0 License"],
        "use_cases": ["fast_and_cost_efficient"],
        "license": {
            "url": "https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md",
            "type": "",
            "name": "Apache 2.0 License",
        },
    },
    {
        "type": "text2text",
        "name": "Mixtral-8x22B-Instruct-v0.1",
        "status": "active",
        "logo_url": "",
        "huggingface_url": "https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1",
        "flavors": [
            {
                "model_id": "mistralai/Mixtral-8x22B-Instruct-v0.1-fast",
                "label": "fast",
                "input_price_per_million_tokens": 0.7,
                "output_price_per_million_tokens": 2.1,
                "tokens_per_second": 35,
                "limits": {
                    "tokens_per_minute": 400000.0,
                    "requests_per_minute": 600.0,
                    "burst_ratio": 1.0,
                },
                "max_model_len": 65536,
            },
            {
                "model_id": "mistralai/Mixtral-8x22B-Instruct-v0.1",
                "label": "cheap",
                "input_price_per_million_tokens": 0.4,
                "output_price_per_million_tokens": 1.2,
                "tokens_per_second": 25,
                "limits": {
                    "tokens_per_minute": 400000.0,
                    "requests_per_minute": 600.0,
                    "burst_ratio": 1.0,
                },
                "max_model_len": 65536,
            },
        ],
        "vendor": "mistralai",
        "quality": 78,
        "context_window_k": 65,
        "size_b": 141,
        "tags": ["65K context", "Apache 2.0 License"],
        "use_cases": ["summarization", "context_and_rag", "complex_writing"],
        "license": {
            "url": "https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md",
            "type": "",
            "name": "Apache 2.0 License",
        },
    },
    {
        "type": "text2text",
        "name": "Qwen2.5-Coder-7B",
        "status": "active",
        "logo_url": "",
        "huggingface_url": "https://huggingface.co/Qwen/Qwen2.5-Coder-7B",
        "flavors": [
            {
                "model_id": "Qwen/Qwen2.5-Coder-7B-fast",
                "label": "fast",
                "input_price_per_million_tokens": 0.03,
                "output_price_per_million_tokens": 0.09,
                "tokens_per_second": 125,
                "limits": {
                    "tokens_per_minute": 100000000.0,
                    "requests_per_minute": 10000.0,
                    "burst_ratio": 1.0,
                },
                "max_model_len": 32768,
            },
            {
                "model_id": "Qwen/Qwen2.5-Coder-7B",
                "label": "cheap",
                "input_price_per_million_tokens": 0.01,
                "output_price_per_million_tokens": 0.03,
                "tokens_per_second": 70,
                "limits": {
                    "tokens_per_minute": 100000000.0,
                    "requests_per_minute": 10000.0,
                    "burst_ratio": 1.0,
                },
                "max_model_len": 32768,
            },
        ],
        "vendor": "Qwen",
        "quality": 74,
        "context_window_k": 32,
        "size_b": 7.62,
        "tags": ["32K context", "Apache 2.0 License"],
        "use_cases": ["code"],
        "license": {
            "url": "https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md",
            "type": "",
            "name": "Apache 2.0 License",
        },
    },
    {
        "type": "text2text",
        "name": "Qwen2.5-Coder-7B-Instruct",
        "status": "active",
        "logo_url": "",
        "huggingface_url": "https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct",
        "flavors": [
            {
                "model_id": "Qwen/Qwen2.5-Coder-7B-Instruct-fast",
                "label": "fast",
                "input_price_per_million_tokens": 0.03,
                "output_price_per_million_tokens": 0.09,
                "tokens_per_second": 125,
                "limits": {
                    "tokens_per_minute": 400000.0,
                    "requests_per_minute": 600.0,
                    "burst_ratio": 1.0,
                },
                "max_model_len": 32768,
            },
            {
                "model_id": "Qwen/Qwen2.5-Coder-7B-Instruct",
                "label": "cheap",
                "input_price_per_million_tokens": 0.01,
                "output_price_per_million_tokens": 0.03,
                "tokens_per_second": 70,
                "limits": {
                    "tokens_per_minute": 400000.0,
                    "requests_per_minute": 600.0,
                    "burst_ratio": 1.0,
                },
                "max_model_len": 32768,
            },
        ],
        "vendor": "Qwen",
        "quality": 74,
        "context_window_k": 32,
        "size_b": 7.62,
        "tags": ["32K context", "Apache 2.0 License"],
        "use_cases": ["code"],
        "license": {
            "url": "https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md",
            "type": "",
            "name": "Apache 2.0 License",
        },
    },
    {
        "type": "text2text",
        "name": "DeepSeek-Coder-V2-Lite-Instruct",
        "status": "active",
        "logo_url": "",
        "huggingface_url": "https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
        "flavors": [
            {
                "model_id": "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct-fast",
                "label": "fast",
                "input_price_per_million_tokens": 0.08,
                "output_price_per_million_tokens": 0.24,
                "tokens_per_second": 50,
                "max_model_len": 128000,
                "limits": {
                    "tokens_per_minute": 200000.0,
                    "requests_per_minute": 600.0,
                    "burst_ratio": 1.0,
                },
            },
            {
                "model_id": "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
                "label": "cheap",
                "input_price_per_million_tokens": 0.04,
                "output_price_per_million_tokens": 0.12,
                "tokens_per_second": 30,
                "max_model_len": 128000,
                "limits": {
                    "tokens_per_minute": 200000.0,
                    "requests_per_minute": 600.0,
                    "burst_ratio": 1.0,
                },
            },
        ],
        "vendor": "deepseek",
        "quality": 60,
        "context_window_k": 128,
        "size_b": 15.7,
        "tags": ["128K context", "DeepSeek License"],
        "use_cases": ["code"],
        "license": {
            "url": "https://github.com/deepseek-ai/deepseek-coder/blob/main/LICENSE-MODEL",
            "type": "",
            "name": "DeepSeek Licence",
        },
    },
    {
        "type": "text2text",
        "name": "Phi-3-mini-4k-instruct",
        "status": "active",
        "logo_url": "",
        "huggingface_url": "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct",
        "flavors": [
            {
                "model_id": "microsoft/Phi-3-mini-4k-instruct-fast",
                "label": "fast",
                "input_price_per_million_tokens": 0.13,
                "output_price_per_million_tokens": 0.4,
                "tokens_per_second": 40,
                "limits": {
                    "tokens_per_minute": 400000.0,
                    "requests_per_minute": 600.0,
                    "burst_ratio": 1.0,
                },
                "max_model_len": 4096,
            },
            {
                "model_id": "microsoft/Phi-3-mini-4k-instruct",
                "label": "cheap",
                "input_price_per_million_tokens": 0.04,
                "output_price_per_million_tokens": 0.13,
                "tokens_per_second": 13,
                "limits": {
                    "tokens_per_minute": 400000.0,
                    "requests_per_minute": 600.0,
                    "burst_ratio": 1.0,
                },
                "max_model_len": 4096,
            },
        ],
        "vendor": "microsoft",
        "quality": 70,
        "context_window_k": 4,
        "size_b": 3.82,
        "tags": ["4K context", "small", "MIT License"],
        "use_cases": ["fast_and_cost_efficient"],
        "license": {
            "url": "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/resolve/main/LICENSE",
            "type": "",
            "name": "MIT Licence",
        },
    },
    {
        "type": "text2text",
        "name": "Phi-3-medium-128k-instruct",
        "status": "active",
        "logo_url": "",
        "huggingface_url": "https://huggingface.co/microsoft/Phi-3-medium-128k-instruct",
        "flavors": [
            {
                "model_id": "microsoft/Phi-3-medium-128k-instruct-fast",
                "label": "fast",
                "input_price_per_million_tokens": 0.15,
                "output_price_per_million_tokens": 0.45,
                "tokens_per_second": 60,
                "limits": {
                    "tokens_per_minute": 400000.0,
                    "requests_per_minute": 600.0,
                    "burst_ratio": 1.0,
                },
                "max_model_len": 131072,
            },
            {
                "model_id": "microsoft/Phi-3-medium-128k-instruct",
                "label": "cheap",
                "input_price_per_million_tokens": 0.1,
                "output_price_per_million_tokens": 0.3,
                "tokens_per_second": 25,
                "limits": {
                    "tokens_per_minute": 400000.0,
                    "requests_per_minute": 600.0,
                    "burst_ratio": 1.0,
                },
                "max_model_len": 131072,
            },
        ],
        "vendor": "microsoft",
        "quality": 70,
        "context_window_k": 128,
        "size_b": 3.82,
        "tags": ["128K context", "small", "MIT License"],
        "use_cases": ["fast_and_cost_efficient", "summarization", "context_and_rag"],
        "license": {
            "url": "https://huggingface.co/microsoft/Phi-3-medium-128k-instruct/blob/main/LICENSE",
            "type": "",
            "name": "MIT Licence",
        },
    },
    {
        "type": "text2text",
        "name": "OLMo-7B-Instruct",
        "status": "active",
        "logo_url": "",
        "huggingface_url": "https://huggingface.co/allenai/OLMo-7B-Instruct-hf",
        "flavors": [
            {
                "model_id": "allenai/OLMo-7B-Instruct-hf",
                "label": "cheap",
                "input_price_per_million_tokens": 0.08,
                "output_price_per_million_tokens": 0.24,
                "tokens_per_second": 25,
                "limits": {
                    "tokens_per_minute": 400000.0,
                    "requests_per_minute": 600.0,
                    "burst_ratio": 1.0,
                },
                "max_model_len": 2048,
            }
        ],
        "vendor": "allenai",
        "quality": 46,
        "context_window_k": 2,
        "size_b": 6.89,
        "tags": ["2K context", "small", "Apache 2.0 License"],
        "use_cases": ["open_training_data"],
        "license": {
            "url": "https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md",
            "type": "",
            "name": "Apache 2.0 License",
        },
    },
    {
        "type": "text2text",
        "name": "Gemma-2-9b-it",
        "status": "active",
        "logo_url": "",
        "huggingface_url": "https://huggingface.co/google/gemma-2-9b-it",
        "flavors": [
            {
                "model_id": "google/gemma-2-9b-it-fast",
                "label": "fast",
                "input_price_per_million_tokens": 0.03,
                "output_price_per_million_tokens": 0.09,
                "tokens_per_second": 90,
                "limits": {
                    "tokens_per_minute": 400000.0,
                    "requests_per_minute": 600.0,
                    "burst_ratio": 1.0,
                },
                "max_model_len": 4096,
            },
            {
                "model_id": "google/gemma-2-9b-it",
                "label": "cheap",
                "input_price_per_million_tokens": 0.02,
                "output_price_per_million_tokens": 0.06,
                "tokens_per_second": 50,
                "limits": {
                    "tokens_per_minute": 400000.0,
                    "requests_per_minute": 600.0,
                    "burst_ratio": 1.0,
                },
                "max_model_len": 4096,
            },
        ],
        "vendor": "google",
        "quality": 73,
        "context_window_k": 8,
        "size_b": 9.24,
        "tags": ["8K context", "Gemma License"],
        "use_cases": ["fast_and_cost_efficient"],
        "policy_url": "https://ai.google.dev/gemma/terms",
        "license": {
            "url": "https://ai.google.dev/gemma/terms",
            "type": "",
            "name": "Gemma",
        },
    },
]


class Nebius_Llama_3_1_70B_fast(BaseLLMModel):
    config = ModelConfig(
        identifier="meta-llama/Meta-Llama-3.1-70B-Instruct-fast",
        verbose_name="Nebius Llama 3.1 70B",
        description="Nebius Llama 3.1 70B is a large-scale model with 128K context and a 70B parameter size.",
        max_input_tokens=131072,
        euro_per_1M_input_tokens=0.25,
        euro_per_1M_output_tokens=0.75,
        capabilities=[ModelCapabilities.chat],
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
    )
    provider = ModelProvider.nebius
