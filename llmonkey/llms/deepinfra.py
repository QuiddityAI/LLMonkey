from ..models import ModelCapabilities, ModelConfig, ModelLocation, ModelProvider
from .base_llm import BaseLLMModel


class Deepinfra_DeepSeek_R1(BaseLLMModel):
    config = ModelConfig(
        identifier="deepseek-ai/DeepSeek-R1",
        verbose_name="Deepinfra DeepSeek R1",
        description="We introduce DeepSeek-R1, which incorporates cold-start data before RL. DeepSeek-R1 achieves performance comparable to OpenAI-o1 across math, code, and reasoning tasks.",
        max_input_tokens=131072,
        euro_per_1M_input_tokens=0.85,
        euro_per_1M_output_tokens=2.5,
        capabilities=[ModelCapabilities.chat],
        location=ModelLocation.US,
        parameters="R1",
    )
    provider = ModelProvider.deepinfra


class Deepinfra_DeepSeek_R1_Distill_Llama_70B(BaseLLMModel):
    config = ModelConfig(
        identifier="deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        verbose_name="Deepinfra DeepSeek R1 Distill Llama 70B",
        description="DeepSeek-R1-Distill-Llama-70B is a highly efficient language model that leverages knowledge distillation to achieve state-of-the-art performance. This model distills the reasoning patterns of larger models into a smaller, more agile architecture, resulting in exceptional results on benchmarks like AIME 2024, MATH-500, and LiveCodeBench. With 70 billion parameters, DeepSeek-R1-Distill-Llama-70B offers a unique balance of accuracy and efficiency, making it an ideal choice for a wide range of natural language processing tasks.",
        max_input_tokens=131072,
        euro_per_1M_input_tokens=0.23,
        euro_per_1M_output_tokens=0.69,
        capabilities=[ModelCapabilities.chat],
        location=ModelLocation.US,
        parameters="70B",
    )
    provider = ModelProvider.deepinfra


class Deepinfra_DeepSeek_V3(BaseLLMModel):
    config = ModelConfig(
        identifier="deepseek-ai/DeepSeek-V3",
        verbose_name="Deepinfra DeepSeek V3",
        description="",
        max_input_tokens=131072,
        euro_per_1M_input_tokens=0.85,
        euro_per_1M_output_tokens=0.9,
        capabilities=[ModelCapabilities.chat],
        location=ModelLocation.US,
        parameters="V3",
    )
    provider = ModelProvider.deepinfra


class Deepinfra_Llama_3_3_70B_Instruct_Turbo(BaseLLMModel):
    config = ModelConfig(
        identifier="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        verbose_name="Deepinfra Llama 3.3 70B Instruct Turbo",
        description="Llama 3.3-70B Turbo is a highly optimized version of the Llama 3.3-70B model, utilizing FP8 quantization to deliver significantly faster inference speeds with a minor trade-off in accuracy. The model is designed to be helpful, safe, and flexible, with a focus on responsible deployment and mitigating potential risks such as bias, toxicity, and misinformation. It achieves state-of-the-art performance on various benchmarks, including conversational tasks, language translation, and text generation.",
        max_input_tokens=131072,
        euro_per_1M_input_tokens=0.12,
        euro_per_1M_output_tokens=0.3,
        capabilities=[ModelCapabilities.chat],
        location=ModelLocation.US,
        parameters="70B",
    )
    provider = ModelProvider.deepinfra


class Deepinfra_Llama_3_3_70B_Instruct(BaseLLMModel):
    config = ModelConfig(
        identifier="meta-llama/Llama-3.3-70B-Instruct",
        verbose_name="Deepinfra Llama 3.3 70B Instruct",
        description="Llama 3.3-70B is a multilingual LLM trained on a massive dataset of 15 trillion tokens, fine-tuned for instruction-following and conversational dialogue. The model is designed to be helpful, safe, and flexible, with a focus on responsible deployment and mitigating potential risks such as bias, toxicity, and misinformation. It achieves state-of-the-art performance on various benchmarks, including conversational tasks, language translation, and text generation.",
        max_input_tokens=131072,
        euro_per_1M_input_tokens=0.23,
        euro_per_1M_output_tokens=0.4,
        capabilities=[ModelCapabilities.chat],
        location=ModelLocation.US,
        parameters="70B",
    )
    provider = ModelProvider.deepinfra


class Deepinfra_phi_4(BaseLLMModel):
    config = ModelConfig(
        identifier="microsoft/phi-4",
        verbose_name="Deepinfra phi 4",
        description="Phi-4 is a model built upon a blend of synthetic datasets, data from filtered public domain websites, and acquired academic books and Q&A datasets. The goal of this approach was to ensure that small capable models were trained with data focused on high quality and advanced reasoning.",
        max_input_tokens=131072,
        euro_per_1M_input_tokens=0.07,
        euro_per_1M_output_tokens=0.14,
        capabilities=[ModelCapabilities.chat],
        location=ModelLocation.US,
        parameters="4",
    )
    provider = ModelProvider.deepinfra


class Deepinfra_Meta_Llama_3_1_70B_Instruct(BaseLLMModel):
    config = ModelConfig(
        identifier="meta-llama/Meta-Llama-3.1-70B-Instruct",
        verbose_name="Deepinfra Meta Llama 3.1 70B Instruct",
        description="Meta developed and released the Meta Llama 3.1 family of large language models (LLMs), a collection of pretrained and instruction tuned generative text models in 8B, 70B and 405B sizes",
        max_input_tokens=131072,
        euro_per_1M_input_tokens=0.23,
        euro_per_1M_output_tokens=0.4,
        capabilities=[ModelCapabilities.chat],
        location=ModelLocation.US,
        parameters="70B",
    )
    provider = ModelProvider.deepinfra


class Deepinfra_Meta_Llama_3_1_8B_Instruct(BaseLLMModel):
    config = ModelConfig(
        identifier="meta-llama/Meta-Llama-3.1-8B-Instruct",
        verbose_name="Deepinfra Meta Llama 3.1 8B Instruct",
        description="Meta developed and released the Meta Llama 3.1 family of large language models (LLMs), a collection of pretrained and instruction tuned generative text models in 8B, 70B and 405B sizes",
        max_input_tokens=131072,
        euro_per_1M_input_tokens=0.03,
        euro_per_1M_output_tokens=0.05,
        capabilities=[ModelCapabilities.chat],
        location=ModelLocation.US,
        parameters="8B",
    )
    provider = ModelProvider.deepinfra


class Deepinfra_Meta_Llama_3_1_405B_Instruct(BaseLLMModel):
    config = ModelConfig(
        identifier="meta-llama/Meta-Llama-3.1-405B-Instruct",
        verbose_name="Deepinfra Meta Llama 3.1 405B Instruct",
        description="Meta developed and released the Meta Llama 3.1 family of large language models (LLMs), a collection of pretrained and instruction tuned generative text models in 8B, 70B and 405B sizes",
        max_input_tokens=131072,
        euro_per_1M_input_tokens=0.8,
        euro_per_1M_output_tokens=0.8,
        capabilities=[ModelCapabilities.chat],
        location=ModelLocation.US,
        parameters="405B",
    )
    provider = ModelProvider.deepinfra


class Deepinfra_Qwen_QwQ_32B(BaseLLMModel):
    config = ModelConfig(
        identifier="Qwen/QwQ-32B-Preview",
        verbose_name="Deepinfra Qwen QwQ 32B Preview",
        description="QwQ is an experimental research model developed by the Qwen Team, designed to advance AI reasoning capabilities. This model embodies the spirit of philosophical inquiry, approaching problems with genuine wonder and doubt. QwQ demonstrates impressive analytical abilities, achieving scores of 65.2% on GPQA, 50.0% on AIME, 90.6% on MATH-500, and 50.0% on LiveCodeBench. With its contemplative approach and exceptional performance on complex problems.",
        max_input_tokens=131072,
        euro_per_1M_input_tokens=0.12,
        euro_per_1M_output_tokens=0.18,
        capabilities=[ModelCapabilities.chat],
        location=ModelLocation.US,
        parameters="32B",
    )
    provider = ModelProvider.deepinfra


class Deepinfra_Meta_Llama_3_1_8B_Instruct_Turbo(BaseLLMModel):
    config = ModelConfig(
        identifier="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        verbose_name="Deepinfra Meta Llama 3.1 8B Instruct Turbo",
        description="Meta developed and released the Meta Llama 3.1 family of large language models (LLMs), a collection of pretrained and instruction tuned generative text models in 8B, 70B and 405B sizes",
        max_input_tokens=131072,
        euro_per_1M_input_tokens=0.02,
        euro_per_1M_output_tokens=0.05,
        capabilities=[ModelCapabilities.chat],
        location=ModelLocation.US,
        parameters="8B",
    )
    provider = ModelProvider.deepinfra


class Deepinfra_Meta_Llama_3_1_70B_Instruct_Turbo(BaseLLMModel):
    config = ModelConfig(
        identifier="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        verbose_name="Deepinfra Meta Llama 3.1 70B Instruct Turbo",
        description="Meta developed and released the Meta Llama 3.1 family of large language models (LLMs), a collection of pretrained and instruction tuned generative text models in 8B, 70B and 405B sizes",
        max_input_tokens=131072,
        euro_per_1M_input_tokens=0.12,
        euro_per_1M_output_tokens=0.3,
        capabilities=[ModelCapabilities.chat],
        location=ModelLocation.US,
        parameters="70B",
    )
    provider = ModelProvider.deepinfra


class Deepinfra_Qwen2_5_Coder_32B_Instruct(BaseLLMModel):
    config = ModelConfig(
        identifier="Qwen/Qwen2.5-Coder-32B-Instruct",
        verbose_name="Deepinfra Qwen2.5 Coder 32B Instruct",
        description="Qwen2.5-Coder is the latest series of Code-Specific Qwen large language models (formerly known as CodeQwen). It has significant improvements in code generation, code reasoning and code fixing. A more comprehensive foundation for real-world applications such as Code Agents. Not only enhancing coding capabilities but also maintaining its strengths in mathematics and general competencies.",
        max_input_tokens=131072,
        euro_per_1M_input_tokens=0.07,
        euro_per_1M_output_tokens=0.16,
        capabilities=[ModelCapabilities.chat],
        location=ModelLocation.US,
        parameters="32B",
    )
    provider = ModelProvider.deepinfra


class Deepinfra_Llama_3_1_Nemotron_70B_Instruct(BaseLLMModel):
    config = ModelConfig(
        identifier="nvidia/Llama-3.1-Nemotron-70B-Instruct",
        verbose_name="Deepinfra Llama 3.1 Nemotron 70B Instruct",
        description="Llama-3.1-Nemotron-70B-Instruct is a large language model customized by NVIDIA to improve the helpfulness of LLM generated responses to user queries. This model reaches Arena Hard of 85.0, AlpacaEval 2 LC of 57.6 and GPT-4-Turbo MT-Bench of 8.98, which are known to be predictive of LMSys Chatbot Arena Elo.  As of 16th Oct 2024, this model is #1 on all three automatic alignment benchmarks (verified tab for AlpacaEval 2 LC), edging out strong frontier models such as GPT-4o and Claude 3.5 Sonnet.",
        max_input_tokens=131072,
        euro_per_1M_input_tokens=0.12,
        euro_per_1M_output_tokens=0.3,
        capabilities=[ModelCapabilities.chat],
        location=ModelLocation.US,
        parameters="70B",
    )
    provider = ModelProvider.deepinfra


class Deepinfra_Qwen2_5_72B_Instruct(BaseLLMModel):
    config = ModelConfig(
        identifier="Qwen/Qwen2.5-72B-Instruct",
        verbose_name="Deepinfra Qwen2.5 72B Instruct",
        description="Qwen2.5 is a model pretrained on a large-scale dataset of up to 18 trillion tokens, offering significant improvements in knowledge, coding, mathematics, and instruction following compared to its predecessor Qwen2. The model also features enhanced capabilities in generating long texts, understanding structured data, and generating structured outputs, while supporting multilingual capabilities for over 29 languages.",
        max_input_tokens=131072,
        euro_per_1M_input_tokens=0.23,
        euro_per_1M_output_tokens=0.4,
        capabilities=[ModelCapabilities.chat],
        location=ModelLocation.US,
        parameters="72B",
    )
    provider = ModelProvider.deepinfra


class Deepinfra_Llama_3_2_90B_Vision_Instruct(BaseLLMModel):
    config = ModelConfig(
        identifier="meta-llama/Llama-3.2-90B-Vision-Instruct",
        verbose_name="Deepinfra Llama 3.2 90B Vision Instruct",
        description="The Llama 90B Vision model is a top-tier, 90-billion-parameter multimodal model designed for the most challenging visual reasoning and language tasks. It offers unparalleled accuracy in image captioning, visual question answering, and advanced image-text comprehension. Pre-trained on vast multimodal datasets and fine-tuned with human feedback, the Llama 90B Vision is engineered to handle the most demanding image-based AI tasks.  This model is perfect for industries requiring cutting-edge multimodal AI capabilities, particularly those dealing with complex, real-time visual and textual analysis.",
        max_input_tokens=131072,
        euro_per_1M_input_tokens=0.35,
        euro_per_1M_output_tokens=0.4,
        capabilities=[ModelCapabilities.chat],
        location=ModelLocation.US,
        parameters="90B",
    )
    provider = ModelProvider.deepinfra


class Deepinfra_Llama_3_2_11B_Vision_Instruct(BaseLLMModel):
    config = ModelConfig(
        identifier="meta-llama/Llama-3.2-11B-Vision-Instruct",
        verbose_name="Deepinfra Llama 3.2 11B Vision Instruct",
        description="Llama 3.2 11B Vision is a multimodal model with 11 billion parameters, designed to handle tasks combining visual and textual data. It excels in tasks such as image captioning and visual question answering, bridging the gap between language generation and visual reasoning. Pre-trained on a massive dataset of image-text pairs, it performs well in complex, high-accuracy image analysis.  Its ability to integrate visual understanding with language processing makes it an ideal solution for industries requiring comprehensive visual-linguistic AI applications, such as content creation, AI-driven customer service, and research.",
        max_input_tokens=131072,
        euro_per_1M_input_tokens=0.055,
        euro_per_1M_output_tokens=0.055,
        capabilities=[ModelCapabilities.chat],
        location=ModelLocation.US,
        parameters="11B",
    )
    provider = ModelProvider.deepinfra


class Deepinfra_Llama_3_2_1B_Instruct(BaseLLMModel):
    config = ModelConfig(
        identifier="meta-llama/Llama-3.2-1B-Instruct",
        verbose_name="Deepinfra Llama 3.2 1B Instruct",
        description="The Meta Llama 3.2 collection of multilingual large language models (LLMs) is a collection of pretrained and instruction-tuned generative models in 1B and 3B sizes (text in/text out).",
        max_input_tokens=131072,
        euro_per_1M_input_tokens=0.01,
        euro_per_1M_output_tokens=0.01,
        capabilities=[ModelCapabilities.chat],
        location=ModelLocation.US,
        parameters="1B",
    )
    provider = ModelProvider.deepinfra


class Deepinfra_Llama_3_2_3B_Instruct(BaseLLMModel):
    config = ModelConfig(
        identifier="meta-llama/Llama-3.2-3B-Instruct",
        verbose_name="Deepinfra Llama 3.2 3B Instruct",
        description="The Meta Llama 3.2 collection of multilingual large language models (LLMs) is a collection of pretrained and instruction-tuned generative models in 1B and 3B sizes (text in/text out)",
        max_input_tokens=131072,
        euro_per_1M_input_tokens=0.015,
        euro_per_1M_output_tokens=0.025,
        capabilities=[ModelCapabilities.chat],
        location=ModelLocation.US,
        parameters="3B",
    )
    provider = ModelProvider.deepinfra
