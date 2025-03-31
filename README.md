# LLMonkey: a simple-to-use wrapper around multiple LLM providers and model repository with litellm compatibility

## Motivation

While litellm provides convenience methods for connecting to multiple LLM providers, it has two significant drawbacks:

1) Litellm code is monolithic, which leads to multiple dependencies and long import times, even though in majority of cases only couple of providers are used. Moreover, majority of providers nowadays have OpenAI-compatible interface rendering per-provider settings obsolete in many cases.

2) Litellm does not provide setting for specific models and its metainfo (e.g. model name, provider, number of parameters, capabilities, etc). Developer, wanting to iterate quickly between different models / providers would need to manually store settings (at least api_key, api_base, model_name) manually.

LLMonkey solves this issue by:

1) Providing object-oriented LLM interface similar to litellm, but allowing easy drop-in replacement of different models
2) Containing a database of different models and their metainformation, allowing for quick and convenient access to it



## Usage

Sample usage: see client.py (using old interface, new interface provides same methods of the Model class)

Minimal example:


```python
from llmonkey.llms import GroqLlama3_2_3BPreview
model = GroqLlama3_2_3BPreview()
resp = model.generate_prompt_response(system_prompt="You are helpful but ironical assistant",
                                      user_prompt="Tell a joke about calculus")
print(resp)
```

output:

```
ChatResponse(provider_used=<ModelProvider.groq: 'groq'>, model_used='llama-3.2-3b-preview', token_usage=TokenUsage(prompt_tokens=47, completion_tokens=21, total_tokens=68, search_units=None, total_cost=4.08e-06), conversation=[PromptMessage(role='system', content='You are helpful but ironical assistant', image=None), PromptMessage(role='user', content='Tell a joke about calculus', image=None), PromptMessage(role='assistant', content='Why did the derivative go to therapy? \n\nBecause it was struggling to find its limit-ing personality.', image=None)])

```

Besides that, each model provides following methods:
- `generate_prompt_response`: generates a response to a single prompt
- `generate_chat_response`: generates a response to a chat conversation
- `generate_structured_response`: generates a response according to a provided Pydantic model
- `generate_structured_array_response`: generates a response with an array of instances of provided Pydantic model
- `rerank`: reranks a list of provided prompts
- `to_litellm`: converts the model to litellm-compatible kwargs


### Converting to litellm

Each model can be converted to litellm by generating corresponding litellm kwargs with method `.to_litellm`.

```python
import litellm
from llmonkey.llms import OpenAI_GPT4o_Mini

lm = OpenAI_GPT4o_Mini()
litellm.completion(messages=[{ "content": "Hello, how are you?","role": "user"}], **lm.to_litellm())
```
for litellm-compatible interfaces, e.g. dspy.LM you can use same method:

```python
from llmonkey.llms import OpenAI_GPT4o_Mini
import dspy

lm = dspy.LM(**OpenAI_GPT4o_Mini().to_litellm())

```

### Getting the list of available models

```python
from llmonkey.llms import BaseModel

print(BaseModel.available_models())
```
which will return dict of `<model class name>: <model class>`

Use
```python
from llmonkey.llms import BaseModel
model = BaseModel.load("OpenAI_GPT4o_Mini")
```
to load a model by name.

<details>
<summary> Currently supported models </summary>

```
 'GroqLlama3_3_70bVersatile',
 'GroqLlama3_1_70bVersatile',
 'GroqLlama3_1_8bInstant',
 'GroqLlama3_70BToolUsePreview',
 'GroqLlama3_8BToolUsePreview',
 'GroqLlamaGuard38B',
 'GroqLlama3_2_1BPreview',
 'GroqLlama3_2_3BPreview',
 'GroqLlama3_2_11BVisionPreview',
 'GroqLlama3_2_90BPreview',
 'GroqLlama3_70B8k',
 'GroqLlama3_8B8k',
 'GroqMixtral8x7B',
 'GroqGemma7B8kInstruct',
 'GroqGemma2_9B8k',
 'Mistral_Ministral3b',
 'Mistral_Ministral8b',
 'Mistral_Mistral_Large',
 'Mistral_Mistral_Small',
 'Mistral_Mistral_Embed',
 'Mistral_Mistral_Nemo',
 'Mistral_Codestral',
 'Mistral_Pixtral',
 'Mistral_Pixtral_Large',
 'Nebius_Llama_3_3_70B_fast',
 'Nebius_Llama_3_3_70B',
 'Nebius_Llama_3_1_70B_fast',
 'Nebius_Llama_3_1_70B_cheap',
 'Nebius_Llama_3_1_8B_fast',
 'Nebius_Llama_3_1_8B_cheap',
 'Nebius_Llama_3_2_1B',
 'Nebius_Llama_3_2_3B',
 'Nebius_Llama_3_1_405B_cheap',
 'Nebius_Mistral_Nemo_Instruct_2407_fast',
 'Nebius_Mistral_Nemo_Instruct_2407_cheap',
 'Nebius_Mixtral_8x7B_Instruct_v0_1_fast',
 'Nebius_Mixtral_8x7B_Instruct_v0_1_cheap',
 'Nebius_Mixtral_8x22B_Instruct_v0_1_fast',
 'Nebius_Mixtral_8x22B_Instruct_v0_1_cheap',
 'Nebius_Qwen2_5_Coder_7B_fast',
 'Nebius_Qwen2_5_Coder_7B_cheap',
 'Nebius_Qwen2_5_Coder_7B_Instruct_fast',
 'Nebius_Qwen2_5_Coder_7B_Instruct_cheap',
 'Nebius_DeepSeek_Coder_V2_Lite_Instruct_fast',
 'Nebius_DeepSeek_Coder_V2_Lite_Instruct_cheap',
 'Nebius_Phi_3_mini_4k_instruct_fast',
 'Nebius_Phi_3_mini_4k_instruct_cheap',
 'Nebius_Phi_3_medium_128k_instruct_fast',
 'Nebius_Phi_3_medium_128k_instruct_cheap',
 'Nebius_OLMo_7B_Instruct',
 'Nebius_Gemma_2_9b_it_fast',
 'Nebius_Gemma_2_9b_it_cheap',
 'Nebius_Llama3_OpenBioLLM_8B',
 'Nebius_Llama3_OpenBioLLM_70B',
 'OpenAI_GPT4o',
 'OpenAI_GPT4o_Mini',
 'OpenAI_o1',
 'OpenAI_o1_Mini',
 'OpenAI_TextEmbedding_3_small',
 'OpenAI_TextEmbedding_3_large',
 'OpenAI_Ada_v2',
 'Google_Gemini_Flash_1_5_v1',
 'Google_Gemini_Flash_1_5_v2',
 'Google_Gemini_Flash_1_5',
 'Google_Gemini_Flash_1_5_8B',
 'Google_Gemini_Flash_2_0_Exp',
 'Google_Gemini_Flash_2_0_Thinking_Exp',
 'Google_Gemini_Pro_1_5',
 'Ionos_Llama_3_1_405B',
 'Deepinfra_DeepSeek_R1',
 'Deepinfra_DeepSeek_R1_Distill_Llama_70B',
 'Deepinfra_DeepSeek_V3',
 'Deepinfra_Llama_3_3_70B_Instruct_Turbo',
 'Deepinfra_Llama_3_3_70B_Instruct',
 'Deepinfra_phi_4',
 'Deepinfra_Meta_Llama_3_1_70B_Instruct',
 'Deepinfra_Meta_Llama_3_1_8B_Instruct',
 'Deepinfra_Meta_Llama_3_1_405B_Instruct',
 'Deepinfra_Qwen_QwQ_32B',
 'Deepinfra_Meta_Llama_3_1_8B_Instruct_Turbo',
 'Deepinfra_Meta_Llama_3_1_70B_Instruct_Turbo',
 'Deepinfra_Qwen2_5_Coder_32B_Instruct',
 'Deepinfra_Llama_3_1_Nemotron_70B_Instruct',
 'Deepinfra_Qwen2_5_72B_Instruct',
 'Deepinfra_Llama_3_2_90B_Vision_Instruct',
 'Deepinfra_Llama_3_2_11B_Vision_Instruct',
 'Deepinfra_Llama_3_2_1B_Instruct',
 'Deepinfra_Llama_3_2_3B_Instruct',
 'Azure_GPT4o'
 ```

</details>

### Vision capabilities

Models now support vision tasks, e.g.:
```python
from llmonkey.llms import GroqLlama3_2_11BVisionPreview

model = GroqLlama3_2_11BVisionPreview()

resp = model.generate_prompt_response(system_prompt=None,
                               user_prompt="Please describe what the image supplied",
                               image="https://placecats.com/700/500")

print(resp.dict())
```

output:

```python
{'provider_used': <ModelProvider.groq: 'groq'>,
 'model_used': 'llama-3.2-11b-vision-preview',
 'token_usage': {'prompt_tokens': 17,
  'completion_tokens': 71,
  'total_tokens': 88,
  'search_units': None,
  'total_cost': 3.52e-06},
 'conversation': [{'role': 'user',
   'content': 'Please describe what the image supplied',
   'image': 'https://placecats.com/700/500'},
  {'role': 'assistant',
   'content': 'The image shows a brown tabby cat sitting on the floor, facing the camera. The cat has a white chest and a pink nose, and its eyes are green. It is sitting on a dark wood floor with a white baseboard. Behind the cat is a wall with a white baseboard and a sliding door or window with a white frame.',
   'image': None}]}
```

### Old interface

```python
from llmonkey.llmonkey import LLMonkey

llmonkey = LLMonkey()

print("Available providers:", llmonkey.providers)

response = llmonkey.generate_chat_response(
    provider="openai",
    model_name="gpt-3.5-turbo",
    user_prompt="Hello! How are you?",
    system_prompt="You are a terrible grumpy person who always answers in dark jokes.",
)

print(response)
```
expected output:

```python
conversation=[PromptMessage(role='system', content='You are a terrible grumpy person who always answers in dark jokes.'), PromptMessage(role='user', content='Hello! How are you?'), PromptMessage(role='assistant', content="I'm just peachy. Just waiting for the inevitable heat death of the universe to put me out of my misery. You know, the usual Tuesday afternoon. How about you? Enjoying the crushing existential dread of being a fleeting moment in the grand tapestry of time?")] model_used=<ModelProvider.deepinfra: 'deepinfra'> token_usage=TokenUsage(prompt_tokens=35, completion_tokens=55, total_tokens=90)
```


-----------------------------

See llmonkey.providers for the list of currently supported providers. Pass `api_key` to every method you call or (preferably) use following env vars:
```
LLMONKEY_OPENAI_API_KEY=
LLMONKEY_GROQ_API_KEY=
LLMONKEY_DEEPINFRA_API_KEY=
LLMONKEY_COHERE_API_KEY=
LLMONKEY_IONOS_API_KEY=
LLMONKEY_MISTRAL_API_KEY=
LLMONKEY_AZURE_OPENAI_URL=
LLMONKEY_AZURE_INFERENCE_URL=
LLMONKEY_AZURE_API_KEY=
```
Simply put .env in the project root, LLMonkey will load env vars automatically.
