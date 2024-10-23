# LLMonkey: a simple-to-use wrapper around multiple LLM providers

Sample usage: see client.py

Minimal example:

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

### New model-centered interface


Alternatively, you can use model-centered interface

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

-----------------------------

See llmonkey.providers for the list of currently supported providers. Pass `api_key` to every method you call or (preferably) use following env vars:
```
LLMONKEY_OPENAI_API_KEY=
LLMONKEY_GROQ_API_KEY=
LLMONKEY_DEEPINFRA_API_KEY=
LLMONKEY_COHERE_API_KEY=
LLMONKEY_IONOS_API_KEY=
LLMONKEY_MISTRAL_API_KEY=
```
Simply put .env in the project root, LLMonkey will load env vars automatically.
