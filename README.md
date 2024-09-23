# LLMonkey: a simple-to-use wrapper around multiple LLM providers

Sample usage: see client.py

Minimal example:
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

See llmonkey.providers for the list of currently supported providers. Pass `api_key` to every method you call or (preferably) use following env vars:
```
LLMONKEY_OPENAI_API_KEY=
LLMONKEY_GROQ_API_KEY=
LLMONKEY_DEEPINFRA_API_KEY=
LLMONKEY_COHERE_API_KEY=
```
Simply put .env in the project root, LLMonkey will load env vars automatically.
