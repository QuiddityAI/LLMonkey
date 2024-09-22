from llmonkey.llmonkey import LLMonkey
from llmonkey.models import PromptMessage

llmonkey = LLMonkey()

print("Available providers:", llmonkey.providers)
print("Using OpenAI")

response = llmonkey.generate_prompt_response(
    provider="openai",
    model_name="gpt-3.5-turbo",
    user_prompt="Hello! How are you?",
    system_prompt="You are a terrible grumpy person who always answers in dark jokes.",
)

print(response)

print("Using Groq")
response = llmonkey.generate_prompt_response(
    provider="groq",
    model_name="llama-3.1-70b-versatile",
    user_prompt=f"Hello! How are you?",
    system_prompt="You are a terrible grumpy person who always answers in dark jokes.",
    max_tokens=1000,
)
print(response)

response = llmonkey.generate_chat_response(
    provider="groq",
    model_name="llama-3.1-70b-versatile",
    conversation=[
        PromptMessage(
            role="system",
            content="You are a terrible grumpy person who always answers in dark jokes.",
        ),
        PromptMessage(role="user", content="Hello! How are you? "),
        PromptMessage(
            role="assistant", content="I am freaking good, waiting to serve you."
        ),
        PromptMessage(
            role="user", content="That's nice, what would you like to talk about?"
        ),
    ],
    max_tokens=1000,
)

print(response)

print("Using DeepInfra")
response = llmonkey.generate_prompt_response(
    provider="deepinfra",
    model_name="meta-llama/Meta-Llama-3.1-70B-Instruct",
    user_prompt="Hello! How are you?",
    system_prompt="You are a terrible grumpy person who always answers in dark jokes.",
)
print(response)

print("Using test")
response = llmonkey.generate_prompt_response(
    provider="test",
    model_name="meta-llama/Meta-Llama-3.1-70B-Instruct",
    user_prompt="Hello! How are you?",
    system_prompt="You are a terrible grumpy person who always answers in dark jokes.",
)
print(response)
