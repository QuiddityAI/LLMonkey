from llmonkey.llmonkey import LLMonkey

llmonkey = LLMonkey()

print("Available providers:", llmonkey.providers)
print("Using OpenAI")

response = llmonkey.generate_chat_response(
    provider="openai",
    model_name="gpt-3.5-turbo",
    user_prompt="Hello! How are you?",
    system_prompt="You are a terrible grumpy person who always answers in dark jokes.",
)

print(response)

print("Using Groq")
response = llmonkey.generate_chat_response(
    provider="groq",
    model_name="llama-3.1-70b-versatile",
    user_prompt="Hello! How are you?",
    system_prompt="You are a terrible grumpy person who always answers in dark jokes.",
)

print(response)

print("Using DeepInfra")
response = llmonkey.generate_chat_response(
    provider="deepinfra",
    model_name="meta-llama/Meta-Llama-3.1-70B-Instruct",
    user_prompt="Hello! How are you?",
    system_prompt="You are a terrible grumpy person who always answers in dark jokes.",
)
print(response)
