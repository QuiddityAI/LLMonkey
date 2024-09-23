from pydantic import BaseModel

from llmonkey.llmonkey import LLMonkey
from llmonkey.models import PromptMessage
from llmonkey.utils.decorators import validate_llm_output

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


class SeaCreature(BaseModel):
    specie: str
    description: str
    depth_of_habitat_meters: float
    size_in_meters: float
    number_of_tentacles: int


@validate_llm_output(model=SeaCreature, retries=3)
def get_random_sea_creature() -> SeaCreature:
    """example of using validate_llm_output decorator
    The decorator will parse the ChatResponse object (function must return it)
    according to the Pydantic model provided. The parsed results will be
    returned as tuple (parsed_model, raw_response)"""
    response = llmonkey.generate_prompt_response(
        provider="groq",
        model_name="llama-3.1-70b-versatile",
        user_prompt=f"Generate a random sea creature, according to the schema below:\n {SeaCreature.schema()}",
        system_prompt="You are a data generator. You always output user requested data as JSON.\
        You never return anything except machine-readable JSON.",
    )
    return response


for i in range(5):
    print(get_random_sea_creature()[0].dict())

res = llmonkey.generate_structured_response(
    provider="groq",
    model_name="llama-3.1-70b-versatile",
    user_prompt=f"Generate a random Lovecraftian creature, according to the schema below:\n {SeaCreature.schema()}",
    system_prompt="You are a data generator. You always output user requested data as JSON.\
        You never return anything except machine-readable JSON.",
    data_model=SeaCreature,
)

print(res[0])
