import pytest
from pydantic import BaseModel, ValidationError

from ..llmonkey import LLMonkey
from ..models import PromptMessage
from ..utils.decorators import validate_llm_output


@pytest.fixture
def llmonkey():
    return LLMonkey()


def test_openai_response(llmonkey):
    response = llmonkey.generate_prompt_response(
        provider="openai",
        model_name="gpt-3.5-turbo",
        user_prompt="Hello! How are you?",
        system_prompt="You are a terrible grumpy person who always answers in dark jokes.",
    )

    assert response.model_used == "openai"
    assert len(response.conversation) > 0
    assert response.token_usage


def test_groq_response(llmonkey):
    response = llmonkey.generate_prompt_response(
        provider="groq",
        model_name="llama-3.1-70b-versatile",
        user_prompt="Hello! How are you?",
        system_prompt="You are a terrible grumpy person who always answers in dark jokes.",
        max_tokens=1000,
    )

    assert response.model_used == "groq"
    assert len(response.conversation) > 0
    assert response.token_usage


def test_groq_chat(llmonkey):
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

    assert response.model_used == "groq"
    assert len(response.conversation) > 0
    assert response.token_usage


def test_deepinfra_response(llmonkey):
    response = llmonkey.generate_prompt_response(
        provider="deepinfra",
        model_name="meta-llama/Meta-Llama-3.1-70B-Instruct",
        user_prompt="Hello! How are you?",
        system_prompt="You are a terrible grumpy person who always answers in dark jokes.",
    )

    assert response.model_used == "deepinfra"
    assert len(response.conversation) > 0
    assert response.token_usage


def test_bad_provider(llmonkey):
    with pytest.raises(ValidationError):
        llmonkey.generate_prompt_response(
            provider="test",
            model_name="meta-llama/Meta-Llama-3.1-70B-Instruct",
            user_prompt="Hello! How are you?",
            system_prompt="You are a terrible grumpy person who always answers in dark jokes.",
        )


def test_validate_llm_output_decorator(llmonkey):
    class SeaCreature(BaseModel):
        specie: str
        description: str
        depth_of_habitat_meters: float
        size_in_meters: float
        number_of_tentacles: int

    @validate_llm_output(model=SeaCreature, retries=3)
    def generate_llm_data(user_prompt: str) -> SeaCreature:
        response = llmonkey.generate_prompt_response(
            provider="groq",
            model_name="llama-3.1-70b-versatile",
            user_prompt=user_prompt,
            system_prompt="You are a data generator. You always output user requested data as JSON.\
            You never return anything except machine-readable JSON.",
        )
        return response

    prompt = f"Generate a random sea creature, according to the schema below:\n {SeaCreature.model_json_schema()}"
    assert isinstance(generate_llm_data(prompt)[0], SeaCreature)

    with pytest.raises(ValueError):
        bad_prompt = "Be friendly, always ask user if they liked it. " + prompt
        generate_llm_data(bad_prompt)[0]
