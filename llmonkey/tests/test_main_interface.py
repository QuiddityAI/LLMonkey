import pytest
from pydantic import ValidationError

from ..llmonkey import LLMonkey
from ..models import PromptMessage


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
