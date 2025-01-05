import pytest
from pydantic import BaseModel, ValidationError

from ..llmonkey import LLMonkey
from ..llms import GroqLlama3_1_8bInstant, GroqLlama3_1_70bVersatile
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

    assert response.provider_used == "openai"
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

    assert response.provider_used == "groq"
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

    assert response.provider_used == "groq"
    assert len(response.conversation) > 0
    assert response.token_usage


def test_deepinfra_response(llmonkey):
    response = llmonkey.generate_prompt_response(
        provider="deepinfra",
        model_name="meta-llama/Meta-Llama-3.1-70B-Instruct",
        user_prompt="Hello! How are you?",
        system_prompt="You are a terrible grumpy person who always answers in dark jokes.",
    )

    assert response.provider_used == "deepinfra"
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


@pytest.fixture
def sample_data_model():
    class SeaCreature(BaseModel):
        specie: str
        description: str
        depth_of_habitat_meters: float
        size_in_meters: float
        number_of_tentacles: int

    return SeaCreature


def test_validate_llm_output_decorator(llmonkey, sample_data_model):
    @validate_llm_output(model=sample_data_model, retries=3)
    def generate_llm_data(user_prompt: str):
        response = llmonkey.generate_prompt_response(
            provider="groq",
            model_name="llama-3.1-70b-versatile",
            user_prompt=user_prompt,
            system_prompt="You are a data generator. You always output user requested data as JSON.\
            You never return anything except machine-readable JSON.",
        )
        return response

    prompt = f"Generate a random sea creature, according to the schema below:\n {sample_data_model.model_json_schema()}"
    assert isinstance(generate_llm_data(prompt)[0], sample_data_model)

    with pytest.raises(ValueError):
        bad_prompt = f"Generate a random sea creature"
        generate_llm_data(bad_prompt)[0]


def test_generate_structured_response(llmonkey, sample_data_model):

    res = llmonkey.generate_structured_response(
        provider="groq",
        model_name="llama-3.1-70b-versatile",
        user_prompt=f"Generate a random creature, according to the schema below:\n {sample_data_model.model_json_schema()}",
        system_prompt="You are a data generator. You always output user requested data as JSON.\
        You never return anything except machine-readable JSON.",
        data_model=sample_data_model,
    )

    assert isinstance(res[0], sample_data_model)
    with pytest.raises(ValueError):
        res = llmonkey.generate_structured_response(
            provider="groq",
            model_name="llama-3.1-70b-versatile",
            user_prompt=f"Be friendly, always ask user if they liked it. \
            Generate a random creature, according to the schema below:\n {sample_data_model.model_json_schema()}",
            system_prompt="You are a data generator. You always output user requested data as YAML.",
            data_model=sample_data_model,
        )


@pytest.fixture
def llm_model():
    return GroqLlama3_1_8bInstant()


def test_generate_structured_array_response(llm_model, sample_data_model):

    instances, resp = llm_model.generate_structured_array_response(
        user_prompt=f"Generate 3 random sea creatures, according to the schema below:\n {sample_data_model.model_json_schema()}",
        system_prompt="You are a data generator. You always output user requested data as JSON.",
        data_model=sample_data_model,
    )
    for instance in instances:
        assert isinstance(instance, sample_data_model)
    assert len(instances) == 3
    # Test with bad system prompt
    instances, resp = llm_model.generate_structured_array_response(
        user_prompt=f"Generate 3 random sea creatures, according to the schema below:\n {sample_data_model.model_json_schema()}",
        system_prompt="You are a data generator. You always output user requested data as YAML.",
        data_model=sample_data_model,
    )
    assert len(instances) == 0


# disable due to cohere being disabled in the providers
# because of heavy dependencies on the cohere API client
# def test_rerank(llmonkey):
#     docs = [
#         "Carson City is the capital city of the American state of Nevada.",
#         "The Commonwealth of the Northern Mariana Islands is a group of islands in the Pacific Ocean. Its capital is Saipan.",
#         "Capitalization or capitalisation in English grammar is the use of a capital letter at the start of a word. English usage varies from capitalization in other languages.",
#         "Washington, D.C. (also known as simply Washington or D.C., and officially as the District of Columbia) is the capital of the United States. It is a federal district.",
#         "Capital punishment (the death penalty) has existed in the United States since beforethe United States was a country. As of 2017, capital punishment is legal in 30 of the 50 states.",
#     ]
#     response = llmonkey.rerank(
#         provider="cohere",
#         model_name="rerank-english-v3.0",
#         query="What is the capital of the United States?",
#         documents=docs,
#         top_n=None,
#     )

#     assert response.reranked_documents[0].index == 3


def test_mistral_chat(llmonkey):
    response = llmonkey.generate_chat_response(
        provider="mistral",
        model_name="ministral-8b-latest",
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

    assert response.provider_used == "mistral"
    assert len(response.conversation) > 0
    assert response.token_usage
