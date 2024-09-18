from enum import Enum
from typing import List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class ModelProvider(str, Enum):
    openai = "openai"
    groq = "groq"
    deepinfra = "deepinfra"
    # for the future:
    # self_hosted = "self_hosted"


class PromptMessage(BaseModel):
    role: Literal["system", "user", "assistant"] = Field(
        ..., description="Role of the prompt, either 'system', 'user', or 'assistant'"
    )
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_provider: ModelProvider = Field(..., description="Model provider to use")
    model_name: str = Field(..., description="Model to use")
    conversation: List[PromptMessage] = Field(..., description="A list of previous")
    temperature: Optional[float] = Field(
        1.0, ge=0.0, le=2.0, description="Temperature of the model"
    )
    max_tokens: Optional[int] = Field(..., description="Maximum tokens to generate")


class EmbeddingRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_provider: ModelProvider = (Field(..., description="Model provider to use"),)
    model_name: str = Field(..., description="Model to use")
    text: str = Field(..., description="Text to generate embeddings from")


class TokenUsage(BaseModel):
    prompt_tokens: Optional[int] = Field(
        ..., description="Number of tokens used in the prompt"
    )
    completion_tokens: Optional[int] = Field(
        ..., description="Number of tokens used in the completion"
    )
    total_tokens: Optional[int] = Field(..., description="Total number of tokens used")


class ChatResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    conversation: List[PromptMessage] = Field(
        ..., description="A list of previous PromptMessages"
    )
    model_used: ModelProvider = Field(
        ..., description="Model used to generate this response"
    )
    token_usage: TokenUsage = Field(..., description="Token usage details")


class EmbeddingResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    embedding: List[float] = Field(..., description="Text embeddings")
    model_used: ModelProvider = Field(
        ..., description="Model used to generate this embedding"
    )
