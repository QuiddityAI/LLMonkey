from enum import Enum
from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class ModelProvider(str, Enum):
    openai = "openai"
    groq = "groq"
    deepinfra = "deepinfra"
    cohere = "cohere"
    ionos = "ionos"
    mistral = "mistral"
    nebius = "nebius"
    # for the future:
    # self_hosted = "self_hosted"


class PromptMessage(BaseModel):
    role: Literal["system", "user", "assistant"] = Field(
        ..., description="Role of the prompt, either 'system', 'user', or 'assistant'"
    )
    content: str = Field(..., description="Message content")
    image: Optional[Union[str, bytes]] = Field(
        None, description="Image URL or raw bytes"
    )


class TokenUsage(BaseModel):
    prompt_tokens: Optional[int] = Field(
        None, description="Number of tokens used in the prompt"
    )
    completion_tokens: Optional[int] = Field(
        None, description="Number of tokens used in the completion"
    )
    total_tokens: Optional[int] = Field(None, description="Total number of tokens used")
    search_units: Optional[int] = Field(
        None, description="Number of search units used e.g. in cohere reranking"
    )
    total_cost: Optional[float] = Field(None, description="Total price of the request")


class LLMonkeyResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    provider_used: ModelProvider = Field(
        ..., description="Provider used to generate this response"
    )
    model_used: str = Field(..., description="Model used to generate this response")
    token_usage: TokenUsage = Field(..., description="Token usage details")


class LLMonkeyRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_provider: ModelProvider = (Field(..., description="Model provider to use"),)
    model_name: str = Field(..., description="Model to use")


class ChatRequest(LLMonkeyRequest):
    conversation: List[PromptMessage] = Field(..., description="A list of previous")
    temperature: Optional[float] = Field(
        1.0, ge=0.0, le=2.0, description="Temperature of the model"
    )
    max_tokens: Optional[int] = Field(..., description="Maximum tokens to generate")


class EmbeddingRequest(LLMonkeyRequest):
    text: str = Field(..., description="Text to generate embeddings from")


class ChatResponse(LLMonkeyResponse):
    conversation: List[PromptMessage] = Field(
        ..., description="A list of previous PromptMessages"
    )


class EmbeddingResponse(LLMonkeyResponse):
    embedding: List[float] = Field(..., description="Text embeddings")


class RerankItem(BaseModel):
    index: int = Field(..., description="The index of the reranked document")
    score: float = Field(..., description="The score of the reranked document")


class RerankRequest(LLMonkeyRequest):
    query: str = Field(..., description="The search query")
    documents: List[str] | Dict[str, str] = Field(
        ..., description="List of documents to rerank"
    )
    top_n: Optional[int] = Field(
        None, description="Number of most relevant documents to return"
    )
    rank_fields: Optional[List[str]] = Field(
        None, description="Fields to rank documents on, only if documents is a dict"
    )
    max_chunks_per_doc: Optional[int] = Field(
        None, description="Maximum number of chunks per document"
    )


class RerankResponse(LLMonkeyResponse):
    reranked_documents: List[RerankItem] = Field(
        ..., description="List of reranked documents"
    )


class ModelCapabilities(str, Enum):
    chat = "chat"
    embeddings = "embeddings"
    rerank = "rerank"
    vision = "vision"


class ModelConfig(BaseModel):
    identifier: str = Field(..., description="Identifier of the model")
    verbose_name: str = Field(..., description="Verbose name of the model")
    description: str = Field(..., description="Description of the model")
    max_input_tokens: int = Field(..., description="Maximum number of tokens in input")
    euro_per_1M_input_tokens: float = Field(..., description="Cost per 1M input tokens")
    euro_per_1M_output_tokens: float = Field(
        ..., description="Cost per 1M output tokens"
    )
    capabilities: List[ModelCapabilities] = Field(..., description="Model capabilities")
