import os

import cohere

from ..models import (
    ChatRequest,
    ChatResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    RerankItem,
    RerankRequest,
    RerankResponse,
    TokenUsage,
)
from .base import BaseModelProvider


class CohereProvider(BaseModelProvider):
    def __init__(self, api_key: str = ""):
        if not api_key:
            api_key = os.environ.get("LLMONKEY_COHERE_API_KEY")
        self.client = cohere.Client(api_key)

    def generate_prompt_response(self, request: ChatRequest) -> ChatResponse:
        raise NotImplementedError

    def generate_chat_response(self, request: ChatRequest) -> ChatResponse:
        raise NotImplementedError

    def get_embedding(self, request: EmbeddingRequest) -> EmbeddingResponse:
        raise NotImplementedError

    def rerank(self, request: RerankRequest) -> RerankResponse:
        resp = self.client.rerank(
            model=request.model_name,
            query=request.query,
            documents=request.documents,
            top_n=request.top_n,
            rank_fields=request.rank_fields if request.rank_fields else None,
            max_chunks_per_doc=request.max_chunks_per_doc,
        )

        return RerankResponse(
            reranked_documents=[
                RerankItem(index=item.index, score=item.relevance_score)
                for item in resp.results
            ],
            token_usage=TokenUsage(search_units=resp.meta.billed_units.search_units),
            provider_used=request.model_provider,
            model_used=request.model_name,
        )
