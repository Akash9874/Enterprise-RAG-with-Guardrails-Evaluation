"""POST /query — retrieve, generate, enforce citations.

A refusal is HTTP 200 with `refused: true` (FR-A6). Refusing is a correct outcome, and
reporting it as a transport error would make it indistinguishable from a broken service.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from rag.api.deps import get_guarded_answerer
from rag.contracts import Answer
from rag.guardrails.guarded import GuardedAnswerer

router = APIRouter()


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=50)
    rerank: bool | None = None
    include_trace: bool = False
    stream: bool = False

    @field_validator("query")
    @classmethod
    def reject_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be blank")
        return value


def _sse(answerer: GuardedAnswerer, request: QueryRequest) -> Iterator[str]:
    """Token events, then one terminal event carrying the enforced answer.

    Citation enforcement needs the completed text, so the terminal event — not the token
    stream — is the authoritative result. Phase 3 adds the trace to the same event.
    """
    for kind, payload in answerer.answer_stream(
        request.query, top_k=request.top_k, rerank=request.rerank
    ):
        if kind == "token":
            body = {"type": "token", "text": payload}
        else:
            body = {"type": "final", "answer": payload.model_dump(mode="json")}
        yield f"data: {json.dumps(body)}\n\n"


@router.post("/query", response_model=Answer)
def query(
    request: QueryRequest,
    answerer: Annotated[GuardedAnswerer, Depends(get_guarded_answerer)],
) -> Answer | StreamingResponse:
    if request.stream:
        return StreamingResponse(_sse(answerer, request), media_type="text/event-stream")

    answer = answerer.answer(request.query, top_k=request.top_k, rerank=request.rerank)
    if not request.include_trace:
        # The trace is verbose and carries rail evidence — matched spans, unsupported
        # sentences. Opt-in keeps the default response small and the evidence private.
        answer = answer.model_copy(update={"trace": None})
    return answer
