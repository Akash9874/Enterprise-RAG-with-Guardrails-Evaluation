from __future__ import annotations

from fastapi import FastAPI

import rag
from rag.api.problems import install_problem_handlers
from rag.api.routes import corpus, health, query


def create_app() -> FastAPI:
    app = FastAPI(
        title="Enterprise RAG",
        description="CPU-only RAG with tiered guardrails and deterministic evaluation",
        version=rag.__version__,
    )
    install_problem_handlers(app)
    app.include_router(health.router)
    app.include_router(query.router)
    app.include_router(corpus.router)
    return app


app = create_app()
