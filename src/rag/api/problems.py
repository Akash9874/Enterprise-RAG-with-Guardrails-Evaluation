"""RFC-7807 problem details for every error response (FR-A6).

Guardrail refusals never come through here — a refusal is HTTP 200 with the verdict in the
trace (CLAUDE.md invariant 5). This module is only for genuine request and server errors.
"""

from __future__ import annotations

from http import HTTPStatus
from typing import Any

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

PROBLEM_JSON = "application/problem+json"


def problem(status: int, detail: Any, instance: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        media_type=PROBLEM_JSON,
        content={
            "type": "about:blank",
            "title": HTTPStatus(status).phrase,
            "status": status,
            "detail": jsonable_encoder(detail),
            "instance": instance,
        },
    )


def install_problem_handlers(app: FastAPI) -> None:
    @app.exception_handler(StarletteHTTPException)
    async def _http_problem(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        return problem(exc.status_code, exc.detail, request.url.path)

    @app.exception_handler(RequestValidationError)
    async def _validation_problem(request: Request, exc: RequestValidationError) -> JSONResponse:
        return problem(422, exc.errors(), request.url.path)
