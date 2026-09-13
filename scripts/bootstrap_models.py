"""Pull the generation model and warm caches. Run once after `uv sync`.

Usage:  uv run python scripts/bootstrap_models.py           # pull the Ollama generator
        uv run python scripts/bootstrap_models.py --warm    # download + load every encoder

In Docker:  docker compose run --rm api python scripts/bootstrap_models.py --warm
"""

from __future__ import annotations

import sys

import ollama

from rag.config import get_settings


def warm_encoders() -> None:
    """Download and load every encoder once, so the first real request pays no download.

    Runs the pipeline's own rails over a throwaway input and output, so exactly the models
    a request will need are the ones cached — nothing guessed from a list.
    """
    from rag.contracts import Chunk, RailContext, Retrieved
    from rag.guardrails.factory import build_pipeline, cached_centroid_provider
    from rag.guardrails.policy import load_policy
    from rag.models.embedder import Embedder

    settings = get_settings()
    embedder = Embedder(settings)
    embedder.embed_documents(["warm"])
    embedder.embed_sparse(["warm"])
    pipeline = build_pipeline(
        settings,
        load_policy(),
        embedder=embedder,
        centroid_provider=cached_centroid_provider(),
        judge=None,
    )
    # Input rails: heuristics, PII (spaCy), injection classifier, topicality.
    pipeline.run_input(RailContext(request_id="warm", query="How does retrieval work?"))
    chunk = Chunk(
        chunk_id="warm",
        doc_id="warm",
        text="RRF fuses rankings.",
        source_path="warm.md",
        language="markdown",
    )
    # Output rails: PII leak scan and HHEM groundedness.
    pipeline.run_output(
        RailContext(
            request_id="warm",
            query="q",
            answer="RRF fuses rankings.",
            retrieved=[Retrieved(chunk=chunk)],
        )
    )
    print("OK: encoders warmed")


def main() -> int:
    if "--warm" in sys.argv:
        # The container warms its own cache; the generator lives with host Ollama (ADR-028).
        warm_encoders()
        return 0

    settings = get_settings()
    model = settings.models.generator

    client = ollama.Client(host=settings.ollama.host)
    try:
        listed = client.list()
    except Exception as exc:  # noqa: BLE001 - bootstrap reports, never crashes
        print(f"ERROR: cannot reach Ollama at {settings.ollama.host}: {exc}")
        print("Install from https://ollama.com/download, then re-run this script.")
        return 1

    if model in {m.get("model", "") for m in listed.get("models", [])}:
        print(f"OK: {model} already present")
        return 0

    print(f"Pulling {model} (~2.0 GB). This runs once.")
    for progress in client.pull(model, stream=True):
        status = progress.get("status", "")
        completed = progress.get("completed")
        total = progress.get("total")
        if completed and total:
            pct = 100 * completed / total
            print(f"\r  {status}: {pct:5.1f}%", end="", flush=True)
    print(f"\nOK: pulled {model}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
