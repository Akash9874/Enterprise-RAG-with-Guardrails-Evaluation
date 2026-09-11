"""Pull the generation model and warm caches. Run once after `uv sync`.

Usage:  uv run python scripts/bootstrap_models.py
"""

from __future__ import annotations

import sys

import ollama

from rag.config import get_settings


def main() -> int:
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
