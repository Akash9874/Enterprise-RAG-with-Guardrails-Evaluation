"""Model adapters for Tier B. Weights load lazily on first use (CLAUDE.md invariant 8)."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import numpy.typing as npt
import structlog

from rag.config import Settings
from rag.eval.metrics.generation import ScoredSentence
from rag.guardrails.rails.groundedness import split_sentences, strip_markers

log = structlog.get_logger(__name__)
_SINGLE_MARKER = re.compile(r"\[\d+\]")


def split_answer(text: str) -> list[tuple[str, list[str]]]:
    """Sentences with the markers each one carries, de-duplicated in order.

    Enforcement (FR-G4) already rewrote grouped markers to `[1][4]`, so single-marker
    matching is sufficient. A marker placed *after* the full stop attaches to the next
    sentence; that is a known limitation of sentence-level scoring (ADR-024).
    """
    return [
        (sentence, list(dict.fromkeys(_SINGLE_MARKER.findall(sentence))))
        for sentence in split_sentences(text)
    ]


class HHEMSupportScorer:
    """HHEM score of each sentence against each chunk it cites — never concatenated."""

    def __init__(self, settings: Settings, model: Any | None = None) -> None:
        self._name = settings.models.groundedness
        self._model = model

    @property
    def model(self) -> Any:
        if self._model is None:
            from transformers import AutoModelForSequenceClassification

            self._model = AutoModelForSequenceClassification.from_pretrained(
                self._name, trust_remote_code=True
            )
            log.info("hhem_loaded", model=self._name, purpose="tier_b")
        return self._model

    def score(
        self, sentences: list[tuple[str, list[str]]], chunk_by_marker: dict[str, str]
    ) -> list[ScoredSentence]:
        pairs: list[tuple[str, str]] = []
        slots: list[tuple[int, str]] = []
        for index, (sentence, markers) in enumerate(sentences):
            hypothesis = strip_markers(sentence)
            for marker in markers:
                if marker in chunk_by_marker:
                    pairs.append((chunk_by_marker[marker], hypothesis))
                    slots.append((index, marker))

        flat = [float(s) for s in self.model.predict(pairs)] if pairs else []
        support: list[dict[str, float]] = [{} for _ in sentences]
        for (index, marker), value in zip(slots, flat, strict=True):
            support[index][marker] = value

        return [
            ScoredSentence(text=sentence, cited_markers=markers, support=support[index])
            for index, (sentence, markers) in enumerate(sentences)
        ]


class TokenEmbedder:
    """Contextual token vectors for BERTScore: one hidden layer, special tokens removed."""

    def __init__(
        self, settings: Settings, model: Any | None = None, tokenizer: Any | None = None
    ) -> None:
        self.model_name = settings.models.bertscore
        self.layer = settings.eval.bertscore_layer
        self._model = model
        self._tokenizer = tokenizer

    def _load(self) -> None:
        from transformers import AutoModel, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)  # type: ignore[no-untyped-call]
        self._model = AutoModel.from_pretrained(self.model_name).eval()
        log.info("bertscore_model_loaded", model=self.model_name, layer=self.layer)

    def embed(self, texts: list[str]) -> list[npt.NDArray[np.float32]]:
        import torch

        if self._model is None or self._tokenizer is None:
            self._load()
        assert self._model is not None and self._tokenizer is not None

        vectors: list[npt.NDArray[np.float32]] = []
        for text in texts:
            encoded = self._tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                return_special_tokens_mask=True,
            )
            special = encoded.pop("special_tokens_mask")[0].bool()
            with torch.no_grad():
                output = self._model(**encoded, output_hidden_states=True)
            # hidden_states[0] is the embedding layer, so index L is the output of layer L.
            hidden = output.hidden_states[self.layer][0]
            vectors.append(hidden[~special].numpy().astype(np.float32))
        return vectors
