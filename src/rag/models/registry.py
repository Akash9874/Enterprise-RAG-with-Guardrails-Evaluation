"""Lazy model loading with LRU eviction.

Never instantiate a transformer at module scope. Import-time construction breaks the
resident-memory budget (PRD NFR-4) and slows every test collection.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from typing import Any

import structlog

log = structlog.get_logger(__name__)


class ModelRegistry:
    def __init__(self, max_resident: int = 4) -> None:
        self._loaders: dict[str, Callable[[], Any]] = {}
        self._resident: OrderedDict[str, Any] = OrderedDict()
        self._max_resident = max_resident

    def register(self, name: str, loader: Callable[[], Any]) -> None:
        """Record how to build a model. Does not build it."""
        self._loaders[name] = loader

    def get(self, name: str) -> Any:
        if name in self._resident:
            self._resident.move_to_end(name)
            return self._resident[name]

        if name not in self._loaders:
            raise KeyError(f"no loader registered for model {name!r}")

        log.info("model_loading", model=name)
        model = self._loaders[name]()
        self._resident[name] = model
        self._evict_if_needed()
        return model

    def resident(self) -> list[str]:
        return list(self._resident.keys())

    def evict_all(self) -> None:
        self._resident.clear()

    def _evict_if_needed(self) -> None:
        while len(self._resident) > self._max_resident:
            name, _ = self._resident.popitem(last=False)
            log.info("model_evicted", model=name)
