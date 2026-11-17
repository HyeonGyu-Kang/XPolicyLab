"""Blackboard: side storage for numpy artifacts the LLM must not see raw.

Tools store masks / point clouds / candidate pose lists here and hand the LLM
a short id ("mask_1"). Later tools take the id back and fetch the array.
"""
from __future__ import annotations

from typing import Any


class Blackboard:
    def __init__(self):
        self._store: dict[str, Any] = {}
        self._counter: dict[str, int] = {}

    def put(self, kind: str, value: Any, meta: dict | None = None) -> str:
        self._counter[kind] = self._counter.get(kind, 0) + 1
        key = f"{kind}_{self._counter[kind]}"
        self._store[key] = {"value": value, "meta": meta or {}}
        return key

    def get(self, key: str) -> Any:
        if key not in self._store:
            raise KeyError(f"unknown blackboard id '{key}'. Known: {sorted(self._store)}")
        return self._store[key]["value"]

    def meta(self, key: str) -> dict:
        return self._store[key]["meta"]

    def keys(self) -> list[str]:
        return sorted(self._store)
