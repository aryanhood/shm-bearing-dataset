"""Small thread-safe TTL cache for inference responses."""
from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Generic, TypeVar

K = TypeVar("K")
V = TypeVar("V")


@dataclass(frozen=True)
class CacheSnapshot:
    hits: int
    misses: int
    evictions: int
    entries: int
    max_entries: int
    ttl_seconds: float


class TTLCache(Generic[K, V]):
    """Ordered TTL cache with deterministic eviction behavior."""

    def __init__(self, *, max_entries: int, ttl_seconds: float) -> None:
        if max_entries <= 0:
            raise ValueError("max_entries must be > 0")
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be > 0")

        self.max_entries = int(max_entries)
        self.ttl_seconds = float(ttl_seconds)
        self._data: OrderedDict[K, tuple[float, V]] = OrderedDict()
        self._lock = threading.Lock()

        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get(self, key: K) -> V | None:
        now = time.monotonic()
        with self._lock:
            item = self._data.get(key)
            if item is None:
                self._misses += 1
                return None

            expires_at, value = item
            if expires_at < now:
                self._data.pop(key, None)
                self._misses += 1
                return None

            self._data.move_to_end(key)
            self._hits += 1
            return value

    def put(self, key: K, value: V) -> None:
        now = time.monotonic()
        expires_at = now + self.ttl_seconds
        with self._lock:
            self._data[key] = (expires_at, value)
            self._data.move_to_end(key)
            self._drop_expired_locked(now)
            while len(self._data) > self.max_entries:
                self._data.popitem(last=False)
                self._evictions += 1

    def snapshot(self) -> CacheSnapshot:
        with self._lock:
            self._drop_expired_locked(time.monotonic())
            return CacheSnapshot(
                hits=self._hits,
                misses=self._misses,
                evictions=self._evictions,
                entries=len(self._data),
                max_entries=self.max_entries,
                ttl_seconds=self.ttl_seconds,
            )

    def _drop_expired_locked(self, now: float) -> None:
        expired_keys = [key for key, (expires_at, _) in self._data.items() if expires_at < now]
        for key in expired_keys:
            self._data.pop(key, None)
