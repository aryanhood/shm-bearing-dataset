"""Unit tests for the inference TTL cache."""
from __future__ import annotations

import time

from src.inference.cache import TTLCache


def test_cache_get_put_roundtrip() -> None:
    cache = TTLCache[str, int](max_entries=2, ttl_seconds=1.0)
    cache.put("a", 10)
    assert cache.get("a") == 10
    stats = cache.snapshot()
    assert stats.hits == 1
    assert stats.entries == 1


def test_cache_evicts_oldest_entry() -> None:
    cache = TTLCache[str, int](max_entries=2, ttl_seconds=10.0)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("c", 3)
    assert cache.get("a") is None
    assert cache.get("c") == 3
    stats = cache.snapshot()
    assert stats.evictions >= 1


def test_cache_expires_entries() -> None:
    cache = TTLCache[str, int](max_entries=2, ttl_seconds=0.05)
    cache.put("a", 1)
    time.sleep(0.08)
    assert cache.get("a") is None
    stats = cache.snapshot()
    assert stats.entries == 0
