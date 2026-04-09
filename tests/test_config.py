"""Tests for configuration loading behavior."""
from __future__ import annotations

from src.utils.config import load_config


def test_config_loader_handles_utf8_bom() -> None:
    cfg = load_config()
    assert cfg["project"]["name"] == "SHM-DSS"
    assert cfg.get("project.version") == "1.0.0"
