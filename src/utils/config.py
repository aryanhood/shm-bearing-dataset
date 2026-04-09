"""
Config loader
=============
Loads ``configs/config.yaml`` once and exposes a shared proxy so every module
reads the same live configuration.

The proxy keeps imports stable while still allowing entrypoints such as the
CLI to swap in a different config path before work starts.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import yaml

_ROOT = Path(__file__).resolve().parents[2]   # project root
_DEFAULT_PATH = _ROOT / "configs" / "config.yaml"
_ACTIVE_CONFIG: "Config | None" = None
_ACTIVE_CONFIG_PATH: Path | None = None


class Config(dict):
    """Thin wrapper that adds dot-notation *get* to a plain dict."""

    def get(self, key: str, default: Any = None) -> Any:  # type: ignore[override]
        """Support 'section.key' look-ups in addition to plain key."""
        parts = key.split(".")
        node: Any = self
        for part in parts:
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node


def load_config(path: str | Path | None = None) -> Config:
    """
    Load and return the project config.

    Parameters
    ----------
    path:
        Explicit path to a YAML config file.
        Defaults to configs/config.yaml relative to the project root.
    """
    path = Path(path) if path else _DEFAULT_PATH
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open(encoding="utf-8-sig") as fh:
        raw = yaml.safe_load(fh)
    return Config(raw)


def init_config(path: str | Path | None = None, *, force: bool = False) -> Config:
    """Initialise the shared config singleton and return it."""
    global _ACTIVE_CONFIG, _ACTIVE_CONFIG_PATH

    resolved = Path(path) if path else _DEFAULT_PATH
    if force or _ACTIVE_CONFIG is None or _ACTIVE_CONFIG_PATH != resolved:
        _ACTIVE_CONFIG = load_config(resolved)
        _ACTIVE_CONFIG_PATH = resolved
    return _ACTIVE_CONFIG


def get_config(path: str | Path | None = None) -> Config:
    """Return the active config, initialising it on first access."""
    return init_config(path)


def get_config_path() -> Path:
    """Return the filesystem path of the active config."""
    init_config()
    assert _ACTIVE_CONFIG_PATH is not None
    return _ACTIVE_CONFIG_PATH


class ConfigProxy:
    """Proxy object so imported ``CFG`` always resolves to the active config."""

    def _cfg(self) -> Config:
        return get_config()

    def __getitem__(self, key: str) -> Any:
        return self._cfg()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._cfg())

    def __len__(self) -> int:
        return len(self._cfg())

    def __contains__(self, item: object) -> bool:
        return item in self._cfg()

    def get(self, key: str, default: Any = None) -> Any:
        return self._cfg().get(key, default)

    def items(self):
        return self._cfg().items()

    def keys(self):
        return self._cfg().keys()

    def values(self):
        return self._cfg().values()


CFG = ConfigProxy()
