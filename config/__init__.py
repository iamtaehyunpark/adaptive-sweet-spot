"""Configuration loading."""

import tomllib
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = CONFIG_DIR / "default.toml"


def load_config(path: str | Path | None = None) -> dict:
    """Load a TOML config; defaults to config/local.toml if present, else default.toml."""
    if path is None:
        local = CONFIG_DIR / "local.toml"
        path = local if local.exists() else DEFAULT_CONFIG
    with open(path, "rb") as f:
        return tomllib.load(f)
