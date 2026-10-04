"""Harness settings, loaded from environment variables / `.env`.

Nothing in this codebase should call `os.environ` directly -- go through
`get_settings()` so every setting has one documented place, a type, and
(where sensible) a default.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    anthropic_api_key: str = ""
    anthropic_model: str = "claude-haiku-4-5-20251001"

    jev_api_key: str = ""
    jev_base_url: str = "https://api.typesafe.ai"
    jev_model: str = "jev-latest"

    cache_dir: str = "runs/cache"
    seed: int = 0

    # Project 1's raw paragraphs, read-only -- confidence-audit never writes here.
    risk_topic_paragraphs_path: str = "../10k-analyst/data/processed/paragraphs.jsonl"
    risk_topic_gold_path: str = "data/gold/risk_topic_sample.jsonl"
    # Confirmed with Hugo: hold out these two tickers entirely for the OOD split.
    risk_topic_ood_tickers: tuple[str, ...] = ("PFE", "V")


@lru_cache
def get_settings() -> Settings:
    return Settings()
