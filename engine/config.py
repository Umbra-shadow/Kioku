"""Kioku configuration — every tunable constant lives here, documented.

Secrets come from the environment (``.env`` is loaded once, never logged).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_dotenv(path: Path | None = None) -> None:
    """Tiny .env loader: KEY=VALUE lines, '#' comments, no expansion.
    Existing environment variables always win."""
    path = path or REPO_ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass(frozen=True, slots=True)
class LLMConfig:
    """One OpenAI-compatible endpoint. Qwen Cloud is the default brain."""

    base_url: str
    api_key: str
    model: str
    embed_model: str
    provider: str = "qwen"
    timeout_s: float = 60.0
    max_retries: int = 3


@dataclass(frozen=True, slots=True)
class Settings:
    llm: LLMConfig
    data_dir: Path
    # Curiosity loop (§3.4): max self-research lookups per turn; async, never
    # blocks the reply.
    curiosity_max_lookups: int = 3
    # Memory pack (§4): strict token budget for recalled context.
    pack_token_budget: int = 1200
    # Retrieval scoring (§4): score = α·similarity + β·importance +
    # γ·recency_decay + δ·access_frequency.
    score_alpha: float = 0.55
    score_beta: float = 0.20
    score_gamma: float = 0.15
    score_delta: float = 0.10
    # Forgetting (§5): retention = importance · e^(−λ·age_days) · log(1+access).
    # λ per memory class — preferences decay slowest, small talk fastest.
    lambda_per_class: dict[str, float] = field(
        default_factory=lambda: {
            "preference": 0.005,
            "semantic": 0.02,
            "episodic": 0.08,
            "smalltalk": 0.5,
        }
    )


def _llm_from_env() -> LLMConfig:
    qwen_key = os.environ.get("QWEN_API_KEY", "")
    generic_key = os.environ.get("GENERIC_API_KEY", "")
    if not qwen_key and generic_key:
        # Secondary mode: any OpenAI-compatible key gets a Kioku memory.
        return LLMConfig(
            base_url=os.environ.get("GENERIC_BASE_URL", "").rstrip("/"),
            api_key=generic_key,
            model=os.environ.get("GENERIC_MODEL", "gpt-4o-mini"),
            embed_model=os.environ.get("GENERIC_EMBED_MODEL", "text-embedding-3-small"),
            provider="generic",
        )
    return LLMConfig(
        base_url=os.environ.get(
            "QWEN_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
        ).rstrip("/"),
        api_key=qwen_key,
        model=os.environ.get("QWEN_MODEL", "qwen-max"),
        embed_model=os.environ.get("QWEN_EMBED_MODEL", "text-embedding-v3"),
        provider="qwen",
    )


@lru_cache(maxsize=1)
def settings() -> Settings:
    load_dotenv()
    return Settings(
        llm=_llm_from_env(),
        data_dir=Path(os.environ.get("KIOKU_DATA_DIR", REPO_ROOT / "kioku_data")),
        curiosity_max_lookups=int(os.environ.get("KIOKU_CURIOSITY_MAX_LOOKUPS", "3")),
        pack_token_budget=int(os.environ.get("KIOKU_PACK_TOKEN_BUDGET", "1200")),
    )
