from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class LLMConfig:
    model: str
    api_key: str | None = None
    api_base: str | None = None


def load_llm_config() -> LLMConfig:
    return LLMConfig(
        model=os.getenv("AGENT_LLM_MODEL", "gpt-4o-mini"),
        api_key=os.getenv("OPENAI_API_KEY"),
        api_base=os.getenv("OPENAI_API_BASE"),
    )
