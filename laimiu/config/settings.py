"""Pydantic configuration model for Laimiu."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class ProviderModelConfig(BaseModel):
    """Configuration for a single LLM provider."""

    base_url: str
    model: str
    api_key_env: str = ""
    api_key: str = ""
    temperature: float = 0.7
    max_tokens: int = 4096


class ProviderConfig(BaseModel):
    """Provider routing configuration."""

    default: str = "deepseek"
    models: dict[str, ProviderModelConfig] = Field(default_factory=dict)


class AgentConfig(BaseModel):
    """Agent behavior configuration."""

    max_turns: int = 50
    temperature: float = 0.7


class DreamConfig(BaseModel):
    """Dream engine configuration."""

    enabled: bool = True
    trigger_after_sessions: int = 5
    trigger_after_hours: int = 24
    model: str = "ollama"


class ProceduralConfig(BaseModel):
    """Procedural memory (Layer 2) configuration."""

    enabled: bool = True
    extract_after_repeats: int = 3  # Legacy, kept for compat
    extract_strength: float = 0.6  # Strength threshold for extraction
    memory_half_life_days: float = 7.0  # Forgetting half-life
    success_bonus: float = 0.2  # Strength bonus on success
    failure_penalty: float = 0.3  # Strength penalty on failure
    auto_register: bool = True
    require_validation: bool = True


class MemoryConfig(BaseModel):
    """Memory system configuration."""

    index_max_lines: int = 200
    recall_max_chars: int = 2000
    vector_search_results: int = 5
    archive_after_days: int = 90


class AdaptationConfig(BaseModel):
    """Model adaptation (Layer 3) configuration."""

    enabled: bool = False
    trigger_interval_days: int = 14
    min_examples: int = 200
    model: str = "ollama"


class AgentRoleConfig(BaseModel):
    """Configuration for a single agent role."""

    provider: str = "deepseek"
    role: str = "brain"           # brain | worker | specialist
    router_task: str = "chat"     # chat | cheap | dream
    enabled: bool = True


class MultiAgentConfig(BaseModel):
    """Multi-agent orchestration configuration."""

    enabled: bool = False
    agents: dict[str, AgentRoleConfig] = Field(default_factory=dict)


class LaimiuConfig(BaseModel):
    """Root configuration for Laimiu."""

    provider: ProviderConfig = Field(default_factory=ProviderConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    dream: DreamConfig = Field(default_factory=DreamConfig)
    procedural: ProceduralConfig = Field(default_factory=ProceduralConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    adaptation: AdaptationConfig = Field(default_factory=AdaptationConfig)
    multi_agent: MultiAgentConfig = Field(default_factory=MultiAgentConfig)

    @classmethod
    def load(cls, path: Path) -> LaimiuConfig:
        """Load config from YAML file, falling back to defaults."""
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            # Resolve API keys from env vars
            models = data.get("provider", {}).get("models", {})
            for _name, mcfg in models.items():
                env_var = mcfg.get("api_key_env", "")
                if env_var and not mcfg.get("api_key"):
                    mcfg["api_key"] = os.environ.get(env_var, "")
            return cls(**data)
        return cls()

    def save(self, path: Path) -> None:
        """Save config to YAML file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, allow_unicode=True)

    def get_default_provider(self) -> tuple[str, ProviderModelConfig]:
        """Get the default provider name and config."""
        name = self.provider.default
        cfg = self.provider.models.get(name)
        if cfg is None:
            # Fall back to first available
            if self.provider.models:
                name = next(iter(self.provider.models))
                cfg = self.provider.models[name]
            else:
                raise ValueError("No providers configured")
        return name, cfg
