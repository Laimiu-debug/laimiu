"""Adaptation engine - orchestrates model fine-tuning (Layer 3)."""

from __future__ import annotations

import logging
from typing import Any

from laimiu.config.settings import AdaptationConfig
from laimiu.constants import ADAPTERS_DIR

logger = logging.getLogger("laimiu.adaptation.engine")


class AdaptationEngine:
    """Manages model adaptation through QLoRA fine-tuning.

    MVP: Interface only. Actual training requires Unsloth + GPU.
    """

    def __init__(self, config: AdaptationConfig):
        self.config = config
        self.adapters_dir = ADAPTERS_DIR
        self.adapters_dir.mkdir(parents=True, exist_ok=True)

    async def run_adaptation(self, dataset_path: str | None = None) -> dict[str, Any]:
        """Run the full adaptation pipeline.

        Pipeline:
        1. Build dataset from conversations
        2. Train QLoRA adapter with Unsloth
        3. Export to GGUF
        4. Import into Ollama
        5. Hot-swap the model

        Returns summary of the adaptation process.
        """
        if not self.config.enabled:
            return {"status": "disabled", "message": "Model adaptation is disabled"}

        return {
            "status": "not_implemented",
            "message": "Layer 3 adaptation will be implemented in a future phase. "
            "Requires Unsloth + GPU for QLoRA training.",
            "config": {
                "trigger_interval_days": self.config.trigger_interval_days,
                "min_examples": self.config.min_examples,
                "model": self.config.model,
            },
        }

    def check_readiness(self, conversation_count: int) -> dict[str, Any]:
        """Check if we have enough data for adaptation."""
        ready = conversation_count >= self.config.min_examples
        return {
            "ready": ready,
            "conversations": conversation_count,
            "min_required": self.config.min_examples,
            "progress": min(1.0, conversation_count / max(self.config.min_examples, 1)),
        }

    def list_adapters(self) -> list[dict[str, Any]]:
        """List available LoRA adapter versions."""
        adapters = []
        for adapter_dir in sorted(self.adapters_dir.iterdir()):
            if adapter_dir.is_dir():
                adapters.append({
                    "version": adapter_dir.name,
                    "path": str(adapter_dir),
                    "size": sum(f.stat().st_size for f in adapter_dir.rglob("*") if f.is_file()),
                })
        return adapters
