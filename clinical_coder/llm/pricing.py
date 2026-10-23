from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Price:
    input_per_mtok: float
    output_per_mtok: float


class PriceTable:
    def __init__(self, prices: dict[str, Price] | None = None, default: Price | None = None):
        self._prices = prices or {}
        self._default = default

    @classmethod
    def from_env(cls, path: str | None = None) -> "PriceTable":
        prices: dict[str, Price] = {}
        default: Price | None = None

        env_input = os.getenv("LLM_PRICE_INPUT_PER_MTOK")
        env_output = os.getenv("LLM_PRICE_OUTPUT_PER_MTOK")
        if env_input is not None and env_output is not None:
            default = Price(float(env_input), float(env_output))

        file_path = path or os.getenv("LLM_PRICES_FILE", "")
        if file_path and Path(file_path).is_file():
            raw = json.loads(Path(file_path).read_text(encoding="utf-8"))
            models = raw.get("models", raw)
            if isinstance(models, dict):
                for slug, value in models.items():
                    if not isinstance(value, dict):
                        continue
                    try:
                        prices[str(slug)] = Price(
                            float(value["input_per_mtok"]),
                            float(value["output_per_mtok"]),
                        )
                    except (KeyError, TypeError, ValueError):
                        continue
        return cls(prices, default)

    def price_for(self, model: str) -> Price | None:
        if model in self._prices:
            return self._prices[model]
        # Routers often return a provider-prefixed actual model; try the suffix.
        if "/" in model:
            suffix = model.rsplit("/", 1)[-1]
            if suffix in self._prices:
                return self._prices[suffix]
        return self._default

    def cost_usd(self, model: str, input_tokens: int, output_tokens: int) -> float | None:
        price = self.price_for(model)
        if price is None:
            return None
        return (
            input_tokens / 1_000_000 * price.input_per_mtok
            + output_tokens / 1_000_000 * price.output_per_mtok
        )
