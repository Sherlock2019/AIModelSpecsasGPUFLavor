"""Customer-facing AI flavor naming: AI-70B-Q4-PROD / ai.llm.70b.int4.prod / ai-70b-q4-prod."""

from __future__ import annotations

from dataclasses import dataclass

from ..schemas import ModelSpec
from .config import EngineSettings, PriorityConfig


@dataclass(frozen=True)
class Flavor:
    ai_flavor: str  # AI-70B-Q4-PROD
    flavor_name: str  # ai.llm.70b.int4.prod
    slug: str  # ai-70b-q4-prod (OpenStack flavor name)


def _b(value: float) -> str:
    # Round half to even so 30.5B -> "30B", matching the published "Qwen3-30B-A3B" name.
    return f"{round(value)}B" if value >= 1 else f"{value:g}B"


def size_class(total_parameters_b: float, settings: EngineSettings) -> str:
    for cls in settings.flavor_size_classes:
        if total_parameters_b <= cls.max_total_b:
            return cls.label
    return "XL"


def ai_flavor(
    model: ModelSpec,
    precision: str,
    kind: str,
    settings: EngineSettings,
    priority: PriorityConfig,
    environment: str,
) -> Flavor:
    tag = settings.precision[precision].flavor_tag
    if model.architecture == "moe":
        base = f"MOE-{_b(model.total_parameters_b)}-A{_b(model.active_parameters_b)}"
    else:
        base = size_class(model.total_parameters_b, settings)
    if model.metadata_status == "user_defined":
        base = f"CUSTOM-{base}"
    if kind == "vgpu":
        tier = "shared"
    elif environment == "development":
        tier = "dev"
    else:
        tier = priority.flavor_tier
    return Flavor(
        ai_flavor=f"AI-{base}-{tag}-{tier.upper()}",
        flavor_name=f"ai.llm.{base.lower()}.{precision}.{tier.lower()}",
        slug=f"ai-{base.lower()}-{tag.lower()}-{tier.lower()}",
    )
