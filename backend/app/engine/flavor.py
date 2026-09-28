"""Customer-facing AI flavor naming.

    AI-70B-Q4-PRO      full name (tier + precision + variant)
    AI-70B-PRO         short name (precision omitted when it equals the tier's baseline precision)
    ai.llm.70b.int4.pro / ai-70b-q4-pro   catalog / OpenStack flavor names

Variant follows the recommended infrastructure: SHARED (vGPU), PRO (one dedicated GPU),
MULTI (multi-GPU). The tier comes from the AI flavor catalog (smallest tier whose parameter
ceiling, plus tolerance, covers the model); settings.flavor_size_classes is the fallback when no
catalog is loaded. Custom ML workloads use AI-CUSTOM-{INF|TRAIN|FT}-{memory} instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..schemas import AIFlavorDefinition, ModelSpec
from .config import EngineSettings

VARIANTS = {"vgpu": "SHARED", "full_gpu": "PRO", "multi_gpu": "MULTI"}


@dataclass(frozen=True)
class Flavor:
    ai_flavor: str
    short_name: str
    flavor_name: str
    slug: str
    variant: str
    tier_id: Optional[str]
    tier_label: str  # "AI-70B" or "AI-XL"


def tier_for(params_b: float, tiers: list[AIFlavorDefinition], settings: EngineSettings) -> Optional[AIFlavorDefinition]:
    """Smallest enabled tier whose ceiling (+ tolerance) covers the parameter count."""
    tolerance = 1 + settings.flavor_tier_tolerance_percent / 100
    for t in sorted((t for t in tiers if t.enabled), key=lambda t: t.parameter_ceiling_b):
        if params_b <= t.parameter_ceiling_b * tolerance:
            return t
    return None


def _token(tier: AIFlavorDefinition) -> str:
    """ai-70b -> 70B (naming token independent of the editable display name)."""
    return tier.id.removeprefix("ai-").upper()


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
    tiers: Optional[list[AIFlavorDefinition]] = None,
    forced_tier: Optional[AIFlavorDefinition] = None,
) -> Flavor:
    """`forced_tier` names the flavor after a tier chosen by the flavor service (e.g. an upgrade
    because the smaller tier's policy cannot host the workload)."""
    tag = settings.precision[precision].flavor_tag
    variant = VARIANTS[kind]
    tier = forced_tier or (tier_for(model.total_parameters_b, tiers, settings) if tiers else None)
    if tier is not None:
        token = _token(tier)
        baseline = tier.baseline_precision
    elif tiers:
        token, baseline = "XL", None  # larger than the largest catalog tier
    else:
        token, baseline = size_class(model.total_parameters_b, settings), "int4"
    short = f"AI-{token}-{variant}" if precision == baseline else f"AI-{token}-{tag}-{variant}"
    return Flavor(
        ai_flavor=f"AI-{token}-{tag}-{variant}",
        short_name=short,
        flavor_name=f"ai.llm.{token.lower()}.{precision}.{variant.lower()}",
        slug=f"ai-{token.lower()}-{tag.lower()}-{variant.lower()}",
        variant=variant,
        tier_id=tier.id if tier else None,
        tier_label=f"AI-{token}",
    )


def custom_ml_flavor(task: str, kind: str, count: int, memory_gb: float) -> tuple[str, str, str]:
    """AI-CUSTOM-INF-24G / AI-CUSTOM-TRAIN-2X80G, a readable name, and an OpenStack slug."""
    task_tag = {"inference": "INF", "training": "TRAIN", "fine_tuning": "FT"}[task]
    size = f"{round(memory_gb)}G"
    multi = count > 1 and kind == "multi_gpu"
    code = f"AI-CUSTOM-{task_tag}-{count}X{size}" if multi else f"AI-CUSTOM-{task_tag}-{size}"
    task_label = {"inference": "Inference", "training": "Training", "fine_tuning": "Fine-tuning"}[task]
    gpu_label = f"{count} × {round(memory_gb)} GB GPUs" if multi else f"{round(memory_gb)} GB {'shared ' if kind == 'vgpu' else ''}GPU"
    return code, f"Custom ML — {task_label} — {gpu_label}", code.lower()
