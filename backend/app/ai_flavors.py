"""AI VM flavor service: customer product (AI-8B, AI-70B, ...) on top of the sizing engine.

Layers stay separate:

    customer product (AIFlavorDefinition)  ->  sizing engine (engine.calculate)
        ->  resource requirement  ->  matcher (vGPU / full / multi)  ->  OpenStack flavor

This module never sizes anything itself. It chooses which tier's *policy* applies, calls the one
sizing engine, and explains how the result relates to the tier the customer picked (fits /
exceeds / can run smaller). It never silently switches the customer's flavor: the recommended
tier is returned next to the requested one with a message.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Optional

from .engine import Catalog, EngineSettings, calculate
from .engine.flavor import tier_for
from .schemas import AIFlavorDefinition, CalculationResult, FlavorRef, FlavorSizeRequest, FlavorSizeResult, ModelSpec

MODE_LABEL = {"vgpu": "Shared GPU", "full_gpu": "Dedicated GPU", "multi_gpu": "Multi-GPU"}


def enabled_tiers(tiers: list[AIFlavorDefinition]) -> list[AIFlavorDefinition]:
    return sorted((t for t in tiers if t.enabled), key=lambda t: t.parameter_ceiling_b)


def models_in_tier(tier: AIFlavorDefinition, models: list[ModelSpec], tiers: list[AIFlavorDefinition], settings: EngineSettings) -> list[str]:
    ordered = enabled_tiers(tiers)
    return [m.id for m in models if (t := tier_for(m.total_parameters_b, ordered, settings)) is not None and t.id == tier.id]


def _ref(t: Optional[AIFlavorDefinition]) -> Optional[FlavorRef]:
    return FlavorRef(id=t.id, display_name=t.display_name) if t else None


def _apply_policy(req: FlavorSizeRequest, tier: Optional[AIFlavorDefinition]) -> FlavorSizeRequest:
    if tier is None:
        return req
    return req.model_copy(
        update={
            "allow_vgpu": req.allow_vgpu and tier.allow_vgpu,
            "allow_single_gpu": req.allow_single_gpu and tier.allow_full_gpu,
            "allow_multi_gpu": req.allow_multi_gpu and tier.allow_multi_gpu,
        }
    )


def size_flavor(
    req: FlavorSizeRequest,
    model: ModelSpec,
    catalog: Catalog,
    settings: EngineSettings,
) -> FlavorSizeResult:
    tiers = enabled_tiers(catalog.ai_flavors)
    requested = next((t for t in tiers if t.id == req.flavor_id), None) if req.flavor_id else None
    model_tier = tier_for(model.total_parameters_b, tiers, settings)

    # Start at the model's own tier; move up only if that tier's infrastructure policy
    # (e.g. no multi-GPU in AI-14B) cannot host the workload.
    start = tiers.index(model_tier) if model_tier else len(tiers)
    chain: list[Optional[AIFlavorDefinition]] = tiers[start:] or [None]
    calc: Optional[CalculationResult] = None
    chosen: Optional[AIFlavorDefinition] = None
    for tier in chain:
        cat = replace(catalog, forced_tier=tier)
        calc = calculate(_apply_policy(req, tier), model, cat, settings, with_suggestions=True)
        chosen = tier
        if calc.recommendation is not None:
            break
    assert calc is not None
    if calc.recommendation is None:  # nothing fits under any tier policy: report the model's own tier
        chosen = model_tier
        calc = calculate(_apply_policy(req, model_tier), model, replace(catalog, forced_tier=model_tier), settings, with_suggestions=True)

    # ---------------------------------------------------------------- relation to the requested tier
    rank = {t.id: i for i, t in enumerate(tiers)}
    message = None
    if chosen is None:
        status = "no_tier"
        message = f"{model.name} ({model.total_parameters_b:g}B) is larger than the biggest AI flavor tier; sized as AI-XL."
    elif requested is None:
        status = "model_tier"
        message = f"{model.name} belongs to {chosen.display_name}."
        if model_tier is not None and chosen.id != model_tier.id:
            message = (
                f"{model.name} is {model_tier.display_name}-class, but this workload needs infrastructure that "
                f"{model_tier.display_name} does not offer; recommended {chosen.display_name}."
            )
    elif rank[chosen.id] > rank[requested.id]:
        status = "exceeds"
        if model_tier is not None and rank[model_tier.id] > rank[requested.id]:
            message = f"This model exceeds the normal {requested.display_name} tier. Recommended: {chosen.display_name}."
        else:
            message = (
                f"This workload exceeds {requested.display_name}: it needs infrastructure "
                f"(e.g. multi-GPU) that {requested.display_name} does not offer. Recommended: {chosen.display_name}."
            )
    elif rank[chosen.id] < rank[requested.id]:
        status = "oversized"
        message = (
            f"This workload can run on a smaller flavor: {chosen.display_name}. "
            "Potential benefit: lower GPU consumption and lower expected cost."
        )
    else:
        status = "fits"

    rec = calc.recommendation
    mem = calc.memory
    allocated = None
    vram_note = None
    if rec is not None:
        c = rec.candidate
        gpu = next((g for g in catalog.gpus if g.id == rec.gpu_id), None)
        profile = next((p for p in catalog.profiles if p.profile_name == rec.vgpu_profile), None)
        if profile:
            allocated = profile.vram_gb
        elif gpu:
            allocated = gpu.vram_gb * c.count
    if chosen is not None:
        if mem.required_vram_gb > chosen.default_vram_max_gb:
            vram_note = (
                f"Above {chosen.display_name}'s typical {chosen.default_vram_min_gb:g}–{chosen.default_vram_max_gb:g} GB "
                "because of your context length and concurrency."
            )
        elif mem.required_vram_gb < chosen.default_vram_min_gb:
            vram_note = f"Below {chosen.display_name}'s typical {chosen.default_vram_min_gb:g}–{chosen.default_vram_max_gb:g} GB range."

    sizing = {
        "model_weight_gb": mem.model_weights_gb,
        "kv_cache_gb": mem.kv_cache_gb,
        "runtime_overhead_gb": mem.runtime_overhead_gb,
        "workspace_gb": mem.workspace_gb,
        "safety_margin_gb": mem.headroom_gb,
        "required_vram_gb": mem.required_vram_gb,
        "compute_class": calc.compute.classification,
        "memory_bandwidth_class": calc.compute.bandwidth_class,
        "recommended_gpu_count": rec.count if rec else None,
        "recommended_gpu_mode": rec.mode if rec else None,
        "multi_gpu_required": bool(rec and rec.count > 1),
        "recommended_ai_flavor": rec.ai_flavor if rec else None,
    }
    return FlavorSizeResult(
        requested_flavor=_ref(requested),
        model_flavor=_ref(model_tier),
        recommended_tier=_ref(chosen),
        tier_status=status,
        tier_message=message,
        recommended_flavor=rec.ai_flavor if rec else None,
        display_flavor=rec.ai_flavor_short if rec else None,
        variant=rec.flavor_variant if rec else None,
        mode_label=MODE_LABEL[rec.mode] if rec else None,
        allocated_gpu_memory_gb=allocated,
        vram_range_note=vram_note,
        sizing=sizing,
        calculation=calc,
    )
