"""Orchestrates a full sizing calculation: model -> memory -> compute -> GPU -> AI flavor."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..schemas import (
    BenchmarkRecord,
    CalculateRequest,
    CalculationResult,
    Candidate,
    GPUSpec,
    KVDetail,
    ModelSpec,
    ModelSummary,
    Recommendation,
    ResolvedWorkload,
    Suggestion,
    TraceStep,
    VGPUProfile,
    Warning_,
    WeightDetail,
    WorkloadInput,
)
from . import compute as compute_mod
from .config import EngineSettings
from .flavor import ai_flavor
from .memory import SizingError, kv_cache, per_gpu_breakdown, resolve_kv_precision, weight_memory
from .openstack import openstack_flavor, to_yaml
from .selection import Capacity, SelectionContext, generate, pick_alternatives
from .units import fmt_tokens, gb, gib, r


@dataclass
class Catalog:
    """Snapshot of everything the engine needs besides the request."""

    gpus: list[GPUSpec]
    profiles: list[VGPUProfile]
    capacity: dict[str, Capacity] = field(default_factory=dict)
    has_inventory: bool = False
    benchmarks: list[BenchmarkRecord] = field(default_factory=list)


def _warn(out: list[Warning_], code: str, severity: str, message: str) -> None:
    out.append(Warning_(code=code, severity=severity, message=message))


def calculate(
    req: WorkloadInput,
    model: ModelSpec,
    catalog: Catalog,
    settings: EngineSettings,
    with_suggestions: bool = False,
) -> CalculationResult:
    result = _calculate(req, model, catalog, settings)
    if with_suggestions:
        result.suggestions = what_if(req, model, catalog, settings, result)
    return result


# Next smaller serving precision to try in what-if suggestions.
_SMALLER_PRECISION = {"fp32": "bf16", "bf16": "fp8", "fp16": "fp8", "fp8": "int4", "int8": "int4"}


def what_if(
    req: WorkloadInput, model: ModelSpec, catalog: Catalog, settings: EngineSettings, base: CalculationResult
) -> list[Suggestion]:
    """Variants of the request that land on less hardware (fewer GPU units, or less provisioned VRAM)."""
    variants: list[tuple[str, str, dict, str]] = []
    if base.workload.kv_cache_precision in ("fp16", "bf16"):
        variants.append(
            (
                "kv_fp8",
                "Use an FP8 KV cache",
                {"kv_cache_precision": "fp8"},
                "Halves KV-cache memory. Enable with --kv-cache-dtype fp8 (vLLM, TensorRT-LLM); small accuracy impact.",
            )
        )
    smaller = _SMALLER_PRECISION.get(req.precision)
    if smaller in settings.precision and (
        not model.supported_serving_precisions or smaller in model.supported_serving_precisions
    ):
        variants.append(
            (
                "precision",
                f"Serve at {settings.precision[smaller].label}",
                {"precision": smaller},
                "Quantization can reduce output quality; validate on your own evaluation set.",
            )
        )
    if req.context_mode == "max":
        variants.append(
            (
                "realistic",
                "Size for typical request lengths",
                {"context_mode": "realistic"},
                "Assumes average-length requests; bursts of very long prompts may queue.",
            )
        )
    if req.concurrent_sequences >= 4:
        half = req.concurrent_sequences // 2
        variants.append(
            (
                "concurrency",
                f"Serve {half} concurrent requests per instance",
                {"concurrent_sequences": half},
                "Run more replicas behind a load balancer to keep the same total capacity.",
            )
        )
    if base.recommendation is None and (not req.allow_multi_gpu or req.max_gpu_count < 16):
        variants.append(
            (
                "multi_gpu",
                "Allow up to 16 GPUs",
                {"allow_multi_gpu": True, "max_gpu_count": 16},
                "Needs tensor and pipeline parallelism, possibly across nodes.",
            )
        )

    out: list[Suggestion] = []
    base_c = base.recommendation.candidate if base.recommendation else None
    for key, title, patch, tradeoff in variants:
        try:
            res = _calculate(req.model_copy(update=patch), model, catalog, settings)
        except SizingError:
            continue
        rec = res.recommendation
        if rec is None:
            continue
        c = rec.candidate
        if base_c is not None and (c.gpu_units, c.vram_per_unit_gb * c.count) >= (
            base_c.gpu_units,
            base_c.vram_per_unit_gb * base_c.count,
        ):
            continue  # no hardware saving
        out.append(
            Suggestion(
                key=key,
                title=title,
                tradeoff=tradeoff,
                patch=patch,
                required_vram_gb=res.memory.required_vram_gb,
                recommendation=rec.summary,
                ai_flavor=rec.ai_flavor,
                saves_gb=r(base.memory.required_vram_gb - res.memory.required_vram_gb, 1),
            )
        )
    return out


def _calculate(req: WorkloadInput, model: ModelSpec, catalog: Catalog, settings: EngineSettings) -> CalculationResult:
    warnings: list[Warning_] = []
    trace: list[TraceStep] = []

    # ------------------------------------------------------------------ resolve inputs
    if req.precision not in settings.precision:
        raise SizingError(f"Unknown precision '{req.precision}'")
    framework = settings.frameworks.get(req.framework)
    if framework is None:
        raise SizingError(f"Unknown framework '{req.framework}'. Known: {', '.join(settings.frameworks)}")
    priority = settings.priorities[req.performance_priority]
    precision_cfg = settings.precision[req.precision]
    if model.max_context_length and req.context_length > model.max_context_length:
        raise SizingError(
            f"Context {req.context_length:,} exceeds {model.name}'s maximum of {model.max_context_length:,} tokens"
        )
    kv_precision = resolve_kv_precision(req.kv_cache_precision, req.precision, settings)

    avg_in = (
        req.average_input_tokens
        if req.average_input_tokens is not None
        else settings.context.default_average_input_tokens
    )
    avg_out = (
        req.average_output_tokens
        if req.average_output_tokens is not None
        else settings.context.default_average_output_tokens
    )
    if req.context_mode == "realistic" and avg_in + avg_out > req.context_length:
        _warn(
            warnings,
            "CONTEXT_CLAMPED",
            "info",
            f"Average request ({avg_in + avg_out:,} tokens) is longer than the {fmt_tokens(req.context_length)} "
            "context; the effective sequence is capped at the context length.",
        )

    if not req.add_safety_headroom:
        safety = 0.0
    elif req.safety_margin_percent is not None:
        safety = req.safety_margin_percent
    elif req.environment == "development":
        safety = settings.environments["development"].default_safety_margin_percent
    else:
        safety = priority.safety_margin_percent

    per_user = priority.tokens_per_second_per_user
    if req.target_tokens_per_second_per_user:
        per_user = req.target_tokens_per_second_per_user
    elif req.tpot_target_ms:
        per_user = 1000.0 / req.tpot_target_ms

    # ------------------------------------------------------------------ weights
    w = weight_memory(model, req.precision, settings, req.quantization_overhead_factor)
    trace.append(
        TraceStep(
            step="Model weights",
            formula=(
                f"{w.parameters_b:g}B params × {w.bytes_per_parameter:g} B/param ({precision_cfg.label}) "
                f"× {w.overhead_factor:g} quantization overhead"
            ),
            value=f"{gb(w.bytes):.2f} GB ({gib(w.bytes):.2f} GiB); raw {gb(w.raw_bytes):.2f} GB",
        )
    )

    # ------------------------------------------------------------------ KV cache
    kv = kv_cache(model, kv_precision, req.context_length, req.concurrent_sequences, req.context_mode, avg_in, avg_out, settings)
    if req.context_mode == "max":
        tokens_formula = f"{fmt_tokens(req.context_length)} context × {req.concurrent_sequences} sequences (worst case)"
    else:
        tokens_formula = (
            f"max({fmt_tokens(req.context_length)} context, {kv.tokens_per_sequence:,} effective tokens "
            f"× {req.concurrent_sequences} sequences)"
        )
    trace.append(TraceStep(step="KV cache bytes per token", formula=kv.formula, value=f"{kv.bytes_per_token / 1024:.1f} KiB/token"))
    trace.append(
        TraceStep(
            step="KV cache",
            formula=f"{kv.bytes_per_token:,.0f} B/token × {kv.sized_tokens:,} tokens = {tokens_formula}",
            value=f"{gb(kv.bytes):.2f} GB",
        )
    )

    # ------------------------------------------------------------------ totals (single GPU view)
    base = per_gpu_breakdown(
        w.bytes, kv, 1, framework, safety, req.runtime_overhead_override_gb, req.workspace_percent_override
    )
    trace.append(
        TraceStep(
            step="Runtime + workspace",
            formula=(
                f"{framework.label}: {gb(base.runtime):g} GB base + "
                f"{(req.workspace_percent_override if req.workspace_percent_override is not None else framework.workspace_percent):g}% "
                "of (weights + KV)"
            ),
            value=f"{gb(base.runtime + base.workspace):.2f} GB",
        )
    )
    trace.append(
        TraceStep(
            step="Required VRAM",
            formula=f"subtotal {gb(base.subtotal):.2f} GB × (1 + {safety:g}% safety margin)",
            value=f"{gb(base.required):.2f} GB ({gib(base.required):.2f} GiB)",
        )
    )

    # ------------------------------------------------------------------ compute demand
    d = compute_mod.demand(
        model, w.bytes, kv.bytes, req.concurrent_sequences, avg_in, avg_out, per_user, req.target_aggregate_tokens_per_second
    )
    cprof = compute_mod.compute_profile(model, d, base.required, settings)
    trace.append(
        TraceStep(
            step="Compute demand (relative)",
            formula=f"{model.active_parameters_b:g}B active params × {d.decode_tps:,.0f} decode tokens/s",
            value=f"{cprof.compute_demand_score:,.0f} → {cprof.classification.replace('_', ' ').upper()}",
        )
    )
    trace.append(
        TraceStep(
            step="Decode bandwidth demand (roofline)",
            formula=(
                f"({gb(w.bytes):.1f} GB weights × {d.moe_fraction:.2f} read/step + {gb(kv.bytes):.1f} GB KV) "
                f"× {d.tokens_per_second_per_user:g} steps/s"
            ),
            value=f"{cprof.required_bandwidth_gbps:,.0f} GB/s",
        )
    )

    # ------------------------------------------------------------------ GPU selection
    use_bench = req.calculation_mode != "estimate"
    ctx = SelectionContext(
        model=model,
        req=req,
        settings=settings,
        priority=priority,
        framework=framework,
        precision_cfg=precision_cfg,
        weights_bytes=w.bytes,
        kv=kv,
        safety_pct=safety,
        demand=d,
        avg_input=avg_in,
        gpus=catalog.gpus,
        profiles=catalog.profiles,
        capacity=catalog.capacity,
        has_inventory=catalog.has_inventory,
        benchmarks=catalog.benchmarks,
        use_benchmarks=use_bench,
    )
    sel = generate(ctx)
    best: Optional[Candidate] = sel.ranked[0] if sel.ranked else None
    alternatives = pick_alternatives(sel.ranked)

    # ------------------------------------------------------------------ recommendation
    recommendation = None
    openstack = openstack_yaml = None
    gpus_by_id = {g.id: g for g in catalog.gpus}
    profiles_by_name = {p.profile_name: p for p in catalog.profiles}
    if best is not None:
        fl = ai_flavor(model, req.precision, best.kind, settings, priority, req.environment)
        mode_label = {"vgpu": "vGPU", "full_gpu": "Dedicated GPU", "multi_gpu": "Multi-GPU (tensor parallel)"}[best.kind]
        recommendation = Recommendation(
            ai_flavor=fl.ai_flavor,
            flavor_name=fl.flavor_name,
            gpu=best.gpu_name,
            gpu_id=best.gpu_id,
            count=best.count,
            mode=best.kind,
            vgpu_profile=best.vgpu_profile,
            tensor_parallel=best.tensor_parallel,
            utilization_percent=best.utilization_percent,
            headroom_gb=best.headroom_gb,
            confidence=best.confidence,
            summary=best.label if best.kind == "vgpu" else f"{best.label} — {mode_label}",
            candidate=best,
        )
        openstack = openstack_flavor(
            best,
            fl.slug,
            model,
            gpus_by_id[best.gpu_id],
            profiles_by_name.get(best.vgpu_profile) if best.vgpu_profile else None,
            req.precision,
            gb(w.bytes),
            req.context_length,
            req.concurrent_sequences,
            req.framework,
            req.performance_priority,
            settings,
        )
        openstack_yaml = to_yaml(openstack)

    calc_mode = "benchmark" if best is not None and best.confidence == "benchmark" else "estimate"

    vgpu_verdict = _vgpu_verdict(req, priority.label, best, sel, gb(base.required))

    _collect_warnings(warnings, req, model, kv, w, base, best, sel, calc_mode, settings)
    explanation = _explain(model, req, precision_cfg.label, framework.label, w, kv, base, safety, best, sel, cprof)

    summary = ModelSummary(
        id=model.id,
        name=model.name,
        family=model.family,
        vendor=model.vendor,
        architecture=model.architecture,
        total_parameters_b=model.total_parameters_b,
        active_parameters_b=model.active_parameters_b,
        max_context_length=model.max_context_length,
        metadata_status=model.metadata_status,
        source_url=model.source_url,
        kv_metadata=kv.method,
    )
    workload = ResolvedWorkload(
        precision=req.precision,
        kv_cache_precision=kv_precision,
        context_length=req.context_length,
        context_mode=req.context_mode,
        concurrent_sequences=req.concurrent_sequences,
        average_input_tokens=avg_in,
        average_output_tokens=avg_out,
        performance_priority=req.performance_priority,
        framework=req.framework,
        environment=req.environment,
        safety_margin_percent=safety,
        tokens_per_second_per_user=r(d.tokens_per_second_per_user, 1),
    )
    return CalculationResult(
        calculation_mode=calc_mode,
        confidence_label="BENCHMARK-BASED" if calc_mode == "benchmark" else "ESTIMATED",
        model=summary,
        workload=workload,
        weights=WeightDetail(
            parameters_b=w.parameters_b,
            bytes_per_parameter=w.bytes_per_parameter,
            quantization_overhead_factor=w.overhead_factor,
            raw_gb=r(gb(w.raw_bytes)),
            gb=r(gb(w.bytes)),
            gib=r(gib(w.bytes)),
        ),
        kv_cache=KVDetail(
            method=kv.method,
            kv_precision=kv_precision,
            bytes_per_element=kv.bytes_per_element,
            bytes_per_token=r(kv.bytes_per_token, 1),
            head_dim=kv.head_dim,
            context_mode=req.context_mode,
            tokens_per_sequence=kv.tokens_per_sequence,
            sized_tokens=kv.sized_tokens,
            gb=r(gb(kv.bytes)),
            gib=r(gib(kv.bytes)),
            formula=kv.formula,
        ),
        memory=base.to_schema("gb"),
        memory_gib=base.to_schema("gib"),
        compute=cprof,
        recommendation=recommendation,
        matches=sel.ranked[:15],
        alternatives=alternatives,
        rejected=sel.rejected[:12],
        vgpu_verdict=vgpu_verdict,
        warnings=warnings,
        explanation=explanation,
        openstack=openstack,
        openstack_yaml=openstack_yaml,
        trace=trace,
    )


# ---------------------------------------------------------------------------- helpers


def _profile_gib(c: Candidate) -> str:
    """vGPU framebuffers are defined in GiB (e.g. 16C = 16 GiB); show them that way."""
    return f"{c.vram_per_unit_gb * 1e9 / 1024**3:.0f} GiB"


def _vgpu_verdict(req: WorkloadInput, priority_label: str, best, sel, required_gb: float) -> str:
    if best is not None and best.kind == "vgpu":
        return f"Recommended: {best.vgpu_profile} ({_profile_gib(best)}) — a full physical GPU is not required."
    if not req.allow_vgpu:
        return "Not considered: vGPU disabled in options."
    if req.dedicated_gpu_required:
        return "Not considered: dedicated GPU required."
    if (req.tensor_parallel_size or 1) > 1:
        return "Not considered: tensor parallelism requires full GPUs."
    fitting_vgpu = [c for c in sel.ranked if c.kind == "vgpu"]
    if fitting_vgpu:
        return (
            f"Possible ({fitting_vgpu[0].label}), but a dedicated option ranked higher for "
            f"{priority_label} priority."
        )
    if sel.vgpu_too_small:
        largest = max(sel.vgpu_too_small, key=lambda c: c.vram_per_unit_gb)
        return (
            f"Not recommended for this workload: needs ~{required_gb:.0f} GB; the largest supported profile is "
            f"{largest.vgpu_profile} ({_profile_gib(largest)})."
        )
    if sel.vgpu_policy_blocked:
        return f"Not recommended for {priority_label} priority: shared GPUs cannot guarantee throughput."
    return "No vGPU profiles in the catalog for the selected GPUs."


def _collect_warnings(warnings, req, model, kv, w, base, best, sel, calc_mode, settings) -> None:
    if kv.method == "heuristic":
        _warn(
            warnings,
            "KV_HEURISTIC",
            "warning",
            "KV cache calculated using estimate mode: layer / KV-head metadata is not available for this model.",
        )
    if kv.method == "mla":
        _warn(warnings, "KV_MLA", "info", "Multi-head Latent Attention: KV cache stores compressed latents (much smaller than standard attention).")
    if model.sliding_window and kv.method == "exact":
        _warn(
            warnings,
            "KV_SLIDING_WINDOW",
            "info",
            f"Model uses sliding-window attention ({model.sliding_window:,} tokens) on some layers; the KV figure assumes "
            "full attention on every layer and is conservative.",
        )
    if model.architecture == "moe":
        _warn(
            warnings,
            "MOE_TOTAL_PARAMS",
            "info",
            f"MoE model memory requirement is based on total parameters ({model.total_parameters_b:g}B), not active "
            f"parameters ({model.active_parameters_b:g}B). Compute per token scales with active parameters.",
        )
    if model.supported_serving_precisions and req.precision not in model.supported_serving_precisions:
        _warn(
            warnings,
            "PRECISION_UNSUPPORTED",
            "warning",
            f"{req.precision.upper()} is not listed as a supported serving precision for {model.name} "
            f"(listed: {', '.join(p.upper() for p in model.supported_serving_precisions)}).",
        )
    native_sizes = [settings.precision[p].bytes_per_parameter for p in model.native_precision if p in settings.precision]
    if native_sizes:
        native_bytes = min(native_sizes)
        if settings.precision[req.precision].bytes_per_parameter > native_bytes:
            _warn(
                warnings,
                "PRECISION_ABOVE_NATIVE",
                "info",
                f"Serving above the native {'/'.join(p.upper() for p in model.native_precision)} precision increases memory with no quality gain.",
            )
    if model.metadata_status == "user_defined":
        _warn(warnings, "USER_DEFINED_MODEL", "info", "Custom model: results depend entirely on the metadata you entered.")
    if kv.bytes > 0.5 * w.bytes:
        _warn(
            warnings,
            "KV_DOMINANT",
            "warning",
            f"Requested context length and concurrency substantially increase KV cache: {gb(kv.bytes):.1f} GB "
            f"({kv.bytes / w.bytes * 100:.0f}% of model weights). Consider FP8 KV cache or lower concurrency.",
        )
    single_weight_only = [c for c in sel.rejected if c.kind == "full_gpu" and c.count == 1 and c.fits_weights_only]
    if single_weight_only:
        c = max(single_weight_only, key=lambda c: c.vram_per_unit_gb)
        _warn(
            warnings,
            "FITS_WEIGHTS_NOT_KV",
            "warning",
            f"Model fits by weights but not after KV-cache allocation on a single {c.gpu_name} "
            f"({c.required_per_unit_gb:.1f} GB needed, {c.vram_per_unit_gb:.1f} GB usable).",
        )
    if best is None:
        _warn(
            warnings,
            "NO_FIT",
            "critical",
            "No GPU configuration in the catalog fits this workload within the allowed GPU count. Lower precision, "
            "context or concurrency, or allow more GPUs.",
        )
        return
    free_after_subtotal = (best.vram_per_unit_gb - best.memory_per_unit.subtotal_gb) / best.vram_per_unit_gb
    if free_after_subtotal < 0.10:
        _warn(
            warnings,
            "LOW_HEADROOM",
            "warning",
            f"Model barely fits {best.label}. Production headroom is below 10% ({free_after_subtotal * 100:.1f}%).",
        )
    if best.compute_ratio is not None and best.compute_ratio < 1:
        _warn(
            warnings,
            "COMPUTE_BOUND",
            "warning",
            "High concurrency may make this GPU compute-bound even though VRAM fits "
            f"(estimated {best.compute_ratio * 100:.0f}% of required prefill+decode compute).",
        )
    if best.bandwidth_ratio is not None and best.bandwidth_ratio < 1 and best.est_decode_tokens_per_second_per_user:
        _warn(
            warnings,
            "BANDWIDTH_BOUND",
            "warning",
            f"Estimated decode ceiling is ~{best.est_decode_tokens_per_second_per_user:.0f} tokens/s per sequence at this "
            "concurrency, below the target. Choose Performance priority or allow more GPUs.",
        )
    if req.ttft_target_ms and best.est_ttft_ms and best.est_ttft_ms > req.ttft_target_ms:
        _warn(
            warnings,
            "TTFT_TARGET",
            "warning",
            f"Estimated time-to-first-token ~{best.est_ttft_ms:.0f} ms exceeds the {req.ttft_target_ms:.0f} ms target.",
        )
    if best.kind == "vgpu":
        _warn(warnings, "VGPU_PERF", "warning", "vGPU profile fits memory but performance is not guaranteed.")
    if best.count > settings.gpus_per_node:
        _warn(warnings, "MULTI_NODE", "warning", "Recommendation spans multiple nodes; validate interconnect and serving-stack support.")
    if best.availability in ("insufficient", "not_in_inventory", "unavailable"):
        text = {
            "insufficient": f"Only {best.available_units} of the required GPUs are currently available in inventory.",
            "not_in_inventory": f"{best.gpu_name} is not in the site inventory.",
            "unavailable": f"{best.gpu_name} is marked unavailable in the catalog.",
        }[best.availability]
        _warn(warnings, "INVENTORY", "warning", text)
    if calc_mode == "estimate":
        if req.calculation_mode == "benchmark":
            _warn(warnings, "BENCHMARK_MISSING", "warning", "Benchmark mode requested but no matching benchmark exists; fell back to estimate mode.")
        _warn(warnings, "NO_BENCHMARK", "info", "Benchmark data unavailable; recommendation is formula-based.")


def _explain(model, req, prec_label, fw_label, w, kv, base, safety, best, sel, cprof) -> list[str]:
    lines: list[str] = []
    if model.architecture == "moe":
        lines.append(
            f"{model.name} is a Mixture-of-Experts model with {model.total_parameters_b:g}B total parameters, of which "
            f"~{model.active_parameters_b:g}B are active per token. All experts must sit in GPU memory, so memory is "
            f"sized on {model.total_parameters_b:g}B; compute is sized on {model.active_parameters_b:g}B."
        )
    else:
        lines.append(f"{model.name} has {model.total_parameters_b:g} billion parameters.")
    lines.append(
        f"At {prec_label}: {model.total_parameters_b:g}B × {w.bytes_per_parameter:g} bytes ≈ {gb(w.raw_bytes):.1f} GB; "
        f"with {w.overhead_factor:g}× quantization/packing overhead ≈ {gb(w.bytes):.1f} GB of model weights."
    )
    ctx_desc = (
        f"{req.concurrent_sequences} sequences × {fmt_tokens(req.context_length)} tokens (worst case)"
        if req.context_mode == "max"
        else f"{req.concurrent_sequences} concurrent requests × ~{kv.tokens_per_sequence:,} tokens"
    )
    method = {"exact": "from the model architecture", "mla": "from the MLA latent size", "heuristic": "HEURISTIC estimate"}[kv.method]
    lines.append(f"Your workload ({ctx_desc}) needs ≈ {gb(kv.bytes):.1f} GB of KV cache ({method}).")
    lines.append(f"Runtime and workspace ({fw_label}) add ≈ {gb(base.runtime + base.workspace):.1f} GB.")
    if safety:
        lines.append(f"With {safety:g}% safety headroom, estimated required VRAM is ≈ {gb(base.required):.0f} GB.")
    else:
        lines.append(f"Without safety headroom, estimated required VRAM is ≈ {gb(base.required):.0f} GB.")
    if best is None:
        lines.append("No catalog configuration fits within the allowed GPU count.")
        return lines
    # Explain the most relevant near-miss: the largest single GPU that was rejected.
    near_miss = [c for c in sel.rejected if c.count == 1 and c.kind == "full_gpu" and c.vram_per_unit_gb < best.vram_per_unit_gb * best.count]
    if near_miss:
        nm = max(near_miss, key=lambda c: c.vram_per_unit_gb)
        lines.append(f"Therefore a single {nm.gpu_name} ({nm.vram_per_unit_gb:.0f} GB usable) does not fit comfortably.")
    if best.kind == "vgpu":
        lines.append(
            f"A {best.vgpu_profile} vGPU profile ({_profile_gib(best)}) is sufficient ({best.utilization_percent:.0f}% "
            "utilization), so a full physical GPU is not required."
        )
    elif best.count > 1:
        lines.append(
            f"{best.label} splits the model with tensor parallelism: ≈ {best.required_per_unit_gb:.1f} GB per GPU "
            f"({best.utilization_percent:.0f}% of each GPU)."
        )
    else:
        lines.append(
            f"{best.gpu_name} provides {best.vram_per_unit_gb:.0f} GB usable, leaving {best.headroom_gb:.1f} GB headroom "
            f"({best.utilization_percent:.0f}% utilization)."
        )
    if best.est_decode_tokens_per_second_per_user:
        basis = "measured" if best.confidence == "benchmark" else "roofline estimate, not a measurement"
        lines.append(
            f"Decode speed ceiling ≈ {best.est_decode_tokens_per_second_per_user:.0f} tokens/s per request at this "
            f"concurrency (target {cprof.tokens_per_second_per_user:g}; {basis})."
        )
    return lines
