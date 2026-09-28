"""Custom ML model sizing: vision, speech, embedding, diffusion, video, tabular, transformers, ...

The memory model differs from LLM serving:

    inference = weights + activations & I/O tensors (+ KV cache only if the model uses one)
                + runtime/workspace + headroom
    training  = weights + gradients + optimizer states (+ FP32 master weights) + activations
                + runtime/workspace + headroom
    LoRA      = frozen base weights + small trainable adapter (its own gradients/optimizer) + activations
    QLoRA     = LoRA with a 4-bit base model

Activation memory is architecture dependent. Unless the user provides a measured value it is an
ESTIMATE from model-type heuristics (engine settings `ml.activation`) and is always labelled so.

GPU matching reuses the LLM calculator's matcher (selection.generate) through a per-GPU memory
function, so vGPU / full GPU / multi-GPU, NVIDIA / AMD, capacity and policy rules are identical.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from ..schemas import (
    Candidate,
    MLComputeProfile,
    MLModelSummary,
    MLRequirement,
    MLSizingRequest,
    MLSizingResult,
    MLWorkloadSummary,
    ModelSpec,
    Recommendation,
    Suggestion,
    TraceStep,
    Warning_,
    WorkloadInput,
)
from .calculator import Catalog
from .compute import Demand
from .config import EngineSettings, PriorityConfig, classify
from .flavor import VARIANTS, custom_ml_flavor
from .memory import Breakdown, KVCache, SizingError
from .openstack import compile_openstack_flavor, deployment_spec, to_yaml
from .selection import SelectionContext, generate
from .units import GB, gb, gib, r


@dataclass(frozen=True)
class CategoryInfo:
    label: str
    input_kind: str  # image | video | audio | sequence | diffusion | generic
    defaults: dict
    emphasis: tuple[float, float, float]  # ranking multipliers: (compute, bandwidth, cost)
    latency_scale: float = 1.0  # x the priority's latency target (a 30-step image is not a 500 ms job)


CATEGORIES: dict[str, CategoryInfo] = {
    "computer_vision": CategoryInfo("Computer Vision", "image", {"image_width": 224, "image_height": 224}, (1.5, 0.7, 1.0)),
    "image_classification": CategoryInfo("Image Classification", "image", {"image_width": 224, "image_height": 224}, (1.5, 0.7, 1.0)),
    "object_detection": CategoryInfo("Object Detection", "image", {"image_width": 640, "image_height": 640, "num_detections": 100}, (1.5, 0.8, 1.0)),
    "segmentation": CategoryInfo("Segmentation", "image", {"image_width": 512, "image_height": 512}, (1.4, 0.9, 1.0), 2),
    "ocr": CategoryInfo("OCR", "image", {"image_width": 1024, "image_height": 1024}, (1.3, 0.9, 1.0), 2),
    "speech_recognition": CategoryInfo("Speech Recognition", "audio", {"audio_seconds": 30, "sample_rate_hz": 16000}, (1.2, 1.0, 1.0), 6),
    "text_to_speech": CategoryInfo("Text-to-Speech", "audio", {"audio_seconds": 10, "sample_rate_hz": 22050}, (1.2, 1.0, 1.0), 4),
    "embedding": CategoryInfo("Embedding Models", "sequence", {"sequence_length": 512}, (0.8, 1.2, 1.5)),
    "diffusion": CategoryInfo("Diffusion / Image Generation", "diffusion", {"image_width": 1024, "image_height": 1024, "diffusion_steps": 30}, (1.3, 1.1, 1.0), 20),
    "video": CategoryInfo("Video Models", "video", {"image_width": 224, "image_height": 224, "frames_per_sample": 16}, (1.3, 1.1, 1.0), 10),
    "recommendation": CategoryInfo("Recommendation Models", "generic", {"input_elements": 1000}, (0.8, 1.3, 1.2)),
    "tabular": CategoryInfo("Tabular Neural Networks", "generic", {"input_elements": 256}, (1.0, 1.0, 1.2)),
    "time_series": CategoryInfo("Time-Series Models", "sequence", {"sequence_length": 512}, (1.0, 1.0, 1.2)),
    "transformer": CategoryInfo("Transformer Models", "sequence", {"sequence_length": 2048}, (1.1, 1.2, 1.0), 2),
    "generic": CategoryInfo("Generic Neural Network", "generic", {"input_elements": 1024}, (1.0, 1.0, 1.0)),
    "other": CategoryInfo("Custom / Other", "generic", {"input_elements": 1024}, (1.0, 1.0, 1.0)),
}

TRAINABLE_PRECISIONS = ("fp32", "tf32", "fp16", "bf16")
TASK_LABEL = {"inference": "Inference", "training": "Training", "fine_tuning": "Fine-tuning"}


def _act_bytes(precision: str) -> float:
    """Activation element size: INT4 weight-only models still compute in 16-bit."""
    return {"fp32": 4, "tf32": 4, "fp16": 2, "bf16": 2, "fp8": 1, "int8": 1, "int4": 2, "mxfp4": 2}.get(precision, 2)


@dataclass
class _Plan:
    """Everything derived from the request before GPU matching (single-GPU totals in bytes)."""

    params: float
    trainable: float
    method: Optional[str]  # full | lora | qlora for training/fine-tuning
    weight_precision: str
    compute_precision: str
    weights: float
    gradients: float
    optimizer: float
    master: float
    activations: float
    activation_method: str
    tokens: float
    hidden: int
    layers: int
    in_flight: int
    micro_batch: int
    input_desc: str
    io_bytes: float
    kv_bytes: float
    flops_per_sample: float
    samples_per_second: float
    bandwidth: float
    safety: float


def _resolve(req: MLSizingRequest) -> dict:
    """Request values with category defaults filled in."""
    info = CATEGORIES[req.model_category]
    v = req.model_dump()
    for k, d in info.defaults.items():
        if v.get(k) is None:
            v[k] = d
    return v


def _arch_dims(params: float, hidden: Optional[int], layers: Optional[int], hpl: float) -> tuple[int, int]:
    """Hidden size and depth; derived from params ~= 12 x layers x hidden^2 when not given."""
    if hidden and layers:
        return hidden, layers
    if hidden:
        return hidden, max(1, round(params / (12 * hidden**2)))
    if layers:
        return max(64, round(math.sqrt(params / (12 * layers)))), layers
    h = min(32768, max(64, (params * hpl / 12) ** (1 / 3)))
    return round(h), max(1, round(h / hpl))


def _plan(req: MLSizingRequest, settings: EngineSettings) -> _Plan:
    ml = settings.ml
    a = ml.activation
    info = CATEGORIES[req.model_category]
    v = _resolve(req)
    if req.precision not in settings.precision:
        raise SizingError(f"Unknown precision '{req.precision}'")
    if req.framework not in ml.frameworks:
        raise SizingError(f"Unknown framework '{req.framework}'. Known: {', '.join(ml.frameworks)}")
    if req.optimizer not in ml.optimizers:
        raise SizingError(f"Unknown optimizer '{req.optimizer}'")

    params = req.parameters_m * 1e6 * (1 + req.model_growth_percent / 100)
    training = req.task != "inference"
    method = (req.fine_tune_method if req.task == "fine_tuning" else "full") if training else None

    # ---------------------------------------------------------------- precisions
    if method == "full" and req.precision not in TRAINABLE_PRECISIONS:
        raise SizingError(
            f"Full training needs FP32, TF32, BF16 or FP16 weights, not {req.precision.upper()}. "
            "Use QLoRA (or LoRA) to fine-tune a quantized base model."
        )
    weight_precision = "int4" if method == "qlora" else req.precision
    default_compute = req.precision if req.precision in TRAINABLE_PRECISIONS else "bf16"
    compute_precision = (req.training_precision or default_compute) if training else req.precision
    for p in (compute_precision, req.gradient_precision, req.optimizer_precision):
        if p and p not in settings.precision:
            raise SizingError(f"Unknown precision '{p}'")
    pcfg = settings.precision

    # ---------------------------------------------------------------- effective tokens per sample
    w, h_img, c = v["image_width"], v["image_height"], req.channels
    kind = info.input_kind
    if kind == "image":
        tokens = max(1.0, w * h_img / a.patch_size**2)
        io_elems = w * h_img * c
        desc = f"{w}×{h_img}×{c} image"
    elif kind == "video":
        frames = v["frames_per_sample"]
        tokens = max(1.0, frames * w * h_img / a.patch_size**2 / 2)
        io_elems = frames * w * h_img * c
        desc = f"{frames} frames of {w}×{h_img}"
    elif kind == "audio":
        secs = v["audio_seconds"]
        tokens = max(1.0, secs * a.audio_tokens_per_second)
        io_elems = secs * (v["sample_rate_hz"] or 16000)
        desc = f"{secs:g} s of audio"
    elif kind == "sequence":
        tokens = float(v["sequence_length"])
        io_elems = tokens
        desc = f"{int(tokens):,}-token sequence"
    elif kind == "diffusion":
        tokens = max(1.0, (w / a.diffusion_latent_downsample) * (h_img / a.diffusion_latent_downsample) / a.diffusion_patch**2)
        io_elems = w * h_img * 3
        desc = f"{w}×{h_img} image, {v['diffusion_steps']} steps"
    else:
        tokens = 1.0
        io_elems = float(v["input_elements"] or 1024)
        desc = f"{int(io_elems):,} input features"
    out_elems = v.get("output_elements") or (
        w * h_img if req.model_category == "segmentation" else (v.get("num_detections") or 0) * 6 or 1000
    )

    hidden, layers = _arch_dims(params, req.hidden_size, req.num_layers, a.hidden_per_layer)

    act_b = _act_bytes(compute_precision)
    scale = act_b / 2  # coefficients are calibrated for 16-bit activations
    spatial_pixels = (w * h_img * (v.get("frames_per_sample") or 1)) if kind in ("image", "video") else 0

    # ---------------------------------------------------------------- weights / gradients / optimizer
    opt = ml.optimizers[req.optimizer]
    opt_bytes_per_param = (
        req.optimizer_bytes_per_param
        if req.optimizer_bytes_per_param is not None
        else (opt.bytes_per_param if opt.bytes_per_param is not None else opt.states * pcfg[req.optimizer_precision].bytes_per_parameter)
    )
    wp = pcfg[weight_precision]
    gradients = optimizer = master = 0.0
    trainable = 0.0
    if not training:
        weights = req.model_file_size_gb * GB if req.model_file_size_gb else params * wp.bytes_per_parameter * wp.default_overhead_factor
    elif method == "full":
        trainable = params
        weights = params * wp.bytes_per_parameter
        grad_b = pcfg[req.gradient_precision or req.precision].bytes_per_parameter
        gradients = params * grad_b
        optimizer = params * opt_bytes_per_param
        if req.master_weights and wp.bytes_per_parameter < 4:
            master = params * 4
    else:  # lora / qlora
        pct = req.adapter_percent if req.adapter_percent is not None else ml.default_adapter_percent
        trainable = params * pct / 100
        tp_b = pcfg[compute_precision].bytes_per_parameter
        weights = params * wp.bytes_per_parameter * wp.default_overhead_factor + trainable * tp_b
        gradients = trainable * pcfg[req.gradient_precision or compute_precision].bytes_per_parameter
        optimizer = trainable * opt_bytes_per_param
        if req.master_weights and tp_b < 4:
            master = trainable * 4

    # ---------------------------------------------------------------- activations (+ I/O, KV)
    in_flight = req.batch_size * req.concurrent_requests
    micro_batch = math.ceil(req.batch_size / req.data_parallel_size)
    kv_bytes = 0.0
    if req.uses_kv_cache and kind == "sequence":
        kv_bytes = 2 * layers * hidden * tokens * act_b * (micro_batch if training else in_flight)
    io_bytes = (io_elems + out_elems) * act_b * (micro_batch if training else in_flight)
    if req.activation_memory_gb is not None:
        activations = req.activation_memory_gb * GB
        act_method = "manual"
    else:
        act_method = "estimated"
        if not training:
            per_sample = tokens * hidden * a.inference_bytes_per_token_hidden * scale
            per_sample += spatial_pixels * a.spatial_channels_inference * act_b
            if kind == "diffusion":
                per_sample += w * h_img * a.vae_decoder_channels * act_b  # VAE decode at full resolution
            activations = per_sample * in_flight
        else:
            if req.gradient_checkpointing:
                per_sample = layers * tokens * hidden * a.checkpointed_bytes_per_token_hidden_layer * scale
                per_sample += tokens * hidden * a.training_bytes_per_token_hidden_layer * scale  # one layer recomputed
                per_sample += spatial_pixels * a.spatial_channels_training * act_b / 4
            else:
                per_sample = layers * tokens * hidden * a.training_bytes_per_token_hidden_layer * scale
                per_sample += spatial_pixels * a.spatial_channels_training * act_b
            activations = per_sample * micro_batch
    activations += io_bytes + kv_bytes

    # ---------------------------------------------------------------- compute demand
    prio = req.performance_priority
    passes = {"full": 3.0, "lora": 2.0, "qlora": 2.0}.get(method or "", 1.0)  # forward (+ backward)
    steps = (v["diffusion_steps"] or 1) if (kind == "diffusion" and not training) else 1
    flops_per_sample = 2 * params * tokens * steps * passes
    if req.target_throughput:
        sps = req.target_throughput
    elif training:
        sps = req.batch_size * ml.training_steps_per_second[prio]
    else:
        latency_s = (req.target_latency_ms or ml.latency_target_ms[prio] * info.latency_scale) / 1000
        sps = in_flight / latency_s
    step_rate = sps / req.batch_size
    bytes_per_step = (weights + master) * (2 if training else 1) + activations * (3 if training else 2)
    bandwidth = bytes_per_step * step_rate

    if req.safety_margin_percent is not None:
        safety = req.safety_margin_percent
    else:
        safety = settings.priorities[prio].safety_margin_percent

    return _Plan(
        params=params,
        trainable=trainable,
        method=method,
        weight_precision=weight_precision,
        compute_precision=compute_precision,
        weights=weights,
        gradients=gradients,
        optimizer=optimizer,
        master=master,
        activations=activations,
        activation_method=act_method,
        tokens=tokens,
        hidden=hidden,
        layers=layers,
        in_flight=in_flight,
        micro_batch=micro_batch,
        input_desc=desc,
        io_bytes=io_bytes,
        kv_bytes=kv_bytes,
        flops_per_sample=flops_per_sample,
        samples_per_second=sps,
        bandwidth=bandwidth,
        safety=safety,
    )


def _breakdown(plan: _Plan, framework, count: int) -> Breakdown:
    """Per-GPU memory when the model and its activations are sharded across `count` GPUs."""
    weights = plan.weights / count
    act = plan.activations / count
    return Breakdown(
        weights=weights,
        kv=0.0,
        runtime=framework.base_overhead_gb * GB,
        workspace=(weights + act) * framework.workspace_percent / 100,
        communication=framework.communication_overhead_gb_per_gpu * GB if count > 1 else 0.0,
        safety_margin_percent=plan.safety,
        activations=act,
        gradients=plan.gradients / count,
        optimizer=(plan.optimizer + plan.master) / count,
    )


def _priority(req: MLSizingRequest, settings: EngineSettings) -> PriorityConfig:
    """The user's priority, re-weighted for what matters for this workload type."""
    prio = settings.priorities[req.performance_priority].model_copy(deep=True)
    c, b, cost = CATEGORIES[req.model_category].emphasis
    if req.task != "inference":
        c, b = c * 1.5, b * 1.2  # training leans on tensor compute and memory bandwidth
    prio.weights.compute *= c
    prio.weights.bandwidth *= b
    prio.weights.cost *= cost
    return prio


def size_ml(req: MLSizingRequest, catalog: Catalog, settings: EngineSettings, with_suggestions: bool = False) -> MLSizingResult:
    plan = _plan(req, settings)
    framework = settings.ml.frameworks[req.framework]
    training = req.task != "inference"
    info = CATEGORIES[req.model_category]

    # ---------------------------------------------------------------- match GPUs (shared matcher)
    model = ModelSpec(
        id="custom-ml",
        name=req.model_name,
        architecture="dense",
        total_parameters_b=max(plan.params / 1e9, 1e-6),
        metadata_status="user_defined",
    )
    tp = None
    if req.tensor_parallel_size:
        tp = req.tensor_parallel_size * req.pipeline_parallel_size
    elif req.pipeline_parallel_size > 1:
        tp = req.pipeline_parallel_size
    wl = WorkloadInput(
        precision=req.precision,
        concurrent_sequences=req.concurrent_requests,
        performance_priority=req.performance_priority,
        framework=req.framework,
        allow_vgpu=req.allow_vgpu,
        allow_single_gpu=req.allow_single_gpu,
        allow_multi_gpu=req.allow_multi_gpu,
        prefer_lowest_cost=req.prefer_lowest_cost,
        max_gpu_count=req.max_gpu_count,
        tensor_parallel_size=tp,
        allowed_gpu_ids=req.allowed_gpu_ids,
        calculation_mode="estimate",
    )
    demand = Demand(
        tokens_per_second_per_user=0.0,
        decode_tps=0.0,
        prefill_tps=0.0,
        required_flops=plan.flops_per_sample * plan.samples_per_second,
        step_bytes=0.0,  # per-token decode speed is an LLM notion; not reported here
        required_bandwidth=plan.bandwidth,
        moe_fraction=1.0,
    )
    no_kv = KVCache("exact", "fp16", 2, 0, None, 1, 0, "", None)
    ctx = SelectionContext(
        model=model,
        req=wl,
        settings=settings,
        priority=_priority(req, settings),
        framework=framework,
        precision_cfg=settings.precision[plan.compute_precision],
        weights_bytes=plan.weights,
        kv=no_kv,
        safety_pct=plan.safety,
        demand=demand,
        avg_input=0,
        gpus=catalog.gpus,
        profiles=catalog.profiles,
        capacity=catalog.capacity,
        has_inventory=catalog.has_inventory,
        benchmarks=[],
        use_benchmarks=False,
        memory_fn=lambda n: _breakdown(plan, framework, n),
    )
    sel = generate(ctx)
    best: Optional[Candidate] = sel.ranked[0] if sel.ranked else None
    base = _breakdown(plan, framework, 1)

    required_tflops = demand.required_flops / 1e12
    compute = MLComputeProfile(
        classification=classify(required_tflops, settings.ml.compute_classes),
        bandwidth_class=classify(plan.bandwidth / GB, settings.bandwidth_classes),
        memory_intensity=classify(base.required / GB, settings.memory_intensity),
        required_tflops=r(required_tflops, 2),
        required_bandwidth_gbps=r(plan.bandwidth / GB, 1),
        gflops_per_sample=r(plan.flops_per_sample / 1e9, 2),
        samples_per_second=r(plan.samples_per_second, 2),
    )

    # ---------------------------------------------------------------- recommendation + flavor
    gpus_by_id = {g.id: g for g in catalog.gpus}
    profiles = {p.profile_name: p for p in catalog.profiles}
    recommendation = display = spec = openstack = openstack_yaml = None
    dp = req.data_parallel_size
    if best is not None:
        gpu = gpus_by_id[best.gpu_id]
        profile = profiles.get(best.vgpu_profile) if best.vgpu_profile else None
        unit_gb = profile.vram_gb if profile else gpu.vram_gb
        code, display, slug = custom_ml_flavor(req.task, best.kind, best.count, unit_gb)
        mode_label = {"vgpu": "Shared vGPU", "full_gpu": "Dedicated GPU", "multi_gpu": "Multi-GPU"}[best.kind]
        recommendation = Recommendation(
            ai_flavor=code,
            ai_flavor_short=display,
            flavor_variant=VARIANTS[best.kind],
            flavor_tier_id=None,
            flavor_name=f"ai.ml.{req.task}.{slug}",
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
        spec = deployment_spec(
            candidate=best,
            gpu=gpu,
            profile=profile,
            ai_flavor=code,
            display_name=display,
            slug=slug,
            workload_type="custom_ml",
            model={
                "name": req.model_name,
                "category": req.model_category,
                "parameters_b": r(plan.params / 1e9, 4),
                "precision": req.precision,
            },
            workload={
                "task": req.task,
                **({"fine_tune_method": plan.method} if req.task == "fine_tuning" else {}),
                "batch_size": req.batch_size,
                "concurrent_requests": req.concurrent_requests,
                "data_parallel_replicas": dp,
                "framework": req.framework,
            },
            weights_gb=gb(plan.weights),
            required_vram_gb=gb(base.required),
            settings=settings,
            gpu_requirement={
                "minimum_vram_gb": r(gb(base.required), 1),
                "compute_class": compute.classification,
                "bandwidth_class": compute.bandwidth_class,
                "gpu_count": best.count * dp,
                "mode": {"vgpu": "shared", "full_gpu": "dedicated", "multi_gpu": "multi_gpu"}[best.kind],
            },
        )
        openstack = compile_openstack_flavor(
            spec,
            pci_alias=gpu.openstack_pci_alias,
            vgpu_trait=profile.openstack_trait if profile else None,
            performance=req.performance_priority in ("performance", "maximum") or training,
        )
        openstack_yaml = to_yaml(openstack)

    # ---------------------------------------------------------------- requirement summary
    if best is None:
        count_label, total, mode = "No fit", 0, "—"
    elif best.kind == "vgpu":
        count_label, total, mode = ("Part of 1 GPU" if dp == 1 else f"{dp} shared slices"), dp, "Shared vGPU"
    else:
        count_label = f"{best.count}" + ("" if dp == 1 else f" × {dp} replicas")
        total, mode = best.count * dp, ("Multi-GPU" if best.count > 1 else "Dedicated GPU")
    if best is not None and best.kind == "multi_gpu":
        interconnect = "Required" if training else "Recommended"
    elif training and dp > 1:
        interconnect = "Recommended"
    else:
        interconnect = "Not required"
    requirement = MLRequirement(gpu_count_label=count_label, total_gpus=total, mode=mode, interconnect=interconnect)

    warnings = _warnings(req, plan, base, best, settings, training, dp)
    explanation, trace = _explain(req, plan, base, best, info, settings, training)

    result = MLSizingResult(
        activation_method=plan.activation_method,
        model=MLModelSummary(
            name=req.model_name,
            category=req.model_category,
            category_label=info.label,
            task=req.task,
            fine_tune_method=plan.method if req.task == "fine_tuning" else None,
            parameters_b=r(plan.params / 1e9, 4),
            trainable_parameters_b=r(plan.trainable / 1e9, 4),
            precision=plan.weight_precision,
            compute_precision=plan.compute_precision,
            framework=framework.label,
        ),
        workload=MLWorkloadSummary(
            batch_size=req.batch_size,
            micro_batch_per_gpu=plan.micro_batch,
            concurrent_requests=req.concurrent_requests,
            in_flight_samples=plan.in_flight,
            input_description=plan.input_desc,
            effective_tokens_per_sample=r(plan.tokens, 1),
            performance_priority=req.performance_priority,
            data_parallel_size=dp,
            samples_per_second=r(plan.samples_per_second, 2),
            safety_margin_percent=plan.safety,
        ),
        memory=base.to_schema("gb"),
        memory_gib=base.to_schema("gib"),
        compute=compute,
        requirement=requirement,
        recommendation=recommendation,
        ai_flavor_display=display,
        matches=sel.ranked[:15],
        rejected=sel.rejected[:12],
        vgpu_verdict=_vgpu_verdict(req, best, sel, gb(base.required)),
        warnings=warnings,
        explanation=explanation,
        deployment_spec=spec,
        openstack=openstack,
        openstack_yaml=openstack_yaml,
        trace=trace,
    )
    if with_suggestions:
        result.suggestions = _suggestions(req, catalog, settings, result)
    return result


# ---------------------------------------------------------------------------- helpers


def _vgpu_verdict(req: MLSizingRequest, best, sel, required_gb: float) -> str:
    if best is not None and best.kind == "vgpu":
        return f"Shared GPU is enough: {best.vgpu_profile} ({best.vram_per_unit_gb * 1e9 / 1024**3:.0f} GiB). A full GPU would waste capacity."
    if not req.allow_vgpu:
        return "Not considered: shared GPUs disabled."
    if sel.vgpu_capacity_blocked:
        c = sel.vgpu_capacity_blocked[0]
        return f"Suitable vGPU profile currently unavailable ({c.vgpu_profile} has no free instances)."
    if any(c.kind == "vgpu" for c in sel.ranked):
        return "Possible, but a dedicated option ranked higher for this workload."
    if sel.vgpu_too_small:
        largest = max(sel.vgpu_too_small, key=lambda c: c.vram_per_unit_gb)
        return f"Not enough: needs ~{required_gb:.0f} GB; the largest shared profile is {largest.vgpu_profile}."
    if sel.vgpu_policy_blocked:
        return "Not offered for this performance priority."
    return "No shared GPU profiles configured."


def _warn(out: list[Warning_], code: str, severity: str, message: str) -> None:
    out.append(Warning_(code=code, severity=severity, message=message))


def _warnings(req, plan: _Plan, base: Breakdown, best, settings, training: bool, dp: int) -> list[Warning_]:
    out: list[Warning_] = []
    if plan.activation_method == "estimated":
        _warn(
            out,
            "ACTIVATION_ESTIMATED",
            "warning" if training else "info",
            f"Activation memory ({gb(plan.activations):.1f} GB) is an ESTIMATE from {CATEGORIES[req.model_category].label.lower()} "
            "heuristics. Enter a measured value under Advanced for accuracy.",
        )
    else:
        _warn(out, "ACTIVATION_MANUAL", "info", f"Using your activation memory value ({req.activation_memory_gb:g} GB).")
    if req.model_file_size_gb and not training:
        _warn(out, "FILE_SIZE", "info", f"Weight memory taken from the model file size ({req.model_file_size_gb:g} GB).")
    if training and plan.activation_method == "estimated" and not req.gradient_checkpointing and plan.activations > 0.5 * base.subtotal:
        _warn(
            out,
            "ACTIVATIONS_DOMINATE",
            "warning",
            f"Activations are {plan.activations / base.subtotal * 100:.0f}% of memory. Gradient checkpointing (Advanced) "
            "would cut them substantially at ~20–30% extra compute.",
        )
    if plan.method in ("lora", "qlora"):
        full = _plan(req.model_copy(update={"task": "training", "fine_tune_method": "full", "precision": "bf16" if req.precision not in TRAINABLE_PRECISIONS else req.precision}), settings)
        full_bytes = (full.weights + full.gradients + full.optimizer + full.master + full.activations) * (1 + plan.safety / 100)
        method = "QLoRA" if plan.method == "qlora" else "LoRA"
        _warn(
            out,
            "PEFT_SAVINGS",
            "info",
            f"{method} can substantially reduce GPU memory compared with full fine-tuning: ~{gb(base.required):.0f} GB here vs "
            f"~{gb(full_bytes):.0f} GB for full fine-tuning of the same model.",
        )
    if best is None:
        _warn(out, "NO_FIT", "critical", "No GPU configuration fits. Reduce batch size, enable gradient checkpointing, use LoRA/QLoRA, or allow more GPUs.")
        return out
    if best.kind == "multi_gpu":
        _warn(
            out,
            "MULTI_GPU",
            "warning",
            "Multi-GPU needs model-parallel support in your framework"
            + (" and a high-speed GPU link (NVLink / Infinity Fabric) for training." if training else "."),
        )
    if dp > 1:
        _warn(out, "DATA_PARALLEL", "info", f"{dp} data-parallel replicas: {best.count * dp} GPUs in total, each with a micro-batch of {plan.micro_batch}.")
    if best.kind == "vgpu" and training:
        _warn(out, "VGPU_TRAINING", "warning", "Training on a shared GPU: performance is not guaranteed and will vary with other tenants.")
    if best.compute_ratio is not None and best.compute_ratio < 1:
        _warn(out, "COMPUTE_BOUND", "warning", f"Estimated compute covers ~{best.compute_ratio * 100:.0f}% of the throughput target; expect lower samples/s.")
    _warn(out, "PERFORMANCE_UNVALIDATED", "info", "Memory fit is calculated; performance fit is estimated and requires benchmark validation.")
    return out


def _explain(req, plan: _Plan, base: Breakdown, best, info: CategoryInfo, settings, training: bool):
    lines: list[str] = []
    trace: list[TraceStep] = []
    p_b = plan.params / 1e9
    wp = settings.precision[plan.weight_precision]
    lines.append(f"{req.model_name}: {info.label.lower()} model with {p_b:g}B parameters, sized for {TASK_LABEL[req.task].lower()}.")
    if req.model_file_size_gb and not training:
        lines.append(f"Model weights: {req.model_file_size_gb:g} GB (from the model file).")
        trace.append(TraceStep(step="Weights", formula="model file size", value=f"{gb(plan.weights):.2f} GB"))
    elif plan.method in ("lora", "qlora"):
        lines.append(
            f"Frozen base weights at {wp.label}: {p_b:g}B × {wp.bytes_per_parameter:g} B ≈ {gb(plan.weights - plan.trainable * settings.precision[plan.compute_precision].bytes_per_parameter):.1f} GB; "
            f"only {plan.trainable / 1e6:,.1f}M adapter parameters are trained."
        )
        trace.append(TraceStep(step="Weights", formula=f"{p_b:g}B × {wp.bytes_per_parameter:g} B ({wp.label}, frozen) + adapter", value=f"{gb(plan.weights):.2f} GB"))
    else:
        lines.append(f"Model weights: {p_b:g}B × {wp.bytes_per_parameter:g} bytes ({wp.label}) ≈ {gb(plan.weights):.1f} GB.")
        trace.append(TraceStep(step="Weights", formula=f"{p_b:g}B × {wp.bytes_per_parameter:g} B/param ({wp.label})", value=f"{gb(plan.weights):.2f} GB"))
    if training:
        opt = settings.ml.optimizers[req.optimizer]
        lines.append(
            f"Training state: gradients ≈ {gb(plan.gradients):.1f} GB and {opt.label} optimizer states"
            + (" plus FP32 master weights" if plan.master else "")
            + f" ≈ {gb(plan.optimizer + plan.master):.1f} GB (for {plan.trainable / 1e9:g}B trainable parameters)."
        )
        trace.append(TraceStep(step="Gradients", formula=f"{plan.trainable / 1e9:g}B trainable × gradient bytes", value=f"{gb(plan.gradients):.2f} GB"))
        trace.append(
            TraceStep(
                step="Optimizer",
                formula=f"{opt.label} states" + (" + FP32 master weights (4 B/param)" if plan.master else ""),
                value=f"{gb(plan.optimizer + plan.master):.2f} GB",
            )
        )
    samples = f"micro-batch of {plan.micro_batch}" if training else f"{plan.in_flight} samples in flight ({req.batch_size} batch × {req.concurrent_requests} requests)"
    if plan.activation_method == "manual":
        lines.append(f"Activations: {gb(plan.activations):.1f} GB (your value).")
        trace.append(TraceStep(step="Activations", formula="user-provided", value=f"{gb(plan.activations):.2f} GB"))
    else:
        lines.append(
            f"Activations & I/O tensors (ESTIMATED) for a {samples} of {plan.input_desc}: ≈ {gb(plan.activations):.1f} GB "
            f"(~{plan.tokens:,.0f} tokens/sample × hidden {plan.hidden:,} × {plan.layers} layers"
            + (", gradient checkpointing" if training and req.gradient_checkpointing else "")
            + ")."
        )
        trace.append(
            TraceStep(
                step="Activations (estimated)",
                formula=f"{plan.tokens:,.0f} tokens × hidden {plan.hidden:,} × {plan.layers} layers × {samples}",
                value=f"{gb(plan.activations):.2f} GB (I/O {gb(plan.io_bytes):.2f} GB)",
            )
        )
    lines.append(f"Framework runtime and workspace add ≈ {gb(base.runtime + base.workspace):.1f} GB.")
    lines.append(f"With {plan.safety:g}% headroom, the requirement is ≈ {gb(base.required):.0f} GB of GPU memory.")
    trace.append(TraceStep(step="Required VRAM", formula=f"subtotal {gb(base.subtotal):.2f} GB × (1 + {plan.safety:g}%)", value=f"{gb(base.required):.2f} GB"))
    trace.append(
        TraceStep(
            step="Compute demand (relative)",
            formula=f"2 × {p_b:g}B params × {plan.tokens:,.0f} tokens × passes × {plan.samples_per_second:,.1f} samples/s",
            value=f"{plan.flops_per_sample * plan.samples_per_second / 1e12:,.2f} TFLOPS",
        )
    )
    if best is not None:
        if best.kind == "vgpu":
            lines.append(f"A {best.vgpu_profile} shared GPU slice is enough; a full GPU would waste capacity.")
        elif best.count > 1:
            lines.append(f"{best.label}: the model state is split across {best.count} GPUs (~{best.required_per_unit_gb:.0f} GB each).")
        else:
            lines.append(f"{best.gpu_name} fits on one GPU with ~{best.headroom_gb:.0f} GB to spare.")
    else:
        lines.append("No catalog configuration fits within the allowed GPU count.")
    return lines, trace


def _suggestions(req: MLSizingRequest, catalog: Catalog, settings: EngineSettings, base: MLSizingResult) -> list[Suggestion]:
    training = req.task != "inference"
    variants: list[tuple[str, str, dict, str]] = []
    if training and not req.gradient_checkpointing and req.activation_memory_gb is None:
        variants.append(("checkpointing", "Enable gradient checkpointing", {"gradient_checkpointing": True}, "Recomputes activations in the backward pass: ~20–30% slower steps, far less memory."))
    if req.task == "fine_tuning" and req.fine_tune_method == "full":
        variants.append(("lora", "Fine-tune with LoRA", {"fine_tune_method": "lora"}, "Trains small adapters only; quality is usually close to full fine-tuning."))
    if req.task == "fine_tuning" and req.fine_tune_method == "lora":
        variants.append(("qlora", "Fine-tune with QLoRA", {"fine_tune_method": "qlora"}, "4-bit base model: least memory, slightly slower and a small quality trade-off."))
    if req.batch_size >= 2:
        half = req.batch_size // 2
        variants.append(("batch", f"Halve the batch size to {half}", {"batch_size": half}, "Use gradient accumulation to keep the effective batch." if training else "Lower throughput per request batch."))
    if req.precision in ("fp32", "tf32"):
        variants.append(("bf16", "Use BF16", {"precision": "bf16"}, "Mixed precision halves weight and activation memory; standard for modern training and inference."))
    if not training and req.precision in ("fp16", "bf16"):
        variants.append(("int8", "Serve in INT8", {"precision": "int8"}, "Post-training quantization; validate accuracy on your data."))
    if base.recommendation is None and (not req.allow_multi_gpu or req.max_gpu_count < 16):
        variants.append(("multi", "Allow up to 16 GPUs", {"allow_multi_gpu": True, "max_gpu_count": 16}, "Needs model-parallel training/serving and a fast GPU link."))

    out: list[Suggestion] = []
    base_c = base.recommendation.candidate if base.recommendation else None
    for key, title, patch, tradeoff in variants:
        try:
            res = size_ml(req.model_copy(update=patch), catalog, settings)
        except SizingError:
            continue
        rec = res.recommendation
        if rec is None:
            continue
        c = rec.candidate
        if base_c is not None and (c.gpu_units, c.vram_per_unit_gb * c.count) >= (base_c.gpu_units, base_c.vram_per_unit_gb * base_c.count):
            continue
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
