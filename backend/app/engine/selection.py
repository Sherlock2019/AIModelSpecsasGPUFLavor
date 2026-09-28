"""GPU selection: evaluate vGPU / single GPU / multi-GPU candidates and rank them."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Optional

from ..schemas import (
    BenchmarkMatch,
    BenchmarkRecord,
    Candidate,
    CostEstimate,
    GPUSpec,
    ModelSpec,
    VGPUProfile,
    WorkloadInput,
)
from .compute import Demand, gpu_tensor_tflops
from .config import EngineSettings, FrameworkConfig, PrecisionConfig, PriorityConfig, classify
from .memory import Breakdown, KVCache, per_gpu_breakdown
from .units import GB, gb, r

# Ranking penalty per availability status (multiplied by the priority's availability weight).
AVAILABILITY_PENALTY = {
    "available": 0.0,
    "limited": 0.5,
    "unknown": 0.5,
    "not_in_inventory": 1.0,
    "insufficient": 2.0,
    "unavailable": 3.0,
}


@dataclass(frozen=True)
class Capacity:
    installed: int
    available: int


@dataclass
class SelectionContext:
    model: ModelSpec
    req: WorkloadInput
    settings: EngineSettings
    priority: PriorityConfig
    framework: FrameworkConfig
    precision_cfg: PrecisionConfig
    weights_bytes: float
    kv: KVCache
    safety_pct: float
    demand: Demand
    avg_input: int
    gpus: list[GPUSpec]
    profiles: list[VGPUProfile]
    capacity: dict[str, Capacity]
    has_inventory: bool
    benchmarks: list[BenchmarkRecord] = field(default_factory=list)
    use_benchmarks: bool = True
    # Per-GPU memory for a split across `count` GPUs. None = LLM inference breakdown (weights + KV).
    # Other workloads (custom ML inference/training) plug in their own breakdown here so every
    # workload shares the same matcher.
    memory_fn: Optional[Callable[[int], Breakdown]] = None


@dataclass
class SelectionResult:
    ranked: list[Candidate]
    rejected: list[Candidate]
    vgpu_policy_blocked: list[str]
    vgpu_too_small: list[Candidate]
    vgpu_capacity_blocked: list[Candidate] = field(default_factory=list)


# --------------------------------------------------------------------------- evaluation


def _find_benchmark(ctx: SelectionContext, gpu_id: str, count: int) -> Optional[BenchmarkRecord]:
    rows = [
        b
        for b in ctx.benchmarks
        if b.model_id == ctx.model.id and b.gpu_id == gpu_id and b.gpu_count == count and b.precision == ctx.req.precision
    ]
    if not rows:
        return None
    per_seq = ctx.kv.tokens_per_sequence
    rows.sort(
        key=lambda b: (
            b.framework != ctx.req.framework,
            b.context_length < per_seq,
            abs(b.concurrency - ctx.req.concurrent_sequences),
            not b.verified,
        )
    )
    return rows[0]


def _nvidia_only(ctx: SelectionContext) -> bool:
    # The id check keeps databases seeded before the nvidia_only flag existed correct.
    return ctx.framework.nvidia_only or ctx.req.framework == "tensorrt_llm"


def _availability(ctx: SelectionContext, gpu: GPUSpec, units_needed: int) -> tuple[str, Optional[int]]:
    if gpu.availability == "unavailable":
        return "unavailable", None
    if not ctx.has_inventory:
        return gpu.availability, None
    cap = ctx.capacity.get(gpu.id)
    if cap is None:
        return "not_in_inventory", None
    return ("available" if cap.available >= units_needed else "insufficient"), cap.available


def _label(gpu: GPUSpec, count: int, profile: Optional[VGPUProfile], tp: int) -> str:
    if profile:
        mode = "MIG-backed vGPU" if profile.sharing_mode == "mig" else "vGPU"
        return f"{profile.profile_name} {mode} on {gpu.display_name}"
    if count == 1:
        return f"1 × {gpu.display_name}"
    return f"{count} × {gpu.display_name} (tensor parallel = {tp})"


def evaluate(ctx: SelectionContext, gpu: GPUSpec, count: int, profile: Optional[VGPUProfile] = None) -> Candidate:
    s = ctx.settings
    req = ctx.req
    notes: list[str] = []
    if ctx.memory_fn is not None:
        bd = ctx.memory_fn(count)
    else:
        bd = per_gpu_breakdown(
            ctx.weights_bytes,
            ctx.kv,
            count,
            ctx.framework,
            ctx.safety_pct,
            req.runtime_overhead_override_gb,
            req.workspace_percent_override,
        )
    usable = profile.usable_vram_bytes if profile else gpu.usable_vram_bytes
    fraction = profile.compute_fraction(s.efficiency.mig_total_slices) if profile else 1.0
    fits = bd.required <= usable
    fits_weights = bd.weights + bd.runtime <= usable
    rejected: Optional[str] = None

    # --- memory bandwidth (decode is bandwidth-bound) -------------------------------------
    bw_ratio = est_tps = None
    if gpu.memory_bandwidth_gbps:
        bw_avail = gpu.memory_bandwidth_gbps * GB * count * fraction * s.efficiency.bandwidth_efficiency
        if ctx.demand.required_bandwidth > 0:
            bw_ratio = bw_avail / ctx.demand.required_bandwidth
        if ctx.demand.step_bytes > 0:
            est_tps = bw_avail / ctx.demand.step_bytes

    # --- tensor compute (prefill is compute-bound) ----------------------------------------
    compute_ratio = ttft = None
    tflops, tf_note = gpu_tensor_tflops(gpu, ctx.precision_cfg.compute)
    if tf_note:
        notes.append(tf_note)
    if tflops:
        flops_avail = tflops * 1e12 * count * fraction * s.efficiency.compute_mfu
        if ctx.demand.required_flops > 0:
            compute_ratio = flops_avail / ctx.demand.required_flops
        if ctx.avg_input:
            ttft = 2.0 * ctx.model.active_parameters_b * 1e9 * ctx.avg_input / flops_avail * 1000.0

    # --- parallelism compatibility ---------------------------------------------------------
    tp = min(count, s.gpus_per_node)
    if count > 1:
        heads = ctx.model.attention_heads
        if heads and heads % tp:
            rejected = f"{heads} attention heads are not divisible by tensor parallel size {tp}"
        if count > s.gpus_per_node:
            nodes = math.ceil(count / s.gpus_per_node)
            notes.append(
                f"Spans {nodes} nodes: tensor parallel {tp} × pipeline parallel {nodes}; needs a high-speed interconnect."
            )
        if not gpu.has_high_speed_link:
            notes.append("No high-speed GPU link (NVLink / Infinity Fabric): the model split runs over PCIe, reducing throughput.")
    if _nvidia_only(ctx) and gpu.vendor.upper() != "NVIDIA":
        rejected = f"{ctx.framework.label} runs only on NVIDIA GPUs"
    if ctx.kv.shard_limit is not None and ctx.kv.method == "exact" and count > ctx.kv.shard_limit:
        notes.append(f"Only {ctx.kv.shard_limit} KV heads: KV cache is replicated beyond {ctx.kv.shard_limit}-way TP.")
    if profile:
        if profile.sharing_mode == "time_sliced" and profile.max_instances_per_gpu > 1:
            notes.append(
                f"Time-sliced: worst case 1/{profile.max_instances_per_gpu} of GPU compute and bandwidth when all slots are busy."
            )
        elif profile.sharing_mode == "mig":
            notes.append(f"MIG slice: dedicated {profile.mig_compute_slices}/{s.efficiency.mig_total_slices} of GPU compute.")

    # --- benchmark data --------------------------------------------------------------------
    confidence = "estimated"
    bench_match = None
    if ctx.use_benchmarks and profile is None:
        b = _find_benchmark(ctx, gpu.id, count)
        if b is not None:
            confidence = "benchmark"
            bench_match = BenchmarkMatch(
                benchmark_id=b.id,
                source=b.source,
                verified=b.verified,
                concurrency=b.concurrency,
                context_length=b.context_length,
                aggregate_tokens_per_second=b.aggregate_tokens_per_second,
                decode_tokens_per_second=b.decode_tokens_per_second,
                ttft_ms=b.ttft_ms,
                tpot_ms=b.tpot_ms,
                peak_vram_gb=b.peak_vram_gb,
            )
            measured = b.decode_tokens_per_second or b.aggregate_tokens_per_second
            if measured and ctx.demand.decode_tps > 0:
                ratio = measured / ctx.demand.decode_tps
                bw_ratio = compute_ratio = ratio
                est_tps = measured / req.concurrent_sequences
                notes.append(f"Throughput ratios from measured benchmark ({b.source}).")
            if b.ttft_ms is not None:
                ttft = b.ttft_ms
            if b.peak_vram_gb is not None and b.peak_vram_gb * GB > usable:
                fits = False
                rejected = rejected or f"Measured peak VRAM {b.peak_vram_gb:g} GB exceeds usable capacity"

    if rejected is None and not fits:
        need, have = gb(bd.required), gb(usable)
        if fits_weights:
            rejected = (
                f"Weights fit, but not the full workload ({'activations, training state' if ctx.memory_fn else 'KV cache'} "
                f"and runtime): needs {need:.1f} GB per GPU, "
                f"{have:.1f} GB usable"
            )
        else:
            rejected = f"Needs {need:.1f} GB per GPU, only {have:.1f} GB usable"
        if profile:
            rejected = f"Profile too small: {rejected[0].lower()}{rejected[1:]}"

    units_needed = 1 if profile else count
    availability, available_units = _availability(ctx, gpu, units_needed)
    if profile is not None and profile.available_instances is not None:
        available_units = profile.available_instances
        availability = "available" if profile.available_instances >= 1 else "insufficient"
    capacity_limited = False
    if s.ranking.exclude_insufficient_capacity and availability == "insufficient" and rejected is None:
        capacity_limited = True
        if profile is not None:
            rejected = f"Suitable vGPU profile currently unavailable ({available_units or 0} free {profile.profile_name})"
        else:
            rejected = f"Capacity: only {available_units or 0} of {units_needed} GPU{'s' if units_needed > 1 else ''} available"

    cost = None
    if gpu.cost_per_hour is not None:
        vm = s.cost.vm_cost_per_hour or 0.0
        if profile:
            share = 1.0 / profile.max_instances_per_gpu
            lic = profile.license_cost_per_hour or 0.0
            hourly = gpu.cost_per_hour * share + lic + vm
            formula = f"{gpu.cost_per_hour:g}/h × 1/{profile.max_instances_per_gpu} share + {lic:g} license + {vm:g} VM"
        else:
            hourly = gpu.cost_per_hour * count + vm
            formula = f"{gpu.cost_per_hour:g}/h × {count} GPU + {vm:g} VM"
        cost = CostEstimate(
            hourly=r(hourly, 4),
            daily=r(hourly * s.cost.hours_per_day, 2),
            monthly=r(hourly * s.cost.hours_per_month, 2),
            formula=formula,
        )

    kind = "vgpu" if profile else ("full_gpu" if count == 1 else "multi_gpu")
    return Candidate(
        key=f"{kind}:{gpu.id}:{profile.profile_name if profile else count}",
        kind=kind,
        gpu_id=gpu.id,
        gpu_name=gpu.display_name,
        gpu_vendor=gpu.vendor,
        gpu_compute_class=(
            classify(gpu.bf16_tflops or gpu.fp16_tflops, s.gpu_compute_classes)
            if (gpu.bf16_tflops or gpu.fp16_tflops)
            else None
        ),
        gpu_bandwidth_class=(
            classify(gpu.memory_bandwidth_gbps, s.gpu_bandwidth_classes) if gpu.memory_bandwidth_gbps else None
        ),
        interconnect=gpu.interconnect_type,
        vgpu_profile=profile.profile_name if profile else None,
        sharing_mode=profile.sharing_mode if profile else None,
        count=count,
        tensor_parallel=tp,
        vram_per_unit_gb=r(gb(usable)),
        required_per_unit_gb=r(gb(bd.required)),
        memory_per_unit=bd.to_schema("gb"),
        utilization_percent=r(bd.required / usable * 100, 1),
        headroom_gb=r(gb(usable - bd.required)),
        fits=fits and rejected is None,
        fits_weights_only=fits_weights,
        gpu_units=r((1.0 / profile.max_instances_per_gpu) if profile else count, 3),
        bandwidth_ratio=None if bw_ratio is None else r(bw_ratio, 2),
        compute_ratio=None if compute_ratio is None else r(compute_ratio, 2),
        est_decode_tokens_per_second_per_user=None if est_tps is None else r(est_tps, 1),
        est_ttft_ms=None if ttft is None else r(ttft, 0),
        availability=availability,
        available_units=available_units,
        cost=cost,
        confidence=confidence,
        benchmark=bench_match,
        notes=notes,
        rejected_reason=rejected,
        capacity_limited=capacity_limited,
        label=_label(gpu, count, profile, tp),
    )


# --------------------------------------------------------------------------- generation


def generate(ctx: SelectionContext) -> SelectionResult:
    req, s, prio = ctx.req, ctx.settings, ctx.priority
    fitting: list[Candidate] = []
    rejected: list[Candidate] = []
    policy_blocked: list[str] = []
    too_small: list[Candidate] = []
    capacity_blocked: list[Candidate] = []

    gpus = [g for g in ctx.gpus if g.enabled and (not req.allowed_gpu_ids or g.id in req.allowed_gpu_ids)]

    # Step 1 - vGPU profiles (smallest fitting profile per physical GPU and sharing mode).
    vgpu_allowed = req.allow_vgpu and not req.dedicated_gpu_required and (req.tensor_parallel_size or 1) == 1
    if vgpu_allowed:
        for gpu in gpus:
            if not gpu.supports_vgpu:
                continue
            profiles = sorted(
                (p for p in ctx.profiles if p.enabled and p.physical_gpu == gpu.id), key=lambda p: p.vram_gb
            )
            for mode in ("time_sliced", "mig"):
                mode_profiles = [p for p in profiles if p.sharing_mode == mode]
                if not mode_profiles:
                    continue
                if mode not in prio.allowed_vgpu_sharing:
                    policy_blocked.append(f"{gpu.model} ({'MIG' if mode == 'mig' else 'time-sliced'})")
                    continue
                chosen = None
                for p in mode_profiles:
                    c = evaluate(ctx, gpu, 1, p)
                    if c.fits:
                        chosen = c
                        break
                    if c.capacity_limited:  # would fit, but none free: record why, try the next profile
                        rejected.append(c)
                        capacity_blocked.append(c)
                if chosen:
                    fitting.append(chosen)
                elif not any(c.gpu_id == gpu.id for c in capacity_blocked):
                    largest = evaluate(ctx, gpu, 1, mode_profiles[-1])
                    rejected.append(largest)
                    too_small.append(largest)

    # Steps 2 & 3 - single full GPU, then multi-GPU.
    counts = [c for c in s.gpu_counts if c <= req.max_gpu_count]
    if not req.allow_multi_gpu:
        counts = [1]
    if not req.allow_single_gpu:
        counts = [c for c in counts if c > 1]
    if req.tensor_parallel_size:
        counts = [req.tensor_parallel_size]
    for gpu in gpus:
        if _nvidia_only(ctx) and gpu.vendor.upper() != "NVIDIA":
            rejected.append(evaluate(ctx, gpu, 1))  # framework incompatibility, not a memory question
            continue
        first_fit = last = single = None
        for n in counts:
            c = evaluate(ctx, gpu, n)
            last = c
            if n == 1:
                single = c
            if c.fits:
                first_fit = c
                break
        if first_fit is None:
            if single is not None and single is not last and single.capacity_limited:
                rejected.append(single)  # e.g. "only 0 of 1 GPU available"
            if last is not None:
                if last.rejected_reason and last.count > 1 and len(counts) > 1 and not last.capacity_limited:
                    last.rejected_reason = f"Does not fit on up to {last.count} GPUs. {last.rejected_reason}"
                rejected.append(last)
            continue
        if single is not None and single is not first_fit:
            rejected.append(single)  # single-GPU near miss, shown as "why not 1 x this GPU"
        fitting.append(first_fit)
        # PERFORMANCE / MAXIMUM: scale out when the bandwidth roofline misses the target.
        if (
            prio.escalate_for_throughput
            and not req.tensor_parallel_size
            and first_fit.bandwidth_ratio is not None
            and first_fit.bandwidth_ratio < 1
        ):
            for n in (c for c in counts if c > first_fit.count):
                c = evaluate(ctx, gpu, n)
                if c.fits and c.bandwidth_ratio is not None and c.bandwidth_ratio >= 1:
                    c.notes.append("Scaled out beyond the memory minimum to meet the per-user throughput target.")
                    fitting.append(c)
                    break

    ranked = rank(ctx, fitting)
    # Capacity rejections first (most actionable), then by size.
    rejected.sort(key=lambda c: (not c.capacity_limited, -c.vram_per_unit_gb * c.count, c.gpu_name))
    return SelectionResult(ranked, rejected, policy_blocked, too_small, capacity_blocked)


# --------------------------------------------------------------------------- ranking


def _shortfall(ratio: Optional[float], unknown: float = 0.5) -> float:
    return unknown if ratio is None else max(0.0, 1.0 - ratio)


def _unknown_penalty(ratios: list[Optional[float]]) -> float:
    """Unknown throughput must never look better than a known shortfall: use the worst known one."""
    known = [_shortfall(r) for r in ratios if r is not None]
    return max([0.5, *known])


def rank(ctx: SelectionContext, candidates: list[Candidate]) -> list[Candidate]:
    """Lower score is better. Components (weights configurable per priority):
    GPU units, wasted VRAM, over-comfortable utilization, bandwidth/compute shortfall,
    availability, relative cost, bandwidth bonus (performance tiers) and a vGPU bias.
    """
    w = ctx.priority.weights
    prio = ctx.priority
    costs = [c.cost.hourly for c in candidates if c.cost and c.cost.hourly > 0]
    min_cost = min(costs) if costs else None
    unknown_bw = _unknown_penalty([c.bandwidth_ratio for c in candidates])
    unknown_compute = _unknown_penalty([c.compute_ratio for c in candidates])
    for c in candidates:
        util = c.utilization_percent / 100.0
        score = w.units * c.gpu_units
        score += w.waste * max(0.0, 1.0 - util)
        score += w.tight * max(0.0, util - prio.comfortable_utilization) * 10
        score += w.bandwidth * _shortfall(c.bandwidth_ratio, unknown_bw)
        score += w.compute * _shortfall(c.compute_ratio, unknown_compute)
        if ctx.settings.ranking.use_inventory or c.availability in ("unavailable", "limited"):
            score += w.availability * AVAILABILITY_PENALTY.get(c.availability, 0.5)
        if ctx.req.prefer_lowest_cost and min_cost:
            if c.cost:
                score += w.cost * min(3.0, c.cost.hourly / min_cost - 1.0)
            else:
                score += w.cost * ctx.settings.cost.unknown_cost_penalty
        if w.bandwidth_bonus and c.bandwidth_ratio is not None:
            score -= w.bandwidth_bonus * min(c.bandwidth_ratio, 4.0) / 4.0
        if c.kind == "vgpu":
            score += w.vgpu
        if c.confidence == "benchmark":
            score -= 0.5  # prefer measured evidence on near-ties
        c.score = r(score, 3)
    return sorted(candidates, key=lambda c: (c.score, c.gpu_units, c.gpu_name))


def pick_alternatives(ranked: list[Candidate], limit: int = 4) -> list[Candidate]:
    """Best remaining option of each kind first (diversity), then fill by score."""
    rest = ranked[1:]
    chosen: list[Candidate] = []
    seen_gpu: set[tuple[str, str]] = set()
    for kind in ("full_gpu", "multi_gpu", "vgpu"):
        for c in rest:
            if c.kind == kind:
                chosen.append(c)
                seen_gpu.add((c.gpu_id, c.kind))
                break
    for c in rest:
        if len(chosen) >= limit:
            break
        if c not in chosen and (c.gpu_id, c.kind) not in seen_gpu:
            chosen.append(c)
            seen_gpu.add((c.gpu_id, c.kind))
    chosen.sort(key=lambda c: c.score if c.score is not None else 0)
    return chosen[:limit]
