"""Relative compute / bandwidth demand. Used for classification and ranking only - never
presented as measured performance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..schemas import ComputeProfile, GPUSpec, ModelSpec
from .config import EngineSettings, classify
from .units import GB, r


@dataclass(frozen=True)
class Demand:
    tokens_per_second_per_user: float
    decode_tps: float  # aggregate generated tokens/s
    prefill_tps: float  # aggregate prompt tokens/s
    required_flops: float  # FLOP/s
    step_bytes: float  # bytes read from HBM per decode step (all running sequences)
    required_bandwidth: float  # bytes/s
    moe_fraction: float


def moe_weight_fraction(model: ModelSpec, concurrency: int) -> float:
    """Approximate share of weights read per decode step.

    Dense: 1. MoE: each sequence touches `a = active/total` of the weights; with B sequences in a
    batch, assuming independent routing, touched share ~= 1 - (1 - a)^B. Conservative for
    shared-expert designs, optimistic for heavily skewed routing.
    """
    if model.architecture == "dense":
        return 1.0
    a = model.active_parameters_b / model.total_parameters_b
    return min(1.0, 1.0 - (1.0 - a) ** max(1, concurrency))


def demand(
    model: ModelSpec,
    weights_bytes: float,
    kv_bytes: float,
    concurrency: int,
    avg_input: int,
    avg_output: int,
    tokens_per_second_per_user: float,
    target_aggregate: Optional[float],
) -> Demand:
    if target_aggregate:
        decode_tps = target_aggregate
        per_user = target_aggregate / concurrency
    else:
        per_user = tokens_per_second_per_user
        decode_tps = per_user * concurrency
    # Each request generates avg_output tokens, so prompt tokens arrive at in/out x decode rate.
    prefill_tps = decode_tps * (avg_input / avg_output) if avg_output > 0 else 0.0
    active_params = model.active_parameters_b * 1e9
    required_flops = 2.0 * active_params * (decode_tps + prefill_tps)
    fraction = moe_weight_fraction(model, concurrency)
    step_bytes = weights_bytes * fraction + kv_bytes
    # One decode step emits one token for every running sequence.
    required_bw = step_bytes * per_user
    return Demand(per_user, decode_tps, prefill_tps, required_flops, step_bytes, required_bw, fraction)


def compute_profile(model: ModelSpec, d: Demand, required_vram_bytes: float, settings: EngineSettings) -> ComputeProfile:
    score = model.active_parameters_b * d.decode_tps
    return ComputeProfile(
        active_parameters_b=model.active_parameters_b,
        total_parameters_b=model.total_parameters_b,
        architecture=model.architecture,
        tokens_per_second_per_user=r(d.tokens_per_second_per_user, 1),
        requested_tokens_per_second=r(d.decode_tps, 1),
        prefill_tokens_per_second=r(d.prefill_tps, 1),
        compute_demand_score=r(score, 1),
        classification=classify(score, settings.compute_classes),
        memory_intensity=classify(required_vram_bytes / GB, settings.memory_intensity),
        bandwidth_class=classify(d.required_bandwidth / GB, settings.bandwidth_classes),
        required_tflops=r(d.required_flops / 1e12, 2),
        decode_bytes_per_step_gb=r(d.step_bytes / GB, 2),
        required_bandwidth_gbps=r(d.required_bandwidth / GB, 1),
        moe_weight_fraction_read_per_step=r(d.moe_fraction, 3),
    )


def gpu_tensor_tflops(gpu: GPUSpec, compute_dtype: str) -> tuple[Optional[float], Optional[str]]:
    """Datasheet dense throughput for the serving dtype, with a note when falling back."""
    if compute_dtype == "fp8":
        if gpu.fp8_tflops:
            return gpu.fp8_tflops, None
        return gpu.fp16_tflops, "No native FP8 tensor cores: FP8 weights run through weight-only kernels at FP16 speed."
    if compute_dtype == "int8":
        if gpu.int8_tops:
            return gpu.int8_tops, None
        return gpu.fp16_tflops, "No INT8 tensor throughput listed; using FP16."
    if compute_dtype == "bf16":
        if gpu.bf16_tflops:
            return gpu.bf16_tflops, None
        return gpu.fp16_tflops, "No BF16 support: model would be served in FP16."
    if compute_dtype == "fp16":
        return gpu.fp16_tflops, None
    return None, f"No {compute_dtype.upper()} throughput data in catalog."
