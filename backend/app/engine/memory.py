"""Weight memory, KV cache and per-GPU memory breakdown."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Optional

from ..schemas import MemoryBreakdown, ModelSpec
from .config import EngineSettings, FrameworkConfig
from .units import GB, gb, gib, r


class SizingError(ValueError):
    """Raised for inputs the engine cannot size (unknown precision, bad context, ...)."""


# --------------------------------------------------------------------------- weights


@dataclass(frozen=True)
class WeightMemory:
    parameters_b: float
    bytes_per_parameter: float
    overhead_factor: float

    @property
    def raw_bytes(self) -> float:
        return self.parameters_b * 1e9 * self.bytes_per_parameter

    @property
    def bytes(self) -> float:
        return self.raw_bytes * self.overhead_factor


def weight_memory(
    model: ModelSpec, precision: str, settings: EngineSettings, overhead_override: Optional[float] = None
) -> WeightMemory:
    """weights = parameters x bytes_per_parameter x quantization_overhead_factor.

    Always uses TOTAL parameters - for MoE every expert must be resident in VRAM.
    """
    cfg = settings.precision.get(precision)
    if cfg is None:
        raise SizingError(f"Unknown precision '{precision}'. Known: {', '.join(settings.precision)}")
    factor = overhead_override if overhead_override is not None else cfg.default_overhead_factor
    return WeightMemory(model.total_parameters_b, cfg.bytes_per_parameter, factor)


# --------------------------------------------------------------------------- KV cache


@dataclass(frozen=True)
class KVCache:
    method: Literal["exact", "mla", "heuristic"]
    kv_precision: str
    bytes_per_element: float
    bytes_per_token: float
    head_dim: Optional[int]
    tokens_per_sequence: int
    sized_tokens: int
    formula: str
    # How many ways the KV cache can be split under tensor parallelism
    # (None = unknown -> assume it shards across every GPU).
    shard_limit: Optional[int]

    @property
    def bytes(self) -> float:
        return self.bytes_per_token * self.sized_tokens


def resolve_kv_precision(requested: str, precision: str, settings: EngineSettings) -> str:
    if requested != "auto":
        if requested not in settings.kv_precision:
            raise SizingError(f"Unknown KV-cache precision '{requested}'")
        return requested
    # Serving stacks keep the KV cache in the activation dtype unless told otherwise.
    return "bf16" if precision == "bf16" else "fp16"


def kv_bytes_per_token(model: ModelSpec, bytes_per_element: float, settings: EngineSettings):
    """Returns (bytes_per_token, method, head_dim, formula, shard_limit)."""
    if model.attention_type == "mla" and model.has_exact_kv_metadata:
        # MLA caches one compressed latent + one decoupled RoPE key per layer, shared by all heads.
        width = model.kv_lora_rank + model.qk_rope_head_dim
        bpt = model.num_layers * width * bytes_per_element
        formula = (
            f"{model.num_layers} layers x ({model.kv_lora_rank} latent + {model.qk_rope_head_dim} rope)"
            f" x {bytes_per_element:g} B = {bpt:,.0f} B/token"
        )
        return bpt, "mla", None, formula, 1  # latent is replicated on every TP rank
    if model.attention_type == "standard" and model.has_exact_kv_metadata:
        hd = model.resolved_head_dim
        bpt = 2 * model.num_layers * model.kv_heads * hd * bytes_per_element
        formula = (
            f"2 (K,V) x {model.num_layers} layers x {model.kv_heads} KV heads x {hd} head_dim"
            f" x {bytes_per_element:g} B = {bpt:,.0f} B/token"
        )
        return bpt, "exact", hd, formula, model.kv_heads
    coeff = settings.heuristic_kv_bytes_per_token_coefficient
    bpt = coeff * math.sqrt(model.total_parameters_b) * (bytes_per_element / 2.0)
    formula = (
        f"HEURISTIC: {coeff:,.0f} x sqrt({model.total_parameters_b:g}B params) x ({bytes_per_element:g}/2)"
        f" = {bpt:,.0f} B/token"
    )
    return bpt, "heuristic", None, formula, None


def kv_cache(
    model: ModelSpec,
    kv_precision: str,
    context_length: int,
    concurrency: int,
    context_mode: str,
    avg_input: int,
    avg_output: int,
    settings: EngineSettings,
) -> KVCache:
    """KV = bytes/token x sized tokens.

    max mode:       every sequence holds the full context -> context x concurrency.
    realistic mode: effective sequence = min(context, avg_in + avg_out); the pool must hold
                    concurrency x effective tokens, and at least one full-context request.
    """
    kv_cfg = settings.kv_precision[kv_precision]
    bpt, method, hd, formula, shard_limit = kv_bytes_per_token(model, kv_cfg.bytes_per_element, settings)
    if context_mode == "max":
        per_seq = context_length
        sized = context_length * concurrency
    else:
        per_seq = max(1, min(context_length, avg_input + avg_output))
        sized = max(context_length, per_seq * concurrency)
    return KVCache(method, kv_precision, kv_cfg.bytes_per_element, bpt, hd, per_seq, sized, formula, shard_limit)


# --------------------------------------------------------------------------- breakdown


@dataclass(frozen=True)
class Breakdown:
    weights: float
    kv: float
    runtime: float
    workspace: float
    communication: float
    safety_margin_percent: float
    activations: float = 0.0
    gradients: float = 0.0
    optimizer: float = 0.0

    @property
    def subtotal(self) -> float:
        return (
            self.weights
            + self.kv
            + self.activations
            + self.gradients
            + self.optimizer
            + self.runtime
            + self.workspace
            + self.communication
        )

    @property
    def headroom(self) -> float:
        return self.subtotal * self.safety_margin_percent / 100.0

    @property
    def required(self) -> float:
        return self.subtotal + self.headroom

    def to_schema(self, unit: str = "gb") -> MemoryBreakdown:
        conv = gb if unit == "gb" else gib
        return MemoryBreakdown(
            model_weights_gb=r(conv(self.weights)),
            kv_cache_gb=r(conv(self.kv)),
            runtime_overhead_gb=r(conv(self.runtime)),
            workspace_gb=r(conv(self.workspace)),
            communication_gb=r(conv(self.communication)),
            subtotal_gb=r(conv(self.subtotal)),
            headroom_gb=r(conv(self.headroom)),
            required_vram_gb=r(conv(self.required)),
            safety_margin_percent=r(self.safety_margin_percent, 1),
            activations_gb=r(conv(self.activations)),
            gradients_gb=r(conv(self.gradients)),
            optimizer_gb=r(conv(self.optimizer)),
        )


def per_gpu_breakdown(
    weights_bytes: float,
    kv: KVCache,
    count: int,
    framework: FrameworkConfig,
    safety_margin_percent: float,
    runtime_override_gb: Optional[float] = None,
    workspace_percent_override: Optional[float] = None,
) -> Breakdown:
    """Memory needed on EACH GPU when the model is split across `count` GPUs (tensor parallel).

    Weights shard evenly. KV shards across min(count, kv_heads) ranks - beyond that KV heads are
    replicated; MLA latents are replicated on every rank. Runtime overhead is per GPU;
    communication buffers (NCCL) only exist when count > 1.
    """
    weights = weights_bytes / count
    kv_ways = count if kv.shard_limit is None else max(1, min(count, kv.shard_limit))
    kv_bytes = kv.bytes / kv_ways
    runtime_gb = runtime_override_gb if runtime_override_gb is not None else framework.base_overhead_gb
    ws_pct = workspace_percent_override if workspace_percent_override is not None else framework.workspace_percent
    workspace = (weights + kv_bytes) * ws_pct / 100.0
    comm = framework.communication_overhead_gb_per_gpu * GB if count > 1 else 0.0
    return Breakdown(weights, kv_bytes, runtime_gb * GB, workspace, comm, safety_margin_percent)
