"""Typed view over the configurable engine settings (seeded from data/defaults.yaml)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


class PrecisionConfig(BaseModel):
    label: str
    bytes_per_parameter: float = Field(gt=0)
    default_overhead_factor: float = Field(default=1.0, ge=1.0)
    flavor_tag: str
    compute: str  # fp32 | fp16 | bf16 | fp8 | int8


class KVPrecisionConfig(BaseModel):
    label: str
    bytes_per_element: float = Field(gt=0)


class FrameworkConfig(BaseModel):
    label: str
    base_overhead_gb: float = Field(ge=0)
    workspace_percent: float = Field(ge=0, le=100)
    communication_overhead_gb_per_gpu: float = Field(default=0.5, ge=0)
    nvidia_only: bool = False  # e.g. TensorRT / TensorRT-LLM


class EnvironmentConfig(BaseModel):
    label: str
    default_safety_margin_percent: float = Field(ge=0, le=100)


class RankingWeights(BaseModel):
    units: float = 10
    waste: float = 4
    tight: float = 4
    bandwidth: float = 6
    compute: float = 4
    availability: float = 3
    cost: float = 4
    bandwidth_bonus: float = 0
    vgpu: float = 0


class PriorityConfig(BaseModel):
    label: str
    safety_margin_percent: float = Field(ge=0, le=100)
    tokens_per_second_per_user: float = Field(gt=0)
    comfortable_utilization: float = Field(gt=0, le=1)
    allowed_vgpu_sharing: list[str] = Field(default_factory=list)
    escalate_for_throughput: bool = False
    flavor_tier: str = "prod"
    weights: RankingWeights = Field(default_factory=RankingWeights)


class ContextConfig(BaseModel):
    default_average_input_tokens: int = 4000
    default_average_output_tokens: int = 1000


class EfficiencyConfig(BaseModel):
    bandwidth_efficiency: float = Field(default=0.7, gt=0, le=1)
    compute_mfu: float = Field(default=0.35, gt=0, le=1)
    mig_total_slices: int = 7


class Threshold(BaseModel):
    name: str
    max: Optional[float] = None


class FlavorSizeClass(BaseModel):
    label: str
    max_total_b: float


class OpenStackConfig(BaseModel):
    vgpu_vcpus: int = 8
    vcpus_per_gpu: int = 16
    max_vcpus: int = 128
    min_ram_gb_vgpu: int = 32
    min_ram_gb_full: int = 64
    ram_weights_factor: float = 1.5
    ram_base_gb: int = 16
    ram_round_gb: int = 16
    min_disk_gb: int = 100
    disk_weights_factor: float = 2.0


class CostConfig(BaseModel):
    hours_per_day: float = 24
    hours_per_month: float = 730
    vm_cost_per_hour: Optional[float] = None
    unknown_cost_penalty: float = 0.5


class RankingConfig(BaseModel):
    use_inventory: bool = True
    # Skip options whose capacity is known to be short right now (vGPU profile with 0 free
    # instances, or fewer free GPUs than needed) and say why, instead of recommending them.
    exclude_insufficient_capacity: bool = True


class OptimizerConfig(BaseModel):
    label: str
    states: float = 2  # optimizer state tensors per parameter (Adam m and v = 2)
    bytes_per_param: Optional[float] = None  # fixed estimate instead of states x optimizer precision


def _ml_frameworks() -> dict[str, FrameworkConfig]:
    return {
        "pytorch": FrameworkConfig(label="PyTorch", base_overhead_gb=1.5, workspace_percent=10, communication_overhead_gb_per_gpu=1.0),
        "tensorflow": FrameworkConfig(label="TensorFlow", base_overhead_gb=2.0, workspace_percent=10, communication_overhead_gb_per_gpu=1.0),
        "jax": FrameworkConfig(label="JAX", base_overhead_gb=1.5, workspace_percent=10, communication_overhead_gb_per_gpu=1.0),
        "onnxruntime": FrameworkConfig(label="ONNX Runtime", base_overhead_gb=1.0, workspace_percent=5, communication_overhead_gb_per_gpu=0.5),
        "tensorrt": FrameworkConfig(label="TensorRT", base_overhead_gb=1.0, workspace_percent=5, communication_overhead_gb_per_gpu=0.5, nvidia_only=True),
        "other": FrameworkConfig(label="Other", base_overhead_gb=2.0, workspace_percent=10, communication_overhead_gb_per_gpu=1.0),
    }


def _optimizers() -> dict[str, OptimizerConfig]:
    return {
        "adamw": OptimizerConfig(label="AdamW", states=2),
        "adam": OptimizerConfig(label="Adam", states=2),
        "sgd": OptimizerConfig(label="SGD (momentum)", states=1),
        "adafactor": OptimizerConfig(label="Adafactor", states=0, bytes_per_param=0.5),
        "other": OptimizerConfig(label="Other", states=2),
    }


class ActivationHeuristics(BaseModel):
    """Coefficients for ESTIMATED activation memory (never presented as exact).

    Transformer-style estimate per sample: tokens x hidden x layers x coefficient x (bytes/2), after
    Korthikanti et al. 2022 (~34 bytes per token per hidden unit per layer for 16-bit training
    without recomputation). Hidden size / layers are derived from the parameter count when unknown
    (params ~= 12 x layers x hidden^2 with hidden ~= hidden_per_layer x layers).
    """

    inference_bytes_per_token_hidden: float = 16  # live tensors of one layer at a time
    training_bytes_per_token_hidden_layer: float = 34
    checkpointed_bytes_per_token_hidden_layer: float = 2  # only layer inputs kept
    spatial_channels_inference: float = 64  # early conv feature maps at input resolution
    spatial_channels_training: float = 256
    hidden_per_layer: float = 80
    patch_size: int = 16
    audio_tokens_per_second: float = 50
    diffusion_latent_downsample: int = 8
    diffusion_patch: int = 2
    vae_decoder_channels: float = 128


class MLConfig(BaseModel):
    frameworks: dict[str, FrameworkConfig] = Field(default_factory=_ml_frameworks)
    optimizers: dict[str, OptimizerConfig] = Field(default_factory=_optimizers)
    activation: ActivationHeuristics = Field(default_factory=ActivationHeuristics)
    # Inference: requests are expected to finish within this latency (sets the compute target).
    latency_target_ms: dict[str, float] = Field(
        default_factory=lambda: {"economy": 1000, "balanced": 500, "performance": 200, "maximum": 100}
    )
    # Training: optimizer steps per second targeted when no throughput is given.
    training_steps_per_second: dict[str, float] = Field(
        default_factory=lambda: {"economy": 0.1, "balanced": 0.25, "performance": 0.5, "maximum": 1}
    )
    compute_classes: list[Threshold] = Field(  # required TFLOPS
        default_factory=lambda: [
            Threshold(name="low", max=5),
            Threshold(name="medium", max=50),
            Threshold(name="high", max=200),
            Threshold(name="very_high", max=800),
            Threshold(name="extreme", max=None),
        ]
    )
    default_adapter_percent: float = 0.5  # LoRA trainable parameters as % of the base model


class EngineSettings(BaseModel):
    precision: dict[str, PrecisionConfig]
    kv_precision: dict[str, KVPrecisionConfig]
    frameworks: dict[str, FrameworkConfig]
    environments: dict[str, EnvironmentConfig]
    priorities: dict[str, PriorityConfig]
    context: ContextConfig = Field(default_factory=ContextConfig)
    efficiency: EfficiencyConfig = Field(default_factory=EfficiencyConfig)
    heuristic_kv_bytes_per_token_coefficient: float = 46000
    compute_classes: list[Threshold]
    memory_intensity: list[Threshold]
    # Defaults keep older stored settings valid after upgrades.
    bandwidth_classes: list[Threshold] = Field(  # required decode bandwidth, GB/s
        default_factory=lambda: [
            Threshold(name="low", max=200),
            Threshold(name="medium", max=800),
            Threshold(name="high", max=2500),
            Threshold(name="very_high", max=None),
        ]
    )
    gpu_compute_classes: list[Threshold] = Field(  # GPU dense FP16/BF16 TFLOPS
        default_factory=lambda: [
            Threshold(name="low", max=100),
            Threshold(name="medium", max=300),
            Threshold(name="high", max=700),
            Threshold(name="very_high", max=1500),
            Threshold(name="extreme", max=None),
        ]
    )
    gpu_bandwidth_classes: list[Threshold] = Field(  # GPU memory bandwidth, GB/s
        default_factory=lambda: [
            Threshold(name="low", max=500),
            Threshold(name="medium", max=1200),
            Threshold(name="high", max=3000),
            Threshold(name="very_high", max=None),
        ]
    )
    gpu_counts: list[int] = Field(default_factory=lambda: [1, 2, 4, 8, 16])
    gpus_per_node: int = 8
    flavor_size_classes: list[FlavorSizeClass]  # fallback naming when no AI flavor catalog is loaded
    flavor_tier_tolerance_percent: float = Field(default=10, ge=0, le=50)
    openstack: OpenStackConfig = Field(default_factory=OpenStackConfig)
    cost: CostConfig = Field(default_factory=CostConfig)
    ranking: RankingConfig = Field(default_factory=RankingConfig)
    ml: MLConfig = Field(default_factory=MLConfig)


def load_default_settings_dict() -> dict:
    with open(DATA_DIR / "defaults.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@lru_cache(maxsize=1)
def default_settings() -> EngineSettings:
    return EngineSettings.model_validate(load_default_settings_dict())


def classify(value: float, thresholds: list[Threshold]) -> str:
    for t in thresholds:
        if t.max is None or value <= t.max:
            return t.name
    return thresholds[-1].name
