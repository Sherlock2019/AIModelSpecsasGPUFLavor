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
    flavor_size_classes: list[FlavorSizeClass]
    openstack: OpenStackConfig = Field(default_factory=OpenStackConfig)
    cost: CostConfig = Field(default_factory=CostConfig)
    ranking: RankingConfig = Field(default_factory=RankingConfig)


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
