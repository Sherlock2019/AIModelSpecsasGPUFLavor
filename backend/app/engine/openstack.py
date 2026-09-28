"""OpenStack flavor recommendation (generated only - never deployed)."""

from __future__ import annotations

import math
import re
from typing import Any

import yaml

from ..schemas import Candidate, GPUSpec, ModelSpec, VGPUProfile
from .config import EngineSettings


def _round_up(value: float, step: int) -> int:
    return int(math.ceil(value / step) * step)


def openstack_flavor(
    candidate: Candidate,
    slug: str,
    model: ModelSpec,
    gpu: GPUSpec,
    profile: VGPUProfile | None,
    precision: str,
    weights_gb: float,
    context_length: int,
    concurrency: int,
    framework: str,
    priority: str,
    settings: EngineSettings,
) -> dict[str, Any]:
    o = settings.openstack
    extra_specs: dict[str, str] = {}
    if candidate.kind == "vgpu" and profile is not None:
        vcpus = o.vgpu_vcpus
        min_ram = o.min_ram_gb_vgpu
        gpu_block: dict[str, Any] = {
            "mode": "vgpu",
            "vgpu_profile": profile.profile_name,
            "physical_gpu": gpu.display_name,
            "sharing": profile.sharing_mode,
            "minimum_vram_gb": profile.vram_gb,
            "count": 1,
        }
        trait = "CUSTOM_VGPU_" + re.sub(r"[^A-Z0-9]", "_", profile.profile_name.upper())
        extra_specs["resources:VGPU"] = "1"
        extra_specs[f"trait:{trait}"] = "required"
    else:
        vcpus = min(o.max_vcpus, o.vcpus_per_gpu * candidate.count)
        min_ram = o.min_ram_gb_full
        gpu_block = {
            "mode": "pci_passthrough",
            "model": gpu.display_name,
            "minimum_vram_gb": int(gpu.vram_gb),
            "count": candidate.count,
        }
        extra_specs["pci_passthrough:alias"] = f"{gpu.openstack_pci_alias or gpu.id}:{candidate.count}"
        extra_specs["hw:mem_page_size"] = "large"
        if priority in ("performance", "maximum"):
            extra_specs["hw:cpu_policy"] = "dedicated"

    ram_gb = _round_up(max(min_ram, weights_gb * o.ram_weights_factor + o.ram_base_gb), o.ram_round_gb)
    disk_gb = max(o.min_disk_gb, _round_up(weights_gb * o.disk_weights_factor, 10))

    model_block: dict[str, Any] = {
        "name": model.name,
        "max_parameters_b": int(math.ceil(model.total_parameters_b)),
        "architecture": model.architecture,
        "precision": precision,
        "max_context": context_length,
    }
    if model.architecture == "moe":
        model_block["active_parameters_b"] = int(math.ceil(model.active_parameters_b))

    return {
        "name": slug,
        "vcpus": vcpus,
        "ram_mb": ram_gb * 1024,
        "disk_gb": disk_gb,
        "gpu": gpu_block,
        "extra_specs": extra_specs,
        "model": model_block,
        "workload": {"type": "inference", "framework": framework, "concurrency": concurrency},
        "note": "Recommendation only - generated, not deployed.",
    }


def to_yaml(data: dict[str, Any]) -> str:
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
