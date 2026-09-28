"""Deployment spec + OpenStack flavor compilation (generated only - never deployed).

The deployment spec is infrastructure-neutral ("what this AI VM needs"). The compiler turns it
into an OpenStack flavor recommendation. PCI aliases and Placement traits are site configuration:
they are used only when configured on the GPU / vGPU profile, never guessed. Anything missing
is listed under `requires_configuration`.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import yaml

from ..schemas import Candidate, GPUSpec, VGPUProfile
from .config import EngineSettings


def _round_up(value: float, step: int) -> int:
    return int(math.ceil(value / step) * step)


def deployment_spec(
    *,
    candidate: Candidate,
    gpu: GPUSpec,
    profile: Optional[VGPUProfile],
    ai_flavor: str,
    display_name: str,
    slug: str,
    workload_type: str,
    model: dict[str, Any],
    workload: dict[str, Any],
    weights_gb: float,
    required_vram_gb: float,
    settings: EngineSettings,
    gpu_requirement: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    o = settings.openstack
    if candidate.kind == "vgpu":
        vcpus, min_ram = o.vgpu_vcpus, o.min_ram_gb_vgpu
    else:
        vcpus, min_ram = min(o.max_vcpus, o.vcpus_per_gpu * candidate.count), o.min_ram_gb_full
    ram_gb = _round_up(max(min_ram, weights_gb * o.ram_weights_factor + o.ram_base_gb), o.ram_round_gb)
    disk_gb = max(o.min_disk_gb, _round_up(weights_gb * o.disk_weights_factor, 10))
    gpu_block: dict[str, Any] = {
        "mode": {"vgpu": "vgpu", "full_gpu": "dedicated", "multi_gpu": "multi_gpu"}[candidate.kind],
        "count": 1 if candidate.kind == "vgpu" else candidate.count,
        "vendor": gpu.vendor,
        "physical_gpu": gpu.display_name,
        "physical_gpu_id": gpu.id,
        "minimum_vram_gb": round(profile.vram_gb, 1) if profile else int(gpu.vram_gb),
        "required_vram_gb": round(required_vram_gb, 1),
    }
    if profile:
        gpu_block["vgpu_profile"] = profile.profile_name
        gpu_block["sharing"] = profile.sharing_mode
    if candidate.kind == "multi_gpu":
        gpu_block["interconnect"] = gpu.interconnect_type or "unknown"
    return {
        "workload_type": workload_type,
        "ai_flavor": ai_flavor,
        "display_name": display_name,
        "flavor_slug": slug,
        "model": model,
        "workload": workload,
        "vm": {"vcpus": vcpus, "ram_gb": ram_gb, "disk_gb": disk_gb},
        "gpu": gpu_block,
        "gpu_requirement": gpu_requirement or {},
        "recommendation": {
            "vendor": gpu.vendor,
            "gpu_model": gpu.model,
            "gpu_count": gpu_block["count"],
            "mode": gpu_block["mode"],
        },
        "confidence": candidate.confidence,
    }


def compile_openstack_flavor(
    spec: dict[str, Any],
    *,
    pci_alias: Optional[str],
    vgpu_trait: Optional[str],
    performance: bool = False,
) -> dict[str, Any]:
    """OpenStackFlavorCompiler core: AIFlavorDeploymentSpec -> Nova flavor recommendation."""
    g = spec["gpu"]
    extra: dict[str, str] = {}
    needs: list[str] = []
    if g["mode"] == "vgpu":
        extra["resources:VGPU"] = "1"  # standard Nova resource class
        if vgpu_trait:
            extra[f"trait:{vgpu_trait}"] = "required"
        else:
            needs.append(
                f"Placement trait selecting vGPU profile {g.get('vgpu_profile')} "
                "(set it on the profile in GPU Catalog → vGPU profiles)"
            )
        gpu_mode = "VGPU"
    else:
        if pci_alias:
            extra["pci_passthrough:alias"] = f"{pci_alias}:{g['count']}"
        else:
            needs.append(
                f"Nova PCI alias for {g['physical_gpu']} (set it on the GPU in GPU Catalog; must match [pci] alias in nova.conf)"
            )
        extra["hw:mem_page_size"] = "large"
        gpu_mode = "PCI_PASSTHROUGH"
    if performance:
        extra["hw:cpu_policy"] = "dedicated"
    return {
        "name": spec["flavor_slug"],
        "vcpus": spec["vm"]["vcpus"],
        "ram_mb": spec["vm"]["ram_gb"] * 1024,
        "disk_gb": spec["vm"]["disk_gb"],
        "gpu": {
            "mode": gpu_mode,
            "count": g["count"],
            "minimum_vram_gb": g["minimum_vram_gb"],
            **({"profile_class": f"{g['minimum_vram_gb']:g} GB vGPU class"} if g["mode"] == "vgpu" else {}),
        },
        "extra_specs": extra,
        "requires_configuration": needs,
        "properties": {
            "ai:flavor": spec["ai_flavor"],
            "ai:workload_type": spec["workload_type"],
        },
        "note": "Recommendation only - generated, not deployed.",
    }


def to_yaml(data: dict[str, Any]) -> str:
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
