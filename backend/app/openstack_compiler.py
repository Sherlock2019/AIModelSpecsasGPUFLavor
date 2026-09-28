"""OpenStackFlavorCompiler: AIFlavorDeploymentSpec / MLDeploymentSpec -> Nova flavor recommendation.

PCI aliases and vGPU Placement traits come from the inventory (site configuration), never from
assumptions. Nothing is provisioned.
"""

from __future__ import annotations

from typing import Any

from .engine.openstack import compile_openstack_flavor
from .inventory import OpenStackGPUInventory


class OpenStackFlavorCompiler:
    def __init__(self, inventory: OpenStackGPUInventory) -> None:
        self.inventory = inventory

    def compile(self, spec: dict[str, Any]) -> dict[str, Any]:
        for key in ("flavor_slug", "vm", "gpu", "ai_flavor", "workload_type"):
            if key not in spec:
                raise ValueError(f"Deployment spec is missing '{key}'")
        traits = self.inventory.get_gpu_traits()
        g = spec["gpu"]
        priority = (spec.get("workload") or {}).get("performance_priority")
        return compile_openstack_flavor(
            spec,
            pci_alias=traits["gpus"].get(g.get("physical_gpu_id", "")),
            vgpu_trait=traits["vgpu_profiles"].get(g.get("vgpu_profile") or ""),
            performance=priority in ("performance", "maximum") or (spec.get("workload") or {}).get("task") in ("training", "fine_tuning"),
        )
