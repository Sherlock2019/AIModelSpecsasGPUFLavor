"""Infrastructure inventory adapters.

`StaticInventoryAdapter` reads the local catalog database. `OpenStackInventoryAdapter` is the
future integration point (Nova / Placement / Cyborg); it is intentionally not implemented so
the calculator never requires a live cloud.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from .schemas import GPUSpec, VGPUProfile


class InfrastructureInventory(ABC):
    @abstractmethod
    def get_available_gpus(self) -> list[GPUSpec]:
        """GPU models that can be offered (enabled in the catalog)."""

    @abstractmethod
    def get_vgpu_profiles(self) -> list[VGPUProfile]:
        """vGPU profiles that can be offered."""

    @abstractmethod
    def get_gpu_capacity(self) -> dict[str, Any]:
        """gpu_id -> Capacity(installed, available). Empty dict = no inventory data."""

    @abstractmethod
    def get_flavors(self) -> list[dict[str, Any]]:
        """Existing GPU flavors (name, vcpus, ram, extra_specs)."""


class StaticInventoryAdapter(InfrastructureInventory):
    def __init__(self, session: Session, site: Optional[str] = None) -> None:
        self.session = session
        self.site = site

    def get_available_gpus(self) -> list[GPUSpec]:
        from .repository import list_gpus

        return [g for g in list_gpus(self.session) if g.enabled]

    def get_vgpu_profiles(self) -> list[VGPUProfile]:
        from .repository import list_profiles

        return [p for p in list_profiles(self.session) if p.enabled]

    def get_gpu_capacity(self) -> dict[str, Any]:
        from .db import InventoryRow
        from .engine.selection import Capacity

        stmt = select(InventoryRow)
        if self.site:
            stmt = stmt.where(InventoryRow.site == self.site)
        totals: dict[str, list[int]] = {}
        for row in self.session.scalars(stmt):
            t = totals.setdefault(row.gpu_id, [0, 0])
            t[0] += row.installed
            t[1] += row.available
        return {gpu_id: Capacity(installed=v[0], available=v[1]) for gpu_id, v in totals.items()}

    def get_flavors(self) -> list[dict[str, Any]]:
        return []


class OpenStackGPUInventory(ABC):
    """What the OpenStack flavor compiler needs to know about a cloud's GPU estate."""

    @abstractmethod
    def get_gpu_resource_providers(self) -> list[dict[str, Any]]:
        """Resource providers that expose GPUs (name, gpu_id, total, used)."""

    @abstractmethod
    def get_available_vgpu_profiles(self) -> list[dict[str, Any]]:
        """vGPU profiles with free capacity (profile_name, physical_gpu, available_instances)."""

    @abstractmethod
    def get_available_pci_gpus(self) -> list[dict[str, Any]]:
        """Pass-through GPUs with free capacity (gpu_id, available)."""

    @abstractmethod
    def get_gpu_traits(self) -> dict[str, dict[str, Optional[str]]]:
        """Site configuration: {"gpus": {gpu_id: pci_alias}, "vgpu_profiles": {profile: trait}}."""

    @abstractmethod
    def get_flavors(self) -> list[dict[str, Any]]:
        """Existing Nova flavors."""

    @abstractmethod
    def get_capacity(self) -> dict[str, Any]:
        """gpu_id -> Capacity(installed, available)."""


class StaticOpenStackGPUInventory(OpenStackGPUInventory):
    """MVP: answers from the calculator's own catalog and inventory tables (no live cloud)."""

    def __init__(self, session: Session, site: Optional[str] = None) -> None:
        self._static = StaticInventoryAdapter(session, site)

    def get_gpu_resource_providers(self) -> list[dict[str, Any]]:
        cap = self._static.get_gpu_capacity()
        return [
            {"name": f"static:{gpu_id}", "gpu_id": gpu_id, "total": c.installed, "used": c.installed - c.available}
            for gpu_id, c in cap.items()
        ]

    def get_available_vgpu_profiles(self) -> list[dict[str, Any]]:
        return [
            {"profile_name": p.profile_name, "physical_gpu": p.physical_gpu, "available_instances": p.available_instances}
            for p in self._static.get_vgpu_profiles()
            if p.available_instances is None or p.available_instances > 0
        ]

    def get_available_pci_gpus(self) -> list[dict[str, Any]]:
        return [{"gpu_id": gpu_id, "available": c.available} for gpu_id, c in self._static.get_gpu_capacity().items() if c.available > 0]

    def get_gpu_traits(self) -> dict[str, dict[str, Optional[str]]]:
        return {
            "gpus": {g.id: g.openstack_pci_alias for g in self._static.get_available_gpus()},
            "vgpu_profiles": {p.profile_name: p.openstack_trait for p in self._static.get_vgpu_profiles()},
        }

    def get_flavors(self) -> list[dict[str, Any]]:
        return []

    def get_capacity(self) -> dict[str, Any]:
        return self._static.get_gpu_capacity()


class PlacementOpenStackGPUInventory(OpenStackGPUInventory):  # pragma: no cover - future work
    """Future: Nova + Placement (+ Cyborg) backed inventory. See OpenStackInventoryAdapter below."""

    def __init__(self, *_, **__) -> None:
        raise NotImplementedError("Placement-backed inventory is not implemented in this version")

    def get_gpu_resource_providers(self):
        raise NotImplementedError

    def get_available_vgpu_profiles(self):
        raise NotImplementedError

    def get_available_pci_gpus(self):
        raise NotImplementedError

    def get_gpu_traits(self):
        raise NotImplementedError

    def get_flavors(self):
        raise NotImplementedError

    def get_capacity(self):
        raise NotImplementedError


class OpenStackInventoryAdapter(InfrastructureInventory):  # pragma: no cover - future work
    """Planned sources:

    - get_available_gpus: Placement resource providers with PCI (`PCI_DEVICE`) or `VGPU`
      inventories, mapped to catalog GPUs via PCI vendor/product IDs or custom traits.
    - get_vgpu_profiles: Nova `[devices] enabled_mdev_types` per compute node + Placement
      `CUSTOM_VGPU_*` traits.
    - get_gpu_capacity: Placement inventories minus allocations (`total - used - reserved`).
    - get_flavors: Nova flavors with `pci_passthrough:alias` or `resources:VGPU` extra specs;
      Cyborg device profiles where accelerators are managed by Cyborg.
    """

    def __init__(self, *_, **__) -> None:
        raise NotImplementedError("OpenStack inventory integration is not implemented in this version")

    def get_available_gpus(self) -> list[GPUSpec]:
        raise NotImplementedError

    def get_vgpu_profiles(self) -> list[VGPUProfile]:
        raise NotImplementedError

    def get_gpu_capacity(self) -> dict[str, Any]:
        raise NotImplementedError

    def get_flavors(self) -> list[dict[str, Any]]:
        raise NotImplementedError
