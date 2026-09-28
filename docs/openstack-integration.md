# OpenStack integration

## Flavor output (available now)

Every calculation with a fit produces a flavor recommendation. It appears on the *Infrastructure output* tab, in `openstack` and
`openstack_yaml` in the API, and in the exports. **It is generated only and never deployed.**

Full GPU (PCI passthrough):

```yaml
name: ai-70b-q4-prod
vcpus: 16
ram_mb: 81920
disk_gb: 100
gpu:
  mode: pci_passthrough
  model: NVIDIA H100 80GB
  minimum_vram_gb: 80
  count: 1
extra_specs:
  pci_passthrough:alias: h100:1
  hw:mem_page_size: large
model:
  name: Llama 3.1 70B Instruct
  max_parameters_b: 71
  architecture: dense
  precision: int4
  max_context: 32768
workload:
  type: inference
  framework: vllm
  concurrency: 20
```

vGPU:

```yaml
gpu:
  mode: vgpu
  vgpu_profile: L40S-16C
  physical_gpu: NVIDIA L40S
  sharing: time_sliced
  minimum_vram_gb: 16
  count: 1
extra_specs:
  resources:VGPU: '1'
  trait:CUSTOM_VGPU_L40S_16C: required
```

These sizing rules are configurable under `openstack` in Settings:

| Setting | Rule |
|---|---|
| vCPUs | 8 for vGPU; 16 per GPU for full GPU, capped at 128 |
| RAM | `max(min_ram, weights × 1.5 + 16 GB)`, rounded up to 16 GB (32 GB minimum for vGPU, 64 GB for full GPU) |
| Disk | `max(100 GB, weights × 2)` |
| Extra specs | Performance tiers add `hw:cpu_policy: dedicated` |

The PCI alias comes from each GPU's `openstack_pci_alias` and must match the `[pci] alias` in your Nova config. The vGPU
trait naming (`CUSTOM_VGPU_<PROFILE>`) assumes you tag resource providers per mdev type.

## Inventory adapter (future)

`app/inventory.py` defines:

```python
class InfrastructureInventory(ABC):
    def get_available_gpus(self) -> list[GPUSpec]: ...
    def get_vgpu_profiles(self) -> list[VGPUProfile]: ...
    def get_gpu_capacity(self) -> dict[str, Capacity]: ...
    def get_flavors(self) -> list[dict]: ...
```

`StaticInventoryAdapter` (in use) reads the local database. `OpenStackInventoryAdapter` is a documented stub that
would read the following:

- **Placement:** resource providers with `PCI_DEVICE` / `VGPU` inventories, with `total − used − reserved` giving capacity
- **Nova:** `enabled_mdev_types` per compute node, and existing flavors with `pci_passthrough:alias` or `resources:VGPU`
- **Cyborg:** device profiles, where accelerators are managed by Cyborg
- **Flavor catalog:** mapping recommended AI flavors to existing flavors

The engine only consumes the adapter's output (the `Catalog` snapshot), so a live adapter can be added without engine
changes. No OpenStack cloud is required today.
