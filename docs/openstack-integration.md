# OpenStack integration

Nothing is provisioned. The application produces a **deployment spec** (what the AI VM needs) and compiles it into a
**Nova flavor recommendation** for your team to review and create.

## 1. Deployment spec

Every LLM, AI flavor and custom ML result carries `deployment_spec`. **Deploy** stores it through `POST /api/v1/deployment-specs`,
and it is listed under *Saved & Specs*. The spec is infrastructure-neutral:

```json
{
  "workload_type": "llm | custom_ml",
  "ai_flavor": "AI-70B-Q4-PRO",
  "display_name": "AI-70B-PRO",
  "flavor_slug": "ai-70b-q4-pro",
  "model": {"id": "llama-3.3-70b-instruct", "precision": "int4", "context": 32768},
  "workload": {"task": "inference", "framework": "vllm", "concurrency": 20},
  "vm": {"vcpus": 16, "ram_gb": 80, "disk_gb": 100},
  "gpu": {"mode": "dedicated", "count": 1, "vendor": "NVIDIA", "physical_gpu": "NVIDIA H100 NVL 94GB",
          "physical_gpu_id": "nvidia-h100-nvl", "minimum_vram_gb": 94, "required_vram_gb": 88.2},
  "gpu_requirement": {"minimum_vram_gb": 88.2, "compute_class": "very_high", "bandwidth_class": "high", "gpu_count": 1, "mode": "full_gpu"},
  "recommendation": {"vendor": "NVIDIA", "gpu_model": "H100 NVL 94GB", "gpu_count": 1, "mode": "dedicated"}
}
```

VM sizing rules live in Settings → `openstack`:

| Setting | Rule |
|---|---|
| vCPUs | 8 for vGPU; 16 per GPU for dedicated, capped at 128 |
| RAM | `max(min, weights × 1.5 + 16 GB)`, rounded up to 16 GB; the minimum is 32 GB for vGPU and 64 GB for dedicated |
| Disk | `max(100 GB, weights × 2)` |

## 2. OpenStackFlavorCompiler

`app/openstack_compiler.py` turns a spec into a flavor. It reads site configuration from the inventory adapter
(`get_gpu_traits()`):

| Mode | Extra specs | Needs from your cloud |
|---|---|---|
| vGPU | `resources:VGPU=1` (standard Nova resource class), `trait:<profile trait>=required` | The profile's **Placement trait** (GPU Inventory → vGPU profiles) |
| Dedicated / multi-GPU | `pci_passthrough:alias=<alias>:<count>`, `hw:mem_page_size=large` | The GPU's **PCI alias**, matching `[pci] alias` in `nova.conf` (GPU Inventory → GPUs) |

Performance tiers and training workloads also get `hw:cpu_policy=dedicated`.

**PCI aliases and traits are never guessed.** The bundled catalog leaves them empty. Until you set them, the compiled
flavor omits those extra specs and lists what is missing under `requires_configuration`. The UI shows these items
after Deploy.

```yaml
name: ai-14b-q4-shared
vcpus: 8
ram_mb: 32768
disk_gb: 100
gpu: {mode: VGPU, count: 1, minimum_vram_gb: 24, profile_class: 24 GB vGPU class}
extra_specs:
  resources:VGPU: '1'
  trait:CUSTOM_VGPU_L40S_24C: required      # present only once configured
requires_configuration: []
properties: {ai:flavor: AI-14B-Q4-SHARED, ai:workload_type: llm}
```

## 3. Inventory adapters (`app/inventory.py`)

```python
class OpenStackGPUInventory(ABC):
    def get_gpu_resource_providers(self): ...
    def get_available_vgpu_profiles(self): ...
    def get_available_pci_gpus(self): ...
    def get_gpu_traits(self): ...        # {"gpus": {gpu_id: pci_alias}, "vgpu_profiles": {name: trait}}
    def get_flavors(self): ...
    def get_capacity(self): ...
```

- **`StaticOpenStackGPUInventory`** (in use) answers from the application's own tables: GPU catalog, vGPU profiles
  (including `available_instances`) and site inventory.
- **`PlacementOpenStackGPUInventory`** (future) will read Nova, Placement and Cyborg: resource providers with `PCI_DEVICE` / `VGPU`
  inventories (`total − used − reserved`), `enabled_mdev_types`, traits and existing flavors.

The sizing engine only consumes a catalog snapshot, so a live adapter needs no engine changes.

## 4. Existing capacity

Recommendations already respect capacity:

- A vGPU profile with `available_instances = 0` is skipped with *"Suitable vGPU profile currently unavailable"*, and the next larger profile
  or a dedicated GPU is recommended.
- A configuration needing more GPUs than are free is skipped with *"Capacity: only N of M GPUs available"*.

This behaviour is controlled by Settings → `ranking.exclude_insufficient_capacity`.
