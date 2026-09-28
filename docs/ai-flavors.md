# AI VM flavors

AI flavors are the **customer product**: AI-1B, AI-3B, AI-8B, AI-14B, AI-32B, AI-70B, AI-120B, AI-200B and AI-400B.
They are defined in `backend/app/data/ai_flavors.yaml` (seeded into the `ai_flavors` table) and managed on
**Admin → AI Flavors** or through `/api/v1/ai-flavors`.

A flavor never sizes anything itself. The layers stay separate:

```
customer product (AIFlavorDefinition) → sizing engine (engine.calculate) → resource requirement
  → matcher (vGPU / full / multi, NVIDIA + AMD, capacity) → deployment spec → OpenStack flavor compiler
```

## Definition

| Field | Meaning |
|---|---|
| `id`, `display_name`, `description`, `group` | `ai-70b`, "AI-70B", "Enterprise AI VM", small / medium / large / ultra |
| `parameter_floor_b`, `parameter_ceiling_b` | Model class bounds (total parameters, billions) |
| `baseline_precision` | Usually INT4. It is omitted from the short name when the request uses it |
| `default_vram_min_gb`, `default_vram_max_gb` | **Catalog guidance only.** The calculator is the source of truth |
| `recommended_use_cases`, `example_model_ids` | Shown on the catalog card |
| `allow_vgpu`, `allow_full_gpu`, `allow_multi_gpu` | Product policy applied to the sizing request |
| `default_context_length`, `default_concurrency`, `default_performance_priority` | Pre-filled workload |
| `performance_tier`, `commercial_sku`, `enabled` | Commercial metadata |

## Which tier does a model belong to?

A model belongs to the smallest enabled tier whose ceiling × (1 + tolerance) covers its **total** parameters (MoE
included). The tolerance defaults to 10% (Settings → `flavor_tier_tolerance_percent`), so a "14B" model with 14.8B
parameters is still AI-14B.

## Sizing a flavor (`POST /api/v1/ai-flavors/size`)

1. The model's own tier is found by parameter count.
2. That tier's policy is applied: `allow_vgpu`, `allow_full_gpu` (→ `allow_single_gpu`) and `allow_multi_gpu`.
3. `engine.calculate()` runs. This is the same function the LLM calculator uses.
4. If nothing fits under that policy (for example AI-14B does not offer multi-GPU), the next larger tier's policy is tried.
5. The result is compared with the tier the customer picked:

| `tier_status` | Meaning | UI |
|---|---|---|
| `fits` | Same tier | — |
| `exceeds` | Model or workload needs a larger tier | "This model exceeds the normal AI-32B tier." **Upgrade to AI-70B** |
| `oversized` | A smaller tier is enough | "This workload can run on a smaller flavor." **Switch to AI-8B** |
| `model_tier` | No tier requested (Mode B) | "Qwen3 14B belongs to AI-14B." |
| `no_tier` | Larger than the biggest tier | Sized as AI-XL |

The customer's flavor is never changed silently.

## Naming

| Variant | Infrastructure |
|---|---|
| `-SHARED` | vGPU (time-sliced or MIG) |
| `-PRO` | One dedicated GPU (PCI passthrough) |
| `-MULTI` | Multi-GPU |

- Full name: `AI-{tier}-{precision tag}-{variant}`, e.g. `AI-70B-Q4-PRO` or `AI-70B-BF16-MULTI`.
- Short name: the precision tag is dropped when it equals the tier baseline, e.g. `AI-70B-PRO` or `AI-14B-SHARED`.
- Catalog name: `ai.llm.70b.int4.pro`. OpenStack flavor name: `ai-70b-q4-pro`.
- Custom ML workloads: `AI-CUSTOM-{INF|TRAIN|FT}-{memory}`, e.g. `AI-CUSTOM-INF-24G` or `AI-CUSTOM-TRAIN-2X80G`.

## Deploy → deployment spec

**Deploy** (`POST /api/v1/deployment-specs`) stores the deployment spec and the OpenStack flavor compiled from it. **Nothing is
provisioned.** Example (abridged):

```json
{
  "workload_type": "llm",
  "ai_flavor": "AI-14B-Q4-SHARED",
  "display_name": "AI-14B-SHARED",
  "model": {"id": "qwen3-14b", "precision": "int4", "context": 32768},
  "vm": {"vcpus": 8, "ram_gb": 32, "disk_gb": 100},
  "gpu": {"mode": "vgpu", "count": 1, "physical_gpu": "NVIDIA L40S", "vgpu_profile": "L40S-24C", "minimum_vram_gb": 24},
  "gpu_requirement": {"minimum_vram_gb": 17.2, "compute_class": "medium", "bandwidth_class": "low", "gpu_count": 1, "mode": "vgpu"},
  "recommendation": {"vendor": "NVIDIA", "gpu_model": "L40S", "gpu_count": 1, "mode": "vgpu"}
}
```

See [openstack-integration.md](openstack-integration.md) for the compiler.
