# GPU catalog

The source is `backend/app/data/gpus.yaml`, and it is editable on the **GPU Catalog** page.

| Field | Notes |
|---|---|
| `vram_gb` | nominal marketing capacity |
| `usable_vram_mib` | framebuffer reported by `nvidia-smi`; used for fit checks when present |
| `memory_bandwidth_gbps` | datasheet HBM/GDDR bandwidth |
| `fp16_tflops`, `bf16_tflops`, `fp8_tflops`, `int8_tops` | **dense** tensor throughput (not the sparsity figure) |
| `supports_vgpu`, `supports_mig`, `supports_nvlink` | `null` = unknown. Unknown vGPU support is treated as "no" |
| `power_watts` | TDP |
| `cost_per_hour` | your price. `null` = not priced (default for all GPUs) |
| `availability` | `available`, `limited`, `unavailable` or `unknown`. Unavailable GPUs are heavily penalised |
| `openstack_pci_alias` | used in `pci_passthrough:alias` |
| `metadata_source`, `metadata_status` | datasheet URL and status |

Unknown values are left `null`, never filled in. B300 and RTX PRO 6000 Blackwell ship without tensor throughput, so their
compute adequacy shows as *unknown* and gets a neutral ranking penalty.

## Included GPUs

- **NVIDIA:** T4, A10, L4, L40S, RTX PRO 6000 Blackwell Server, A100 40GB, A100 80GB, H100 80GB (SXM), H100 NVL
  94GB, H200 141GB, B200 180GB, GB200 (per GPU), B300 288GB.
- **AMD Instinct:** MI210, MI250 (per GCD), MI250X (per GCD), MI300X, MI325X.

MI250 and MI250X cards contain two GCDs that ROCm exposes as two separate 64 GB GPUs, so they are catalogued per GCD.
A "2 × MI250X (per GCD)" match is therefore one physical card. TensorRT-LLM is NVIDIA-only, so choosing it as the
framework removes AMD options. vLLM, SGLang and TGI run on both vendors.

`interconnect_type` (PCIe, NVLink, NVLink bridge, Infinity Fabric …) decides whether a multi-GPU split gets a
"no high-speed GPU link" warning.

## Plain-language classes

The UI shows classes instead of raw numbers. Thresholds are in Settings:

| Class | Based on | Thresholds |
|---|---|---|
| **AI compute** (GPU) | dense FP16/BF16 TFLOPS | low <100, medium <300, high <700, very high <1500, extreme |
| **GPU memory speed** (GPU) | memory bandwidth, GB/s | low <500, medium <1200, high <3000, very high |
| **GPU memory speed** (workload) | required decode bandwidth, GB/s | low <200, medium <800, high <2500, very high |

## Inventory

The **Inventory** tab stores `installed` and `available` counts per GPU and site. The static adapter sums them per GPU
(or filters by `?site=` on `/calculate`). When inventory exists, options are ranked by availability:
available 0, not in inventory 1, too few free 2. Options are penalised, not removed, because this is a sizing tool
and not a scheduler. Set `ranking.use_inventory: false` in Settings to ignore inventory.

The seeded demo site (L40S 12/7, H100 4/1) is illustrative. Replace it with your own data.

## Costs

Set `cost_per_hour` per GPU (and optionally `license_cost_per_hour` per vGPU profile and `cost.vm_cost_per_hour` in
Settings) to enable hourly, daily and monthly estimates and cost-aware ranking. See [formulas.md](formulas.md#7-cost).
