# vGPU

A vGPU profile gives a VM a fixed slice of a physical GPU's framebuffer. The catalog (`vgpu_profiles.yaml`)
contains NVIDIA C-series (compute) profiles for T4, A10, L4, L40S, A100 80GB and H100 80GB.

| Field | Meaning |
|---|---|
| `profile_name` | e.g. `L40S-24C`, `H100XM-3-40C` |
| `physical_gpu` | catalog GPU id |
| `vram_gb` | framebuffer in **GiB** |
| `max_instances_per_gpu` | how many of this profile fit on one GPU |
| `sharing_mode` | `time_sliced` or `mig` |
| `mig_compute_slices` | MIG compute slices out of 7 |
| `license_required`, `license_cost_per_hour` | vGPU software licensing |

Profile names and limits are transcribed from the NVIDIA Virtual GPU Software User Guide. Verify them against
the vGPU release you run, then disable profiles your policy does not offer.

## Suitability rules

Equal VRAM does **not** mean equal performance. A profile is considered only if all of the following hold:

1. **VRAM fit:** `required VRAM (single GPU, with safety margin) ≤ profile framebuffer`.
2. **Policy:**
   - vGPU is allowed in the options.
   - No dedicated GPU is required.
   - There is no tensor parallelism.
   - The profile's sharing mode is allowed by the priority:

| Priority | Allowed sharing |
|---|---|
| Economy | time-sliced, MIG |
| Balanced | time-sliced, MIG |
| Performance | MIG only |
| Maximum | none |

3. **Performance tier:** the profile's worst-case share of the GPU is used for bandwidth and compute adequacy.
   A time-sliced profile gets 1/`max_instances` and a MIG profile gets slices/7. A slow share lowers the ranking, and
   a vGPU recommendation always carries the warning "vGPU profile fits memory but performance is not guaranteed".
4. **Availability:** the physical GPU's inventory.

Only the smallest fitting profile per GPU and sharing mode becomes a candidate. The largest profile of each
non-fitting mode is listed under *Considered and rejected*, for example "L40S-48C: profile too small".

For example, Qwen3 14B at INT4 with 8K context, 2 concurrent requests and Economy priority needs:

- 8.1 GB of weights
- 1.6 GB of KV cache
- 2.0 GB of runtime
- 15% headroom

The total is about 14 GB. That fits the **L40S-16C** profile (16 GiB) and gives the **AI-14B-Q4-SHARED** flavor. A full GPU is not required.
