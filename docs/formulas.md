# Formulas

Units: the engine works in bytes. **GB = 10⁹ bytes, GiB = 2³⁰ bytes.** Results are reported in both.
GPU capacity uses the `nvidia-smi` framebuffer when known (for example, H100 80GB = 81,559 MiB = 85.5 GB). vGPU
profile sizes are GiB (L40S-24C = 24 GiB = 25.8 GB).

## 1. Model weights

```
weights = total_parameters × bytes_per_parameter × quantization_overhead_factor
```

| Precision | bytes/param | default overhead |
|---|---|---|
| FP32 | 4 | 1.00 |
| FP16 / BF16 | 2 | 1.00 |
| FP8 | 1 | 1.02 |
| INT8 | 1 | 1.05 |
| INT4 | 0.5 | 1.10 |
| MXFP4 | 0.53125 (4.25 bits) | 1.05 |

MoE models always use **total** parameters here.

For example, Llama 3.1 70B (70.55B) at INT4 is 35.3 GB raw and 38.8 GB with overhead.

## 2. KV cache

**Standard attention (MHA / GQA / MQA)**, which uses KV heads, not attention heads:

```
kv_bytes_per_token = 2 × num_layers × num_kv_heads × head_dim × kv_bytes_per_element
head_dim           = head_dim  (if given)  else  hidden_size / attention_heads
```

**Multi-head Latent Attention (DeepSeek-V3/R1)**, where one compressed latent and one RoPE key are stored per layer:

```
kv_bytes_per_token = num_layers × (kv_lora_rank + qk_rope_head_dim) × kv_bytes_per_element
```

**Heuristic** (only when layer/KV metadata is missing; flagged in the UI):

```
kv_bytes_per_token ≈ 46,000 × sqrt(total_parameters_b) × (kv_bytes_per_element / 2)
```

This is calibrated on GQA models (Llama 3.x, Qwen2.5/3) and is conservative for MoE and MLA models.

**Tokens sized:**

| Mode | Tokens |
|---|---|
| Max context (worst case) | `context_length × concurrency` |
| Realistic (default) | `max(context_length, min(context, avg_in + avg_out) × concurrency)` |

In realistic mode, the pool must hold every active request plus at least one request at full context.

The KV precision is independent of the weight precision. `auto` means FP16/BF16 (the serving-stack default). FP8 and INT8 halve the cache.

For example, Llama 3.1 70B uses 2 × 80 × 8 × 128 × 2 = 327,680 B/token. With 20 × 5,000 tokens that is 32.8 GB (FP16) or 16.4 GB (FP8).

## 3. Runtime overhead and total

Per GPU:

```
weights_per_gpu   = weights / N
kv_per_gpu        = kv / min(N, kv_heads)       (MLA: kv, replicated; unknown heads: kv / N)
runtime           = framework.base_overhead_gb             (CUDA context, allocator, CUDA graphs)
workspace         = (weights_per_gpu + kv_per_gpu) × framework.workspace_percent
communication     = framework.communication_overhead_gb_per_gpu   if N > 1 else 0   (NCCL)
subtotal          = weights_per_gpu + kv_per_gpu + runtime + workspace + communication
required_per_gpu  = subtotal × (1 + safety_margin)
```

The headline **required VRAM** is the N = 1 case.

| Framework | base GB | workspace % | comm GB/GPU |
|---|---|---|---|
| vLLM | 1.5 | 5 | 0.5 |
| SGLang | 1.5 | 5 | 0.5 |
| TensorRT-LLM | 2.0 | 7 | 0.75 |
| TGI | 1.5 | 5 | 0.5 |
| PyTorch / HF | 2.0 | 10 | 1.0 |

These are configurable estimates.

Default safety margin: the production default comes from the priority (15% for Economy/Balanced, 20% for Performance/Maximum). Development defaults to 10%.

## 4. Compute and bandwidth (relative estimates)

```
tokens/s per request   = target, or 1000 / TPOT target, or priority default (10 / 15 / 25 / 40)
decode tokens/s        = per-request × concurrency          (or aggregate target)
prefill tokens/s       = decode tokens/s × avg_in / avg_out
required FLOP/s        = 2 × active_parameters × (decode + prefill tokens/s)
compute class          = active_parameters_b × decode tokens/s   → LOW <500, MEDIUM <2.5k, HIGH <10k, VERY HIGH <40k, EXTREME
```

Decode reads the weights and the KV cache once per step:

```
MoE weight share read per step ≈ 1 − (1 − active/total)^concurrency     (dense: 1)
bytes per step        = weights × share + kv
required bandwidth    = bytes per step × tokens/s per request
available bandwidth   = datasheet GB/s × N × share_of_gpu × 0.70
decode ceiling        = available bandwidth / bytes per step   (tokens/s per request)
available compute     = datasheet dense TFLOPS (serving dtype) × N × share_of_gpu × 0.35 MFU
est. TTFT (idle)      = 2 × active × avg_in / available compute
```

INT4 and MXFP4 weights use FP16 tensor throughput (weight-only kernels). FP8 without FP8 tensor cores falls back to FP16.

These numbers are only used to classify and rank options. They are **not** performance predictions.
When a benchmark matches, measured tokens/s replaces them.

## 5. GPU selection and ranking

1. **vGPU:** for each GPU and sharing mode allowed by the priority, take the smallest profile whose framebuffer
   holds `required_per_gpu (N = 1)`.
2. **Single GPU:** N = 1.
3. **Multi-GPU:** N ∈ {2, 4, 8, 16} up to the maximum GPU count. `attention_heads mod min(N, 8)` must be 0.
   Performance and Maximum also add the smallest N whose bandwidth meets the target.

Score (lower is better, weights configurable per priority):

```
score = w_units × gpu_units                      (vGPU: 1/max_instances)
      + w_waste × (1 − utilization)
      + w_tight × max(0, utilization − comfortable) × 10
      + w_bw × max(0, 1 − bandwidth_ratio) + w_compute × max(0, 1 − compute_ratio)
      + w_avail × availability_penalty            (available 0 … unavailable 3)
      + w_cost × min(3, cost / cheapest − 1)       (only when prices exist)
      − w_bw_bonus × min(bandwidth_ratio, 4) / 4   (performance tiers)
      + w_vgpu (bias for or against sharing)
```

## 6. AI flavor

```
Dense: AI-{size class}-{precision tag}-{tier}      e.g. AI-70B-Q4-PROD, ai.llm.70b.int4.prod
MoE:   AI-MOE-{total}B-A{active}B-{tag}-{tier}     e.g. AI-MOE-30B-A3B-Q4-PROD
```

- Size classes: 1B, 3B, 8B, 14B, 32B, 70B, 120B, 200B, 400B, XL.
- Tier: SHARED (vGPU), DEV (development environment), PROD, PERFORMANCE or MAX.
- User-defined models get a `CUSTOM-` prefix.

## 7. Cost

```
full GPU: cost_per_hour × N + vm_cost
vGPU:     cost_per_hour × 1/max_instances + license_cost_per_hour + vm_cost
daily = hourly × 24,  monthly = hourly × 730
```

No prices are shipped. Cost appears only after you set prices.
