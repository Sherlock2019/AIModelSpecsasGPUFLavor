# Assumptions and limitations

The calculator gives **sizing estimates**. Where it can be wrong:

## Memory

- **Quantization overhead** (INT4 ×1.10, INT8 ×1.05, FP8 ×1.02) is an average. Actual checkpoint sizes depend on group size,
  which layers stay in 16-bit, and the format (AWQ, GPTQ, GGUF, NVFP4 and so on). Override it per calculation if you know the file size.
- **KV cache precision** defaults to the 16-bit activation dtype. FP8 KV cache halves it, but it must be enabled in the serving stack.
- **Sliding-window and chunked attention** (Gemma 2/3, Llama 4, Ministral, gpt-oss) are sized as full attention on
  every layer. This is conservative. Hybrid KV allocators in recent serving stacks can use much less.
- **Heuristic KV** (models without layer metadata) can be off by ±50%. It is flagged every time.
- **Realistic context mode** assumes average-length requests and at least one full-context request. Bursty or
  long-tail traffic needs *Max context* mode or a higher safety margin.
- **Runtime and workspace** (for example 1.5 GB + 5% for vLLM) are estimates. CUDA graphs, large batch prefill and
  speculative decoding can use more. All of these are configurable in Settings.
- **Vision encoders and multimodal inputs** are included in the parameter count, but image-token KV is not modelled separately.
- **MoE expert offloading** (experts in CPU memory) is not modelled. All experts are assumed resident in GPU memory.

## Compute and performance

- Throughput numbers are **roofline estimates**, using 70% of datasheet bandwidth and 35% MFU. They are used only to classify and rank
  options. Real performance depends on kernels, batch scheduling, prefix caching, interconnect and the framework version.
  Add benchmarks for anything that matters.
- The MoE bandwidth model assumes independent expert routing, `1 − (1 − a)^B`.
- Tensor-parallel communication cost is modelled as fixed memory only, not as throughput loss. The UI notes that
  there is no NVLink, but it does not quantify the slowdown.
- Time-sliced vGPU performance assumes the worst case (all slots busy).

## Catalog data

- Model architecture values were transcribed from official configs and model cards and are marked **IMPORTED**. Run
  `python -m app.tools.verify_catalog` to check them and promote them to VERIFIED.
- GPU figures are dense datasheet peaks. Some newer GPUs have unknown values, which are left empty.
- vGPU profile names and limits depend on the vGPU software release.
- No prices are included.

## About the spec's example

The product brief's example ("Llama 70B, INT4, 32K context, 20 users → 55 GB, 8.2 GB KV") understates the KV
cache. With Llama 3.1 70B's real architecture (80 layers, 8 KV heads, head_dim 128), 20 concurrent requests of 5,000
tokens need about 33 GB of FP16 KV cache (or about 16 GB in FP8), not 8 GB. At a full 32K context per request, they need about 215 GB. The
calculator therefore gives about 88 GB (FP16 KV) or about 68 GB (FP8 KV → 1 × H100 80GB) for that workload.
