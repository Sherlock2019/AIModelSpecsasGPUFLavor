# Benchmark methodology

Benchmark mode replaces formula estimates with measured results wherever a record matches. No benchmark data
ships with the calculator, so every recommendation is labelled **ESTIMATED** until you add your own.

## Record fields

`model_id`, `gpu_id`, `gpu_count`, `precision`, `framework`, `tensor_parallel`, `context_length`, `batch_size`,
`concurrency`, `prefill_tokens_per_second`, `decode_tokens_per_second`, `aggregate_tokens_per_second`, `ttft_ms`,
`tpot_ms`, `peak_vram_gb`, `average_gpu_utilization`, `power_watts`, `source`, `date`, `verified`.

## Matching

A record matches a candidate when **model, GPU, GPU count and precision** are equal. Among matches, the calculator
prefers records that, in order:

1. Use the same framework.
2. Have `context_length` ≥ the effective sequence length.
3. Have the closest concurrency.
4. Are verified.

When a record matches:

- The measured decode (or total) tokens/s replaces the bandwidth and compute estimates, as `measured / requested`.
- The measured TTFT replaces the estimate.
- A `peak_vram_gb` above the usable capacity rejects the option.
- Benchmark-backed options get a small tie-break bonus.
- If the recommended option is benchmark-backed, the result is labelled **BENCHMARK-BASED**.

Calculation mode:

| Mode | Behaviour |
|---|---|
| `auto` (default) | uses benchmarks when available |
| `estimate` | ignores benchmarks |
| `benchmark` | uses benchmarks, and warns if it had to fall back to estimates |

## How to measure

1. Serve the model with the target framework, precision and tensor parallel size. Record the framework version.
2. Run a load generator at the target concurrency with realistic prompt and output lengths. For example, use vLLM's
   `benchmark_serving.py` or `genai-perf`, with a fixed random seed.
3. Warm up, then measure for at least 5 minutes of steady state.
4. Record the following:
   - output tokens/s (decode, aggregate), total tokens/s, and median/P90 TTFT and TPOT
   - peak VRAM from `nvidia-smi --query-gpu=memory.used`, noting `gpu_memory_utilization` if set
   - average GPU utilization and power
5. Put the tool, versions and date in `source` and `date`. Mark `verified` only after a second run
   reproduces the numbers within about 10%.

Note that vLLM preallocates KV cache up to `gpu_memory_utilization`, so `nvidia-smi` shows the reservation, not
the need. Record the configured value, or measure with profiling, when you use `peak_vram_gb`.
