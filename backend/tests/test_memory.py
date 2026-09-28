import pytest
from pydantic import ValidationError

from app.engine.config import default_settings
from app.engine.memory import SizingError, kv_cache, per_gpu_breakdown, weight_memory
from app.schemas import ModelSpec

S = default_settings()


def dense(params_b: float, **kw) -> ModelSpec:
    return ModelSpec(id=f"dense-{params_b}b", name=f"Dense {params_b}B", architecture="dense", total_parameters_b=params_b, **kw)


# ------------------------------------------------------------------ weights


@pytest.mark.parametrize("params_b", [1, 3, 8, 14, 32, 70, 120])
def test_dense_int4_raw_weights_are_half_a_byte_per_param(params_b):
    w = weight_memory(dense(params_b), "int4", S)
    assert w.raw_bytes / 1e9 == pytest.approx(params_b * 0.5)
    assert w.bytes / 1e9 == pytest.approx(params_b * 0.5 * 1.10)


@pytest.mark.parametrize(
    "precision,expected_raw_gb",
    [("int4", 35), ("int8", 70), ("fp8", 70), ("fp16", 140), ("bf16", 140)],
)
def test_70b_raw_weights_by_precision(precision, expected_raw_gb):
    assert weight_memory(dense(70), precision, S).raw_bytes / 1e9 == pytest.approx(expected_raw_gb)


@pytest.mark.parametrize("params_b,expected", [(14, 7), (32, 16)])
def test_small_models_int4(params_b, expected):
    assert weight_memory(dense(params_b), "int4", S).raw_bytes / 1e9 == pytest.approx(expected)


def test_quantization_overhead_is_configurable():
    w = weight_memory(dense(70), "int4", S, overhead_override=1.0)
    assert w.bytes == w.raw_bytes
    assert S.precision["int8"].default_overhead_factor == 1.05
    assert S.precision["fp8"].default_overhead_factor == 1.02


def test_unknown_precision_rejected():
    with pytest.raises(SizingError):
        weight_memory(dense(8), "int3", S)


# ------------------------------------------------------------------ KV cache


def test_llama_70b_exact_kv_bytes_per_token(models):
    kv = kv_cache(models["llama-3.1-70b-instruct"], "fp16", 32768, 1, "max", 0, 0, S)
    assert kv.method == "exact"
    assert kv.bytes_per_token == 2 * 80 * 8 * 128 * 2  # 327,680 B - uses KV heads, not attention heads
    assert kv.bytes == 327680 * 32768


def test_kv_precision_halves_cache(models):
    m = models["llama-3.1-8b-instruct"]
    fp16 = kv_cache(m, "fp16", 8192, 4, "max", 0, 0, S)
    fp8 = kv_cache(m, "fp8", 8192, 4, "max", 0, 0, S)
    assert fp8.bytes == pytest.approx(fp16.bytes / 2)


def test_head_dim_derived_from_hidden_size():
    m = dense(8, hidden_size=4096, num_layers=32, attention_heads=32, kv_heads=8)
    kv = kv_cache(m, "fp16", 1024, 1, "max", 0, 0, S)
    assert kv.head_dim == 128
    assert kv.bytes_per_token == 2 * 32 * 8 * 128 * 2


def test_mla_kv_uses_latent_width(models):
    kv = kv_cache(models["deepseek-v3"], "fp16", 4096, 1, "max", 0, 0, S)
    assert kv.method == "mla"
    assert kv.bytes_per_token == 61 * (512 + 64) * 2
    assert kv.shard_limit == 1  # latent replicated across TP ranks


def test_heuristic_kv_when_metadata_missing(models):
    kv = kv_cache(models["ministral-3-8b"], "fp16", 8192, 1, "max", 0, 0, S)
    assert kv.method == "heuristic"
    assert "HEURISTIC" in kv.formula


def test_realistic_vs_max_context(models):
    m = models["llama-3.1-70b-instruct"]
    realistic = kv_cache(m, "fp16", 32768, 20, "realistic", 4000, 1000, S)
    worst = kv_cache(m, "fp16", 32768, 20, "max", 4000, 1000, S)
    assert realistic.tokens_per_sequence == 5000
    assert realistic.sized_tokens == 100_000
    assert worst.sized_tokens == 32768 * 20
    # A single full-context request must always fit.
    single = kv_cache(m, "fp16", 32768, 1, "realistic", 4000, 1000, S)
    assert single.sized_tokens == 32768


# ------------------------------------------------------------------ per-GPU breakdown


def test_tensor_parallel_shards_weights_and_replicates_kv_beyond_kv_heads(models):
    m = models["qwen3-30b-a3b"]  # 4 KV heads
    kv = kv_cache(m, "fp16", 32768, 8, "max", 0, 0, S)
    fw = S.frameworks["vllm"]
    one = per_gpu_breakdown(10e9, kv, 1, fw, 15)
    four = per_gpu_breakdown(10e9, kv, 4, fw, 15)
    eight = per_gpu_breakdown(10e9, kv, 8, fw, 15)
    assert four.weights == pytest.approx(one.weights / 4)
    assert four.kv == pytest.approx(one.kv / 4)
    assert eight.kv == pytest.approx(one.kv / 4)  # only 4 KV heads to split
    assert one.communication == 0 and four.communication > 0


def test_safety_margin_applied_to_subtotal(models):
    kv = kv_cache(models["llama-3.1-8b-instruct"], "fp16", 8192, 1, "max", 0, 0, S)
    bd = per_gpu_breakdown(16e9, kv, 1, S.frameworks["vllm"], 15)
    assert bd.required == pytest.approx(bd.subtotal * 1.15)


# ------------------------------------------------------------------ model validation


def test_dense_active_defaults_to_total():
    assert dense(70).active_parameters_b == 70


def test_dense_active_must_equal_total():
    with pytest.raises(ValidationError):
        dense(70, active_parameters_b=10)


def test_moe_requires_active():
    with pytest.raises(ValidationError):
        ModelSpec(id="moe-x", name="MoE", architecture="moe", total_parameters_b=30)
