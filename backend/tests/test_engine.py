import pytest

from app.engine import Catalog, SizingError, calculate
from app.schemas import BenchmarkRecord, CalculateRequest, ModelSpec


def req(**kw) -> CalculateRequest:
    base = {"model_id": "x", "precision": "int4", "context_length": 32768, "concurrent_sequences": 1}
    base.update(kw)
    return CalculateRequest(**base)


# ------------------------------------------------------------------ MoE


@pytest.mark.parametrize(
    "model_id,total,active",
    [
        ("qwen3-30b-a3b", 30.5, 3.3),
        ("llama-4-scout-17b-16e-instruct", 109, 17),
        ("qwen3-235b-a22b", 235, 22),
        ("deepseek-v3", 671, 37),
    ],
)
def test_moe_memory_uses_total_and_compute_uses_active(models, catalog, settings, model_id, total, active):
    m = models[model_id]
    res = calculate(req(context_length=8192), m, catalog, settings)
    assert res.weights.parameters_b == total
    assert res.weights.raw_gb == pytest.approx(total * 0.5, rel=1e-3)
    assert res.compute.active_parameters_b == active
    assert res.compute.compute_demand_score == pytest.approx(active * res.compute.requested_tokens_per_second, rel=1e-3)
    assert any(w.code == "MOE_TOTAL_PARAMS" for w in res.warnings)


def test_qwen3_30b_a3b_flavor(models, catalog, settings):
    res = calculate(req(context_length=8192, performance_priority="performance"), models["qwen3-30b-a3b"], catalog, settings)
    assert res.recommendation.ai_flavor.startswith("AI-MOE-30B-A3B-Q4-")
    assert res.recommendation.flavor_name.startswith("ai.llm.moe-30b-a3b.int4.")


# ------------------------------------------------------------------ recommendation


def test_llama_70b_int4_balanced(models, catalog, settings):
    res = calculate(
        req(concurrent_sequences=20, average_input_tokens=4000, average_output_tokens=1000),
        models["llama-3.1-70b-instruct"],
        catalog,
        settings,
    )
    assert res.weights.raw_gb == pytest.approx(35.3, abs=0.1)
    rec = res.recommendation
    assert rec is not None and rec.candidate.fits
    assert rec.candidate.required_per_unit_gb <= rec.candidate.vram_per_unit_gb
    assert rec.ai_flavor.startswith("AI-70B-Q4-")
    assert res.confidence_label == "ESTIMATED"
    # A 48 GB L40S cannot hold it on its own.
    assert any(c.gpu_id == "nvidia-l40s" and c.count == 1 and not c.fits for c in res.rejected)
    assert res.openstack["gpu"]["mode"] in ("pci_passthrough", "vgpu")
    assert res.explanation and res.trace


def test_llama_70b_fp8_kv_fits_single_h100(models, catalog, settings):
    res = calculate(
        req(concurrent_sequences=20, kv_cache_precision="fp8", allowed_gpu_ids=["nvidia-h100-80gb"]),
        models["llama-3.1-70b-instruct"],
        catalog,
        settings,
    )
    assert res.recommendation.count == 1
    assert res.recommendation.ai_flavor == "AI-70B-Q4-PROD"
    assert res.recommendation.flavor_name == "ai.llm.70b.int4.prod"
    assert res.openstack["name"] == "ai-70b-q4-prod"


def test_small_model_gets_vgpu_on_economy(models, catalog, settings):
    res = calculate(
        req(concurrent_sequences=2, context_length=8192, performance_priority="economy"),
        models["qwen3-14b"],
        catalog,
        settings,
    )
    rec = res.recommendation
    assert rec.mode == "vgpu"
    assert rec.ai_flavor == "AI-14B-Q4-SHARED"
    assert res.openstack["gpu"]["mode"] == "vgpu"
    assert res.openstack["extra_specs"]["resources:VGPU"] == "1"
    assert any(w.code == "VGPU_PERF" for w in res.warnings)


def test_maximum_priority_never_uses_vgpu(models, catalog, settings):
    res = calculate(
        req(concurrent_sequences=2, context_length=8192, performance_priority="maximum"),
        models["qwen3-8b"],
        catalog,
        settings,
    )
    assert res.recommendation.mode != "vgpu"
    assert all(c.kind != "vgpu" for c in res.alternatives)


def test_vgpu_disabled(models, catalog, settings):
    res = calculate(req(allow_vgpu=False, context_length=4096), models["qwen3-1.7b"], catalog, settings)
    assert res.recommendation.mode != "vgpu"
    assert "disabled" in res.vgpu_verdict


def test_large_model_needs_multi_gpu_and_rejects_vgpu(models, catalog, settings):
    res = calculate(req(precision="bf16", concurrent_sequences=8, context_length=8192), models["llama-3.1-70b-instruct"], catalog, settings)
    assert res.memory.model_weights_gb > 140
    assert res.recommendation.count >= 1
    assert res.recommendation.mode != "vgpu"
    assert "Not recommended" in res.vgpu_verdict


def test_no_fit_without_multi_gpu(models, catalog, settings):
    res = calculate(req(precision="bf16", allow_multi_gpu=False), models["llama-3.1-405b-instruct"], catalog, settings)
    assert res.recommendation is None
    assert any(w.code == "NO_FIT" and w.severity == "critical" for w in res.warnings)


def test_tensor_parallel_must_divide_heads(catalog, settings):
    m = ModelSpec(
        id="odd-heads",
        name="Odd",
        architecture="dense",
        total_parameters_b=20,
        hidden_size=5120,
        num_layers=40,
        attention_heads=20,
        kv_heads=4,
        max_context_length=8192,
    )
    res = calculate(req(precision="bf16", context_length=4096, tensor_parallel_size=8), m, catalog, settings)
    assert res.recommendation is None
    assert res.rejected and all("not divisible" in c.rejected_reason for c in res.rejected if c.kind != "vgpu")


def test_context_above_model_max_rejected(models, catalog, settings):
    with pytest.raises(SizingError):
        calculate(req(context_length=65536), models["qwen3-0.6b"], catalog, settings)


def test_safety_headroom_toggle(models, catalog, settings):
    on = calculate(req(), models["qwen3-8b"], catalog, settings)
    off = calculate(req(add_safety_headroom=False), models["qwen3-8b"], catalog, settings)
    assert off.memory.headroom_gb == 0
    assert on.memory.required_vram_gb == pytest.approx(on.memory.subtotal_gb * 1.15, abs=0.02)


def test_development_environment_uses_10_percent_and_dev_flavor(models, catalog, settings):
    res = calculate(req(environment="development", allow_vgpu=False), models["llama-3.1-8b-instruct"], catalog, settings)
    assert res.memory.safety_margin_percent == 10
    assert res.recommendation.flavor_name.endswith(".dev")


# ------------------------------------------------------------------ NVIDIA + AMD matching


def test_amd_gpus_are_matched_alongside_nvidia(models, catalog, settings):
    res = calculate(req(concurrent_sequences=20), models["llama-3.1-70b-instruct"], catalog, settings)
    vendors = {c.gpu_vendor for c in res.matches}
    assert {"NVIDIA", "AMD"} <= vendors
    mi300x = next(c for c in res.matches if c.gpu_id == "amd-instinct-mi300x")
    assert mi300x.count == 1 and mi300x.fits
    assert mi300x.gpu_bandwidth_class == "very_high"
    assert res.compute.bandwidth_class in ("low", "medium", "high", "very_high")


def test_tensorrt_llm_excludes_amd(models, catalog, settings):
    res = calculate(req(framework="tensorrt_llm"), models["llama-3.1-8b-instruct"], catalog, settings)
    assert all(c.gpu_vendor == "NVIDIA" for c in res.matches)
    assert any(c.gpu_vendor == "AMD" and "NVIDIA" in c.rejected_reason for c in res.rejected)


def test_multi_gpu_without_fabric_is_flagged(models, catalog, settings):
    res = calculate(req(allowed_gpu_ids=["nvidia-l40s"], allow_vgpu=False), models["llama-3.1-70b-instruct"], catalog, settings)
    assert res.recommendation.count >= 2
    assert any("PCIe" in n for n in res.recommendation.candidate.notes)


# ------------------------------------------------------------------ what-if suggestions


def test_suggestions_only_offer_hardware_savings(models, catalog, settings):
    m = models["llama-3.1-70b-instruct"]
    res = calculate(req(concurrent_sequences=20), m, catalog, settings, with_suggestions=True)
    by_key = {s.key: s for s in res.suggestions}
    assert "kv_fp8" in by_key
    assert by_key["kv_fp8"].patch == {"kv_cache_precision": "fp8"}
    assert by_key["kv_fp8"].required_vram_gb < res.memory.required_vram_gb
    base = res.recommendation.candidate
    for s in res.suggestions:
        variant = calculate(req(**{"concurrent_sequences": 20, **s.patch}), m, catalog, settings).recommendation.candidate
        assert (variant.gpu_units, variant.vram_per_unit_gb * variant.count) < (base.gpu_units, base.vram_per_unit_gb * base.count)


def test_suggestions_off_by_default(models, catalog, settings):
    assert calculate(req(), models["qwen3-8b"], catalog, settings).suggestions == []


def test_no_fit_suggests_more_gpus(models, catalog, settings):
    res = calculate(req(precision="bf16", allow_multi_gpu=False), models["llama-3.1-405b-instruct"], catalog, settings, with_suggestions=True)
    assert any(s.key == "multi_gpu" for s in res.suggestions)


# ------------------------------------------------------------------ benchmark mode


def test_benchmark_based_when_record_matches(models, catalog, settings):
    bench = BenchmarkRecord(
        id=1,
        model_id="llama-3.1-8b-instruct",
        gpu_id="nvidia-l40s",
        gpu_count=1,
        precision="int4",
        context_length=32768,
        concurrency=4,
        decode_tokens_per_second=400,
        source="unit-test",
    )
    cat = Catalog(gpus=catalog.gpus, profiles=catalog.profiles, benchmarks=[bench])
    r = req(concurrent_sequences=4, allowed_gpu_ids=["nvidia-l40s"], allow_vgpu=False)
    res = calculate(r, models["llama-3.1-8b-instruct"], cat, settings)
    assert res.calculation_mode == "benchmark"
    assert res.confidence_label == "BENCHMARK-BASED"
    assert res.recommendation.candidate.benchmark.source == "unit-test"

    est = calculate(r.model_copy(update={"calculation_mode": "estimate"}), models["llama-3.1-8b-instruct"], cat, settings)
    assert est.calculation_mode == "estimate"
    assert any(w.code == "NO_BENCHMARK" for w in est.warnings)


# ------------------------------------------------------------------ whole catalog smoke test


def test_every_catalog_model_calculates(models, catalog, settings):
    for m in models.values():
        ctx = min(m.max_context_length or 8192, 8192)
        prec = "int4" if "int4" in m.supported_serving_precisions else m.supported_serving_precisions[0]
        res = calculate(req(precision=prec, context_length=ctx, concurrent_sequences=4), m, catalog, settings)
        assert res.memory.required_vram_gb > res.memory.model_weights_gb
        if res.kv_cache.method == "heuristic":
            assert any(w.code == "KV_HEURISTIC" for w in res.warnings)
