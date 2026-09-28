import pytest

from app.engine import SizingError, size_ml
from app.schemas import MLSizingRequest


def ml(**kw) -> MLSizingRequest:
    base = {"model_name": "m", "model_category": "computer_vision", "parameters_m": 500, "precision": "fp16"}
    base.update(kw)
    return MLSizingRequest(**base)


def test_inference_weights_and_no_kv_cache(catalog, settings):
    r = size_ml(ml(parameters_m=350, batch_size=32, concurrent_requests=5), catalog, settings)
    assert r.memory.model_weights_gb == pytest.approx(0.7, abs=0.01)  # 350M x 2 bytes
    assert r.memory.kv_cache_gb == 0
    assert r.memory.gradients_gb == 0 and r.memory.optimizer_gb == 0
    assert r.memory.activations_gb > 0
    assert r.activation_method == "estimated"
    assert any(w.code == "ACTIVATION_ESTIMATED" for w in r.warnings)
    assert r.recommendation.mode == "vgpu"  # a small vision model must not take a full GPU
    assert r.recommendation.ai_flavor.startswith("AI-CUSTOM-INF-")
    assert r.ai_flavor_display.startswith("Custom ML — Inference — ")


def test_training_has_gradients_and_optimizer_states(catalog, settings):
    r = size_ml(ml(task="training", parameters_m=2000, precision="bf16", batch_size=8, gradient_checkpointing=True), catalog, settings)
    m = r.memory
    assert m.model_weights_gb == pytest.approx(4.0, abs=0.01)  # 2B x 2 B
    assert m.gradients_gb == pytest.approx(4.0, abs=0.01)  # BF16 gradients
    assert m.optimizer_gb == pytest.approx(24.0, abs=0.01)  # AdamW 2 x FP32 states (16) + FP32 master (8)
    no_master = size_ml(ml(task="training", parameters_m=2000, precision="bf16", batch_size=8, gradient_checkpointing=True, master_weights=False), catalog, settings)
    assert no_master.memory.optimizer_gb == pytest.approx(16.0, abs=0.01)


def test_training_needs_much_more_than_inference(catalog, settings):
    inf = size_ml(ml(parameters_m=2000, precision="bf16", batch_size=8), catalog, settings)
    train = size_ml(ml(task="training", parameters_m=2000, precision="bf16", batch_size=8), catalog, settings)
    assert train.memory.required_vram_gb > 3 * inf.memory.required_vram_gb


def test_gradient_checkpointing_cuts_activations(catalog, settings):
    kw = dict(task="training", parameters_m=2000, precision="bf16", batch_size=64, image_width=1024, image_height=1024)
    full = size_ml(ml(**kw), catalog, settings)
    ckpt = size_ml(ml(**kw, gradient_checkpointing=True), catalog, settings)
    assert ckpt.memory.activations_gb < full.memory.activations_gb / 4


def test_peft_is_much_smaller_than_full_fine_tuning(catalog, settings):
    kw = dict(model_category="transformer", task="fine_tuning", parameters_m=70000, precision="bf16", batch_size=4,
              sequence_length=2048, gradient_checkpointing=True, max_gpu_count=16)
    full = size_ml(ml(**kw, fine_tune_method="full"), catalog, settings)
    lora = size_ml(ml(**kw, fine_tune_method="lora"), catalog, settings)
    qlora = size_ml(ml(**kw, fine_tune_method="qlora"), catalog, settings)
    assert qlora.memory.required_vram_gb < lora.memory.required_vram_gb < full.memory.required_vram_gb / 3
    assert qlora.model.precision == "int4"
    assert qlora.model.trainable_parameters_b == pytest.approx(0.35, rel=0.01)  # 0.5% adapters
    assert any(w.code == "PEFT_SAVINGS" and "QLoRA can substantially reduce" in w.message for w in qlora.warnings)


def test_manual_activation_memory_is_used(catalog, settings):
    r = size_ml(ml(activation_memory_gb=10, batch_size=1), catalog, settings)
    assert r.activation_method == "manual"
    assert r.memory.activations_gb >= 10


def test_quantized_full_training_is_rejected(catalog, settings):
    with pytest.raises(SizingError):
        size_ml(ml(task="training", precision="int4"), catalog, settings)


def test_multi_gpu_training_requires_interconnect(catalog, settings):
    r = size_ml(ml(model_category="transformer", task="training", parameters_m=30000, precision="bf16", batch_size=8,
                   sequence_length=4096, gradient_checkpointing=True, allow_vgpu=False), catalog, settings)
    assert r.recommendation.count > 1
    assert r.requirement.interconnect == "Required"
    assert "X" in r.recommendation.ai_flavor.split("-")[-1]  # AI-CUSTOM-TRAIN-2X180G
    assert r.deployment_spec["gpu_requirement"]["gpu_count"] == r.recommendation.count


def test_category_defaults_and_kv_opt_in(catalog, settings):
    diff = size_ml(ml(model_category="diffusion", parameters_m=12000, batch_size=4), catalog, settings)
    assert "1024×1024" in diff.workload.input_description
    assert diff.memory.model_weights_gb == pytest.approx(24.0, abs=0.01)
    assert diff.memory.required_vram_gb > 30  # weights alone are not the answer
    base = size_ml(ml(model_category="transformer", parameters_m=7000, batch_size=4), catalog, settings)
    kv = size_ml(ml(model_category="transformer", parameters_m=7000, batch_size=4, uses_kv_cache=True), catalog, settings)
    assert kv.memory.activations_gb > base.memory.activations_gb


def test_tensorrt_is_nvidia_only(catalog, settings):
    r = size_ml(ml(framework="tensorrt"), catalog, settings)
    assert all(c.gpu_vendor == "NVIDIA" for c in r.matches)


def test_suggestions_save_hardware(catalog, settings):
    r = size_ml(ml(task="training", parameters_m=2000, precision="bf16", batch_size=64, image_width=1024, image_height=1024), catalog, settings, with_suggestions=True)
    assert any(s.key == "checkpointing" for s in r.suggestions)
    for s in r.suggestions:
        assert s.required_vram_gb < r.memory.required_vram_gb or s.key == "multi"


def test_ml_api(client):
    meta = client.get("/api/v1/ml/meta").json()
    assert len(meta["categories"]) == 16 and any(f["id"] == "tensorrt" for f in meta["frameworks"])
    body = {"model_name": "Vision", "model_category": "image_classification", "parameters_m": 350, "precision": "fp16",
            "batch_size": 32, "concurrent_requests": 5}
    r = client.post("/api/v1/ml/size", json=body)
    assert r.status_code == 200, r.text
    assert r.json()["workload_type"] == "custom_ml"
    assert client.post("/api/v1/ml/size", json={**body, "precision": "int3"}).status_code == 422
    assert client.post("/api/v1/ml/size", json={**body, "task": "training", "precision": "int4"}).status_code == 422
