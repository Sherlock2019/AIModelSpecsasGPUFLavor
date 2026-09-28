from dataclasses import replace

import pytest

from app.ai_flavors import models_in_tier, size_flavor
from app.engine import calculate
from app.engine.flavor import tier_for
from app.engine.openstack import compile_openstack_flavor
from app.schemas import AIFlavorDefinition, CalculateRequest, FlavorSizeRequest


def fsr(**kw) -> FlavorSizeRequest:
    base = {"precision": "int4", "context_length": 32768, "concurrent_sequences": 20}
    base.update(kw)
    return FlavorSizeRequest(**base)


# ------------------------------------------------------------------ tier catalog


@pytest.mark.parametrize(
    "params,tier",
    [(0.6, "ai-1b"), (1.24, "ai-1b"), (3.21, "ai-3b"), (8.03, "ai-8b"), (14.8, "ai-14b"), (32.8, "ai-32b"),
     (70.55, "ai-70b"), (72.7, "ai-70b"), (116.8, "ai-120b"), (140.6, "ai-200b"), (405.85, "ai-400b"), (671, None)],
)
def test_tier_for_uses_ceiling_with_tolerance(catalog, settings, params, tier):
    t = tier_for(params, catalog.ai_flavors, settings)
    assert (t.id if t else None) == tier


def test_examples_belong_to_their_tier(catalog, settings, models):
    for t in catalog.ai_flavors:
        in_tier = set(models_in_tier(t, list(models.values()), catalog.ai_flavors, settings))
        assert set(t.example_model_ids) <= in_tier, t.id


def test_flavor_definition_validation():
    with pytest.raises(ValueError):
        AIFlavorDefinition(id="ai-x", display_name="X", parameter_floor_b=10, parameter_ceiling_b=5, default_vram_min_gb=1, default_vram_max_gb=2)
    with pytest.raises(ValueError):
        AIFlavorDefinition(id="ai-x", display_name="X", parameter_floor_b=0, parameter_ceiling_b=5, default_vram_min_gb=1,
                           default_vram_max_gb=2, allow_vgpu=False, allow_full_gpu=False, allow_multi_gpu=False)


# ------------------------------------------------------------------ sizing through the one engine


def test_small_flavor_gets_shared_gpu(catalog, settings, models):
    r = size_flavor(fsr(flavor_id="ai-8b", model_id="llama-3.1-8b-instruct", context_length=16384, concurrent_sequences=5),
                    models["llama-3.1-8b-instruct"], catalog, settings)
    assert r.tier_status == "fits"
    assert r.variant == "SHARED" and r.display_flavor == "AI-8B-SHARED"
    assert r.recommended_flavor == "AI-8B-Q4-SHARED"
    assert r.calculation.recommendation.mode == "vgpu"
    # the flavor layer only relays engine output
    direct = calculate(fsr(model_id="llama-3.1-8b-instruct", context_length=16384, concurrent_sequences=5),
                       models["llama-3.1-8b-instruct"], catalog, settings)
    assert direct.memory.required_vram_gb == r.sizing["required_vram_gb"]


def test_bigger_model_than_selected_tier_is_flagged_not_forced(catalog, settings, models):
    r = size_flavor(fsr(flavor_id="ai-32b", model_id="llama-3.3-70b-instruct"), models["llama-3.3-70b-instruct"], catalog, settings)
    assert r.tier_status == "exceeds"
    assert r.requested_flavor.id == "ai-32b" and r.recommended_tier.id == "ai-70b"
    assert "exceeds the normal AI-32B tier" in r.tier_message
    assert r.recommended_flavor.startswith("AI-70B-")


def test_smaller_model_than_selected_tier_suggests_downsizing(catalog, settings, models):
    r = size_flavor(fsr(flavor_id="ai-70b", model_id="llama-3.1-8b-instruct"), models["llama-3.1-8b-instruct"], catalog, settings)
    assert r.tier_status == "oversized"
    assert r.recommended_tier.id == "ai-8b"
    assert "smaller flavor" in r.tier_message


def test_tier_policy_blocks_multi_gpu_and_upgrades(catalog, settings, models):
    # AI-14B does not offer multi-GPU; a workload no single GPU can hold must move up a tier.
    # ~430 GB (64 x 32K worst-case KV): more than any single GPU, a few GPUs are enough.
    r = size_flavor(fsr(flavor_id="ai-14b", model_id="qwen3-14b", precision="bf16", concurrent_sequences=64, context_mode="max",
                        context_length=32768), models["qwen3-14b"], catalog, settings)
    assert r.calculation.memory.required_vram_gb > 300
    assert r.recommended_tier.id != "ai-14b"
    assert r.tier_status == "exceeds"
    assert r.variant == "MULTI"


def test_multi_gpu_only_tier(catalog, settings, models):
    r = size_flavor(fsr(flavor_id="ai-400b", model_id="llama-3.1-405b-instruct", concurrent_sequences=1, context_length=8192),
                    models["llama-3.1-405b-instruct"], catalog, settings)
    assert r.calculation.recommendation.count >= 2
    assert r.variant == "MULTI"


def test_exhausted_vgpu_profile_is_skipped_with_reason(catalog, settings, models):
    profiles = [p.model_copy(update={"available_instances": 0}) if p.physical_gpu == "nvidia-l40s" else p for p in catalog.profiles]
    cat = replace(catalog, profiles=profiles)
    res = calculate(CalculateRequest(model_id="x", precision="int4", context_length=8192, concurrent_sequences=2, performance_priority="economy"),
                    models["qwen3-14b"], cat, settings)
    assert res.recommendation.gpu_id != "nvidia-l40s"
    assert any(c.capacity_limited and "currently unavailable" in c.rejected_reason for c in res.rejected)


def test_short_gpu_inventory_is_rejected(catalog, settings, models):
    from app.engine.selection import Capacity

    cat = replace(catalog, capacity={"nvidia-h100-80gb": Capacity(installed=4, available=0)}, has_inventory=True)
    res = calculate(CalculateRequest(model_id="x", precision="int4", context_length=32768, concurrent_sequences=20,
                                     kv_cache_precision="fp8", allowed_gpu_ids=["nvidia-h100-80gb", "nvidia-h200-141gb"], allow_vgpu=False),
                    models["llama-3.3-70b-instruct"], cat, settings)
    assert res.recommendation.gpu_id == "nvidia-h200-141gb"
    assert any(c.gpu_id == "nvidia-h100-80gb" and c.capacity_limited for c in res.rejected)


# ------------------------------------------------------------------ OpenStack compiler


def test_compiler_uses_only_configured_alias_and_trait():
    spec = {
        "flavor_slug": "ai-70b-q4-pro",
        "ai_flavor": "AI-70B-Q4-PRO",
        "workload_type": "llm",
        "vm": {"vcpus": 16, "ram_gb": 64, "disk_gb": 100},
        "gpu": {"mode": "dedicated", "count": 1, "physical_gpu": "NVIDIA H100 80GB", "physical_gpu_id": "nvidia-h100-80gb", "minimum_vram_gb": 80},
    }
    bare = compile_openstack_flavor(spec, pci_alias=None, vgpu_trait=None)
    assert "pci_passthrough:alias" not in bare["extra_specs"] and bare["requires_configuration"]
    configured = compile_openstack_flavor(spec, pci_alias="gpu-h100", vgpu_trait=None)
    assert configured["extra_specs"]["pci_passthrough:alias"] == "gpu-h100:1"
    assert not configured["requires_configuration"]
    assert configured["ram_mb"] == 65536


# ------------------------------------------------------------------ API


def test_flavor_api(client):
    flavors = client.get("/api/v1/ai-flavors").json()
    assert [f["id"] for f in flavors][:3] == ["ai-1b", "ai-3b", "ai-8b"]
    assert "llama-3.1-8b-instruct" in next(f for f in flavors if f["id"] == "ai-8b")["model_ids"]
    assert client.get("/api/v1/ai-flavors/ai-70b").json()["display_name"] == "AI-70B"
    assert client.get("/api/v1/ai-flavors/nope").status_code == 404

    body = {"flavor_id": "ai-70b", "model_id": "llama-3.3-70b-instruct", "precision": "int4",
            "context_length": 32768, "concurrent_sequences": 20, "performance_priority": "balanced"}
    r = client.post("/api/v1/ai-flavors/size", json=body)
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["recommended_flavor"].startswith("AI-70B-Q4-")
    assert out["sizing"]["required_vram_gb"] > out["sizing"]["model_weight_gb"]
    assert client.post("/api/v1/ai-flavors/size", json={**body, "flavor_id": "nope"}).status_code == 404

    spec = out["calculation"]["deployment_spec"]
    created = client.post("/api/v1/deployment-specs", json={"name": "test", "spec": spec})
    assert created.status_code == 201
    assert created.json()["openstack"]["name"] == spec["flavor_slug"]
    assert any(s["id"] == created.json()["id"] for s in client.get("/api/v1/deployment-specs").json())
    assert client.delete(f"/api/v1/deployment-specs/{created.json()['id']}").status_code == 204
    assert client.post("/api/v1/deployment-specs", json={"spec": {"x": 1}}).status_code == 422


def test_flavor_admin_api(client):
    f = client.get("/api/v1/ai-flavors/ai-14b").json()
    f["description"] = "Medium AI VM (edited)"
    assert client.put("/api/v1/ai-flavors/ai-14b", json=f).json()["description"] == "Medium AI VM (edited)"
    bad = {**f, "parameter_floor_b": 50}
    assert client.put("/api/v1/ai-flavors/ai-14b", json=bad).status_code == 422
