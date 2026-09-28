import yaml

CALC = {
    "model_id": "llama-3.1-70b-instruct",
    "precision": "int4",
    "context_length": 32768,
    "concurrent_sequences": 20,
    "average_input_tokens": 4000,
    "average_output_tokens": 1000,
    "performance_priority": "balanced",
    "framework": "vllm",
    "allow_vgpu": True,
    "allow_multi_gpu": True,
    "safety_margin_percent": 15,
}


def test_catalog_endpoints(client):
    models = client.get("/api/v1/models").json()
    assert len(models) >= 40
    assert {m["family"] for m in models} >= {"Llama", "Qwen", "Mistral", "Ministral", "Mixtral", "Gemma", "DeepSeek", "Phi", "GLM", "Command R"}
    assert client.get("/api/v1/models/qwen3-30b-a3b").json()["architecture"] == "moe"
    assert client.get("/api/v1/models/nope").status_code == 404
    assert client.get("/api/v1/models", params={"q": "llama 70b"}).json()
    assert len(client.get("/api/v1/gpus").json()) >= 12
    assert client.get("/api/v1/vgpu-profiles").json()
    assert client.get("/api/v1/benchmarks").json() == []
    assert client.get("/api/v1/meta").json()["precisions"]


def test_calculate(client):
    r = client.post("/api/v1/calculate", json=CALC)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["calculation_mode"] == "estimate"
    assert body["model"]["architecture"] == "dense"
    assert body["memory"]["required_vram_gb"] > body["memory"]["subtotal_gb"]
    assert body["recommendation"]["ai_flavor"].startswith("AI-70B-Q4")
    assert body["openstack_yaml"]


def test_calculate_validation(client):
    assert client.post("/api/v1/calculate", json={**CALC, "context_length": 10_000_000}).status_code == 422
    assert client.post("/api/v1/calculate", json={**CALC, "precision": "int3"}).status_code == 422
    assert client.post("/api/v1/calculate", json={**CALC, "model_id": "missing"}).status_code == 404
    assert client.post("/api/v1/calculate", json={**CALC, "bogus": 1}).status_code == 422


def test_exports(client):
    for fmt in ("json", "yaml", "csv", "markdown"):
        r = client.post(f"/api/v1/calculate/export?format={fmt}", json=CALC)
        assert r.status_code == 200 and "attachment" in r.headers["content-disposition"]
    assert "trace" in client.post("/api/v1/calculate/export?format=json", json=CALC).json()


def test_custom_model(client):
    custom = {
        "id": "my-model",
        "name": "My 20B",
        "architecture": "dense",
        "total_parameters_b": 20,
        "hidden_size": 6144,
        "num_layers": 44,
        "attention_heads": 48,
        "kv_heads": 8,
        "max_context_length": 32768,
    }
    r = client.post("/api/v1/custom-model", json={"model": custom})
    assert r.json()["kv_metadata"] == "exact" and r.json()["resolved_head_dim"] == 128
    calc = {k: v for k, v in CALC.items() if k != "model_id"}
    r = client.post("/api/v1/calculate", json={**calc, "custom_model": custom})
    assert r.status_code == 200
    assert r.json()["model"]["metadata_status"] == "user_defined"
    assert r.json()["recommendation"]["ai_flavor"].startswith("AI-32B-")  # 20B custom LLM -> AI-32B tier


def test_compare(client):
    workload = {k: v for k, v in CALC.items() if k != "model_id"}
    rows = client.post(
        "/api/v1/compare/models",
        json={"model_ids": ["llama-3.1-8b-instruct", "qwen3-14b", "qwen3-32b", "llama-3.1-70b-instruct"], "workload": workload},
    ).json()
    assert len(rows) == 4 and all(r.get("required_vram_gb") for r in rows)
    precision_rows = client.post("/api/v1/compare/precisions", json={"request": CALC}).json()
    weights = {r["precision"]: r["weights_gb"] for r in precision_rows}
    assert weights["int4"] < weights["int8"] < weights["bf16"]
    gpu_rows = client.post("/api/v1/compare/gpus", json={"request": CALC, "gpu_ids": ["nvidia-l40s", "nvidia-h200-141gb"]}).json()
    assert {r["gpu_id"] for r in gpu_rows} == {"nvidia-l40s", "nvidia-h200-141gb"}


def test_model_admin_and_yaml_roundtrip(client):
    exported = client.get("/api/v1/models/export").text
    doc = yaml.safe_load(exported)
    assert len(doc["models"]) >= 40
    r = client.patch("/api/v1/models/phi-4", json={"enabled": False})
    assert r.json()["enabled"] is False
    assert all(m["id"] != "phi-4" for m in client.get("/api/v1/models").json())
    client.patch("/api/v1/models/phi-4", json={"enabled": True})
    result = client.post("/api/v1/models/import", json={"yaml": exported, "overwrite": False}).json()
    assert result["skipped"] >= 40 and result["created"] == 0
    bad = client.post("/api/v1/models/import", json={"yaml": "models: [{id: x}]"}).json()
    assert bad["errors"]


def test_benchmarks_and_saved(client):
    bench = {
        "model_id": "llama-3.1-8b-instruct",
        "gpu_id": "nvidia-l40s",
        "precision": "int4",
        "context_length": 8192,
        "concurrency": 8,
        "decode_tokens_per_second": 500,
        "source": "test",
    }
    created = client.post("/api/v1/benchmarks", json=bench)
    assert created.status_code == 201
    assert client.delete(f"/api/v1/benchmarks/{created.json()['id']}").status_code == 204

    saved = client.post("/api/v1/calculations", json={"name": "70B plan", "request": CALC}).json()
    assert client.get(f"/api/v1/calculations/{saved['id']}").json()["result"]["recommendation"]
    assert client.get("/api/v1/calculations").json()
    assert client.delete(f"/api/v1/calculations/{saved['id']}").status_code == 204


def test_settings_roundtrip(client):
    s = client.get("/api/v1/settings").json()
    s["frameworks"]["vllm"]["base_overhead_gb"] = 2.5
    assert client.put("/api/v1/settings", json=s).json()["frameworks"]["vllm"]["base_overhead_gb"] == 2.5
    s["precision"]["int4"]["bytes_per_parameter"] = -1
    assert client.put("/api/v1/settings", json=s).status_code == 422
    assert client.post("/api/v1/settings/reset").json()["frameworks"]["vllm"]["base_overhead_gb"] == 1.5


def test_inventory(client):
    inv = client.get("/api/v1/inventory").json()
    assert inv
    assert client.put("/api/v1/inventory", json=[{"gpu_id": "nope", "site": "a", "installed": 1, "available": 1}]).status_code == 422
    assert client.put("/api/v1/inventory", json=inv).status_code == 200
