# Architecture

```
┌──────────────── frontend (React + TS + Vite + Tailwind + Recharts) ───────────────┐
│ Calculator · Models · GPU Catalog · Compare · Benchmarks · Saved · Settings       │
└───────────────────────────────┬────────────────────────────────────────────────────┘
                                │ /api/v1 (JSON)
┌───────────────────────────────▼────────────────────────────────────────────────────┐
│ FastAPI  app/api.py            request validation (Pydantic), admin token, exports │
│ Repository app/repository.py   SQLAlchemy CRUD, YAML seed/import/export, settings  │
│ Inventory  app/inventory.py    InfrastructureInventory → StaticInventoryAdapter    │
│ Engine     app/engine/         pure functions, no DB / HTTP                        │
│   memory.py     weights, KV cache (exact / MLA / heuristic), per-GPU breakdown     │
│   compute.py    throughput demand, bandwidth roofline, compute classification      │
│   selection.py  vGPU → single GPU → multi-GPU candidates, ranking                  │
│   flavor.py     AI-70B-Q4-PROD / ai.llm.70b.int4.prod                              │
│   openstack.py  flavor recommendation YAML                                         │
│   calculator.py orchestration, warnings, explanation, calculation trace            │
└───────────────────────────────┬────────────────────────────────────────────────────┘
                                │ SQLAlchemy 2 (SQLite default, PostgreSQL via DATABASE_URL)
                    catalog_models · gpus · vgpu_profiles · inventory
                    benchmarks · saved_calculations · app_settings
```

## Data flow of `POST /api/v1/calculate`

1. The request is validated by `CalculateRequest`. It takes exactly one of `model_id` or `custom_model`, and unknown fields are rejected.
2. The model is loaded from the catalog, or the custom model is used and marked `user_defined`.
3. `catalog_snapshot()` asks the inventory adapter for GPUs, vGPU profiles and capacity, and adds benchmarks.
4. `engine.calculate()` does the following:
   1. Resolves defaults: KV precision, average tokens, safety margin (priority / environment / override) and throughput target.
   2. Computes weights, KV cache and the single-GPU memory breakdown, which gives the required VRAM.
   3. Computes demand (FLOP/s, decode bandwidth), then the compute class and memory intensity.
   4. Selects GPUs: the smallest fitting vGPU profile per GPU and sharing mode, then the minimum GPU count per
      GPU model (1, 2, 4, 8, 16). Performance tiers can also scale out for bandwidth.
   5. Ranks the candidates and picks the recommendation, alternatives (diverse by kind) and rejected options.
   6. Generates the AI flavor and OpenStack YAML, then the warnings, explanation and trace.

The engine is pure, so it is tested directly (`tests/test_memory.py`, `tests/test_engine.py`) as well as
through the API (`tests/test_api.py`).

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/v1/models` (`?q=`, `?include_disabled=`) | Model catalog |
| GET | `/api/v1/models/{id}` | One model |
| POST / PUT / PATCH | `/api/v1/models`, `/api/v1/models/{id}` | Admin: add / edit / disable / status |
| GET / POST | `/api/v1/models/export`, `/api/v1/models/import` | YAML export / import (safe_load, validated) |
| POST | `/api/v1/custom-model` | Validate (and optionally persist) a user-defined model |
| GET / POST / PUT | `/api/v1/gpus` | GPU catalog |
| GET / POST | `/api/v1/vgpu-profiles` | vGPU profile catalog |
| GET / PUT | `/api/v1/inventory` | Site inventory |
| POST | `/api/v1/calculate` | Full calculation |
| POST | `/api/v1/calculate/export?format=json\|yaml\|csv\|markdown` | Download a calculation |
| POST | `/api/v1/compare/models`, `/compare/gpus`, `/compare/precisions` | Comparisons |
| GET / POST / DELETE | `/api/v1/benchmarks` | Benchmark records |
| GET / POST / DELETE | `/api/v1/calculations` | Saved calculations |
| GET / PUT / POST | `/api/v1/settings`, `/settings/reset` | Engine settings |
| GET | `/api/v1/meta` | Enumerations for the UI |

Interactive docs are available at `/docs`.

## Security notes

- Catalog imports use `yaml.safe_load` and per-item Pydantic validation. Only metadata is accepted, and nothing is executed or fetched remotely.
- Set `ADMIN_TOKEN` in any shared deployment. Without it, catalog and settings mutations are open (this is the local-mode default).
- `app/tools/verify_catalog.py` is an offline CLI that only downloads `config.json` files from Hugging Face.
