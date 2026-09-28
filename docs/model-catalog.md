# Model catalog

The source of truth is `backend/app/data/models.yaml`. It is seeded into the database on first start. After that,
use the **Models** page (or the API) to add, edit, disable, import or export models.

## Fields

| Field | Required | Notes |
|---|---|---|
| `id` | ✓ | lowercase, `[a-z0-9._-]` |
| `name`, `vendor`, `family` | ✓ | family drives grouping in the model picker |
| `architecture` | ✓ | `dense` or `moe` |
| `total_parameters_b` | ✓ | includes embeddings, and vision towers for multimodal checkpoints |
| `active_parameters_b` | MoE | dense models default to the total and must equal it |
| `hidden_size`, `num_layers`, `attention_heads`, `kv_heads`, `head_dim` | for exact KV | `head_dim` falls back to `hidden_size / attention_heads` |
| `attention_type` | | `standard` (default) or `mla` with `kv_lora_rank`, `qk_rope_head_dim` |
| `sliding_window` | | informational; KV is sized as full attention (conservative) |
| `num_experts`, `experts_per_token` | | informational for MoE |
| `max_context_length` | | requests above it are rejected |
| `native_precision`, `supported_serving_precisions` | | unsupported choices produce a warning |
| `metadata_status` | ✓ | `verified`, `imported` or `user_defined` |
| `source_url`, `license`, `notes` | | |

Validation rejects inconsistent data. That includes dense models with active ≠ total, MoE models without active parameters,
`kv_heads > attention_heads`, attention heads not divisible by KV heads, and MLA models without latent sizes.

## Metadata status

- **VERIFIED:** each architecture field was checked against the official `config.json` / model card.
- **IMPORTED:** transcribed from the official config / model card, but not yet re-verified by the tool below.
- **USER-DEFINED:** entered by a user (custom model).

All bundled models ship as **IMPORTED**. Nothing is guessed: models whose layer and KV metadata has not been
imported (Ministral 3, Mistral Large 3) omit those fields. They use the heuristic KV estimate, and the UI shows
"KV cache calculated using estimate mode".

### Promoting to VERIFIED

```bash
cd backend
HF_TOKEN=hf_xxx python -m app.tools.verify_catalog            # report differences
HF_TOKEN=hf_xxx python -m app.tools.verify_catalog --promote  # mark fully matching models verified
```

The tool downloads each model's `config.json` from its `source_url` (gated repos need `HF_TOKEN`) and compares
hidden size, layers, heads, KV heads, head_dim, MLA ranks and expert counts. It then rewrites
`models.yaml`. To update an existing database, re-import the file on the Models page with *Overwrite* enabled.

## Families included

- **Llama:** 3.2 1B/3B; 3.1 8B/70B/405B; 3.3 70B; 4 Scout; 4 Maverick
- **Qwen:** 2.5 7B/14B/32B/72B; 3 0.6B/1.7B/4B/8B/14B/32B; 3 30B-A3B; 3 235B-A22B
- **Mistral:** 7B v0.3, Small 3.1 24B, Large 2 (123B), Large 3 (675B MoE)
- **Ministral:** 8B (2410); Ministral 3 3B/8B/14B
- **Mixtral:** 8x7B, 8x22B
- **Gemma:** Gemma 7B; Gemma 2 2B/9B/27B; Gemma 3 1B/4B/12B/27B. No open-weight Gemma MoE is catalogued.
- **DeepSeek:** R1-Distill Qwen 1.5B/7B/14B/32B; R1-Distill Llama 8B/70B; V3; R1
- **Phi:** Phi-3 Mini (4K, 128K), Phi-3 Medium 128K, Phi-4 14B
- **GLM:** 4.5 Air, 4.5, 4.6
- **Command R:** Command R (32B), Command R+ (104B)
- **gpt-oss:** 20b, 120b (native MXFP4)
- **Custom model:** any architecture, entered in the calculator

## Adding a model

On the Models page, choose **Add model**, or `POST /api/v1/models`, or import YAML in the same format as `models.yaml`.
Prefer copying values from the model's `config.json`. For multimodal models, read the `text_config` block.
