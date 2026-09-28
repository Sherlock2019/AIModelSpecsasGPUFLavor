# LLM GPU Sizing Calculator

**Customers should not have to ask "which NVIDIA GPU do I need?". They should be able to say
"I want to run Llama 70B."** This calculator turns that request into infrastructure.

```
MODEL → MODEL METADATA → PRECISION → CONTEXT → CONCURRENCY
      → MEMORY + COMPUTE → vGPU / FULL GPU / MULTI-GPU → GPU → AI FLAVOR
```

Pick a model from the curated catalog and set precision, context, concurrency, priority and framework. The calculator returns:

```
Llama 3.1 70B Instruct / INT4                              ESTIMATED
Estimated VRAM required: 68 GB
Recommended: 1 × NVIDIA H100 80GB — Dedicated GPU
Alternative: 1 × NVIDIA H100 NVL 94GB
vGPU: Not recommended for this workload: needs ~68 GB; the largest supported profile is L40S-48C (48 GiB).
AI flavor: AI-70B-Q4-PROD
Why: 39 GB model weights + 16 GB KV cache + 4 GB runtime + 15% production headroom.
```

(That run uses 20 concurrent requests of about 5,000 tokens each, with an FP8 KV cache. With the default
FP16 KV cache, the KV cache doubles to 33 GB and the recommendation moves to a 94 GB H100 NVL.)

Every result is labelled **ESTIMATED** or **BENCHMARK-BASED**. Estimates are never presented as measurements.

## Quick start (any Linux server, e.g. an AWS EC2 instance)

```bash
git clone https://github.com/Sherlock2019/Model2GPUCalculator.git
cd Model2GPUCalculator
./start.sh
```

The launcher sets everything up on the first run: Python 3.10+ via the OS package manager if needed, a Python
virtualenv, and Node.js downloaded to `./.tools`, which is only used to build the UI. It then serves the web UI **and**
the API from one process on `0.0.0.0:8080`, and prints the URL to open, e.g. `http://<ec2-public-ip>:8080`.

**EC2 checklist**

1. Use Amazon Linux 2023 or Ubuntu 22.04/24.04, x86 or Graviton. A t3.small or larger is enough. On 1 GB instances, add swap for the one-time UI build.
2. Security group: add an inbound rule for **TCP 8080**, ideally restricted to your IP ("My IP" in the console).
3. `./start.sh install-service` keeps it running across reboots and SSH logouts (systemd).

| Command | What it does |
|---|---|
| `./start.sh` | Set up if needed, start in the background, print the public URL |
| `./start.sh stop` / `restart` / `status` / `logs` | Manage the running server |
| `./start.sh install-service` | Run at boot with systemd (`uninstall-service` to remove) |
| `./start.sh build` | Force re-install dependencies and rebuild the UI (e.g. after `git pull`) |

Settings live in `.env`, created on the first run: `PORT` (default 8080), `HOST` (default 0.0.0.0) and `ADMIN_TOKEN`.
The admin token is generated automatically. Anyone who can reach the page can use the calculator, but changing the catalog
or settings requires that token (paste it under **Settings → Admin access**). It is plain HTTP; put it behind a load
balancer or reverse proxy with TLS if you need HTTPS.

Docker Compose is also supported: `docker compose up --build`, then open http://localhost:8080.

Local development:

```bash
# backend (Python 3.12+)
cd backend
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt
uvicorn app.main:app --reload         # http://localhost:8000/docs
pytest

# frontend (Node 20+)
cd frontend
npm install
npm run dev                           # http://localhost:5173, proxies /api to :8000
npm test                              # Vitest
npm run e2e                           # Playwright (optional; needs the app running)
```

If the API runs on a different port, set `VITE_API_TARGET=http://localhost:8010 npm run dev`.

| Environment variable | Purpose |
|---|---|
| `DATABASE_URL` | SQLAlchemy URL. Default is SQLite in `backend/var/`. PostgreSQL: `postgresql+psycopg://user:pass@host/db` |
| `ADMIN_TOKEN` | When set, catalog and settings changes require the `X-Admin-Token` header (enter it under Settings in the UI) |
| `CORS_ORIGINS` | Comma-separated origins for the API |

## Concepts in one page

**Parameter count.** The number of learned weights, in billions (B). A "70B" model has about 70.6 billion.

**Precision.** The number of bytes that store each weight: BF16/FP16 = 2, FP8/INT8 = 1, INT4 = 0.5. The
*serving* precision (what you deploy) can differ from the *native* precision (what the model was
released in).

**Quantization.** Storing weights in fewer bits (INT8, INT4, FP8). Real quantized checkpoints also store
scales and zero-points, and keep some layers (embeddings, norms, `lm_head`) in 16-bit. That is why the
calculator applies a configurable **overhead factor**: 1.10 for INT4, 1.05 for INT8 and 1.02 for FP8.

**Weights.** `parameters × bytes/parameter × overhead`. For example, 70B at INT4 is 35 GB raw, or about 38.8 GB with overhead.

**KV cache.** For every token in flight, each layer caches a key and a value vector for each *KV head*:
`2 × layers × kv_heads × head_dim × bytes`. Llama 3.1 70B uses 320 KiB per token at FP16. Twenty
concurrent 5,000-token requests therefore need 33 GB, almost as much as the INT4 weights. Models
with grouped-query attention (GQA) have few KV heads, which keeps this small. DeepSeek's MLA caches
a compressed latent instead.

**Dense vs MoE.** In a dense model every parameter is used for every token. A Mixture-of-Experts
model routes each token through a few experts. For example, Qwen3 30B-A3B has 30.5B parameters but uses only about 3.3B
per token. **Memory is sized on total parameters, because every expert must be resident. Compute is sized on active parameters.**

**vGPU.** One physical GPU split into several virtual GPUs with fixed framebuffers (for example, L40S-24C is
24 GiB). *Time-sliced* profiles share all compute, so in the worst case each gets 1/N of it. *MIG* profiles get dedicated
hardware slices. Suitable when the model needs much less than a full GPU. Performance is not guaranteed.

**Full GPU.** A dedicated physical GPU passed through to the VM (PCI passthrough).

**Tensor parallelism.** Splitting each layer across N GPUs, so weights and KV cache divide by N. Every GPU
still needs its own runtime overhead and communication buffers. The attention-head count must divide by N.
Beyond one node (8 GPUs), pipeline parallelism is added.

## Using the calculator

1. **Model:** open the model list and search (for example "llama 70b"), or pick from *Popular*. *Model details* shows
   the official metadata. *Edit* overrides it, and the result is marked **USER OVERRIDE**.
2. **Precision, context length, concurrent requests** (presets: Development 1, Small 5, Medium 20, Large 100) and
   **performance priority** (Lowest cost, Balanced, Performance, Max performance).
3. Read the right-hand side. Results update live, with no Calculate button:
   - **Your GPU requirement:** GPU memory, AI compute, GPU memory speed, GPU count, sharing and multi-GPU link.
   - **Matching GPUs:** NVIDIA and AMD options, ranked. Each card shows **Memory fit** ("does it fit?") separately
     from **Performance fit** ("will it be fast enough?", estimated unless benchmarked), plus *Technical details*.
4. Below: **From model to GPU** (the derivation), **Why this recommendation?** (with *why not* for other GPUs),
   **Ways to need less hardware** (one-click what-ifs) and **Technical details** (memory chart, formulas,
   OpenStack flavor, precision comparison, exports).

The address bar always holds the current calculation, so *Copy link* shares it. The same guide is in the app under
**How it works**.

## What's in the box

- **Calculator:** a one-page, model-first flow with five default inputs. Advanced options cover framework, request
  lengths, KV precision, headroom, GPU limits, throughput and TTFT/TPOT targets, and overrides.
- **Exports:** JSON with the full trace, YAML, CSV, a Markdown report, and copy to clipboard.
- **Models:** a curated catalog of 50+ models covering Llama, Qwen, Mistral, Ministral, Mixtral, Gemma, DeepSeek, Phi,
  GLM, Command R and gpt-oss, plus custom models. You can add, edit and disable models, and import or export YAML.
- **GPU catalog:** 13 NVIDIA GPUs (T4 to B300) and 5 AMD Instinct GPUs (MI210, MI250, MI250X, MI300X, MI325X), plus 30
  vGPU profiles (time-sliced and MIG) and site inventory. Costs and availability are editable. MI250 and MI250X are
  catalogued per GCD, because software sees each card as two 64 GB GPUs.
- **Compare:** models side by side, GPUs side by side, and precisions side by side.
- **Benchmarks:** record measured results. Matching records switch the result to BENCHMARK-BASED.
- **Saved calculations** and **Settings.** Every engine constant is editable and validated.
- **OpenStack:** generates flavor YAML (`pci_passthrough:alias` or `resources:VGPU`). Nothing is deployed. The
  `InfrastructureInventory` adapter interface is ready for a future Nova/Placement/Cyborg integration.

## Documentation

- [docs/architecture.md](docs/architecture.md): components, data flow, API
- [docs/formulas.md](docs/formulas.md): every formula with a worked example
- [docs/model-catalog.md](docs/model-catalog.md): catalog fields, metadata status, verification tool
- [docs/gpu-catalog.md](docs/gpu-catalog.md): GPU fields, costs, inventory
- [docs/vgpu.md](docs/vgpu.md): vGPU profiles and suitability rules
- [docs/openstack-integration.md](docs/openstack-integration.md): flavor output and the adapter
- [docs/benchmark-methodology.md](docs/benchmark-methodology.md): how to record benchmarks
- [docs/assumptions.md](docs/assumptions.md): what the estimates assume and where they can be wrong
