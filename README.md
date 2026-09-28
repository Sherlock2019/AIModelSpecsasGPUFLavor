# AI Model → GPU Flavor Calculator

**Customers should not have to ask "which NVIDIA GPU do I need?". They should be able to say
"I want to run Llama 70B", "I need an AI-32B VM" or "I train a 2B vision model".**
This application turns that request into infrastructure: GPU memory, compute and memory-speed class, shared vGPU,
dedicated GPU or multi-GPU (NVIDIA or AMD), a customer-facing AI flavor, and an OpenStack flavor recommendation.

```
CUSTOMER LANGUAGE        "I need an AI-32B VM" · "Run Qwen 14B" · "Train a 2B ViT"
        ↓
PRODUCT                  AI flavor tier (AI-1B … AI-400B) or custom LLM / custom ML model
        ↓
SIZING ENGINE            weights + KV cache / activations + gradients + optimizer + runtime + headroom
        ↓
RESOURCE REQUIREMENT     VRAM · AI compute · GPU memory speed · GPU count · sharing · GPU link
        ↓
RESOURCE MATCHER         shared vGPU → one dedicated GPU → multi-GPU   (NVIDIA + AMD, capacity-aware)
        ↓
IMPLEMENTATION           AI flavor (AI-14B-SHARED, AI-70B-Q4-PRO, AI-CUSTOM-TRAIN-80G)
                         + deployment spec + OpenStack flavor recommendation (never auto-provisioned)
```

Every result is labelled **ESTIMATED** or **BENCHMARK-BASED**, and **memory fit** ("does it fit?") is always shown
separately from **performance fit** ("will it be fast enough?").

---

## Quick start on an AWS EC2 instance (public IP)

```bash
git clone git@github.com:Sherlock2019/AIModelSpecsasGPUFLavor.git
cd AIModelSpecsasGPUFLavor
./start.sh
```

The first run sets everything up:

- Python 3.10+ from the OS package manager, if it is missing
- a Python virtualenv
- Node.js downloaded to `./.tools` and checksum-verified (it is only used to build the UI)

It then serves the web UI **and** the API from one process on `0.0.0.0:8080` and prints the URL to open,
e.g. `http://<ec2-public-ip>:8080`.

1. Use Amazon Linux 2023 or Ubuntu 22.04/24.04, x86 or Graviton. A t3.small or larger is enough. On 1 GB instances,
   add swap for the one-time UI build.
2. **Security group:** add an inbound rule for **TCP 8080**, ideally restricted to your IP ("My IP").
3. Run `./start.sh install-service` to keep it running across reboots and SSH logouts (systemd).

| Command | What it does |
|---|---|
| `./start.sh` | Set up if needed, start in the background, print the public URL |
| `./start.sh stop` / `restart` / `status` / `logs` | Manage the running server |
| `./start.sh install-service` / `uninstall-service` | Run at boot with systemd |
| `./start.sh build` | Re-install dependencies and rebuild the UI (after `git pull`) |

If port 8080 is already taken by another program, the launcher uses the next free port (8081, 8082, …), saves it
in `.env` and prints it. Open that port in the security group, or pin one with `PORT=9000 ./start.sh`. An explicit
port is never changed; if it is busy, the launcher stops with a message.

Settings live in `.env`, created on the first run:

- `PORT` (default 8080)
- `HOST` (default 0.0.0.0)
- `ADMIN_TOKEN`, generated automatically and printed

Anyone who can reach the page can size workloads. Changing catalogs, flavors or settings requires the admin token,
entered under **Settings → Admin access**. The server speaks plain HTTP; put a TLS load balancer or proxy in front if needed.

---

## The full workflow

### 1. Choose what to size

The home page (**AI VM Catalog**) asks *"What do you want to size?"* and offers three paths:

| Path | For | Opens |
|---|---|---|
| **Select an LLM** | Llama, Qwen, DeepSeek, Mistral, Gemma, Phi, GLM, gpt-oss … (57 catalogued models) | LLM Calculator |
| **Custom LLM** | Your own language model: parameters, optional layers/heads | LLM Calculator (custom model) |
| **Custom ML model** | Vision, detection, segmentation, OCR, speech, TTS, embeddings, diffusion, video, recommendation, tabular, time-series, transformers, any network | Custom ML Calculator |

Below the three paths is the **AI VM flavor catalog**, grouped as:

- **Small:** AI-1B, AI-3B, AI-8B
- **Medium:** AI-14B, AI-32B
- **Large:** AI-70B, AI-120B
- **Ultra:** AI-200B, AI-400B

Each card shows the model class, example models, typical GPU memory and best use cases. Hardware internals
(PCI alias, vGPU profile, Nova resource classes) stay hidden until *Technical details* is opened.

### 2a. Start from an AI flavor (Mode A: "I need an AI-32B VM")

1. **Configure** a flavor card. You get the tier's models, or *Choose another model*, or *Custom model*. Precision,
   context, concurrency and priority are pre-filled from the tier defaults.
2. **Size my VM.** The same sizing engine as the calculator computes the real requirement. Tier VRAM ranges
   are catalog guidance only; the calculator is the source of truth.
3. Read **Your AI VM**, for example **AI-8B-SHARED** (SKU AI-8B-Q4-SHARED). It shows the GPU memory allocated,
   the mode (shared / dedicated / multi-GPU) and the model it is recommended for.
4. The tier is never switched silently:
   - *"This model exceeds the normal AI-32B tier."* comes with **Upgrade to AI-70B**.
   - *"This workload can run on a smaller flavor."* comes with **Switch to AI-8B**.
   - If a tier's policy can't host the workload (for example AI-14B doesn't offer multi-GPU), the next tier that can is recommended, with the reason.
5. Use the tabs:
   - **Why AI-…?** shows the memory math, the checks and "why not" for other GPUs.
   - **Compare options** shows the L40S vs H100 vs H200 style table: GPU count, VRAM, memory fit, mode, headroom, cost and availability.
   - **Technical details** shows the physical GPU, vGPU profile, OpenStack flavor, deployment spec and calculation trace.
6. **Deploy** creates an **AIFlavorDeploymentSpec** and the compiled OpenStack flavor. **Nothing is provisioned.**
   Specs are listed under **Saved & Specs**.

Examples, using the bundled catalog, balanced priority and INT4:

| Request | Required | Result |
|---|---|---|
| AI-8B · Llama 3.1 8B · 16K · 5 concurrent | ≈ 11 GB | **AI-8B-SHARED**: L40S-12C vGPU (a full GPU would waste capacity) |
| AI-70B · Llama 3.3 70B · 32K · 20 concurrent | ≈ 88 GB (FP16 KV cache) | **AI-70B-PRO**: 1 × H100 NVL 94GB |
| AI-32B · Llama 3.3 70B | — | flagged *exceeds AI-32B* → upgrade to AI-70B |
| AI-70B · Llama 3.1 8B | — | flagged *can run on AI-8B* |
| AI-400B · Llama 3.1 405B | ≈ 300 GB | **AI-400B-MULTI**: 2 × B200 (multi-GPU tier) |

### 2b. Start from a model (Mode B: "I want to run Qwen 14B")

**LLM Calculator:** pick a model, then set the five inputs. They are precision, context length,
concurrent requests (with presets) and performance priority, and results update live.

- **Your GPU requirement:** VRAM, AI compute, GPU memory speed, GPU count, sharing and multi-GPU link.
- **Matching GPUs:** ranked NVIDIA and AMD cards (top 3 plus the best AMD, multi-GPU and shared options), each with memory fit,
  performance fit and technical details.
- **From model to GPU**, **Why this recommendation?**, **Ways to need less hardware** (one-click what-ifs such as an FP8
  KV cache) and **Technical details** (memory chart, formulas, OpenStack flavor, precision comparison, exports).
- The banner *"This is an AI-14B-SHARED workload. Open it as an AI VM flavor"* jumps to Mode A with the same inputs.
  Both modes use the same engine.

### 2c. Custom ML model

1. Simple mode asks for:
   - model name and model type
   - task (**inference / training / fine-tuning**; fine-tuning is **full / LoRA / QLoRA**)
   - parameters (millions), precision and batch size
   - the input size for that model type: image W×H×C, video frames, audio seconds, sequence length, diffusion steps or input features
   - concurrent requests and performance priority
2. **Advanced model settings** cover:
   - framework (PyTorch, TensorFlow, JAX, ONNX Runtime, TensorRT)
   - layers, hidden size and I/O tensors
   - **measured activation memory**, gradient checkpointing and optimizer (AdamW/Adam/SGD/Adafactor) with its precision
   - FP32 master weights, training and gradient precision, LoRA adapter size
   - data-, tensor- and pipeline-parallel sizes, model growth and safety margin
3. **Calculate GPU configuration.** The **Custom model analysis** splits memory into:
   - weights
   - gradients
   - optimizer states
   - **activations (estimated unless you enter a measured value)**
   - runtime and headroom

   KV cache is only added when the model uses one.
4. The same requirement panel, NVIDIA/AMD matches, why and compare views, and what-ifs follow. What-ifs include enabling
   gradient checkpointing, LoRA → QLoRA, halving the batch, BF16 and INT8 serving.
5. The flavor is generated automatically: **AI-CUSTOM-INF-10G**, **AI-CUSTOM-TRAIN-80G** or **AI-CUSTOM-TRAIN-2X180G**,
   with a readable name such as *"Custom ML — Training — 80 GB GPU"*. **Deploy** creates an **MLDeploymentSpec**.

Example: a 2B-parameter vision model, training, BF16 weights, batch 16, 224×224×3. It needs:

- 4 GB weights
- 4 GB gradients
- 24 GB AdamW states plus FP32 master weights
- about 8 GB of estimated activations
- 2.7 GB runtime and 15% headroom

The total is about 49 GB. That gives a dedicated H100 80GB and **AI-CUSTOM-TRAIN-80G**.

### 3. Administer the catalogs (Admin section)

| Page | What you manage |
|---|---|
| **Models** | LLM catalog: add, edit and disable models, import and export YAML, metadata status (VERIFIED / IMPORTED / USER-DEFINED) |
| **AI Flavors** | Tiers, parameter bounds, baseline precision, default workload, allowed vGPU / full / multi-GPU, description, commercial SKU, enable/disable |
| **GPU Inventory** | Physical GPUs (NVIDIA + AMD specs, **PCI alias**, GPU link, cost, availability); vGPU profiles (**free instances**, **Placement trait**); site inventory (installed / available) |
| **Benchmarks** | Measured results. A matching record makes the result BENCHMARK-BASED |
| **Settings** | Every engine constant: precisions, frameworks, priorities, class thresholds, ML heuristics, OpenStack sizing |

**Capacity-aware.** If a suitable vGPU profile has 0 free instances, or fewer GPUs are free than a configuration needs,
that option is skipped with the reason stated, for example *"Suitable vGPU profile currently unavailable"*, and the
next suitable option is recommended.

**OpenStack.** PCI aliases and vGPU traits are site configuration and are never guessed. Until they are set in
GPU Inventory, compiled flavors list them under `requires_configuration`.

---

## Concepts in one page

**Parameters and precision.** Weight memory is `parameters × bytes/parameter × quantization overhead`: 0.5 bytes for INT4, 1 for FP8/INT8
and 2 for BF16/FP16. For example, 70B at INT4 is 35 GB raw, or about 38.8 GB with overhead.

**KV cache (LLMs).** Each token in flight keeps a key and a value for every KV head in every layer:
`2 × layers × kv_heads × head_dim × bytes`. For Llama 70B, 20 concurrent requests of 5,000 tokens add about 33 GB at FP16.

**Activations (general ML).** Intermediate tensors grow with batch × input size × depth. For training they are often
the largest consumer of memory, so the calculator never sizes training from parameters alone. The estimates are
heuristics and are always labelled; enter a measured value when you have one.

**Training.** Training memory is `weights + gradients + optimizer states (+ FP32 master weights) + activations + runtime + headroom`.
Mixed-precision AdamW is about 16 bytes per trainable parameter before activations. LoRA trains small adapters only;
QLoRA also stores the frozen base model in 4-bit.

**Dense vs MoE.** MoE memory follows *total* parameters (every expert is resident). Compute follows *active* parameters.

**vGPU / dedicated / multi-GPU.**

- A **shared vGPU** is a fixed slice of a GPU (time-sliced or MIG).
- A **dedicated GPU** is PCI passthrough of one whole GPU.
- **Multi-GPU** splits the model (tensor or pipeline parallelism) and wants NVLink or AMD Infinity Fabric.

---

## Development

```bash
# backend (Python 3.10+)
cd backend && python -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt
uvicorn app.main:app --reload         # http://localhost:8000/docs
pytest                                # engine, flavors, custom ML, API

# frontend (Node 20+)
cd frontend && npm install
npm run dev                           # http://localhost:5173 (proxies /api to :8000; VITE_API_TARGET to change)
npm test                              # Vitest
npm run e2e                           # Playwright (needs the app running; E2E_BASE_URL to change)
```

Docker Compose is also available: `docker compose up --build`, then open http://localhost:8080.

| Environment variable | Purpose |
|---|---|
| `DATABASE_URL` | SQLAlchemy URL. Default is SQLite in `backend/var/`. PostgreSQL: `postgresql+psycopg://user:pass@host/db` |
| `ADMIN_TOKEN` | Required `X-Admin-Token` for catalog, flavor and settings changes |
| `PORT`, `HOST` | Where `start.sh` serves (default `0.0.0.0:8080`) |

### Main API (`/api/v1`, interactive docs at `/docs`)

| Endpoint | Purpose |
|---|---|
| `POST /calculate` | LLM sizing (known or custom model) |
| `GET /ai-flavors`, `GET /ai-flavors/{id}` | AI VM flavor tiers (with the catalog models in each tier) |
| `POST /ai-flavors/size` | Size an AI flavor: tier status, recommended flavor, sizing, full calculation |
| `POST /ai-flavors`, `PUT/DELETE /ai-flavors/{id}` | Admin: manage flavor tiers |
| `GET /ml/meta`, `POST /ml/size` | Custom ML model sizing |
| `POST/GET/DELETE /deployment-specs` | "Deploy": store a deployment spec and compile its OpenStack flavor |
| `GET /models`, `/gpus`, `/vgpu-profiles`, `/inventory`, `/benchmarks` | Catalogs (admin writes need the token) |
| `POST /compare/models`, `/compare/gpus`, `/compare/precisions` | Comparisons |

## Documentation

- [docs/architecture.md](docs/architecture.md): layers, data flow, API
- [docs/ai-flavors.md](docs/ai-flavors.md): AI VM flavor tiers, naming, upgrade and downsize logic, deployment specs
- [docs/custom-ml.md](docs/custom-ml.md): custom ML sizing (inference, training, LoRA/QLoRA, activations)
- [docs/formulas.md](docs/formulas.md): every LLM formula with a worked example
- [docs/model-catalog.md](docs/model-catalog.md), [docs/gpu-catalog.md](docs/gpu-catalog.md), [docs/vgpu.md](docs/vgpu.md)
- [docs/openstack-integration.md](docs/openstack-integration.md): deployment spec, flavor compiler, inventory adapters
- [docs/benchmark-methodology.md](docs/benchmark-methodology.md), [docs/assumptions.md](docs/assumptions.md)
