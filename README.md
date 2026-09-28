# AI Model → GPU Flavor Calculator

Tell it **what AI you want to run**. It tells you **what GPU infrastructure you need**.

> "I want to run Llama 70B." · "I need an AI-32B VM." · "I train a 2B-parameter vision model."

For each request you get:

- the GPU memory required (with the full breakdown)
- AI compute and GPU memory-speed classes
- whether a **shared vGPU**, a **dedicated GPU** or **multi-GPU** is right
- ranked **NVIDIA and AMD** GPU matches
- a customer-facing **AI flavor** (e.g. `AI-14B-SHARED`, `AI-70B-Q4-PRO`, `AI-CUSTOM-TRAIN-80G`)
- a **deployment spec** and an **OpenStack flavor** recommendation (never auto-provisioned)

Every result is labelled **ESTIMATED** or **BENCHMARK-BASED**, and **memory fit** ("does it fit?") is always shown
separately from **performance fit** ("will it be fast enough?").

```
CUSTOMER LANGUAGE     "Run Qwen 14B" · "AI-32B VM" · "Train a 2B ViT"
      ↓
PRODUCT               AI flavor tier (AI-1B … AI-400B) · custom LLM · custom ML model
      ↓
SIZING ENGINE         weights + KV cache or activations + gradients + optimizer + runtime + headroom
      ↓
REQUIREMENT           VRAM · AI compute · GPU memory speed · GPU count · sharing · GPU link
      ↓
MATCHER               shared vGPU → one dedicated GPU → multi-GPU   (NVIDIA + AMD, capacity-aware)
      ↓
OUTPUT                AI flavor + deployment spec + OpenStack flavor recommendation
```

**Contents:** [Install & run](#1-install--run) · [Use it](#2-use-it) · [Administer](#3-administer) ·
[Troubleshooting](#4-troubleshooting) · [How sizing works](#5-how-sizing-works) · [Development & API](#6-development--api) ·
[Documentation](#7-documentation)

---

## 1. Install & run

### Quick start (any Linux server, e.g. AWS EC2)

```bash
git clone git@github.com:Sherlock2019/AIModelSpecsasGPUFLavor.git
cd AIModelSpecsasGPUFLavor
./start.sh
```

The first run takes a few minutes. After that it starts in seconds.

| Step | What `./start.sh` does |
|---|---|
| Python | Uses Python 3.10+, or installs it with `dnf` / `apt` when missing |
| Backend | Creates `backend/.venv` and installs dependencies |
| Web UI | Downloads Node.js into `./.tools` (checksum-verified, build-time only) and builds the UI |
| Config | Creates `.env` with `PORT`, `HOST` and a generated `ADMIN_TOKEN` |
| Serve | Starts **one process** serving the web UI and the API on `0.0.0.0:<PORT>` |
| Report | Prints the local and **public** URL and the admin token |

Example output:

```
✓ Healthy

AI Model → GPU Flavor Calculator is running
  Local:   http://localhost:8080
  Public:  http://203.0.113.10:8080
  API docs: /docs    Logs: ./start.sh logs
```

### AWS EC2 checklist

1. **Instance:** Amazon Linux 2023 or Ubuntu 22.04/24.04, x86 or Graviton. A t3.small or larger is enough. On 1 GB
   instances, add swap before the first run, because the one-time UI build needs memory.
2. **Security group:** add an inbound rule for **TCP 8080** (or the port the launcher printed), ideally with source
   *My IP*.
3. **Keep it running:** `./start.sh install-service` runs it with systemd across reboots and SSH logouts.
4. **HTTPS (optional):** the server speaks plain HTTP. Put an ALB or a reverse proxy with TLS in front of it if needed.

### Launcher commands

| Command | What it does |
|---|---|
| `./start.sh` | Set up if needed, start in the background, print the URLs |
| `./start.sh status` | Is it running and healthy, and on which port |
| `./start.sh logs` | Follow the server log (`backend/var/gpucalc.log`) |
| `./start.sh stop` / `./start.sh restart` | Stop, or stop and start (rebuilding anything that changed) |
| `./start.sh install-service` / `uninstall-service` | Run at boot with systemd (`journalctl -u gpucalc -f` for logs) |
| `./start.sh build` | Force a dependency re-install and UI rebuild |
| `./start.sh help` | Show this list |

### Ports

- The default is **8080**.
- **If 8080 is already used by another program**, the launcher picks the next free port (8081, 8082, …),
  saves it in `.env` so it stays the same next time, and prints it. Open that port in the security group.
- **To pin a port**, run `PORT=9000 ./start.sh`. A port given this way is never changed; if it is busy, the launcher stops
  and tells you. Ports below 1024 (e.g. 80) need `sudo ./start.sh`.

### Configuration (`.env`, created on the first run)

| Variable | Default | Purpose |
|---|---|---|
| `PORT` | `8080` | Port to serve on |
| `HOST` | `0.0.0.0` | Bind address (`127.0.0.1` for local-only) |
| `ADMIN_TOKEN` | generated | Required to change catalogs, AI flavors and settings |
| `DATABASE_URL` | SQLite in `backend/var/` | e.g. `postgresql+psycopg://user:pass@host/gpucalc` |

Anyone who can open the page can size workloads. Changes to catalogs, flavors or settings need the admin token: paste it
under **Settings → Admin access** (see it again with `grep ADMIN_TOKEN .env`).

### Updating

```bash
git pull
./start.sh restart     # re-installs changed dependencies and rebuilds the UI when needed
```

Your database and edits are kept. Models, GPUs, vGPU profiles and flavor tiers added in a new release appear
automatically. Existing rows are never overwritten.

---

## 2. Use it

Open the URL. The home page (**AI VM Catalog**) asks **"What do you want to size?"**

| Path | Use it for |
|---|---|
| **Select an LLM** | Catalogued LLMs: Llama, Qwen, DeepSeek, Mistral, Gemma, Phi, GLM, gpt-oss … (57 models) |
| **Custom LLM** | Your own language model: parameters, optionally layers and heads |
| **Custom ML model** | Vision, detection, segmentation, OCR, speech, TTS, embeddings, diffusion, video, recommendation, tabular, time-series, transformers, any network |

Below the three paths are the **AI VM flavors**:

- **Small:** AI-1B, AI-3B, AI-8B
- **Medium:** AI-14B, AI-32B
- **Large:** AI-70B, AI-120B
- **Ultra:** AI-200B, AI-400B

### A. Start from an AI flavor: "I need an AI-32B VM"

1. Click **Configure** on a flavor card.
2. Pick a model from the tier, *Choose another model*, or *Custom model*. Precision, context, concurrency and
   priority are pre-filled from the tier.
3. Click **Size my VM**. The same sizing engine as the calculator computes the real requirement; the VRAM
   ranges on the cards are guidance only.
4. Read **Your AI VM**:
   - the flavor, e.g. **AI-8B-SHARED** (SKU `AI-8B-Q4-SHARED`)
   - the GPU memory allocated
   - the mode: shared, dedicated or multi-GPU
5. The tier is never changed silently:
   - A model that is too big shows *"This model exceeds the normal AI-32B tier"* with **Upgrade to AI-70B**.
   - A model that is too small shows *"This workload can run on a smaller flavor"* with **Switch to AI-8B**.
   - If a tier doesn't offer what the workload needs (e.g. multi-GPU), the next tier that does is recommended.
6. Explore the tabs:
   - **Why AI-…?** shows the memory math and why other GPUs ranked lower.
   - **Compare options** lays out GPU count, VRAM, fit, mode, headroom, cost and availability side by side.
   - **Technical details** shows the physical GPU, vGPU profile, OpenStack flavor, deployment spec and formulas.
7. **Deploy** saves an AIFlavorDeploymentSpec and its OpenStack flavor under **Saved & Specs**.
   **Nothing is provisioned.**

| Request (balanced, INT4) | Needs | Result |
|---|---|---|
| AI-8B · Llama 3.1 8B · 16K · 5 users | ≈ 11 GB | **AI-8B-SHARED**: 12 GB vGPU slice of an L40S |
| AI-70B · Llama 3.3 70B · 32K · 20 users | ≈ 88 GB | **AI-70B-PRO**: 1 × H100 NVL 94GB |
| AI-32B · Llama 3.3 70B | — | flagged *exceeds AI-32B* → **Upgrade to AI-70B** |
| AI-70B · Llama 3.1 8B | — | flagged *can run on AI-8B* |
| AI-400B · Llama 3.1 405B · 8 users | ≈ 300 GB | **AI-400B-MULTI**: 2 × B200 |

### B. Start from a model: "I want to run Qwen 14B"

In the **LLM Calculator**, pick a model and set five values:

- precision
- context length
- concurrent requests (presets: 1, 5, 20 or 100)
- performance priority (Lowest cost, Balanced, Performance, Max performance)

Results update live:

- **Your GPU requirement:** VRAM, AI compute, GPU memory speed, GPU count, sharing and multi-GPU link.
- **Matching GPUs:** ranked NVIDIA and AMD cards (the top 3 plus the best AMD, multi-GPU and shared options).
- **Why this recommendation?** and **Ways to need less hardware:** one-click what-ifs such as an FP8 KV cache.
- **Technical details:** memory chart, formulas, OpenStack flavor, precision comparison, exports (JSON, YAML, CSV, report).
- *"This is an AI-14B-SHARED workload. Open it as an AI VM flavor"* jumps to path A with the same inputs.
- The address bar always holds the calculation, so **Copy link** shares it.

### C. Custom ML model: "I train a 2B vision model"

1. Enter:
   - model name and model type
   - **task:** inference, training, or fine-tuning (full, LoRA or QLoRA)
   - parameters (millions), precision and batch size
   - the **input size** for that model type: image W×H×C, video frames, audio seconds, sequence length, diffusion steps or input features
   - concurrent requests (for inference) and priority
2. Optional **Advanced model settings**:
   - framework (PyTorch, TensorFlow, JAX, ONNX Runtime, TensorRT)
   - layers and hidden size
   - **measured activation memory**
   - gradient checkpointing, optimizer and its precision, FP32 master weights, LoRA adapter size
   - data-, tensor- and pipeline-parallel sizes, safety margin
3. Click **Calculate GPU configuration**. The **Custom model analysis** splits memory into:
   - weights
   - gradients
   - optimizer states
   - **activations** (labelled estimated unless you entered a measured value)
   - runtime and headroom
4. The GPU requirement, NVIDIA/AMD matches, why and compare views and what-ifs follow, e.g. *enable gradient checkpointing*
   or *LoRA → QLoRA*.
5. A flavor is generated, such as **AI-CUSTOM-TRAIN-80G** ("Custom ML — Training — 80 GB GPU"). **Deploy** saves an
   MLDeploymentSpec.

Example: a 2B vision model, training, BF16, batch 16, 224×224×3. It needs:

- 4 GB weights
- 4 GB gradients
- 24 GB optimizer states (AdamW plus FP32 master weights)
- ≈ 8 GB activations (estimated)
- runtime and 15% headroom

That is **≈ 49 GB**, which gives 1 × H100 80GB and **AI-CUSTOM-TRAIN-80G**.

---

## 3. Administer

The **Admin** section of the sidebar needs the admin token for changes.

| Page | What you manage |
|---|---|
| **Models** | The LLM catalog: add, edit and disable models, import and export YAML, metadata status (VERIFIED / IMPORTED / USER-DEFINED) |
| **AI Flavors** | Flavor tiers: parameter range, baseline precision, default workload, allowed modes (vGPU / dedicated / multi-GPU), description, commercial SKU, enable/disable |
| **GPU Inventory** | GPUs (NVIDIA and AMD specs, **OpenStack PCI alias**, GPU link, **cost per hour**, availability); vGPU profiles (**free instances**, **Placement trait**); site inventory (installed / available) |
| **Benchmarks** | Measured results. A matching record turns a result into BENCHMARK-BASED |
| **Settings** | Every engine constant: precisions, frameworks, priorities, class thresholds, ML heuristics, OpenStack VM sizing |

- **Capacity-aware:** a vGPU profile with 0 free instances, or too few free GPUs, is skipped with the reason stated
  (*"Suitable vGPU profile currently unavailable"*), and the next suitable option is recommended.
- **Cost:** no prices ship with the app. Set `cost per hour` on GPUs to get hourly, daily and monthly estimates and cost-aware ranking.
- **OpenStack:** PCI aliases and vGPU traits are your cloud's configuration and are never guessed. Until you set them, generated
  flavors list them under `requires_configuration`.

---

## 4. Troubleshooting

| Symptom | Fix |
|---|---|
| `✗ Port 8080 is already in use. Set another one, e.g. PORT=8090 ./start.sh` | This message comes from an older launcher. `git pull` and run `./start.sh` again; it now moves to the next free port automatically. To see what holds the port: `sudo ss -ltnp 'sport = :8080'`. |
| `Port 9000 is already in use … Choose another` | You pinned a busy port with `PORT=9000`. Pick another, or stop the other program. |
| The public URL doesn't load | Check `./start.sh status`. Then check that the security group allows the printed port from your IP. Also check that the instance has a public IP (or use its Elastic IP). |
| `Did not become healthy` | Run `./start.sh logs` for the reason. On very small instances, the first start can be slow; run `./start.sh` again. |
| The UI build fails (`JavaScript heap out of memory` / killed) | Add swap (e.g. 2 GB) on 1 GB instances, then run `./start.sh build`. |
| "Admin token required" when saving | Paste the token from `grep ADMIN_TOKEN .env` into **Settings → Admin access**. |
| Start over with a clean database | Run `./start.sh stop`, then `rm backend/var/gpucalc.db*`, then `./start.sh`. The catalogs are re-seeded. |
| Port 80 without `sudo` | Use the default 8080 and a load balancer, or run `sudo PORT=80 ./start.sh`. |

---

## 5. How sizing works

**Weights:**

```
weights = parameters × bytes per parameter × quantization overhead
```

INT4 is 0.5 bytes, FP8/INT8 is 1 and BF16/FP16 is 2. For example, 70B at INT4 is 35 GB raw, or ≈ 38.8 GB with overhead.

**KV cache (LLMs):** every token in flight stores a key and a value per KV head per layer:

```
2 × layers × kv_heads × head_dim × bytes
```

For Llama 70B, 20 concurrent requests of 5,000 tokens add ≈ 33 GB at FP16.

**Activations (general ML):** intermediate tensors grow with batch × input size × depth. For training they are often
the biggest consumer of memory. They are estimated from model-type heuristics and always labelled; enter a measured
value for accuracy.

**Training:**

```
weights + gradients + optimizer states (+ FP32 master weights) + activations + runtime + headroom
```

Mixed-precision AdamW is ≈ 16 bytes per trainable parameter before activations. LoRA trains only small adapters;
QLoRA also keeps the frozen base model in 4-bit.

**Dense vs MoE:** memory follows *total* parameters, because every expert is resident. Compute follows *active* parameters.

**GPU modes:**

- **Shared vGPU:** a fixed slice of a GPU, time-sliced or MIG. Its performance is not guaranteed.
- **Dedicated:** one whole GPU passed through to the VM.
- **Multi-GPU:** the model is split across GPUs and benefits from NVLink or AMD Infinity Fabric.

**Matching:**

1. Each GPU is tried as the smallest fitting shared vGPU, then one dedicated GPU, then 2, 4 or 8 GPUs.
2. Options are ranked by GPU count, wasted memory, memory speed and compute versus the target, availability, cost and your priority.
3. Unknown performance data is never treated as better than known data.

**AI flavor names:**

| Pattern | Meaning |
|---|---|
| `AI-{tier}-{precision}-{SHARED \| PRO \| MULTI}` | Full name, e.g. `AI-70B-Q4-PRO`. The short name drops the precision when it is the tier baseline: `AI-70B-PRO` |
| `AI-CUSTOM-{INF \| TRAIN \| FT}-{memory}` | Custom ML workloads |

Details and every formula are in [docs/](docs/).

---

## 6. Development & API

```bash
# backend (Python 3.10+)
cd backend && python -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt
uvicorn app.main:app --reload     # API on http://localhost:8000 (docs at /docs)
pytest                            # engine, AI flavors, custom ML, API tests

# frontend (Node 20+)
cd frontend && npm install
npm run dev                       # http://localhost:5173, proxies /api to :8000 (VITE_API_TARGET to change)
npm test                          # Vitest
npm run e2e                       # Playwright; needs the app running (E2E_BASE_URL to change)
```

Docker Compose is also available: `docker compose up --build`, then open http://localhost:8080.

**Main API** (`/api/v1`, interactive docs at `/docs`):

| Endpoint | Purpose |
|---|---|
| `POST /calculate` | LLM sizing (catalog or custom model) |
| `GET /ai-flavors`, `GET /ai-flavors/{id}` | AI VM flavor tiers, with the catalog models in each tier |
| `POST /ai-flavors/size` | Size an AI flavor: tier status, recommended flavor, sizing and the full calculation |
| `POST /ai-flavors`, `PUT` / `DELETE /ai-flavors/{id}` | Admin: manage flavor tiers |
| `GET /ml/meta`, `POST /ml/size` | Custom ML model sizing |
| `POST` / `GET` / `DELETE /deployment-specs` | "Deploy": store a deployment spec and compile its OpenStack flavor |
| `GET /models`, `/gpus`, `/vgpu-profiles`, `/inventory`, `/benchmarks` | Catalogs (writes need `X-Admin-Token`) |
| `POST /compare/models`, `/compare/gpus`, `/compare/precisions` | Comparisons |
| `GET/PUT /settings` | Engine settings |

**Layout:**

- `backend/app/engine/`: the sizing engine (LLM `calculator.py`, custom ML `ml.py`, shared `selection.py` matcher)
- `backend/app/ai_flavors.py`: the flavor layer
- `backend/app/openstack_compiler.py`: the OpenStack compiler
- `backend/app/data/`: catalogs (YAML)
- `frontend/src/`: the React UI
- `start.sh`: the launcher

---

## 7. Documentation

| Document | Covers |
|---|---|
| [docs/architecture.md](docs/architecture.md) | Layers, data flow, API |
| [docs/ai-flavors.md](docs/ai-flavors.md) | Flavor tiers, naming, upgrade and downsize logic, deployment specs |
| [docs/custom-ml.md](docs/custom-ml.md) | Custom ML sizing: inference, training, LoRA/QLoRA, activations |
| [docs/formulas.md](docs/formulas.md) | Every LLM formula with a worked example |
| [docs/model-catalog.md](docs/model-catalog.md) | Model fields, metadata status, verification tool |
| [docs/gpu-catalog.md](docs/gpu-catalog.md) | GPU fields (NVIDIA and AMD), costs, inventory |
| [docs/vgpu.md](docs/vgpu.md) | vGPU profiles and suitability rules |
| [docs/openstack-integration.md](docs/openstack-integration.md) | Deployment spec, flavor compiler, inventory adapters |
| [docs/benchmark-methodology.md](docs/benchmark-methodology.md) | Recording measured results |
| [docs/assumptions.md](docs/assumptions.md) | What the estimates assume and where they can be wrong |
