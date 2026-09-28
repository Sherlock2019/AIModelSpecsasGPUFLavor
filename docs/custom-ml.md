# Custom ML model sizing

`POST /api/v1/ml/size` (engine: `backend/app/engine/ml.py`) sizes non-LLM workloads and custom networks. GPU matching,
the NVIDIA/AMD catalog, capacity rules, flavors and the OpenStack compiler are **shared with the LLM calculator**. The
ML module supplies its own per-GPU memory breakdown and compute demand to the same matcher.

## Inputs

**Simple mode:**

- model name and model type (16 categories)
- task: `inference`, `training` or `fine_tuning` (`full`, `lora` or `qlora`)
- parameters (millions), precision and batch size
- concurrent requests (inference) and performance priority
- input size for the model type:

| Input kind | Categories | Fields (defaults) |
|---|---|---|
| image | computer vision, classification (224²), detection (640²), segmentation (512²), OCR (1024²) | width, height, channels |
| diffusion | diffusion / image generation | width, height (1024²), denoising steps (30) |
| video | video models | frame width × height (224²), frames per sample (16) |
| audio | speech recognition (30 s), text-to-speech (10 s) | duration, sample rate |
| sequence | transformer (2,048), embedding (512), time-series (512) | sequence length, optional hidden size |
| generic | recommendation, tabular, generic, other | input features, output elements |

**Advanced:**

- framework (PyTorch, TensorFlow, JAX, ONNX Runtime, TensorRT; TensorRT is NVIDIA-only)
- model file size, layers, hidden size, I/O tensor sizes
- manual activation memory, gradient checkpointing
- optimizer and its precision, bytes/param override, FP32 master weights
- training and gradient precision, LoRA adapter %
- data-, tensor- and pipeline-parallel sizes
- expected model growth, target latency or throughput, safety margin, vGPU and multi-GPU policy

## Memory

```
inference   = weights + activations & I/O (+ KV cache only if "uses KV cache") + runtime/workspace + headroom
training    = weights + gradients + optimizer states (+ FP32 master weights) + activations + runtime/workspace + headroom
LoRA        = frozen base weights + adapter (weights, gradients, optimizer) + activations + …
QLoRA       = LoRA with the base model at INT4
```

- **Weights:** `parameters × bytes(precision)`, or the model file size when given (inference).
- **Gradients:** `trainable parameters × bytes(gradient precision)`.
- **Optimizer:** AdamW/Adam use 2 states, SGD (momentum) 1, and Adafactor about 0.5 B/param (estimate).
  The states are multiplied by the bytes of the optimizer precision (default FP32), plus 4 B/param FP32 master weights for 16-bit training.
  Mixed-precision AdamW is therefore 2 + 2 + 8 + 4 = 16 B per trainable parameter.
- Full training requires FP32/TF32/BF16/FP16 weights. Quantized models are fine-tuned with QLoRA (or LoRA).

### Activations: estimated unless you provide them

Activation memory depends on the architecture, so by default it is an **estimate** and the UI labels it as such.
Every coefficient lives in Settings → `ml.activation`.

- **Tokens per sample:**
  - image: (W × H) / patch², with patch = 16
  - video: frames × that ÷ 2
  - audio: seconds × 50
  - sequence: its length
  - diffusion: latent (W/8 × H/8) / 2²
  - generic: 1
- **Hidden size and depth:** taken from your input; otherwise derived from `params ≈ 12 × layers × hidden²` with hidden ≈ 80 × layers.
- **Inference:** `tokens × hidden × 16 B` per sample, plus early feature maps at input resolution (64 channels) for image and video.
  Diffusion adds the VAE decoder at full resolution (128 channels). All of this is multiplied by samples in flight (batch × concurrent requests).
- **Training:**
  - without checkpointing: `layers × tokens × hidden × 34 B` per sample (Korthikanti et al., 16-bit)
  - with gradient checkpointing: `layers × tokens × hidden × 2 B` plus one recomputed layer
  - plus feature maps
  - multiplied by the micro-batch (batch ÷ data-parallel replicas)
- Everything scales with the activation precision.

A measured value (`activation_memory_gb`) replaces the estimate. Where benchmark data exists, measured peak VRAM
should be preferred.

## Compute and memory speed (relative classes)

```
FLOPs per sample = 2 × parameters × tokens × (diffusion steps, inference) × (3 for training, 2 for LoRA/QLoRA)
samples/s        = target throughput, or
                   inference: in-flight samples / (priority latency × model-type scale)
                   training:  batch × priority steps/s (0.1 / 0.25 / 0.5 / 1)
required TFLOPS  → compute class  (low < 5, medium < 50, high < 200, very high < 800, extreme)
memory traffic   → GPU memory speed class
```

Latency scales by model type: diffusion ×20, video ×10, speech ×6, TTS ×4, OCR, segmentation and transformers ×2.

Ranking emphasis follows the workload type:

- Vision leans on compute.
- Embeddings lean on memory speed and cost.
- Training raises both compute and memory speed.

## Output

The output contains:

- **Required GPU memory**, AI compute, GPU memory speed, GPU count (× data-parallel replicas) and GPU mode.
- **Interconnect:** *Required* for multi-GPU training, *Recommended* for multi-GPU inference or data-parallel training.
- **Ranked NVIDIA and AMD matches** (shared vGPU first, then one dedicated GPU, then multi-GPU), plus why and why-not explanations.
- **What-ifs:** gradient checkpointing, LoRA → QLoRA, halve the batch, BF16, INT8 serving, more GPUs. They are shown only when they save hardware.
- **Flavor:** `AI-CUSTOM-INF-10G`, `AI-CUSTOM-TRAIN-80G` or `AI-CUSTOM-TRAIN-2X180G`, with a readable name
  ("Custom ML — Training — 80 GB GPU").
- **Deploy** creates an **MLDeploymentSpec** (`workload_type: custom_ml`) with the compiled OpenStack flavor. It is never provisioned.
