"""Pydantic schemas shared by the API, the catalog and the sizing engine."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

Architecture = Literal["dense", "moe"]
MetadataStatus = Literal["verified", "imported", "user_defined"]
Priority = Literal["economy", "balanced", "performance", "maximum"]
Environment = Literal["production", "development"]
ContextMode = Literal["realistic", "max"]
CalculationModeRequest = Literal["auto", "estimate", "benchmark"]
SharingMode = Literal["time_sliced", "mig"]
Availability = Literal["available", "limited", "unavailable", "unknown"]

ID_PATTERN = r"^[a-z0-9][a-z0-9._-]{0,99}$"


# --------------------------------------------------------------------------- catalogs


class ModelSpec(BaseModel):
    """Model metadata. Missing architecture fields stay None - they are never guessed."""

    model_config = ConfigDict(extra="ignore")

    id: str = Field(pattern=ID_PATTERN)
    vendor: str = "Custom"
    family: str = "Custom"
    name: str = Field(min_length=1, max_length=200)
    architecture: Architecture
    total_parameters_b: float = Field(gt=0, le=20000)
    active_parameters_b: Optional[float] = Field(default=None, gt=0)

    hidden_size: Optional[int] = Field(default=None, gt=0)
    num_layers: Optional[int] = Field(default=None, gt=0)
    attention_heads: Optional[int] = Field(default=None, gt=0)
    kv_heads: Optional[int] = Field(default=None, gt=0)
    head_dim: Optional[int] = Field(default=None, gt=0)
    attention_type: Literal["standard", "mla"] = "standard"
    kv_lora_rank: Optional[int] = Field(default=None, gt=0)
    qk_rope_head_dim: Optional[int] = Field(default=None, gt=0)
    sliding_window: Optional[int] = Field(default=None, gt=0)
    num_experts: Optional[int] = Field(default=None, gt=0)
    experts_per_token: Optional[int] = Field(default=None, gt=0)

    max_context_length: Optional[int] = Field(default=None, gt=0)
    native_precision: list[str] = Field(default_factory=list)
    supported_serving_precisions: list[str] = Field(default_factory=list)

    metadata_status: MetadataStatus = "user_defined"
    source_url: Optional[str] = None
    license: Optional[str] = None
    notes: Optional[str] = None
    enabled: bool = True

    @model_validator(mode="after")
    def _validate(self) -> "ModelSpec":
        if self.architecture == "dense":
            if self.active_parameters_b is None:
                self.active_parameters_b = self.total_parameters_b
            elif abs(self.active_parameters_b - self.total_parameters_b) > 1e-6:
                raise ValueError("Dense models: active_parameters_b must equal total_parameters_b")
        else:
            if self.active_parameters_b is None:
                raise ValueError("MoE models require active_parameters_b")
            if self.active_parameters_b > self.total_parameters_b:
                raise ValueError("active_parameters_b cannot exceed total_parameters_b")
        if self.kv_heads and self.attention_heads and self.kv_heads > self.attention_heads:
            raise ValueError("kv_heads cannot exceed attention_heads")
        if self.attention_heads and self.kv_heads and self.attention_heads % self.kv_heads:
            raise ValueError("attention_heads must be a multiple of kv_heads (GQA/MQA)")
        if self.attention_type == "mla" and (self.kv_lora_rank is None or self.qk_rope_head_dim is None):
            raise ValueError("MLA models require kv_lora_rank and qk_rope_head_dim")
        return self

    @property
    def resolved_head_dim(self) -> Optional[int]:
        if self.head_dim:
            return self.head_dim
        if self.hidden_size and self.attention_heads:
            return self.hidden_size // self.attention_heads
        return None

    @property
    def has_exact_kv_metadata(self) -> bool:
        if self.attention_type == "mla":
            return bool(self.num_layers and self.kv_lora_rank and self.qk_rope_head_dim)
        return bool(self.num_layers and self.kv_heads and self.resolved_head_dim)


class GPUSpec(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = Field(pattern=ID_PATTERN)
    vendor: str = "NVIDIA"
    model: str
    architecture_generation: Optional[str] = None
    form_factor: Optional[str] = None
    vram_gb: float = Field(gt=0)
    usable_vram_mib: Optional[int] = Field(default=None, gt=0)
    memory_bandwidth_gbps: Optional[float] = Field(default=None, gt=0)
    fp16_tflops: Optional[float] = Field(default=None, gt=0)
    bf16_tflops: Optional[float] = Field(default=None, gt=0)
    fp8_tflops: Optional[float] = Field(default=None, gt=0)
    int8_tops: Optional[float] = Field(default=None, gt=0)
    int4_tops: Optional[float] = Field(default=None, gt=0)
    supports_vgpu: Optional[bool] = None
    supports_mig: Optional[bool] = None
    supports_nvlink: Optional[bool] = None
    interconnect_type: Optional[str] = None
    power_watts: Optional[float] = Field(default=None, gt=0)
    cost_per_hour: Optional[float] = Field(default=None, ge=0)
    availability: Availability = "unknown"
    openstack_pci_alias: Optional[str] = None
    metadata_source: Optional[str] = None
    metadata_status: MetadataStatus = "user_defined"
    notes: Optional[str] = None
    enabled: bool = True

    @property
    def display_name(self) -> str:
        return f"{self.vendor} {self.model}".strip()

    @property
    def has_high_speed_link(self) -> bool:
        """NVLink, Infinity Fabric or similar GPU-to-GPU fabric (anything faster than plain PCIe)."""
        if self.supports_nvlink:
            return True
        return bool(self.interconnect_type) and self.interconnect_type.strip().lower() != "pcie"

    @property
    def usable_vram_bytes(self) -> float:
        if self.usable_vram_mib:
            return self.usable_vram_mib * 1024**2
        return self.vram_gb * 1e9


class VGPUProfile(BaseModel):
    model_config = ConfigDict(extra="ignore")

    profile_name: str = Field(min_length=1, max_length=64)
    physical_gpu: str
    vram_gb: float = Field(gt=0, description="Profile framebuffer in GiB")
    max_instances_per_gpu: int = Field(gt=0)
    compute_mode: str = "compute"
    sharing_mode: SharingMode = "time_sliced"
    mig_compute_slices: Optional[int] = Field(default=None, gt=0)
    supports_cuda: bool = True
    license_required: bool = True
    license_cost_per_hour: Optional[float] = Field(default=None, ge=0)
    # Free instances right now (None = unknown). 0 means a suitable profile is exhausted and is skipped.
    available_instances: Optional[int] = Field(default=None, ge=0)
    # Placement trait that selects this profile (from your OpenStack config; never guessed).
    openstack_trait: Optional[str] = None
    source_url: Optional[str] = "https://docs.nvidia.com/vgpu/latest/grid-vgpu-user-guide/index.html"
    metadata_status: MetadataStatus = "imported"
    notes: Optional[str] = None
    enabled: bool = True

    def compute_fraction(self, mig_total_slices: int = 7) -> float:
        """Worst-case share of the physical GPU's compute and bandwidth."""
        if self.sharing_mode == "mig" and self.mig_compute_slices:
            return min(1.0, self.mig_compute_slices / mig_total_slices)
        return 1.0 / self.max_instances_per_gpu

    @property
    def usable_vram_bytes(self) -> float:
        return self.vram_gb * 1024**3


class AIFlavorDefinition(BaseModel):
    """Customer-facing AI VM tier (AI-8B, AI-70B, ...). Product catalog only - never hardware sizing.

    The VRAM range is catalog guidance; the sizing engine is the source of truth.
    """

    model_config = ConfigDict(extra="ignore")

    id: str = Field(pattern=ID_PATTERN)
    display_name: str = Field(min_length=1, max_length=40)
    description: str = ""
    group: Literal["small", "medium", "large", "ultra"] = "small"
    parameter_floor_b: float = Field(ge=0)
    parameter_ceiling_b: float = Field(gt=0)
    baseline_precision: str = "int4"
    default_vram_min_gb: float = Field(ge=0)
    default_vram_max_gb: float = Field(ge=0)
    recommended_use_cases: list[str] = Field(default_factory=list)
    example_model_ids: list[str] = Field(default_factory=list)
    allow_vgpu: bool = True
    allow_full_gpu: bool = True
    allow_multi_gpu: bool = False
    performance_tier: str = "standard"
    default_context_length: int = Field(default=32768, gt=0)
    default_concurrency: int = Field(default=20, gt=0)
    default_performance_priority: Priority = "balanced"
    commercial_sku: Optional[str] = None
    enabled: bool = True

    @model_validator(mode="after")
    def _check(self) -> "AIFlavorDefinition":
        if self.parameter_floor_b >= self.parameter_ceiling_b:
            raise ValueError("parameter_floor_b must be below parameter_ceiling_b")
        if self.default_vram_min_gb > self.default_vram_max_gb:
            raise ValueError("default_vram_min_gb cannot exceed default_vram_max_gb")
        if not (self.allow_vgpu or self.allow_full_gpu or self.allow_multi_gpu):
            raise ValueError("Allow at least one of vGPU, full GPU or multi-GPU")
        return self


class InventoryRecord(BaseModel):
    id: Optional[int] = None
    gpu_id: str
    site: str = "default"
    installed: int = Field(ge=0)
    available: int = Field(ge=0)

    @model_validator(mode="after")
    def _check(self) -> "InventoryRecord":
        if self.available > self.installed:
            raise ValueError("available cannot exceed installed")
        return self


class BenchmarkRecord(BaseModel):
    id: Optional[int] = None
    model_id: str
    gpu_id: str
    gpu_count: int = Field(default=1, gt=0)
    precision: str
    framework: str = "vllm"
    tensor_parallel: int = Field(default=1, gt=0)
    context_length: int = Field(gt=0)
    batch_size: Optional[int] = Field(default=None, gt=0)
    concurrency: int = Field(gt=0)
    prefill_tokens_per_second: Optional[float] = Field(default=None, ge=0)
    decode_tokens_per_second: Optional[float] = Field(default=None, ge=0)
    aggregate_tokens_per_second: Optional[float] = Field(default=None, ge=0)
    ttft_ms: Optional[float] = Field(default=None, ge=0)
    tpot_ms: Optional[float] = Field(default=None, ge=0)
    peak_vram_gb: Optional[float] = Field(default=None, ge=0)
    average_gpu_utilization: Optional[float] = Field(default=None, ge=0, le=100)
    power_watts: Optional[float] = Field(default=None, ge=0)
    source: str
    date: Optional[str] = None
    verified: bool = False


# --------------------------------------------------------------------------- requests


class WorkloadInput(BaseModel):
    """Everything except the model. Only precision/context/concurrency/priority are needed."""

    model_config = ConfigDict(extra="forbid")

    precision: str = "int4"
    context_length: int = Field(default=32768, gt=0, le=16_777_216)
    concurrent_sequences: int = Field(default=1, gt=0, le=100_000)
    performance_priority: Priority = "balanced"
    framework: str = "vllm"
    environment: Environment = "production"

    context_mode: ContextMode = "realistic"
    average_input_tokens: Optional[int] = Field(default=None, ge=0)
    average_output_tokens: Optional[int] = Field(default=None, ge=0)
    kv_cache_precision: str = "auto"

    allow_vgpu: bool = True
    allow_single_gpu: bool = True  # one dedicated GPU; AI flavor tiers can switch it off (e.g. multi-GPU-only tiers)
    allow_multi_gpu: bool = True
    prefer_lowest_cost: bool = True
    add_safety_headroom: bool = True
    safety_margin_percent: Optional[float] = Field(default=None, ge=0, le=100)
    dedicated_gpu_required: bool = False
    max_gpu_count: int = Field(default=8, ge=1, le=64)
    tensor_parallel_size: Optional[int] = Field(default=None, ge=1, le=64)
    allowed_gpu_ids: Optional[list[str]] = None

    target_tokens_per_second_per_user: Optional[float] = Field(default=None, gt=0)
    target_aggregate_tokens_per_second: Optional[float] = Field(default=None, gt=0)
    ttft_target_ms: Optional[float] = Field(default=None, gt=0)
    tpot_target_ms: Optional[float] = Field(default=None, gt=0)
    batch_size: Optional[int] = Field(default=None, gt=0)

    runtime_overhead_override_gb: Optional[float] = Field(default=None, ge=0)
    workspace_percent_override: Optional[float] = Field(default=None, ge=0, le=100)
    quantization_overhead_factor: Optional[float] = Field(default=None, ge=1, le=3)

    calculation_mode: CalculationModeRequest = "auto"


class CalculateRequest(WorkloadInput):
    model_id: Optional[str] = None
    custom_model: Optional[ModelSpec] = None

    @model_validator(mode="after")
    def _one_model(self) -> "CalculateRequest":
        if bool(self.model_id) == bool(self.custom_model):
            raise ValueError("Provide exactly one of model_id or custom_model")
        return self


class CompareModelsRequest(BaseModel):
    model_ids: list[str] = Field(min_length=1, max_length=12)
    workload: WorkloadInput


class CompareGPUsRequest(BaseModel):
    request: CalculateRequest
    gpu_ids: Optional[list[str]] = None


class ComparePrecisionRequest(BaseModel):
    request: CalculateRequest
    precisions: Optional[list[str]] = None


class CustomModelRequest(BaseModel):
    model: ModelSpec
    persist: bool = False


class SaveCalculationRequest(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    request: CalculateRequest


class ModelStatusPatch(BaseModel):
    enabled: Optional[bool] = None
    metadata_status: Optional[MetadataStatus] = None
    source_url: Optional[str] = None


class YamlImportRequest(BaseModel):
    yaml: str = Field(max_length=2_000_000)
    overwrite: bool = False


# --------------------------------------------------------------------------- results


class Warning_(BaseModel):
    code: str
    severity: Literal["info", "warning", "critical"]
    message: str


class TraceStep(BaseModel):
    step: str
    formula: str
    value: str


class WeightDetail(BaseModel):
    parameters_b: float
    bytes_per_parameter: float
    quantization_overhead_factor: float
    raw_gb: float
    gb: float
    gib: float


class KVDetail(BaseModel):
    method: Literal["exact", "mla", "heuristic"]
    kv_precision: str
    bytes_per_element: float
    bytes_per_token: float
    head_dim: Optional[int]
    context_mode: ContextMode
    tokens_per_sequence: int
    sized_tokens: int
    gb: float
    gib: float
    formula: str


class MemoryBreakdown(BaseModel):
    model_weights_gb: float
    kv_cache_gb: float
    runtime_overhead_gb: float
    workspace_gb: float
    communication_gb: float
    subtotal_gb: float
    headroom_gb: float
    required_vram_gb: float
    safety_margin_percent: float
    # Non-LLM / training components (0 for LLM inference).
    activations_gb: float = 0
    gradients_gb: float = 0
    optimizer_gb: float = 0


class ComputeProfile(BaseModel):
    active_parameters_b: float
    total_parameters_b: float
    architecture: Architecture
    tokens_per_second_per_user: float
    requested_tokens_per_second: float
    prefill_tokens_per_second: float
    compute_demand_score: float
    classification: str
    memory_intensity: str
    bandwidth_class: str = "medium"
    required_tflops: float
    decode_bytes_per_step_gb: float
    required_bandwidth_gbps: float
    moe_weight_fraction_read_per_step: float


class CostEstimate(BaseModel):
    hourly: float
    daily: float
    monthly: float
    formula: str


class BenchmarkMatch(BaseModel):
    benchmark_id: Optional[int]
    source: str
    verified: bool
    concurrency: int
    context_length: int
    aggregate_tokens_per_second: Optional[float]
    decode_tokens_per_second: Optional[float]
    ttft_ms: Optional[float]
    tpot_ms: Optional[float]
    peak_vram_gb: Optional[float]


class Candidate(BaseModel):
    key: str
    kind: Literal["vgpu", "full_gpu", "multi_gpu"]
    gpu_id: str
    gpu_name: str
    gpu_vendor: str = ""
    gpu_compute_class: Optional[str] = None
    gpu_bandwidth_class: Optional[str] = None
    interconnect: Optional[str] = None
    vgpu_profile: Optional[str] = None
    sharing_mode: Optional[str] = None
    count: int
    tensor_parallel: int
    vram_per_unit_gb: float
    required_per_unit_gb: float
    memory_per_unit: MemoryBreakdown
    utilization_percent: float
    headroom_gb: float
    fits: bool
    fits_weights_only: bool
    gpu_units: float
    bandwidth_ratio: Optional[float]
    compute_ratio: Optional[float]
    est_decode_tokens_per_second_per_user: Optional[float]
    est_ttft_ms: Optional[float]
    availability: str
    available_units: Optional[int]
    cost: Optional[CostEstimate]
    confidence: Literal["estimated", "benchmark"]
    benchmark: Optional[BenchmarkMatch] = None
    notes: list[str] = Field(default_factory=list)
    rejected_reason: Optional[str] = None
    capacity_limited: bool = False  # rejected only because capacity is short right now
    score: Optional[float] = None
    label: str


class Recommendation(BaseModel):
    ai_flavor: str  # AI-70B-Q4-PRO
    ai_flavor_short: str = ""  # AI-70B-PRO (precision shown only when it differs from the tier baseline)
    flavor_variant: str = ""  # SHARED | PRO | MULTI
    flavor_tier_id: Optional[str] = None  # ai-70b (None when above the largest tier)
    flavor_name: str
    gpu: str
    gpu_id: str
    count: int
    mode: Literal["vgpu", "full_gpu", "multi_gpu"]
    vgpu_profile: Optional[str]
    tensor_parallel: int
    utilization_percent: float
    headroom_gb: float
    confidence: Literal["estimated", "benchmark"]
    summary: str
    candidate: Candidate


class ModelSummary(BaseModel):
    id: str
    name: str
    family: str
    vendor: str
    architecture: Architecture
    total_parameters_b: float
    active_parameters_b: float
    max_context_length: Optional[int]
    metadata_status: MetadataStatus
    source_url: Optional[str]
    kv_metadata: Literal["exact", "mla", "heuristic"]


class ResolvedWorkload(BaseModel):
    precision: str
    kv_cache_precision: str
    context_length: int
    context_mode: ContextMode
    concurrent_sequences: int
    average_input_tokens: int
    average_output_tokens: int
    performance_priority: Priority
    framework: str
    environment: Environment
    safety_margin_percent: float
    tokens_per_second_per_user: float


class Suggestion(BaseModel):
    """A what-if variant of the request that needs less hardware."""

    key: str
    title: str
    tradeoff: str
    patch: dict[str, Any]
    required_vram_gb: float
    recommendation: str
    ai_flavor: str
    saves_gb: float


class CalculationResult(BaseModel):
    calculation_mode: Literal["estimate", "benchmark"]
    confidence_label: Literal["ESTIMATED", "BENCHMARK-BASED"]
    model: ModelSummary
    workload: ResolvedWorkload
    weights: WeightDetail
    kv_cache: KVDetail
    memory: MemoryBreakdown
    memory_gib: MemoryBreakdown
    compute: ComputeProfile
    recommendation: Optional[Recommendation]
    matches: list[Candidate] = Field(default_factory=list)
    alternatives: list[Candidate]
    rejected: list[Candidate]
    vgpu_verdict: str
    warnings: list[Warning_]
    explanation: list[str]
    deployment_spec: Optional[dict[str, Any]] = None
    openstack: Optional[dict[str, Any]]
    openstack_yaml: Optional[str]
    trace: list[TraceStep]
    suggestions: list[Suggestion] = Field(default_factory=list)


# --------------------------------------------------------------------------- custom ML models

MLCategory = Literal[
    "computer_vision",
    "image_classification",
    "object_detection",
    "segmentation",
    "ocr",
    "speech_recognition",
    "text_to_speech",
    "embedding",
    "diffusion",
    "video",
    "recommendation",
    "tabular",
    "time_series",
    "transformer",
    "generic",
    "other",
]
MLTask = Literal["inference", "training", "fine_tuning"]


class MLSizingRequest(BaseModel):
    """A non-LLM (or custom) ML workload. Only the simple-mode fields are needed; the rest are
    advanced settings with sensible defaults per model type."""

    model_config = ConfigDict(extra="forbid")

    model_name: str = Field(default="Custom ML model", min_length=1, max_length=200)
    model_category: MLCategory = "computer_vision"
    task: MLTask = "inference"
    fine_tune_method: Literal["full", "lora", "qlora"] = "full"
    framework: str = "pytorch"
    parameters_m: float = Field(gt=0, le=10_000_000, description="Parameter count in millions")
    model_file_size_gb: Optional[float] = Field(default=None, gt=0)
    precision: str = "fp16"
    batch_size: int = Field(default=1, ge=1, le=100_000)
    concurrent_requests: int = Field(default=1, ge=1, le=100_000)
    performance_priority: Priority = "balanced"
    target_latency_ms: Optional[float] = Field(default=None, gt=0)
    target_throughput: Optional[float] = Field(default=None, gt=0, description="Samples per second")

    # Model-type inputs (defaults depend on the category when omitted)
    image_width: Optional[int] = Field(default=None, gt=0, le=32768)
    image_height: Optional[int] = Field(default=None, gt=0, le=32768)
    channels: int = Field(default=3, ge=1, le=64)
    frames_per_sample: Optional[int] = Field(default=None, gt=0, le=10_000)
    audio_seconds: Optional[float] = Field(default=None, gt=0, le=36_000)
    sample_rate_hz: Optional[int] = Field(default=None, gt=0)
    sequence_length: Optional[int] = Field(default=None, gt=0, le=10_000_000)
    diffusion_steps: Optional[int] = Field(default=None, gt=0, le=1000)
    num_detections: Optional[int] = Field(default=None, gt=0)
    input_elements: Optional[int] = Field(default=None, gt=0, description="Input tensor elements per sample")
    output_elements: Optional[int] = Field(default=None, gt=0, description="Output tensor elements per sample")

    # Advanced
    num_layers: Optional[int] = Field(default=None, gt=0)
    hidden_size: Optional[int] = Field(default=None, gt=0)
    uses_kv_cache: bool = False
    activation_memory_gb: Optional[float] = Field(default=None, ge=0, description="Manual total activation memory")
    gradient_checkpointing: bool = False
    optimizer: str = "adamw"
    optimizer_bytes_per_param: Optional[float] = Field(default=None, ge=0)
    optimizer_precision: str = "fp32"
    master_weights: bool = True
    training_precision: Optional[str] = None
    gradient_precision: Optional[str] = None
    adapter_percent: Optional[float] = Field(default=None, gt=0, le=100)
    data_parallel_size: int = Field(default=1, ge=1, le=1024)
    tensor_parallel_size: Optional[int] = Field(default=None, ge=1, le=64)
    pipeline_parallel_size: int = Field(default=1, ge=1, le=64)
    model_growth_percent: float = Field(default=0, ge=0, le=1000)
    safety_margin_percent: Optional[float] = Field(default=None, ge=0, le=100)

    # Infrastructure policy
    allow_vgpu: bool = True
    allow_single_gpu: bool = True
    allow_multi_gpu: bool = True
    max_gpu_count: int = Field(default=8, ge=1, le=64)
    allowed_gpu_ids: Optional[list[str]] = None
    prefer_lowest_cost: bool = True


class MLModelSummary(BaseModel):
    name: str
    category: MLCategory
    category_label: str
    task: MLTask
    fine_tune_method: Optional[str]
    parameters_b: float
    trainable_parameters_b: float
    precision: str
    compute_precision: str
    framework: str


class MLWorkloadSummary(BaseModel):
    batch_size: int
    micro_batch_per_gpu: int
    concurrent_requests: int
    in_flight_samples: int
    input_description: str
    effective_tokens_per_sample: float
    performance_priority: Priority
    data_parallel_size: int
    samples_per_second: float
    safety_margin_percent: float


class MLComputeProfile(BaseModel):
    classification: str
    bandwidth_class: str
    memory_intensity: str
    required_tflops: float
    required_bandwidth_gbps: float
    gflops_per_sample: float
    samples_per_second: float


class MLRequirement(BaseModel):
    gpu_count_label: str
    total_gpus: int
    mode: str
    interconnect: Literal["Not required", "Recommended", "Required"]


class MLSizingResult(BaseModel):
    workload_type: Literal["custom_ml"] = "custom_ml"
    calculation_mode: Literal["estimate", "benchmark"] = "estimate"
    confidence_label: Literal["ESTIMATED", "BENCHMARK-BASED"] = "ESTIMATED"
    activation_method: Literal["estimated", "manual"]
    model: MLModelSummary
    workload: MLWorkloadSummary
    memory: MemoryBreakdown
    memory_gib: MemoryBreakdown
    compute: MLComputeProfile
    requirement: MLRequirement
    recommendation: Optional[Recommendation]
    ai_flavor_display: Optional[str]
    matches: list[Candidate]
    rejected: list[Candidate]
    vgpu_verdict: str
    warnings: list[Warning_]
    explanation: list[str]
    suggestions: list[Suggestion] = Field(default_factory=list)
    deployment_spec: Optional[dict[str, Any]]
    openstack: Optional[dict[str, Any]]
    openstack_yaml: Optional[str]
    trace: list[TraceStep]


# --------------------------------------------------------------------------- AI flavor sizing


class FlavorSizeRequest(WorkloadInput):
    """Size an AI VM flavor. `flavor_id` is optional (Mode B: start from a model)."""

    flavor_id: Optional[str] = None
    model_id: Optional[str] = None
    custom_model: Optional[ModelSpec] = None

    @model_validator(mode="after")
    def _one_model(self) -> "FlavorSizeRequest":
        if bool(self.model_id) == bool(self.custom_model):
            raise ValueError("Provide exactly one of model_id or custom_model")
        return self


class FlavorRef(BaseModel):
    id: str
    display_name: str


class FlavorSizeResult(BaseModel):
    requested_flavor: Optional[FlavorRef]
    model_flavor: Optional[FlavorRef]  # tier the model belongs to by parameter count
    recommended_tier: Optional[FlavorRef]
    tier_status: Literal["fits", "exceeds", "oversized", "no_tier", "model_tier"]
    tier_message: Optional[str]
    recommended_flavor: Optional[str]  # AI-70B-Q4-PRO
    display_flavor: Optional[str]  # AI-70B-PRO
    variant: Optional[str]
    mode_label: Optional[str]
    allocated_gpu_memory_gb: Optional[float]
    vram_range_note: Optional[str]
    sizing: dict[str, Any]  # SizingResult fields in product terms
    calculation: CalculationResult


class DeploymentSpecCreate(BaseModel):
    name: Optional[str] = Field(default=None, max_length=200)
    spec: dict[str, Any]


class DeploymentSpecRecord(BaseModel):
    id: int
    name: Optional[str]
    created_at: str
    workload_type: str
    ai_flavor: str
    spec: dict[str, Any]
    openstack: dict[str, Any]
