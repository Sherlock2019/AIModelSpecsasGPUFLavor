// Mirrors backend/app/schemas.py

export type Architecture = 'dense' | 'moe'
export type MetadataStatus = 'verified' | 'imported' | 'user_defined'
export type Priority = 'economy' | 'balanced' | 'performance' | 'maximum'
export type ContextMode = 'realistic' | 'max'

export interface ModelSpec {
  id: string
  vendor: string
  family: string
  name: string
  architecture: Architecture
  total_parameters_b: number
  active_parameters_b?: number | null
  hidden_size?: number | null
  num_layers?: number | null
  attention_heads?: number | null
  kv_heads?: number | null
  head_dim?: number | null
  attention_type?: 'standard' | 'mla'
  kv_lora_rank?: number | null
  qk_rope_head_dim?: number | null
  sliding_window?: number | null
  num_experts?: number | null
  experts_per_token?: number | null
  max_context_length?: number | null
  native_precision: string[]
  supported_serving_precisions: string[]
  metadata_status: MetadataStatus
  source_url?: string | null
  license?: string | null
  notes?: string | null
  enabled: boolean
}

export interface GPUSpec {
  id: string
  vendor: string
  model: string
  architecture_generation?: string | null
  form_factor?: string | null
  vram_gb: number
  usable_vram_mib?: number | null
  memory_bandwidth_gbps?: number | null
  fp16_tflops?: number | null
  bf16_tflops?: number | null
  fp8_tflops?: number | null
  int8_tops?: number | null
  int4_tops?: number | null
  supports_vgpu?: boolean | null
  supports_mig?: boolean | null
  supports_nvlink?: boolean | null
  interconnect_type?: string | null
  power_watts?: number | null
  cost_per_hour?: number | null
  availability: 'available' | 'limited' | 'unavailable' | 'unknown'
  openstack_pci_alias?: string | null
  metadata_source?: string | null
  metadata_status: MetadataStatus
  notes?: string | null
  enabled: boolean
}

export interface VGPUProfile {
  profile_name: string
  physical_gpu: string
  vram_gb: number
  max_instances_per_gpu: number
  compute_mode: string
  sharing_mode: 'time_sliced' | 'mig'
  mig_compute_slices?: number | null
  supports_cuda: boolean
  license_required: boolean
  license_cost_per_hour?: number | null
  source_url?: string | null
  metadata_status: MetadataStatus
  notes?: string | null
  enabled: boolean
}

export interface InventoryRecord {
  id?: number | null
  gpu_id: string
  site: string
  installed: number
  available: number
}

export interface BenchmarkRecord {
  id?: number | null
  model_id: string
  gpu_id: string
  gpu_count: number
  precision: string
  framework: string
  tensor_parallel: number
  context_length: number
  batch_size?: number | null
  concurrency: number
  prefill_tokens_per_second?: number | null
  decode_tokens_per_second?: number | null
  aggregate_tokens_per_second?: number | null
  ttft_ms?: number | null
  tpot_ms?: number | null
  peak_vram_gb?: number | null
  average_gpu_utilization?: number | null
  power_watts?: number | null
  source: string
  date?: string | null
  verified: boolean
}

export interface WorkloadInput {
  precision: string
  context_length: number
  concurrent_sequences: number
  performance_priority: Priority
  framework: string
  environment: 'production' | 'development'
  context_mode: ContextMode
  average_input_tokens?: number | null
  average_output_tokens?: number | null
  kv_cache_precision: string
  allow_vgpu: boolean
  allow_multi_gpu: boolean
  prefer_lowest_cost: boolean
  add_safety_headroom: boolean
  safety_margin_percent?: number | null
  dedicated_gpu_required: boolean
  max_gpu_count: number
  tensor_parallel_size?: number | null
  target_tokens_per_second_per_user?: number | null
  target_aggregate_tokens_per_second?: number | null
  ttft_target_ms?: number | null
  tpot_target_ms?: number | null
  runtime_overhead_override_gb?: number | null
  workspace_percent_override?: number | null
  quantization_overhead_factor?: number | null
  calculation_mode: 'auto' | 'estimate' | 'benchmark'
}

export interface CalculateRequest extends WorkloadInput {
  model_id?: string | null
  custom_model?: ModelSpec | null
}

export interface MemoryBreakdown {
  model_weights_gb: number
  kv_cache_gb: number
  runtime_overhead_gb: number
  workspace_gb: number
  communication_gb: number
  subtotal_gb: number
  headroom_gb: number
  required_vram_gb: number
  safety_margin_percent: number
}

export interface CostEstimate {
  hourly: number
  daily: number
  monthly: number
  formula: string
}

export interface BenchmarkMatch {
  benchmark_id?: number | null
  source: string
  verified: boolean
  concurrency: number
  context_length: number
  aggregate_tokens_per_second?: number | null
  decode_tokens_per_second?: number | null
  ttft_ms?: number | null
  tpot_ms?: number | null
  peak_vram_gb?: number | null
}

export interface Candidate {
  key: string
  kind: 'vgpu' | 'full_gpu' | 'multi_gpu'
  gpu_id: string
  gpu_name: string
  gpu_vendor: string
  gpu_compute_class?: string | null
  gpu_bandwidth_class?: string | null
  interconnect?: string | null
  vgpu_profile?: string | null
  sharing_mode?: string | null
  count: number
  tensor_parallel: number
  vram_per_unit_gb: number
  required_per_unit_gb: number
  memory_per_unit: MemoryBreakdown
  utilization_percent: number
  headroom_gb: number
  fits: boolean
  fits_weights_only: boolean
  gpu_units: number
  bandwidth_ratio?: number | null
  compute_ratio?: number | null
  est_decode_tokens_per_second_per_user?: number | null
  est_ttft_ms?: number | null
  availability: string
  available_units?: number | null
  cost?: CostEstimate | null
  confidence: 'estimated' | 'benchmark'
  benchmark?: BenchmarkMatch | null
  notes: string[]
  rejected_reason?: string | null
  score?: number | null
  label: string
}

export interface Recommendation {
  ai_flavor: string
  flavor_name: string
  gpu: string
  gpu_id: string
  count: number
  mode: 'vgpu' | 'full_gpu' | 'multi_gpu'
  vgpu_profile?: string | null
  tensor_parallel: number
  utilization_percent: number
  headroom_gb: number
  confidence: 'estimated' | 'benchmark'
  summary: string
  candidate: Candidate
}

export interface CalcWarning {
  code: string
  severity: 'info' | 'warning' | 'critical'
  message: string
}

export interface CalculationResult {
  calculation_mode: 'estimate' | 'benchmark'
  confidence_label: 'ESTIMATED' | 'BENCHMARK-BASED'
  model: {
    id: string
    name: string
    family: string
    vendor: string
    architecture: Architecture
    total_parameters_b: number
    active_parameters_b: number
    max_context_length?: number | null
    metadata_status: MetadataStatus
    source_url?: string | null
    kv_metadata: 'exact' | 'mla' | 'heuristic'
  }
  workload: {
    precision: string
    kv_cache_precision: string
    context_length: number
    context_mode: ContextMode
    concurrent_sequences: number
    average_input_tokens: number
    average_output_tokens: number
    performance_priority: Priority
    framework: string
    environment: string
    safety_margin_percent: number
    tokens_per_second_per_user: number
  }
  weights: {
    parameters_b: number
    bytes_per_parameter: number
    quantization_overhead_factor: number
    raw_gb: number
    gb: number
    gib: number
  }
  kv_cache: {
    method: 'exact' | 'mla' | 'heuristic'
    kv_precision: string
    bytes_per_element: number
    bytes_per_token: number
    head_dim?: number | null
    context_mode: ContextMode
    tokens_per_sequence: number
    sized_tokens: number
    gb: number
    gib: number
    formula: string
  }
  memory: MemoryBreakdown
  memory_gib: MemoryBreakdown
  compute: {
    active_parameters_b: number
    total_parameters_b: number
    architecture: Architecture
    tokens_per_second_per_user: number
    requested_tokens_per_second: number
    prefill_tokens_per_second: number
    compute_demand_score: number
    classification: string
    memory_intensity: string
    bandwidth_class: string
    required_tflops: number
    decode_bytes_per_step_gb: number
    required_bandwidth_gbps: number
    moe_weight_fraction_read_per_step: number
  }
  recommendation?: Recommendation | null
  matches: Candidate[]
  alternatives: Candidate[]
  rejected: Candidate[]
  vgpu_verdict: string
  warnings: CalcWarning[]
  explanation: string[]
  openstack?: Record<string, unknown> | null
  openstack_yaml?: string | null
  trace: { step: string; formula: string; value: string }[]
  suggestions: Suggestion[]
}

export interface Suggestion {
  key: string
  title: string
  tradeoff: string
  patch: Partial<WorkloadInput>
  required_vram_gb: number
  recommendation: string
  ai_flavor: string
  saves_gb: number
}

export interface Meta {
  precisions: { id: string; label: string; bytes_per_parameter: number; default_overhead_factor: number }[]
  kv_precisions: { id: string; label: string }[]
  frameworks: { id: string; label: string }[]
  priorities: { id: Priority; label: string; safety_margin_percent: number; tokens_per_second_per_user: number }[]
  environments: { id: string; label: string; default_safety_margin_percent: number }[]
  context_defaults: { default_average_input_tokens: number; default_average_output_tokens: number }
  gpu_counts: number[]
}

export interface SummaryRow {
  model_id?: string
  name?: string
  architecture?: Architecture
  total_parameters_b?: number
  active_parameters_b?: number
  precision?: string
  context_length?: number
  weights_gb?: number
  kv_cache_gb?: number
  required_vram_gb?: number
  recommended?: string | null
  gpu_count?: number | null
  mode?: string | null
  ai_flavor?: string | null
  cost?: CostEstimate | null
  confidence?: string
  note?: string | null
  supported?: boolean
  error?: string
}

export interface GpuCompareRow {
  gpu_id: string
  gpu: string
  vram_gb: number
  memory_bandwidth_gbps?: number | null
  fits: boolean
  gpu_count?: number | null
  utilization_percent?: number | null
  required_per_gpu_gb?: number | null
  bandwidth_ratio?: number | null
  compute_ratio?: number | null
  est_decode_tokens_per_second_per_user?: number | null
  cost?: CostEstimate | null
  confidence: string
  benchmark?: BenchmarkMatch | null
  reason?: string | null
  notes: string[]
}

export interface SavedCalculationSummary {
  id: number
  name: string
  created_at: string
  model?: string
  precision?: string
  required_vram_gb?: number
  recommended?: string
  ai_flavor?: string
}
