import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from 'react'
import { api } from './api'
import type { CalculateRequest, CalculationResult, GPUSpec, Meta, ModelSpec, WorkloadInput } from './types'

export const CUSTOM_MODEL_ID = '__custom__'
const STORAGE_KEY = 'gpucalc.form.v1'

export const DEFAULT_WORKLOAD: WorkloadInput = {
  precision: 'int4',
  context_length: 32768,
  concurrent_sequences: 20,
  performance_priority: 'balanced',
  framework: 'vllm',
  environment: 'production',
  context_mode: 'realistic',
  average_input_tokens: 4000,
  average_output_tokens: 1000,
  kv_cache_precision: 'auto',
  allow_vgpu: true,
  allow_multi_gpu: true,
  prefer_lowest_cost: true,
  add_safety_headroom: true,
  safety_margin_percent: null,
  dedicated_gpu_required: false,
  max_gpu_count: 8,
  tensor_parallel_size: null,
  target_tokens_per_second_per_user: null,
  target_aggregate_tokens_per_second: null,
  ttft_target_ms: null,
  tpot_target_ms: null,
  runtime_overhead_override_gb: null,
  workspace_percent_override: null,
  quantization_overhead_factor: null,
  calculation_mode: 'auto',
}

export const EMPTY_CUSTOM_MODEL: ModelSpec = {
  id: 'custom-model',
  vendor: 'Custom',
  family: 'Custom',
  name: 'My custom model',
  architecture: 'dense',
  total_parameters_b: 8,
  active_parameters_b: null,
  hidden_size: null,
  num_layers: null,
  attention_heads: null,
  kv_heads: null,
  head_dim: null,
  max_context_length: 32768,
  native_precision: ['bf16'],
  supported_serving_precisions: ['bf16', 'fp16', 'fp8', 'int8', 'int4'],
  metadata_status: 'user_defined',
  enabled: true,
}

export interface CalcForm {
  modelId: string
  custom: ModelSpec
  workload: WorkloadInput
  /** User edits to a catalog model's metadata ("USER OVERRIDE"); cleared when the model changes. */
  override?: ModelSpec | null
}

const DEFAULT_FORM: CalcForm = {
  modelId: 'llama-3.1-70b-instruct',
  custom: EMPTY_CUSTOM_MODEL,
  workload: DEFAULT_WORKLOAD,
  override: null,
}

function loadForm(): CalcForm {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return DEFAULT_FORM
    const parsed = JSON.parse(raw) as Partial<CalcForm>
    return {
      modelId: parsed.modelId ?? DEFAULT_FORM.modelId,
      custom: { ...EMPTY_CUSTOM_MODEL, ...parsed.custom },
      workload: { ...DEFAULT_WORKLOAD, ...parsed.workload },
      override: parsed.override && parsed.override.id === parsed.modelId ? parsed.override : null,
    }
  } catch {
    return DEFAULT_FORM
  }
}

export function toRequest(form: CalcForm): CalculateRequest {
  const w = form.workload
  if (form.modelId === CUSTOM_MODEL_ID) return { ...w, custom_model: { ...form.custom, metadata_status: 'user_defined' } }
  if (form.override) return { ...w, custom_model: { ...form.override, metadata_status: 'user_defined' } }
  return { ...w, model_id: form.modelId }
}

interface Ctx {
  form: CalcForm
  setForm: (f: CalcForm | ((f: CalcForm) => CalcForm)) => void
  setWorkload: (patch: Partial<WorkloadInput>) => void
  result: CalculationResult | null
  /** The request that produced `result` (the form may have changed since). */
  resultRequest: CalculateRequest | null
  setResult: (r: CalculationResult | null, req?: CalculateRequest | null) => void
  models: ModelSpec[]
  gpus: GPUSpec[]
  meta: Meta | null
  reloadModels: () => void
  loadError: string | null
}

const CalcContext = createContext<Ctx | null>(null)

export function CalcProvider({ children }: { children: ReactNode }) {
  const [form, setForm] = useState<CalcForm>(loadForm)
  const [calc, setCalc] = useState<{ result: CalculationResult | null; request: CalculateRequest | null }>({
    result: null,
    request: null,
  })
  const setResult = useCallback(
    (result: CalculationResult | null, request: CalculateRequest | null = null) => setCalc({ result, request }),
    [],
  )
  const [models, setModels] = useState<ModelSpec[]>([])
  const [gpus, setGpus] = useState<GPUSpec[]>([])
  const [meta, setMeta] = useState<Meta | null>(null)
  const [loadError, setLoadError] = useState<string | null>(null)

  const reloadModels = useCallback(() => {
    api
      .models()
      .then(setModels)
      .catch((e: Error) => setLoadError(`Could not load the model catalog: ${e.message}`))
    api
      .gpus()
      .then(setGpus)
      .catch(() => setGpus([]))
  }, [])

  useEffect(() => {
    reloadModels()
    api
      .meta()
      .then(setMeta)
      .catch((e: Error) => setLoadError(`Could not reach the API: ${e.message}`))
  }, [reloadModels])

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(form))
    } catch {
      /* storage unavailable - form resets on reload */
    }
  }, [form])

  const setWorkload = useCallback(
    (patch: Partial<WorkloadInput>) => setForm((f) => ({ ...f, workload: { ...f.workload, ...patch } })),
    [],
  )

  const value = useMemo(
    () => ({
      form,
      setForm,
      setWorkload,
      result: calc.result,
      resultRequest: calc.request,
      setResult,
      models,
      gpus,
      meta,
      reloadModels,
      loadError,
    }),
    [form, setWorkload, calc, setResult, models, gpus, meta, reloadModels, loadError],
  )
  return <CalcContext.Provider value={value}>{children}</CalcContext.Provider>
}

export function useCalc(): Ctx {
  const ctx = useContext(CalcContext)
  if (!ctx) throw new Error('useCalc must be used inside CalcProvider')
  return ctx
}

export function useSelectedModel(): ModelSpec | null {
  const { form, models } = useCalc()
  if (form.modelId === CUSTOM_MODEL_ID) return form.custom
  if (form.override) return form.override
  return models.find((m) => m.id === form.modelId) ?? null
}

/** The official catalog entry for the selected model (ignores user overrides). */
export function useCatalogModel(): ModelSpec | null {
  const { form, models } = useCalc()
  return models.find((m) => m.id === form.modelId) ?? null
}
