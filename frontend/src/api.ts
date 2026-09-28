import type {
  AIFlavor,
  BenchmarkRecord,
  CalculateRequest,
  CalculationResult,
  DeploymentSpec,
  DeploymentSpecRecord,
  FlavorSizeResult,
  MLMeta,
  MLSizingRequest,
  MLSizingResult,
  GPUSpec,
  GpuCompareRow,
  InventoryRecord,
  Meta,
  ModelSpec,
  SavedCalculationSummary,
  SummaryRow,
  VGPUProfile,
  WorkloadInput,
} from './types'

const BASE = '/api/v1'
const TOKEN_KEY = 'gpucalc.adminToken'

export class ApiError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

export function getAdminToken(): string {
  try {
    return localStorage.getItem(TOKEN_KEY) ?? ''
  } catch {
    return ''
  }
}

export function setAdminToken(token: string) {
  try {
    if (token) localStorage.setItem(TOKEN_KEY, token)
    else localStorage.removeItem(TOKEN_KEY)
  } catch {
    /* storage unavailable - token only lives for this page */
  }
}

function formatDetail(detail: unknown): string {
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail)) {
    return detail
      .map((d) => {
        const e = d as { loc?: unknown[]; msg?: string }
        const loc = Array.isArray(e.loc) ? e.loc.filter((p) => p !== 'body').join('.') : ''
        return loc ? `${loc}: ${e.msg}` : String(e.msg ?? JSON.stringify(d))
      })
      .join('; ')
  }
  return JSON.stringify(detail)
}

async function request<T>(path: string, init: RequestInit = {}, raw = false): Promise<T> {
  const headers: Record<string, string> = { ...(init.headers as Record<string, string>) }
  if (init.body && !headers['Content-Type']) headers['Content-Type'] = 'application/json'
  const token = getAdminToken()
  if (token) headers['X-Admin-Token'] = token
  const res = await fetch(`${BASE}${path}`, { ...init, headers })
  if (!res.ok) {
    let message = res.statusText
    try {
      const body = await res.json()
      message = formatDetail(body.detail ?? body)
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(res.status, message)
  }
  if (res.status === 204) return undefined as T
  return (raw ? res.text() : res.json()) as Promise<T>
}

const post = <T>(path: string, body: unknown, raw = false) =>
  request<T>(path, { method: 'POST', body: JSON.stringify(body) }, raw)
const put = <T>(path: string, body: unknown) => request<T>(path, { method: 'PUT', body: JSON.stringify(body) })
const patch = <T>(path: string, body: unknown) => request<T>(path, { method: 'PATCH', body: JSON.stringify(body) })

export const api = {
  meta: () => request<Meta>('/meta'),
  models: (includeDisabled = false) => request<ModelSpec[]>(`/models?include_disabled=${includeDisabled}`),
  createModel: (m: ModelSpec) => post<ModelSpec>('/models', m),
  updateModel: (m: ModelSpec) => put<ModelSpec>(`/models/${encodeURIComponent(m.id)}`, m),
  patchModel: (id: string, p: Partial<Pick<ModelSpec, 'enabled' | 'metadata_status' | 'source_url'>>) =>
    patch<ModelSpec>(`/models/${encodeURIComponent(id)}`, p),
  exportModels: () => request<string>('/models/export', {}, true),
  importModels: (yaml: string, overwrite: boolean) =>
    post<{ created: number; updated: number; skipped: number; errors: string[] }>('/models/import', { yaml, overwrite }),
  validateCustomModel: (model: ModelSpec) =>
    post<{ model: ModelSpec; kv_metadata: string; resolved_head_dim: number | null }>('/custom-model', { model }),

  gpus: () => request<GPUSpec[]>('/gpus'),
  createGpu: (g: GPUSpec) => post<GPUSpec>('/gpus', g),
  updateGpu: (g: GPUSpec) => put<GPUSpec>(`/gpus/${encodeURIComponent(g.id)}`, g),
  vgpuProfiles: () => request<VGPUProfile[]>('/vgpu-profiles'),
  upsertVgpuProfile: (p: VGPUProfile) => post<VGPUProfile>('/vgpu-profiles', p),
  inventory: () => request<InventoryRecord[]>('/inventory'),
  putInventory: (rows: InventoryRecord[]) => put<InventoryRecord[]>('/inventory', rows),

  benchmarks: () => request<BenchmarkRecord[]>('/benchmarks'),
  createBenchmark: (b: BenchmarkRecord) => post<BenchmarkRecord>('/benchmarks', b),
  deleteBenchmark: (id: number) => request<void>(`/benchmarks/${id}`, { method: 'DELETE' }),

  calculate: (req: CalculateRequest) => post<CalculationResult>('/calculate', req),
  exportCalculation: (req: CalculateRequest, format: 'json' | 'yaml' | 'csv' | 'markdown') =>
    post<string>(`/calculate/export?format=${format}`, req, true),
  compareModels: (model_ids: string[], workload: WorkloadInput) =>
    post<SummaryRow[]>('/compare/models', { model_ids, workload }),
  comparePrecisions: (req: CalculateRequest) => post<SummaryRow[]>('/compare/precisions', { request: req }),
  compareGpus: (req: CalculateRequest, gpu_ids?: string[]) =>
    post<GpuCompareRow[]>('/compare/gpus', { request: req, gpu_ids }),

  calculations: () => request<SavedCalculationSummary[]>('/calculations'),
  saveCalculation: (name: string, req: CalculateRequest) => post<{ id: number }>('/calculations', { name, request: req }),
  calculation: (id: number) =>
    request<{ id: number; name: string; request: CalculateRequest; result: CalculationResult }>(`/calculations/${id}`),
  deleteCalculation: (id: number) => request<void>(`/calculations/${id}`, { method: 'DELETE' }),

  aiFlavors: (includeDisabled = false) => request<AIFlavor[]>(`/ai-flavors?include_disabled=${includeDisabled}`),
  aiFlavor: (id: string) => request<AIFlavor>(`/ai-flavors/${encodeURIComponent(id)}`),
  createAIFlavor: (f: AIFlavor) => post<AIFlavor>('/ai-flavors', f),
  updateAIFlavor: (f: AIFlavor) => put<AIFlavor>(`/ai-flavors/${encodeURIComponent(f.id)}`, f),
  sizeFlavor: (req: CalculateRequest & { flavor_id?: string | null }) => post<FlavorSizeResult>('/ai-flavors/size', req),

  mlMeta: () => request<MLMeta>('/ml/meta'),
  sizeML: (req: MLSizingRequest) => post<MLSizingResult>('/ml/size', req),

  deploymentSpecs: () => request<DeploymentSpecRecord[]>('/deployment-specs'),
  createDeploymentSpec: (spec: DeploymentSpec, name?: string) => post<DeploymentSpecRecord>('/deployment-specs', { spec, name }),
  deleteDeploymentSpec: (id: number) => request<void>(`/deployment-specs/${id}`, { method: 'DELETE' }),

  settings: () => request<Record<string, unknown>>('/settings'),
  putSettings: (s: Record<string, unknown>) => put<Record<string, unknown>>('/settings', s),
  resetSettings: () => post<Record<string, unknown>>('/settings/reset', {}),
}
