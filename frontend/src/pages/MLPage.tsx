import { ChevronDown, Rocket } from 'lucide-react'
import { useEffect, useState, type ReactNode } from 'react'
import { api } from '../api'
import { PriorityPicker } from '../components/calculator/WorkloadPanel'
import { FlowGraphic, WhyPanel, mlBreakdown, mlFlow } from '../components/results/Explain'
import { MatchList } from '../components/results/MatchList'
import { RequirementPanel, mlRequirement } from '../components/results/Requirement'
import { InfrastructureTab, TraceTab } from '../components/results/ResultExtras'
import { WarningsSummary, WhatIfPanel } from '../components/results/WhatIf'
import { Alert, Badge, Button, Field, Input, NumberInput, Segmented, Select, Spinner, Tabs, Toggle, cx } from '../components/ui'
import { fmtMem, fmtNum } from '../lib/format'
import type { DeploymentSpecRecord, MLMeta, MLSizingRequest, MLSizingResult, MLTask } from '../types'
import { CompareInfrastructure, DeployedNotice } from './FlavorPage'

const STORAGE_KEY = 'gpucalc.ml.v1'

const DEFAULT_ML: MLSizingRequest = {
  model_name: 'Custom vision model',
  model_category: 'computer_vision',
  task: 'inference',
  fine_tune_method: 'full',
  framework: 'pytorch',
  parameters_m: 500,
  precision: 'fp16',
  batch_size: 16,
  concurrent_requests: 10,
  performance_priority: 'balanced',
  image_width: 224,
  image_height: 224,
  channels: 3,
}

function load(): MLSizingRequest {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    return raw ? { ...DEFAULT_ML, ...(JSON.parse(raw) as Partial<MLSizingRequest>) } : DEFAULT_ML
  } catch {
    return DEFAULT_ML
  }
}

type ResultTab = 'why' | 'compare' | 'tech'

export function MLPage() {
  const [meta, setMeta] = useState<MLMeta | null>(null)
  const [req, setReq] = useState<MLSizingRequest>(load)
  const [result, setResult] = useState<MLSizingResult | null>(null)
  const [sizedKey, setSizedKey] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [tab, setTab] = useState<ResultTab>('why')
  const unit = 'GB' as const
  const set = (patch: Partial<MLSizingRequest>) => setReq((r) => ({ ...r, ...patch }))
  const key = JSON.stringify(req)

  useEffect(() => {
    api.mlMeta().then(setMeta).catch((e: Error) => setError(e.message))
  }, [])
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, key)
    } catch {
      /* storage unavailable */
    }
  }, [key])

  const calculate = async (body: MLSizingRequest = req) => {
    setLoading(true)
    setError(null)
    try {
      setResult(await api.sizeML(body))
      setSizedKey(JSON.stringify(body))
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setLoading(false)
    }
  }
  useEffect(() => {
    void calculate()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const category = meta?.categories.find((c) => c.id === req.model_category)
  const kind = category?.input_kind ?? 'image'
  const d = category?.defaults ?? {}
  const onCategory = (id: string) => {
    const c = meta?.categories.find((x) => x.id === id)
    // Reset the model-type inputs to the new type's defaults.
    set({
      model_category: id,
      image_width: c?.defaults.image_width ?? null,
      image_height: c?.defaults.image_height ?? null,
      frames_per_sample: c?.defaults.frames_per_sample ?? null,
      audio_seconds: c?.defaults.audio_seconds ?? null,
      sample_rate_hz: c?.defaults.sample_rate_hz ?? null,
      sequence_length: c?.defaults.sequence_length ?? null,
      diffusion_steps: c?.defaults.diffusion_steps ?? null,
      input_elements: c?.defaults.input_elements ?? null,
      uses_kv_cache: false,
    })
  }
  const training = req.task !== 'inference'
  const stale = result && sizedKey !== key

  return (
    <div className="mx-auto max-w-[1400px]">
      <header className="mb-6">
        <h1 className="text-2xl font-semibold tracking-tight text-ink sm:text-3xl">Custom ML model</h1>
        <p className="mt-1 text-sm text-ink-2">Describe your model and workload. We’ll calculate the GPU configuration: shared vGPU, dedicated GPU or multi-GPU.</p>
      </header>
      <div className="grid gap-6 lg:grid-cols-[minmax(340px,420px)_minmax(0,1fr)]">
        {/* ------------------------------------------------------------ inputs */}
        <div className="lg:sticky lg:top-6 lg:self-start">
          <form
            className="rounded-2xl border border-line bg-surface p-5 shadow-[0_1px_3px_rgba(15,23,42,0.06)]"
            onSubmit={(e) => {
              e.preventDefault()
              void calculate()
            }}
          >
            <div className="space-y-4">
              <Field label="Model name">
                <Input value={req.model_name} onChange={(e) => set({ model_name: e.target.value })} />
              </Field>
              <Field label="Model type">
                <Select value={req.model_category} onChange={(e) => onCategory(e.target.value)} aria-label="Model type">
                  {(meta?.categories ?? []).map((c) => (
                    <option key={c.id} value={c.id}>
                      {c.label}
                    </option>
                  ))}
                </Select>
              </Field>
              <Field label="Task">
                <Segmented<MLTask>
                  value={req.task}
                  onChange={(t) => set({ task: t, precision: t !== 'inference' && ['int4', 'int8', 'fp8'].includes(req.precision) ? 'bf16' : req.precision })}
                  options={[
                    { value: 'inference', label: 'Inference' },
                    { value: 'training', label: 'Training' },
                    { value: 'fine_tuning', label: 'Fine-tuning' },
                  ]}
                />
              </Field>
              {req.task === 'fine_tuning' && (
                <Field label="Fine-tuning method" hint={req.fine_tune_method === 'full' ? 'All weights trained (like training).' : req.fine_tune_method === 'lora' ? 'Only small adapters are trained.' : '4-bit base model + trained adapters: least memory.'}>
                  <Segmented<'full' | 'lora' | 'qlora'>
                    value={req.fine_tune_method}
                    onChange={(v) => set({ fine_tune_method: v })}
                    options={[
                      { value: 'full', label: 'Full' },
                      { value: 'lora', label: 'LoRA' },
                      { value: 'qlora', label: 'QLoRA' },
                    ]}
                  />
                </Field>
              )}
              <div className="grid grid-cols-2 gap-3">
                <Field label="Parameters (millions)" hint={`= ${fmtNum(req.parameters_m / 1000, req.parameters_m < 1000 ? 2 : 1)}B`}>
                  <NumberInput min={0.001} step="any" value={req.parameters_m} onChange={(v) => set({ parameters_m: v ?? 1 })} />
                </Field>
                <Field label="Precision">
                  <Select value={req.precision} onChange={(e) => set({ precision: e.target.value })}>
                    {(meta?.precisions ?? []).map((p) => (
                      <option key={p.id} value={p.id} disabled={training && req.fine_tune_method === 'full' && ['int4', 'int8', 'fp8'].includes(p.id)}>
                        {p.label}
                      </option>
                    ))}
                  </Select>
                </Field>
                <Field label="Batch size">
                  <NumberInput min={1} value={req.batch_size} onChange={(v) => set({ batch_size: Math.max(1, Math.round(v ?? 1)) })} />
                </Field>
                {!training && (
                  <Field label="Concurrent requests">
                    <NumberInput min={1} value={req.concurrent_requests} onChange={(v) => set({ concurrent_requests: Math.max(1, Math.round(v ?? 1)) })} />
                  </Field>
                )}
              </div>

              <InputFields kind={kind} req={req} set={set} defaults={d} />

              <PriorityPicker value={req.performance_priority} onChange={(v) => set({ performance_priority: v })} />
              <AdvancedML req={req} set={set} meta={meta} />

              <Button type="submit" variant="primary" size="lg" className="w-full" disabled={loading}>
                {loading && <Spinner />} Calculate GPU configuration
              </Button>
            </div>
          </form>
        </div>

        {/* ------------------------------------------------------------ result */}
        <div className="min-w-0 space-y-5">
          {error && <Alert tone="critical">{error}</Alert>}
          {stale && <Alert tone="info">Inputs changed — press “Calculate GPU configuration” to update.</Alert>}
          {!result && !error && (
            <div className="flex items-center gap-2 rounded-2xl border border-line bg-surface p-6 text-sm text-ink-3">
              <Spinner /> Calculating…
            </div>
          )}
          {result && (
            <div className={cx('space-y-5 transition-opacity', (loading || stale) && 'opacity-60')}>
              <Analysis result={result} />
              <RequirementPanel view={mlRequirement(result, unit)} unit={unit} />
              <WarningsSummary warnings={result.warnings} />
              <MatchList matches={result.matches} priority={result.workload.performance_priority} flavor={result.recommendation?.ai_flavor} unit={unit} />
              <Tabs<ResultTab>
                value={tab}
                onChange={setTab}
                tabs={[
                  { value: 'why', label: 'Why?' },
                  { value: 'compare', label: 'Compare options' },
                  { value: 'tech', label: 'Technical details' },
                ]}
              />
              {tab === 'why' && (
                <div className="space-y-5">
                  <FlowGraphic steps={mlFlow(result, unit)} title="From model to GPU" />
                  <WhyPanel
                    rec={result.recommendation}
                    matches={result.matches}
                    rejected={result.rejected}
                    breakdown={mlBreakdown(result.memory, result.activation_method === 'estimated')}
                    requiredGb={result.memory.required_vram_gb}
                    workloadBandwidthClass={result.compute.bandwidth_class}
                    priority={result.workload.performance_priority}
                    unit={unit}
                    noFitLines={result.explanation}
                  />
                  <section className="rounded-2xl border border-line bg-surface p-5">
                    <h3 className="mb-2 text-sm font-semibold text-ink">How it was calculated</h3>
                    <ol className="list-decimal space-y-1 pl-5 text-sm text-ink-2">
                      {result.explanation.map((l, i) => (
                        <li key={i}>{l}</li>
                      ))}
                    </ol>
                  </section>
                </div>
              )}
              {tab === 'compare' && <CompareInfrastructure matches={result.matches} rejected={result.rejected} />}
              {tab === 'tech' && (
                <div className="space-y-4">
                  <InfrastructureTab result={result} />
                  <details className="rounded-xl border border-line bg-surface p-4">
                    <summary className="cursor-pointer text-sm font-medium text-ink">Deployment spec (JSON)</summary>
                    <pre className="mt-3 overflow-x-auto rounded-lg bg-surface-2 p-4 font-mono text-xs text-ink">{JSON.stringify(result.deployment_spec, null, 2)}</pre>
                  </details>
                  <TraceTab result={result} />
                </div>
              )}
              <WhatIfPanel<Partial<MLSizingRequest>>
                suggestions={result.suggestions}
                unit={unit}
                onApply={(patch) => {
                  const next = { ...req, ...patch }
                  setReq(next)
                  void calculate(next)
                }}
              />
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

function InputFields({
  kind,
  req,
  set,
  defaults,
}: {
  kind: string
  req: MLSizingRequest
  set: (p: Partial<MLSizingRequest>) => void
  defaults: Record<string, number>
}) {
  const num = (key: keyof MLSizingRequest, label: string, hint?: string) => (
    <Field label={label} hint={hint}>
      <NumberInput
        min={1}
        step="any"
        value={(req[key] as number | null | undefined) ?? null}
        placeholder={defaults[key as string] != null ? String(defaults[key as string]) : ''}
        onChange={(v) => set({ [key]: v } as Partial<MLSizingRequest>)}
      />
    </Field>
  )
  const box = (children: ReactNode) => (
    <div className="rounded-lg border border-line bg-surface-2 p-3">
      <div className="mb-2 text-[11px] font-semibold tracking-wider text-ink-3 uppercase">Input size</div>
      <div className="grid grid-cols-2 gap-3">{children}</div>
    </div>
  )
  if (kind === 'image' || kind === 'diffusion')
    return box(
      <>
        {num('image_width', 'Width (px)')}
        {num('image_height', 'Height (px)')}
        {kind === 'diffusion' ? num('diffusion_steps', 'Denoising steps') : num('channels', 'Channels')}
        {req.model_category === 'object_detection' && num('num_detections', 'Detections / image')}
      </>,
    )
  if (kind === 'video')
    return box(
      <>
        {num('image_width', 'Frame width')}
        {num('image_height', 'Frame height')}
        {num('frames_per_sample', 'Frames per sample')}
      </>,
    )
  if (kind === 'audio')
    return box(
      <>
        {num('audio_seconds', 'Audio duration (s)')}
        {num('sample_rate_hz', 'Sample rate (Hz)')}
      </>,
    )
  if (kind === 'sequence')
    return box(
      <>
        {num('sequence_length', 'Sequence length (tokens)')}
        {num('hidden_size', 'Hidden size', 'Optional')}
      </>,
    )
  return box(
    <>
      {num('input_elements', 'Input features', 'Elements per sample')}
      {num('output_elements', 'Output elements', 'Optional')}
    </>,
  )
}

function AdvancedML({ req, set, meta }: { req: MLSizingRequest; set: (p: Partial<MLSizingRequest>) => void; meta: MLMeta | null }) {
  const [open, setOpen] = useState(false)
  const training = req.task !== 'inference'
  const precisions = meta?.precisions ?? []
  const nullableNum = (key: keyof MLSizingRequest, label: string, hint?: string, step = 'any') => (
    <Field label={label} hint={hint}>
      <NumberInput min={0} step={step} value={(req[key] as number | null | undefined) ?? null} placeholder="Auto" onChange={(v) => set({ [key]: v } as Partial<MLSizingRequest>)} />
    </Field>
  )
  return (
    <div className="border-t border-line pt-3">
      <button type="button" aria-expanded={open} onClick={() => setOpen(!open)} className="flex w-full items-center justify-between text-left">
        <span>
          <span className="block text-[13px] font-medium text-ink-2">Advanced model settings</span>
          <span className="block text-xs text-ink-3">Framework, layers, activations, optimizer, parallelism, headroom</span>
        </span>
        <ChevronDown className={cx('h-4 w-4 text-ink-3 transition', open && 'rotate-180')} aria-hidden />
      </button>
      {open && (
        <div className="fade-up mt-4 space-y-4">
          <div className="grid grid-cols-2 gap-3">
            <Field label="Framework">
              <Select value={req.framework} onChange={(e) => set({ framework: e.target.value })}>
                {(meta?.frameworks ?? []).map((f) => (
                  <option key={f.id} value={f.id}>
                    {f.label}
                    {f.nvidia_only ? ' (NVIDIA only)' : ''}
                  </option>
                ))}
              </Select>
            </Field>
            {nullableNum('model_file_size_gb', 'Model file size (GB)', 'Overrides weight estimate')}
            {nullableNum('num_layers', 'Number of layers')}
            {nullableNum('hidden_size', 'Hidden dimension')}
            {nullableNum('input_elements', 'Input tensor elements')}
            {nullableNum('output_elements', 'Output tensor elements')}
            {nullableNum('activation_memory_gb', 'Activation memory (GB)', 'Measured value replaces the estimate')}
            {!training && nullableNum('target_latency_ms', 'Target latency (ms)')}
            {nullableNum('target_throughput', 'Target throughput (samples/s)')}
            {nullableNum('safety_margin_percent', 'Safety margin %')}
            {nullableNum('model_growth_percent', 'Expected model growth %')}
          </div>
          {req.model_category === 'transformer' && (
            <Toggle checked={!!req.uses_kv_cache} onChange={(v) => set({ uses_kv_cache: v })} label="Model uses a KV cache" hint="Autoregressive generation with attention caching" />
          )}
          {training && (
            <>
              <Toggle
                checked={!!req.gradient_checkpointing}
                onChange={(v) => set({ gradient_checkpointing: v })}
                label="Gradient checkpointing"
                hint="Recompute activations in the backward pass: far less memory, ~20–30% slower"
              />
              <div className="grid grid-cols-2 gap-3">
                <Field label="Optimizer">
                  <Select value={req.optimizer ?? 'adamw'} onChange={(e) => set({ optimizer: e.target.value })}>
                    {(meta?.optimizers ?? []).map((o) => (
                      <option key={o.id} value={o.id}>
                        {o.label}
                      </option>
                    ))}
                  </Select>
                </Field>
                <Field label="Optimizer precision">
                  <Select value={req.optimizer_precision ?? 'fp32'} onChange={(e) => set({ optimizer_precision: e.target.value })}>
                    {precisions.filter((p) => ['fp32', 'bf16', 'fp16', 'fp8', 'int8'].includes(p.id)).map((p) => (
                      <option key={p.id} value={p.id}>
                        {p.label}
                      </option>
                    ))}
                  </Select>
                </Field>
                {nullableNum('optimizer_bytes_per_param', 'Optimizer bytes / param', 'Override')}
                <Field label="Training precision">
                  <Select value={req.training_precision ?? ''} onChange={(e) => set({ training_precision: e.target.value || null })}>
                    <option value="">Auto</option>
                    {precisions.filter((p) => ['fp32', 'tf32', 'bf16', 'fp16', 'fp8'].includes(p.id)).map((p) => (
                      <option key={p.id} value={p.id}>
                        {p.label}
                      </option>
                    ))}
                  </Select>
                </Field>
                <Field label="Gradient precision">
                  <Select value={req.gradient_precision ?? ''} onChange={(e) => set({ gradient_precision: e.target.value || null })}>
                    <option value="">Auto</option>
                    {precisions.filter((p) => ['fp32', 'bf16', 'fp16'].includes(p.id)).map((p) => (
                      <option key={p.id} value={p.id}>
                        {p.label}
                      </option>
                    ))}
                  </Select>
                </Field>
                {req.task === 'fine_tuning' && req.fine_tune_method !== 'full' && nullableNum('adapter_percent', 'Adapter size (% of params)', `Default ${meta?.default_adapter_percent ?? 0.5}%`)}
                <Field label="Data-parallel replicas">
                  <NumberInput min={1} value={req.data_parallel_size ?? 1} onChange={(v) => set({ data_parallel_size: Math.max(1, Math.round(v ?? 1)) })} />
                </Field>
                <Field label="Pipeline-parallel size">
                  <NumberInput min={1} value={req.pipeline_parallel_size ?? 1} onChange={(v) => set({ pipeline_parallel_size: Math.max(1, Math.round(v ?? 1)) })} />
                </Field>
              </div>
              <Toggle checked={req.master_weights ?? true} onChange={(v) => set({ master_weights: v })} label="FP32 master weights" hint="Standard for mixed-precision training (+4 bytes per trainable parameter)" />
            </>
          )}
          <div className="grid grid-cols-2 gap-3">
            {nullableNum('tensor_parallel_size', 'Tensor-parallel size', 'Forces the GPU split', '1')}
            <Field label="Max GPU count">
              <Select value={req.max_gpu_count ?? 8} onChange={(e) => set({ max_gpu_count: Number(e.target.value) })}>
                {[1, 2, 4, 8, 16].map((n) => (
                  <option key={n} value={n}>
                    {n}
                  </option>
                ))}
              </Select>
            </Field>
          </div>
          <Toggle checked={req.allow_vgpu ?? true} onChange={(v) => set({ allow_vgpu: v })} label="Allow shared GPU (vGPU)" />
          <Toggle checked={req.allow_multi_gpu ?? true} onChange={(v) => set({ allow_multi_gpu: v })} label="Allow multiple GPUs" />
        </div>
      )}
    </div>
  )
}

function Analysis({ result }: { result: MLSizingResult }) {
  const m = result.memory
  const training = result.model.task !== 'inference'
  const [deploying, setDeploying] = useState(false)
  const [deployed, setDeployed] = useState<DeploymentSpecRecord | null>(null)
  const [error, setError] = useState<string | null>(null)
  const rows: [string, number][] = [['Weights', m.model_weights_gb]]
  if (training) rows.push(['Gradients', m.gradients_gb], ['Optimizer', m.optimizer_gb])
  rows.push([`Activations${result.activation_method === 'estimated' ? ' (est.)' : ''}`, m.activations_gb], ['Runtime', m.runtime_overhead_gb + m.workspace_gb + m.communication_gb], ['Headroom', m.headroom_gb])
  const deploy = async () => {
    if (!result.deployment_spec) return
    setDeploying(true)
    try {
      setDeployed(await api.createDeploymentSpec(result.deployment_spec, result.ai_flavor_display ?? result.model.name))
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setDeploying(false)
    }
  }
  return (
    <section className="overflow-hidden rounded-2xl border border-brand/40 bg-surface shadow-[0_1px_3px_rgba(15,23,42,0.06)]">
      <div className="flex flex-wrap items-start justify-between gap-3 bg-brand-soft px-5 py-4">
        <div>
          <div className="text-xs font-semibold tracking-wider text-brand-text uppercase">Custom model analysis</div>
          <div className="text-xl font-semibold text-brand-text">{result.ai_flavor_display ?? 'No fit'}</div>
          {result.recommendation && <div className="font-mono text-xs text-brand-text/80">{result.recommendation.ai_flavor}</div>}
        </div>
        <Badge tone={result.activation_method === 'estimated' ? 'warning' : 'good'}>
          {result.activation_method === 'estimated' ? '≈ Estimated activations' : '✓ Your activation value'}
        </Badge>
      </div>
      <div className="grid gap-6 p-5 md:grid-cols-2">
        <dl className="grid grid-cols-2 gap-x-4 gap-y-2 text-sm">
          <dt className="text-ink-3">Model</dt>
          <dd className="font-medium text-ink">{result.model.name}</dd>
          <dt className="text-ink-3">Type</dt>
          <dd className="text-ink">{result.model.category_label}</dd>
          <dt className="text-ink-3">Task</dt>
          <dd className="text-ink">
            {result.model.task.replace('_', '-')}
            {result.model.fine_tune_method ? ` (${result.model.fine_tune_method.toUpperCase()})` : ''}
          </dd>
          <dt className="text-ink-3">Parameters</dt>
          <dd className="text-ink">{fmtNum(result.model.parameters_b, result.model.parameters_b < 10 ? 2 : 0)}B</dd>
          <dt className="text-ink-3">Precision</dt>
          <dd className="text-ink">
            {result.model.precision.toUpperCase()}
            {training && result.model.compute_precision !== result.model.precision ? ` · compute ${result.model.compute_precision.toUpperCase()}` : ''}
          </dd>
          <dt className="text-ink-3">Batch</dt>
          <dd className="text-ink">
            {result.workload.batch_size}
            {!training ? ` × ${result.workload.concurrent_requests} requests` : ''} · {result.workload.input_description}
          </dd>
        </dl>
        <dl className="tabular divide-y divide-line text-sm">
          {rows.map(([k, v]) => (
            <div key={k} className="flex justify-between py-1">
              <dt className="text-ink-2">{k}</dt>
              <dd className="text-ink">{fmtMem(v, 'GB', 1)}</dd>
            </div>
          ))}
          <div className="flex justify-between py-1.5 font-semibold">
            <dt className="text-ink">Required VRAM</dt>
            <dd className="text-ink">{fmtMem(m.required_vram_gb, 'GB', 1)}</dd>
          </div>
        </dl>
      </div>
      {result.recommendation && (
        <div className="flex flex-wrap items-center gap-2 border-t border-line px-5 py-3">
          <Button variant="primary" onClick={deploy} disabled={deploying}>
            {deploying ? <Spinner /> : <Rocket className="h-4 w-4" />} Deploy
          </Button>
          <span className="text-xs text-ink-3">Creates an ML deployment spec and OpenStack flavor recommendation. Nothing is provisioned.</span>
        </div>
      )}
      {error && (
        <div className="px-5 pb-4">
          <Alert tone="critical">{error}</Alert>
        </div>
      )}
      {deployed && <DeployedNotice record={deployed} />}
    </section>
  )
}
