import { ArrowDownRight, ArrowLeft, ArrowUpRight, Check, Rocket } from 'lucide-react'
import { useEffect, useMemo, useRef, useState } from 'react'
import { Link, useNavigate, useParams, useSearchParams } from 'react-router-dom'
import { api } from '../api'
import { ModelSpecFields } from '../components/calculator/CustomModelForm'
import { ConcurrencyStepper, ContextSelect, PrecisionSelect, PrioritySelect } from '../components/calculator/WorkloadPanel'
import { FlowGraphic, LlmWhyPanel, llmFlow } from '../components/results/Explain'
import { MatchList } from '../components/results/MatchList'
import { RequirementPanel, llmRequirement } from '../components/results/Requirement'
import { InfrastructureTab, TraceTab } from '../components/results/ResultExtras'
import { WarningsSummary } from '../components/results/WhatIf'
import { Alert, Badge, Button, Select, Spinner, Table, Tabs, cx } from '../components/ui'
import { downloadText } from '../lib/download'
import { clampContext, fmtContext, fmtMem, fmtMoney, fmtParams } from '../lib/format'
import { candidateTitle, memoryFit, modeLabel } from '../lib/plain'
import { CUSTOM_MODEL_ID, toRequest, useCalc } from '../state'
import type { AIFlavor, Candidate, DeploymentSpecRecord, FlavorSizeResult, Priority } from '../types'

type Tab = 'why' | 'compare' | 'tech'

export function FlavorPage() {
  const { id = '' } = useParams()
  const [params] = useSearchParams()
  const navigate = useNavigate()
  const { form, setForm, models } = useCalc()
  const [flavors, setFlavors] = useState<AIFlavor[] | null>(null)
  const [result, setResult] = useState<FlavorSizeResult | null>(null)
  const [sizedKey, setSizedKey] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [showAll, setShowAll] = useState(false)
  const [tab, setTab] = useState<Tab>('why')
  const autoSized = useRef<string | null>(null)
  const unit = 'GB' as const

  useEffect(() => {
    api.aiFlavors().then(setFlavors).catch((e: Error) => setError(e.message))
  }, [])
  const flavor = flavors?.find((f) => f.id === id) ?? null

  // Arriving on a flavor: pick its model (from ?m= or the tier's first example) and its default workload.
  useEffect(() => {
    if (!flavor || !models.length || autoSized.current === `${id}?${params}`) return
    const fromLink = params.get('m')
    const inTier = flavor.model_ids ?? []
    const modelId = fromLink ?? (form.modelId !== CUSTOM_MODEL_ID && inTier.includes(form.modelId) ? form.modelId : flavor.example_model_ids.find((m) => inTier.includes(m)) ?? inTier[0])
    const model = models.find((m) => m.id === modelId)
    const num = (k: string) => (params.get(k) ? Number(params.get(k)) : null)
    setForm((f) => ({
      ...f,
      modelId: modelId ?? f.modelId,
      override: null,
      workload: {
        ...f.workload,
        precision: params.get('p') ?? flavor.baseline_precision,
        context_length: clampContext(num('ctx') ?? flavor.default_context_length, model?.max_context_length),
        concurrent_sequences: num('n') ?? flavor.default_concurrency,
        performance_priority: (params.get('pr') as Priority) ?? flavor.default_performance_priority,
        context_mode: 'realistic',
      },
    }))
    autoSized.current = `${id}?${params}`
    setResult(null)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [flavor, models.length, id, params])

  const request = { ...toRequest(form), flavor_id: id }
  const requestKey = JSON.stringify(request)

  const size = async () => {
    setLoading(true)
    setError(null)
    try {
      setResult(await api.sizeFlavor(request))
      setSizedKey(requestKey)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setLoading(false)
    }
  }

  // Size automatically once the flavor's defaults are in place (the button re-sizes after edits).
  // `result` is a dependency so switching tiers (same page, result cleared) sizes again.
  useEffect(() => {
    if (flavor && autoSized.current === `${id}?${params}` && !result && !loading && form.modelId) void size()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [requestKey, flavor, result])

  const tierModels = useMemo(() => {
    const ids = new Set(flavor?.model_ids ?? [])
    return { inTier: models.filter((m) => ids.has(m.id)), other: models.filter((m) => !ids.has(m.id)) }
  }, [flavor, models])

  if (!flavors) return <div className="flex items-center gap-2 text-sm text-ink-3"><Spinner /> Loading…</div>
  if (!flavor) return <Alert tone="critical">AI flavor “{id}” not found. <Link to="/" className="underline">Back to the catalog</Link></Alert>

  const stale = result && sizedKey !== requestKey
  const selected = models.find((m) => m.id === form.modelId)
  const isCustom = form.modelId === CUSTOM_MODEL_ID

  const goTier = (tierId: string) => {
    const w = form.workload
    const q = new URLSearchParams({ ctx: String(w.context_length), n: String(w.concurrent_sequences), p: w.precision, pr: w.performance_priority })
    if (!isCustom) q.set('m', form.modelId)
    navigate(`/flavors/${tierId}?${q}`)
  }

  return (
    <div className="mx-auto max-w-[1400px]">
      <Link to="/" className="mb-4 inline-flex items-center gap-1.5 text-sm text-ink-3 hover:text-ink">
        <ArrowLeft className="h-4 w-4" /> AI VM catalog
      </Link>
      <div className="grid gap-6 lg:grid-cols-[minmax(340px,400px)_minmax(0,1fr)]">
        {/* ------------------------------------------------------------ configuration */}
        <div className="lg:sticky lg:top-6 lg:self-start">
          <section className="rounded-2xl border border-line bg-surface p-5 shadow-[0_1px_3px_rgba(15,23,42,0.06)]">
            <div className="mb-4">
              <div className="font-mono text-2xl font-semibold text-ink">{flavor.display_name}</div>
              <div className="text-sm text-ink-2">
                {flavor.description} · up to ~{flavor.parameter_ceiling_b}B · typical {flavor.default_vram_min_gb}–{flavor.default_vram_max_gb} GB
              </div>
            </div>
            <div className="space-y-5">
              <div>
                <label htmlFor="flavor-model" className="mb-1.5 block text-[13px] font-medium text-ink-2">
                  Model
                </label>
                <Select
                  id="flavor-model"
                  value={form.modelId}
                  onChange={(e) => {
                    const m = models.find((x) => x.id === e.target.value)
                    setForm((f) => ({
                      ...f,
                      modelId: e.target.value,
                      override: null,
                      workload: { ...f.workload, context_length: clampContext(f.workload.context_length, m?.max_context_length) },
                    }))
                  }}
                >
                  <optgroup label={`${flavor.display_name} models`}>
                    {tierModels.inTier.map((m) => (
                      <option key={m.id} value={m.id}>
                        {m.name}
                      </option>
                    ))}
                  </optgroup>
                  {(showAll || (selected && !tierModels.inTier.includes(selected))) && (
                    <optgroup label="Other models">
                      {tierModels.other.map((m) => (
                        <option key={m.id} value={m.id}>
                          {m.name} ({fmtParams(m.total_parameters_b)})
                        </option>
                      ))}
                    </optgroup>
                  )}
                  <optgroup label="Custom">
                    <option value={CUSTOM_MODEL_ID}>Custom {flavor.display_name.replace('AI-', '')} model…</option>
                  </optgroup>
                </Select>
                <div className="mt-1.5 flex justify-between text-xs">
                  <button type="button" className="text-brand-text hover:underline" onClick={() => setShowAll(!showAll)}>
                    {showAll ? 'Only this tier' : 'Choose another model'}
                  </button>
                  {selected && !isCustom && <span className="text-ink-3">{fmtParams(selected.total_parameters_b)} · {fmtContext(selected.max_context_length)} context</span>}
                </div>
                {isCustom && (
                  <div className="mt-3 rounded-lg border border-line bg-surface-2 p-3">
                    <ModelSpecFields value={form.custom} onChange={(custom) => setForm((f) => ({ ...f, custom }))} />
                  </div>
                )}
              </div>
              <PrecisionSelect />
              <ContextSelect />
              <ConcurrencyStepper />
              <PrioritySelect />
              <Button variant="primary" size="lg" className="w-full" onClick={size} disabled={loading}>
                {loading && <Spinner />} Size my VM
              </Button>
              <p className="text-center text-xs text-ink-3">No GPU choice needed — the calculator decides the infrastructure.</p>
            </div>
          </section>
        </div>

        {/* ------------------------------------------------------------ result */}
        <div className="min-w-0 space-y-5">
          {error && <Alert tone="critical">{error}</Alert>}
          {stale && <Alert tone="info">Inputs changed — press “Size my VM” to update.</Alert>}
          {!result && !error && (
            <div className="flex items-center gap-2 rounded-2xl border border-line bg-surface p-6 text-sm text-ink-3">
              <Spinner /> Sizing {flavor.display_name}…
            </div>
          )}
          {result && (
            <div className={cx('space-y-5 transition-opacity', (loading || stale) && 'opacity-60')}>
              <TierBanner result={result} onGo={goTier} />
              <YourVM result={result} />
              <RequirementPanel view={llmRequirement(result.calculation, unit)} unit={unit} />
              <WarningsSummary warnings={result.calculation.warnings} />
              <Tabs<Tab>
                value={tab}
                onChange={setTab}
                tabs={[
                  { value: 'why', label: `Why ${result.display_flavor ?? 'this'}?` },
                  { value: 'compare', label: 'Compare options' },
                  { value: 'tech', label: 'Technical details' },
                ]}
              />
              {tab === 'why' && (
                <div className="space-y-5">
                  <FlowGraphic steps={llmFlow(result.calculation, unit, result.display_flavor)} title="From AI flavor to infrastructure" />
                  <LlmWhyPanel result={result.calculation} unit={unit} title={`Why ${result.display_flavor}?`} />
                </div>
              )}
              {tab === 'compare' && (
                <div className="space-y-5">
                  <CompareInfrastructure matches={result.calculation.matches} rejected={result.calculation.rejected} />
                  <MatchList matches={result.calculation.matches} priority={result.calculation.workload.performance_priority} flavor={result.recommended_flavor} unit={unit} />
                </div>
              )}
              {tab === 'tech' && <TechnicalDetails result={result} />}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

function TierBanner({ result, onGo }: { result: FlavorSizeResult; onGo: (tierId: string) => void }) {
  const target = result.recommended_tier
  if (result.tier_status === 'fits' || result.tier_status === 'model_tier') {
    return result.tier_message && result.tier_status === 'model_tier' ? <Alert tone="info">{result.tier_message}</Alert> : null
  }
  if (result.tier_status === 'no_tier') return <Alert tone="warning">{result.tier_message}</Alert>
  const up = result.tier_status === 'exceeds'
  return (
    <div className={cx('flex flex-wrap items-center justify-between gap-3 rounded-xl px-4 py-3', up ? 'bg-warning-soft text-warning-text' : 'bg-info-soft text-ink-2')}>
      <span className="flex items-start gap-2 text-sm">
        {up ? <ArrowUpRight className="mt-0.5 h-4 w-4 shrink-0" /> : <ArrowDownRight className="mt-0.5 h-4 w-4 shrink-0" />}
        {result.tier_message}
      </span>
      {target && (
        <Button size="sm" variant={up ? 'primary' : 'secondary'} onClick={() => onGo(target.id)}>
          {up ? `Upgrade to ${target.display_name}` : `Switch to ${target.display_name}`}
        </Button>
      )}
    </div>
  )
}

function YourVM({ result }: { result: FlavorSizeResult }) {
  const calc = result.calculation
  const rec = calc.recommendation
  const w = calc.workload
  const [deploying, setDeploying] = useState(false)
  const [deployed, setDeployed] = useState<DeploymentSpecRecord | null>(null)
  const [error, setError] = useState<string | null>(null)
  const deploy = async () => {
    if (!calc.deployment_spec) return
    setDeploying(true)
    setError(null)
    try {
      setDeployed(await api.createDeploymentSpec(calc.deployment_spec, `${result.display_flavor} · ${calc.model.name}`))
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setDeploying(false)
    }
  }
  return (
    <section className="overflow-hidden rounded-2xl border border-brand/40 bg-surface shadow-[0_1px_3px_rgba(15,23,42,0.06)]">
      <div className="flex flex-wrap items-start justify-between gap-4 bg-brand-soft px-5 py-4">
        <div>
          <div className="text-xs font-semibold tracking-wider text-brand-text uppercase">Your AI VM</div>
          <div className="font-mono text-3xl font-semibold text-brand-text">{result.display_flavor ?? 'No fit'}</div>
          {result.recommended_flavor && result.recommended_flavor !== result.display_flavor && (
            <div className="font-mono text-xs text-brand-text/80">SKU {result.recommended_flavor}</div>
          )}
        </div>
        {result.mode_label && <Badge tone={result.variant === 'SHARED' ? 'good' : result.variant === 'MULTI' ? 'warning' : 'brand'}>{result.mode_label}</Badge>}
      </div>
      {rec ? (
        <div className="grid gap-4 p-5 sm:grid-cols-2 lg:grid-cols-4">
          <div>
            <div className="text-xs text-ink-3">Model</div>
            <div className="font-semibold text-ink">{calc.model.name}</div>
          </div>
          <div>
            <div className="text-xs text-ink-3">Configuration</div>
            <div className="font-semibold text-ink">
              {w.precision.toUpperCase()} · {fmtContext(w.context_length)} · {w.concurrent_sequences} concurrent
            </div>
          </div>
          <div>
            <div className="text-xs text-ink-3">GPU memory</div>
            <div className="font-semibold text-ink">
              {result.allocated_gpu_memory_gb != null ? `${Math.round(result.allocated_gpu_memory_gb)} GB` : '—'}
              <span className="ml-1 text-xs font-normal text-ink-3">(needs {fmtMem(calc.memory.required_vram_gb, 'GB', 0)})</span>
            </div>
          </div>
          <div>
            <div className="text-xs text-ink-3">Mode</div>
            <div className="font-semibold text-ink">{modeLabel(rec.candidate)}</div>
          </div>
        </div>
      ) : (
        <div className="p-5">
          <Alert tone="critical">No infrastructure in the catalog fits this workload.</Alert>
        </div>
      )}
      {result.vram_range_note && <p className="px-5 pb-3 text-xs text-ink-3">{result.vram_range_note}</p>}
      {rec && (
        <div className="flex flex-wrap items-center gap-2 border-t border-line px-5 py-3">
          <Button variant="primary" onClick={deploy} disabled={deploying || !calc.deployment_spec}>
            {deploying ? <Spinner /> : <Rocket className="h-4 w-4" />} Deploy
          </Button>
          <span className="text-xs text-ink-3">Creates a deployment spec and OpenStack flavor recommendation. Nothing is provisioned.</span>
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

export function DeployedNotice({ record }: { record: DeploymentSpecRecord }) {
  const needs = record.openstack.requires_configuration ?? []
  return (
    <div className="space-y-2 border-t border-line bg-good-soft px-5 py-4 text-sm text-good-text">
      <div className="flex items-center gap-2 font-semibold">
        <Check className="h-4 w-4" /> Deployment spec #{record.id} created for {record.ai_flavor}
      </div>
      <div className="text-ink-2">
        OpenStack flavor <code className="font-mono">{record.openstack.name}</code> ·{' '}
        <Link to="/saved" className="text-brand-text underline">
          View all specs
        </Link>{' '}
        ·{' '}
        <button type="button" className="text-brand-text underline" onClick={() => downloadText(`${record.openstack.name}.json`, JSON.stringify(record, null, 2), 'application/json')}>
          Download JSON
        </button>
      </div>
      {needs.length > 0 && (
        <div className="text-warning-text">
          ⚠ Before creating the Nova flavor, configure:
          <ul className="list-disc pl-5">
            {needs.map((n) => (
              <li key={n}>{n}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}

/** Side-by-side infrastructure options for the same workload (spec: "Compare Infrastructure"). */
export function CompareInfrastructure({ matches, rejected }: { matches: Candidate[]; rejected: Candidate[] }) {
  const cols = [...matches.slice(0, 4), ...rejected.filter((r) => r.capacity_limited || r.fits_weights_only).slice(0, 2)]
  if (!cols.length) return null
  const rec = matches[0]
  const rows: [string, (c: Candidate) => React.ReactNode][] = [
    ['GPU count', (c) => (c.kind === 'vgpu' ? `1/${Math.round(1 / c.gpu_units)} slice` : c.count)],
    ['VRAM', (c) => fmtMem(c.kind === 'multi_gpu' ? c.vram_per_unit_gb * c.count : c.vram_per_unit_gb, 'GB', 0)],
    [
      'Memory fit',
      (c) => {
        const f = memoryFit(c)
        return <span className={f.tone === 'good' ? 'text-good-text' : f.tone === 'warning' ? 'text-warning-text' : 'text-critical-text'}>{f.tone === 'critical' ? '✗' : f.tone === 'warning' ? '⚠' : '✓'} {f.label}</span>
      },
    ],
    ['Mode', (c) => (c.kind === 'vgpu' ? 'Shared' : c.kind === 'multi_gpu' ? 'Multi' : 'Full')],
    ['Headroom', (c) => (c.fits ? fmtMem(Math.max(0, c.headroom_gb) * (c.kind === 'multi_gpu' ? c.count : 1), 'GB', 0) : '—')],
    ['Cost', (c) => (c.cost ? `${fmtMoney(c.cost.hourly)}/h` : '—')],
    ['Available', (c) => (c.available_units != null ? c.available_units : c.availability.replace(/_/g, ' '))],
  ]
  return (
    <div className="space-y-2">
      <Table>
        <thead>
          <tr>
            <th />
            {cols.map((c) => (
              <th key={c.key} className={c === rec ? 'text-brand-text' : ''}>
                {candidateTitle(c)}
                {c === rec && <span className="ml-1">★</span>}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map(([label, fn]) => (
            <tr key={label}>
              <td className="font-medium text-ink-2">{label}</td>
              {cols.map((c) => (
                <td key={c.key}>{fn(c)}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </Table>
      {rec && (
        <p className="text-xs text-ink-3">
          Recommendation: <strong className="text-ink-2">{candidateTitle(rec)}</strong> —{' '}
          {rec.kind === 'vgpu' ? 'shared capacity is enough' : rec.count === 1 ? 'single GPU, sufficient capacity, less complexity' : 'smallest multi-GPU set that fits'}.
        </p>
      )}
    </div>
  )
}

function TechnicalDetails({ result }: { result: FlavorSizeResult }) {
  const [open, setOpen] = useState<'infra' | 'spec' | 'trace'>('infra')
  const rec = result.calculation.recommendation
  return (
    <div className="space-y-4">
      {rec && (
        <div className="grid gap-3 rounded-xl border border-line bg-surface p-4 text-sm sm:grid-cols-3">
          <div>
            <div className="text-xs text-ink-3">Physical GPU</div>
            <div className="font-semibold text-ink">{rec.gpu}</div>
          </div>
          <div>
            <div className="text-xs text-ink-3">Mode</div>
            <div className="font-semibold text-ink">{rec.mode === 'vgpu' ? `vGPU (${rec.vgpu_profile})` : rec.mode === 'multi_gpu' ? `PCI passthrough × ${rec.count} (TP ${rec.tensor_parallel})` : 'PCI passthrough'}</div>
          </div>
          <div>
            <div className="text-xs text-ink-3">GPU memory used</div>
            <div className="font-semibold text-ink">{Math.round(rec.utilization_percent)}%</div>
          </div>
        </div>
      )}
      <div className="flex gap-2">
        {(['infra', 'spec', 'trace'] as const).map((k) => (
          <Button key={k} size="sm" variant={open === k ? 'primary' : 'secondary'} onClick={() => setOpen(k)}>
            {{ infra: 'OpenStack flavor', spec: 'Deployment spec', trace: 'Calculation' }[k]}
          </Button>
        ))}
      </div>
      {open === 'infra' && <InfrastructureTab result={result.calculation} />}
      {open === 'spec' && (
        <pre className="overflow-x-auto rounded-lg bg-surface-2 p-4 font-mono text-xs text-ink">{JSON.stringify(result.calculation.deployment_spec, null, 2)}</pre>
      )}
      {open === 'trace' && <TraceTab result={result.calculation} />}
    </div>
  )
}
