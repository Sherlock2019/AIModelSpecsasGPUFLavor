import { Check, ChevronDown, Link2 } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { api } from '../api'
import { ModelCombobox } from '../components/calculator/ModelCombobox'
import { ModelDetails } from '../components/calculator/ModelDetails'
import {
  AdvancedOptions,
  ConcurrencyStepper,
  ContextSelect,
  PrecisionSelect,
  PrioritySelect,
} from '../components/calculator/WorkloadPanel'
import { FlowGraphic, WhyPanel } from '../components/results/Explain'
import { MatchList } from '../components/results/MatchList'
import { MemoryChart } from '../components/results/MemoryChart'
import { RequirementPanel } from '../components/results/Requirement'
import { ExportBar, InfrastructureTab, PrecisionCompareTab, TraceTab } from '../components/results/ResultExtras'
import { WarningsSummary, WhatIfPanel } from '../components/results/WhatIf'
import { Alert, Button, Input, Segmented, Tabs, cx } from '../components/ui'
import { copyText } from '../lib/download'
import { clampContext, type Unit } from '../lib/format'
import { fromQuery, toQuery } from '../lib/share'
import { CUSTOM_MODEL_ID, toRequest, useCalc } from '../state'
import type { CalculateRequest, CalculationResult, ModelSpec, WorkloadInput } from '../types'

const DEBOUNCE_MS = 300

export function CalculatorPage() {
  const { form, setForm, setWorkload, result, resultRequest, setResult, loadError } = useCalc()
  const [updating, setUpdating] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [unit, setUnit] = useState<Unit>('GB')
  const seq = useRef(0)
  const resultsRef = useRef<HTMLDivElement>(null)

  const request = toRequest(form)
  const requestKey = JSON.stringify(request)
  const shownRequest = resultRequest ?? request

  // Apply a shared link (?m=…&p=…) once on arrival.
  useEffect(() => {
    const shared = fromQuery(window.location.search)
    if (shared) setForm((f) => ({ ...f, modelId: shared.modelId, override: null, workload: { ...f.workload, ...shared.patch } }))
  }, [setForm])

  // Live calculation: every change recalculates after a short pause; stale responses are dropped.
  useEffect(() => {
    const id = ++seq.current
    setUpdating(true)
    const timer = setTimeout(async () => {
      try {
        const r = await api.calculate(request)
        if (id !== seq.current) return
        setResult(r, request)
        setError(null)
      } catch (e) {
        if (id === seq.current) setError((e as Error).message)
      } finally {
        if (id === seq.current) setUpdating(false)
      }
    }, DEBOUNCE_MS)
    return () => clearTimeout(timer)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [requestKey])

  // Keep the address bar shareable for catalog models without user overrides.
  useEffect(() => {
    if (form.modelId === CUSTOM_MODEL_ID || form.override) return
    window.history.replaceState(null, '', `?${toQuery(form.modelId, form.workload)}`)
  }, [requestKey, form.modelId, form.override, form.workload])

  const onModelChange = (m: ModelSpec | null) => {
    if (!m) return
    const patch: Partial<WorkloadInput> = { context_length: clampContext(form.workload.context_length, m.max_context_length) }
    const supported = m.supported_serving_precisions
    if (supported.length && !supported.includes(form.workload.precision)) {
      patch.precision = supported.includes('int4') ? 'int4' : supported[0]
    }
    setWorkload(patch)
  }

  const applySuggestion = (patch: Partial<WorkloadInput>) => {
    setWorkload(patch)
    resultsRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }

  return (
    <div className="mx-auto max-w-[1400px] pb-20 lg:pb-0">
      <header className="mb-6">
        <h1 className="text-2xl font-semibold tracking-tight text-ink sm:text-3xl">LLM GPU Calculator</h1>
        <p className="mt-1 text-sm text-ink-2">Choose your model. We’ll calculate the GPU.</p>
      </header>

      <div className="grid gap-6 lg:grid-cols-[minmax(340px,400px)_minmax(0,1fr)]">
        {/* ------------------------------------------------------------ your model */}
        <div className="lg:sticky lg:top-6 lg:self-start">
          <section className="rounded-2xl border border-line bg-surface p-5 shadow-[0_1px_3px_rgba(15,23,42,0.06)]">
            <h2 className="mb-4 text-[15px] font-semibold text-ink">Your model</h2>
            {loadError && (
              <div className="mb-3">
                <Alert tone="critical">{loadError}</Alert>
              </div>
            )}
            <div className="space-y-5">
              <div>
                <div className="mb-1.5 text-[13px] font-medium text-ink-2">Model</div>
                <ModelCombobox onModelChange={onModelChange} />
                <ModelDetails />
              </div>
              <PrecisionSelect />
              <ContextSelect />
              <ConcurrencyStepper />
              <PrioritySelect />
              <AdvancedOptions />
            </div>
          </section>
          <p className="mt-2 px-1 text-xs text-ink-3">Results update automatically as you change any value.</p>
        </div>

        {/* ------------------------------------------------------------ requirement + matches */}
        <div ref={resultsRef} id="results" className="min-w-0 scroll-mt-6 space-y-5">
          {error && (
            <Alert tone="critical">
              {error}
              {result && <span className="block text-xs opacity-80">Showing the last valid result.</span>}
            </Alert>
          )}
          {!result && !error && <ResultSkeleton />}
          {result && (
            <>
              <RequirementPanel result={result} unit={unit} updating={updating} />
              <WarningsSummary result={result} />
              <div className={cx('transition-opacity', updating && 'opacity-60')}>
                <MatchList result={result} unit={unit} />
              </div>
            </>
          )}
        </div>
      </div>

      {/* ------------------------------------------------------------ explanation (full width) */}
      {result && (
        <div className="mt-6 space-y-5">
          <FlowGraphic result={result} unit={unit} />
          <WhyPanel result={result} unit={unit} />
          <WhatIfPanel suggestions={result.suggestions} unit={unit} onApply={applySuggestion} />
          <TechnicalSection result={result} request={shownRequest} unit={unit} onUnit={setUnit} />
          <Toolbar result={result} request={shownRequest} canLink={form.modelId !== CUSTOM_MODEL_ID && !form.override} />
        </div>
      )}

      {/* ------------------------------------------------------------ mobile summary bar */}
      {result && (
        <button
          type="button"
          onClick={() => resultsRef.current?.scrollIntoView({ behavior: 'smooth' })}
          className="fixed inset-x-3 bottom-3 z-30 flex items-center justify-between gap-3 rounded-xl border border-line bg-surface px-4 py-3 text-left shadow-xl lg:hidden"
        >
          <span>
            <span className="block text-xs text-ink-3">Needs ≥ {Math.round(result.memory.required_vram_gb)} GB GPU memory</span>
            <span className="block text-sm font-semibold text-ink">{result.recommendation?.candidate.label ?? 'No fit'}</span>
          </span>
          <span className="text-xs font-medium text-brand-text">View ↓</span>
        </button>
      )}
    </div>
  )
}

type TechTab = 'memory' | 'trace' | 'infra' | 'precision'

function TechnicalSection({
  result,
  request,
  unit,
  onUnit,
}: {
  result: CalculationResult
  request: CalculateRequest
  unit: Unit
  onUnit: (u: Unit) => void
}) {
  const [open, setOpen] = useState(false)
  const [tab, setTab] = useState<TechTab>('memory')
  const rec = result.recommendation
  const single = rec && rec.count === 1
  return (
    <section className="rounded-2xl border border-line bg-surface">
      <button
        type="button"
        aria-expanded={open}
        onClick={() => setOpen(!open)}
        className="flex w-full items-center justify-between gap-3 rounded-2xl px-5 py-3.5 text-left hover:bg-surface-2"
      >
        <span>
          <span className="block text-[15px] font-semibold text-ink">Technical details</span>
          <span className="block text-xs text-ink-3">Memory breakdown, formulas, OpenStack flavor, precision comparison</span>
        </span>
        <ChevronDown className={cx('h-4 w-4 text-ink-3 transition', open && 'rotate-180')} aria-hidden />
      </button>
      {open && (
        <div className="border-t border-line p-5">
          <div className="mb-4 flex flex-wrap items-end justify-between gap-3">
            <Tabs<TechTab>
              value={tab}
              onChange={setTab}
              tabs={[
                { value: 'memory', label: 'Memory breakdown' },
                { value: 'trace', label: 'Calculation' },
                { value: 'infra', label: 'Infrastructure output' },
                { value: 'precision', label: 'Compare precision' },
              ]}
            />
            <Segmented<Unit>
              value={unit}
              onChange={onUnit}
              options={[
                { value: 'GB', label: 'GB' },
                { value: 'GiB', label: 'GiB' },
              ]}
            />
          </div>
          {tab === 'memory' && (
            <div className="space-y-4">
              <MemoryChart
                memory={result.memory}
                unit={unit}
                capacityGb={single ? rec.candidate.vram_per_unit_gb : null}
                capacityLabel={single ? `${rec.vgpu_profile ?? rec.gpu}` : undefined}
              />
              <ol className="list-decimal space-y-1 pl-5 text-sm text-ink-2">
                {result.explanation.map((l, i) => (
                  <li key={i}>{l}</li>
                ))}
              </ol>
            </div>
          )}
          {tab === 'trace' && <TraceTab result={result} />}
          {tab === 'infra' && <InfrastructureTab result={result} />}
          {tab === 'precision' && <PrecisionCompareTab request={request} unit={unit} />}
        </div>
      )}
    </section>
  )
}

function Toolbar({ result, request, canLink }: { result: CalculationResult; request: CalculateRequest; canLink: boolean }) {
  const [saveName, setSaveName] = useState('')
  const [saveMsg, setSaveMsg] = useState<string | null>(null)
  const [linkCopied, setLinkCopied] = useState(false)
  const save = async () => {
    if (!saveName.trim()) return
    try {
      await api.saveCalculation(saveName.trim(), request)
      setSaveMsg('Saved')
      setSaveName('')
    } catch (e) {
      setSaveMsg((e as Error).message)
    }
    setTimeout(() => setSaveMsg(null), 2500)
  }
  return (
    <div className="flex flex-col gap-3 border-t border-line pt-4 lg:flex-row lg:items-center lg:justify-between">
      <div className="flex flex-wrap items-center gap-2">
        <Button
          size="sm"
          disabled={!canLink}
          title={canLink ? undefined : 'Links are available for catalog models without overrides'}
          onClick={async () => {
            setLinkCopied(await copyText(window.location.href))
            setTimeout(() => setLinkCopied(false), 1500)
          }}
        >
          {linkCopied ? <Check className="h-3.5 w-3.5" /> : <Link2 className="h-3.5 w-3.5" />}
          {linkCopied ? 'Link copied' : 'Copy link'}
        </Button>
        <ExportBar request={request} result={result} />
      </div>
      <div className="flex items-center gap-2">
        <Input
          className="w-56"
          aria-label="Calculation name"
          placeholder="Name this calculation"
          value={saveName}
          onChange={(e) => setSaveName(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && void save()}
        />
        <Button onClick={save} disabled={!saveName.trim()}>
          Save
        </Button>
        {saveMsg && <span className="text-xs text-ink-3">{saveMsg}</span>}
      </div>
    </div>
  )
}

function ResultSkeleton() {
  return (
    <div className="animate-pulse space-y-4 rounded-2xl border border-line bg-surface p-6" aria-label="Calculating">
      <div className="h-4 w-1/3 rounded bg-surface-2" />
      <div className="h-12 w-1/4 rounded bg-surface-2" />
      <div className="h-4 w-full rounded bg-surface-2" />
      <div className="h-4 w-2/3 rounded bg-surface-2" />
    </div>
  )
}
