import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { api } from '../api'
import { PrecisionCompareTab, SummaryTable } from '../components/results/ResultExtras'
import { Alert, Badge, Button, Input, PageHeader, Segmented, Spinner, Table, Tabs } from '../components/ui'
import { fmtContext, fmtMem, fmtNum, fmtPct, titleCase, type Unit } from '../lib/format'
import { toRequest, useCalc, useSelectedModel } from '../state'
import type { GPUSpec, GpuCompareRow, SummaryRow } from '../types'

type Tab = 'models' | 'gpus' | 'precision'

const DEFAULT_MODELS = ['llama-3.1-8b-instruct', 'qwen2.5-14b-instruct', 'qwen2.5-32b-instruct', 'llama-3.1-70b-instruct']
const DEFAULT_GPUS = ['nvidia-l40s', 'nvidia-h100-80gb', 'nvidia-h200-141gb', 'nvidia-b200']

export function ComparePage() {
  const { form } = useCalc()
  const model = useSelectedModel()
  const [tab, setTab] = useState<Tab>('models')
  const [unit, setUnit] = useState<Unit>('GB')
  const w = form.workload

  return (
    <div>
      <PageHeader
        title="Compare"
        description="Compare models, GPUs or precisions using the workload currently set on the Calculator."
        actions={
          <Segmented<Unit>
            value={unit}
            onChange={setUnit}
            options={[
              { value: 'GB', label: 'GB' },
              { value: 'GiB', label: 'GiB' },
            ]}
          />
        }
      />
      <div className="mb-4 flex flex-wrap items-center gap-2 text-sm text-ink-2">
        <span>Workload:</span>
        <Badge>{w.precision}</Badge>
        <Badge>{fmtContext(w.context_length)} context</Badge>
        <Badge>{w.concurrent_sequences} concurrent</Badge>
        <Badge>{titleCase(w.performance_priority)}</Badge>
        <Badge>{w.framework}</Badge>
        <Link to="/" className="text-brand-text hover:underline">
          Edit on Calculator
        </Link>
      </div>
      <div className="mb-5">
        <Tabs<Tab>
          value={tab}
          onChange={setTab}
          tabs={[
            { value: 'models', label: 'Compare models' },
            { value: 'gpus', label: 'Compare GPUs' },
            { value: 'precision', label: 'Compare precision' },
          ]}
        />
      </div>
      {tab === 'models' && <CompareModels unit={unit} />}
      {tab === 'gpus' && <CompareGpus unit={unit} />}
      {tab === 'precision' && (
        <div className="space-y-3">
          <p className="text-sm text-ink-2">
            Model: <strong className="text-ink">{model?.name ?? 'none selected'}</strong>
          </p>
          <PrecisionCompareTab request={toRequest(form)} unit={unit} />
        </div>
      )}
    </div>
  )
}

function CompareModels({ unit }: { unit: Unit }) {
  const { form, models } = useCalc()
  const [selected, setSelected] = useState<string[]>(DEFAULT_MODELS)
  const [query, setQuery] = useState('')
  const [rows, setRows] = useState<SummaryRow[] | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const toggle = (id: string) => setSelected(selected.includes(id) ? selected.filter((x) => x !== id) : [...selected, id].slice(0, 12))
  const run = async () => {
    setLoading(true)
    setError(null)
    try {
      setRows(await api.compareModels(selected, form.workload))
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setLoading(false)
    }
  }
  const list = models.filter((m) => !query || `${m.name} ${m.family}`.toLowerCase().includes(query.toLowerCase()))
  return (
    <div className="grid gap-5 lg:grid-cols-[280px_minmax(0,1fr)]">
      <div className="rounded-xl border border-line bg-surface p-3">
        <Input placeholder="Filter models" value={query} onChange={(e) => setQuery(e.target.value)} />
        <ul className="mt-2 max-h-[420px] space-y-0.5 overflow-y-auto text-sm">
          {list.map((m) => (
            <li key={m.id}>
              <label className="flex cursor-pointer items-center gap-2 rounded-md px-2 py-1 hover:bg-surface-2">
                <input type="checkbox" checked={selected.includes(m.id)} onChange={() => toggle(m.id)} className="accent-[var(--brand)]" />
                <span className="text-ink">{m.name}</span>
              </label>
            </li>
          ))}
        </ul>
        <Button variant="primary" className="mt-3 w-full" onClick={run} disabled={!selected.length || loading}>
          {loading && <Spinner />}Compare {selected.length} models
        </Button>
      </div>
      <div className="min-w-0 space-y-3">
        {error && <Alert tone="critical">{error}</Alert>}
        {rows ? <SummaryTable rows={rows} unit={unit} firstCol="model" /> : <p className="text-sm text-ink-3">Pick up to 12 models and run the comparison.</p>}
      </div>
    </div>
  )
}

function CompareGpus({ unit }: { unit: Unit }) {
  const { form } = useCalc()
  const [gpus, setGpus] = useState<GPUSpec[]>([])
  const [selected, setSelected] = useState<string[]>(DEFAULT_GPUS)
  const [rows, setRows] = useState<GpuCompareRow[] | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  useEffect(() => {
    api.gpus().then(setGpus).catch((e: Error) => setError(e.message))
  }, [])
  const run = async () => {
    setLoading(true)
    setError(null)
    try {
      setRows(await api.compareGpus(toRequest(form), selected))
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setLoading(false)
    }
  }
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap gap-2">
        {gpus
          .filter((g) => g.enabled)
          .map((g) => {
            const on = selected.includes(g.id)
            return (
              <button
                key={g.id}
                type="button"
                aria-pressed={on}
                onClick={() => setSelected(on ? selected.filter((x) => x !== g.id) : [...selected, g.id])}
                className={`h-8 rounded-lg border px-3 text-sm transition ${on ? 'border-brand bg-brand-soft text-brand-text' : 'border-line-strong text-ink-2 hover:bg-surface-2'}`}
              >
                {g.model}
              </button>
            )
          })}
      </div>
      <Button variant="primary" onClick={run} disabled={!selected.length || loading}>
        {loading && <Spinner />}Compare GPUs
      </Button>
      {error && <Alert tone="critical">{error}</Alert>}
      {rows && (
        <Table>
          <thead>
            <tr>
              <th>GPU</th>
              <th className="text-right">VRAM</th>
              <th>Fits?</th>
              <th className="text-right">GPUs needed</th>
              <th className="text-right">Per GPU</th>
              <th className="text-right">Utilization</th>
              <th className="text-right">Bandwidth</th>
              <th className="text-right">Decode ceiling / req</th>
              <th className="text-right">Est. cost</th>
              <th>Basis</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.gpu_id}>
                <td className="font-medium">{r.gpu}</td>
                <td className="text-right">{r.vram_gb} GB</td>
                <td>{r.fits ? <Badge tone="good">✓ Fits</Badge> : <Badge tone="critical">✕ No</Badge>}</td>
                <td className="text-right">{r.gpu_count ?? '—'}</td>
                <td className="text-right">{fmtMem(r.required_per_gpu_gb, unit)}</td>
                <td className="text-right">{fmtPct(r.utilization_percent)}</td>
                <td className="text-right">{r.memory_bandwidth_gbps ? `${fmtNum(r.memory_bandwidth_gbps)} GB/s` : '—'}</td>
                <td className="text-right">{r.est_decode_tokens_per_second_per_user ? `~${fmtNum(r.est_decode_tokens_per_second_per_user)} tok/s` : '—'}</td>
                <td className="text-right">{r.cost ? `$${r.cost.hourly.toFixed(2)}/h` : '—'}</td>
                <td>
                  {r.benchmark ? <Badge tone="good">Benchmark</Badge> : <Badge>Estimate</Badge>}
                  {!r.fits && r.reason && <div className="mt-1 max-w-xs text-xs text-ink-3">{r.reason}</div>}
                </td>
              </tr>
            ))}
          </tbody>
        </Table>
      )}
    </div>
  )
}
