import { useState } from 'react'
import { api } from '../../api'
import { copyText, downloadText } from '../../lib/download'
import { fmtMem, type Unit } from '../../lib/format'
import type { CalculateRequest, CalculationResult, SummaryRow } from '../../types'
import { Alert, Badge, Button, Card, Spinner, Table } from '../ui'

export function InfrastructureTab({ result }: { result: Pick<CalculationResult, 'openstack' | 'openstack_yaml'> }) {
  const [copied, setCopied] = useState(false)
  if (!result.openstack_yaml) {
    return <Alert tone="warning">No configuration fits, so no OpenStack flavor can be generated.</Alert>
  }
  const yaml = result.openstack_yaml
  return (
    <Card
      eyebrow="Infrastructure output"
      title="OpenStack flavor recommendation"
      actions={
        <div className="flex gap-2">
          <Button
            size="sm"
            onClick={async () => {
              setCopied(await copyText(yaml))
              setTimeout(() => setCopied(false), 1500)
            }}
          >
            {copied ? 'Copied ✓' : 'Copy'}
          </Button>
          <Button size="sm" onClick={() => downloadText(`${String(result.openstack?.name ?? 'flavor')}.yaml`, yaml, 'application/yaml')}>
            Download
          </Button>
        </div>
      }
    >
      <Alert tone="info">Generated recommendation only. Nothing is deployed to OpenStack.</Alert>
      <pre className="mt-3 overflow-x-auto rounded-lg bg-surface-2 p-4 font-mono text-[13px] leading-relaxed text-ink">{yaml}</pre>
    </Card>
  )
}

export function TraceTab({ result }: { result: Pick<CalculationResult, 'trace'> }) {
  return (
    <div className="space-y-4">
      <Table>
        <thead>
          <tr>
            <th>Step</th>
            <th>Formula</th>
            <th className="text-right">Result</th>
          </tr>
        </thead>
        <tbody>
          {result.trace.map((t) => (
            <tr key={t.step}>
              <td className="font-medium whitespace-nowrap">{t.step}</td>
              <td className="font-mono text-xs text-ink-2">{t.formula}</td>
              <td className="text-right whitespace-nowrap">{t.value}</td>
            </tr>
          ))}
        </tbody>
      </Table>
      <p className="text-xs text-ink-3">
        GB = 10⁹ bytes, GiB = 2³⁰ bytes. Runtime and workspace defaults are configurable estimates (Settings).
      </p>
    </div>
  )
}

export function ExportBar({ request, result }: { request: CalculateRequest; result: CalculationResult }) {
  const [busy, setBusy] = useState<string | null>(null)
  const [msg, setMsg] = useState<string | null>(null)
  const base = `gpu-sizing-${result.model.id}-${result.workload.precision}`

  const exportAs = async (format: 'json' | 'yaml' | 'csv' | 'markdown') => {
    setBusy(format)
    try {
      const text = await api.exportCalculation(request, format)
      const ext = { json: 'json', yaml: 'yaml', csv: 'csv', markdown: 'md' }[format]
      downloadText(`${base}.${ext}`, text, format === 'markdown' ? 'text/markdown' : `text/${ext}`)
    } catch (e) {
      setMsg((e as Error).message)
    } finally {
      setBusy(null)
    }
  }

  const copySummary = async () => {
    const r = result.recommendation
    const text = [
      `${result.model.name} / ${result.workload.precision.toUpperCase()}`,
      `Estimated VRAM required: ${fmtMem(result.memory.required_vram_gb, 'GB', 0)}`,
      `Recommended: ${r ? r.summary : 'no fit'}`,
      result.alternatives[0] ? `Alternative: ${result.alternatives[0].label}` : null,
      `vGPU: ${result.vgpu_verdict}`,
      r ? `AI Flavor: ${r.ai_flavor}` : null,
      `Confidence: ${result.confidence_label}`,
    ]
      .filter(Boolean)
      .join('\n')
    setMsg((await copyText(text)) ? 'Summary copied' : 'Clipboard unavailable')
    setTimeout(() => setMsg(null), 1500)
  }

  return (
    <div className="flex flex-wrap items-center gap-2">
      <Button size="sm" onClick={copySummary}>
        Copy calculation
      </Button>
      {(['json', 'yaml', 'csv'] as const).map((f) => (
        <Button key={f} size="sm" onClick={() => exportAs(f)} disabled={!!busy}>
          {busy === f ? <Spinner /> : null}Export {f.toUpperCase()}
        </Button>
      ))}
      <Button size="sm" onClick={() => exportAs('markdown')} disabled={!!busy}>
        {busy === 'markdown' ? <Spinner /> : null}Download report
      </Button>
      {msg && <span className="text-xs text-ink-3">{msg}</span>}
    </div>
  )
}

export function SummaryTable({ rows, unit, firstCol }: { rows: SummaryRow[]; unit: Unit; firstCol: 'model' | 'precision' }) {
  return (
    <Table>
      <thead>
        <tr>
          <th>{firstCol === 'model' ? 'Model' : 'Precision'}</th>
          {firstCol === 'model' && <th>Params</th>}
          {firstCol === 'model' && <th>Arch</th>}
          {firstCol === 'model' && <th>Precision</th>}
          <th className="text-right">Weights</th>
          <th className="text-right">KV cache</th>
          <th className="text-right">Required VRAM</th>
          <th>Recommended GPU</th>
          <th>AI flavor</th>
          <th className="text-right">Est. cost</th>
        </tr>
      </thead>
      <tbody className="[&_td]:whitespace-nowrap">
        {rows.map((r, i) => (
          <tr key={i}>
            <td className="min-w-[160px] font-medium !whitespace-normal">
              {firstCol === 'model' ? r.name ?? r.model_id : r.precision?.toUpperCase()}
              {r.supported === false && <Badge tone="warning" className="ml-2">Not listed</Badge>}
              {r.note && <div className="text-xs font-normal text-ink-3">{r.note}</div>}
            </td>
            {r.error ? (
              <td colSpan={firstCol === 'model' ? 9 : 6} className="text-critical-text">
                {r.error}
              </td>
            ) : (
              <>
                {firstCol === 'model' && (
                  <td>
                    {r.total_parameters_b}B{r.architecture === 'moe' ? ` / ${r.active_parameters_b}B` : ''}
                  </td>
                )}
                {firstCol === 'model' && <td>{r.architecture === 'moe' ? 'MoE' : 'Dense'}</td>}
                {firstCol === 'model' && <td>{r.precision?.toUpperCase()}</td>}
                <td className="text-right">{fmtMem(r.weights_gb, unit)}</td>
                <td className="text-right">{fmtMem(r.kv_cache_gb, unit)}</td>
                <td className="text-right font-semibold">{fmtMem(r.required_vram_gb, unit)}</td>
                <td className="min-w-[200px] !whitespace-normal">{r.recommended ?? <span className="text-critical-text">No fit</span>}</td>
                <td className="font-mono text-xs">{r.ai_flavor ?? '—'}</td>
                <td className="text-right">{r.cost ? `$${r.cost.hourly.toFixed(2)}/h` : '—'}</td>
              </>
            )}
          </tr>
        ))}
      </tbody>
    </Table>
  )
}

export function PrecisionCompareTab({ request, unit }: { request: CalculateRequest; unit: Unit }) {
  const [rows, setRows] = useState<SummaryRow[] | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const run = async () => {
    setLoading(true)
    setError(null)
    try {
      setRows(await api.comparePrecisions(request))
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setLoading(false)
    }
  }
  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3">
        <Button variant="primary" onClick={run} disabled={loading}>
          {loading && <Spinner />}Compare precision
        </Button>
        <span className="text-sm text-ink-3">Same model and workload at INT4, INT8, FP8 and BF16.</span>
      </div>
      {error && <Alert tone="critical">{error}</Alert>}
      {rows && <SummaryTable rows={rows} unit={unit} firstCol="precision" />}
    </div>
  )
}
