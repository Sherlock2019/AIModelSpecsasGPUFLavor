import { Check, ChevronDown, Copy, ExternalLink } from 'lucide-react'
import { useState, type ReactNode } from 'react'
import { copyText } from '../../lib/download'
import { fmtMem, fmtNum, type Unit } from '../../lib/format'
import { candidateTitle, classLabel, memoryFit, modeLabel, performanceFit, type Tone } from '../../lib/plain'
import { useCalc } from '../../state'
import type { CalculationResult, Candidate, GPUSpec } from '../../types'
import { Badge, cx } from '../ui'

const TONE_TEXT: Record<Tone, string> = {
  good: 'text-good-text',
  warning: 'text-warning-text',
  critical: 'text-critical-text',
  neutral: 'text-ink-2',
}
const TONE_ICON: Record<Tone, string> = { good: '✓', warning: '⚠', critical: '✗', neutral: '?' }
const TONE_BORDER: Record<Tone, string> = {
  good: 'border-l-good',
  warning: 'border-l-warning',
  critical: 'border-l-critical',
  neutral: 'border-l-line-strong',
}

function Stat({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="min-w-0">
      <div className="text-[11px] text-ink-3">{label}</div>
      <div className="truncate text-sm font-semibold text-ink">{children}</div>
    </div>
  )
}

// Neutral styling on purpose: green/amber/red are reserved for fit status.
function VendorBadge({ vendor }: { vendor: string }) {
  return (
    <span className="rounded border border-line-strong px-1.5 py-0.5 text-[10px] font-bold tracking-wider text-ink-2">
      {vendor.toUpperCase()}
    </span>
  )
}

export function MatchList({ result, unit }: { result: CalculationResult; unit: Unit }) {
  const [showAll, setShowAll] = useState(false)
  const matches = result.matches
  if (!matches.length) {
    return (
      <div className="rounded-2xl border border-critical/40 bg-critical-soft p-5 text-sm text-critical-text">
        ✗ No GPU in the catalog fits this workload. Try INT4 precision, fewer concurrent requests, a shorter context, or allow
        more GPUs under Advanced options.
      </div>
    )
  }
  // Top 3 by score, then the best option of each missing kind (AMD, NVIDIA, multi-GPU, shared) with its true rank.
  const top = matches.slice(0, 3)
  const extras: { c: Candidate; tag: string }[] = []
  const addBest = (pred: (c: Candidate) => boolean, tag: string) => {
    if (top.some(pred)) return
    const c = matches.find((x) => pred(x) && !extras.some((e) => e.c === x))
    if (c) extras.push({ c, tag })
  }
  addBest((c) => c.gpu_vendor.toUpperCase() === 'AMD', 'Best AMD option')
  addBest((c) => c.gpu_vendor.toUpperCase() === 'NVIDIA', 'Best NVIDIA option')
  addBest((c) => c.kind === 'multi_gpu', 'Best multi-GPU option')
  addBest((c) => c.kind === 'vgpu', 'Best shared-GPU option')
  const shown: { c: Candidate; tag?: string }[] = showAll
    ? matches.map((c) => ({ c }))
    : [...top.map((c) => ({ c })), ...extras.sort((a, b) => matches.indexOf(a.c) - matches.indexOf(b.c))]
  const hidden = matches.length - shown.length
  return (
    <section aria-labelledby="matches-title">
      <div className="mb-2 flex items-baseline justify-between">
        <h2 id="matches-title" className="text-[15px] font-semibold text-ink">
          Matching GPUs
        </h2>
        <span className="text-xs text-ink-3">
          {matches.length} option{matches.length > 1 ? 's' : ''} · NVIDIA &amp; AMD · ranked for {result.workload.performance_priority === 'economy' ? 'lowest cost' : result.workload.performance_priority.replace('maximum', 'max performance')}
        </span>
      </div>
      <ol className="space-y-3">
        {shown.map(({ c, tag }) => {
          const rank = matches.indexOf(c) + 1
          return (
            <li key={c.key}>
              <GpuCard candidate={c} rank={rank} tag={tag} unit={unit} flavor={rank === 1 ? result.recommendation?.ai_flavor : undefined} />
            </li>
          )
        })}
      </ol>
      {(hidden > 0 || showAll) && (
        <button type="button" onClick={() => setShowAll(!showAll)} className="mt-3 text-sm font-medium text-brand-text hover:underline">
          {showAll ? 'Show fewer' : `Show all ${matches.length} matching options`}
        </button>
      )}
    </section>
  )
}

export function GpuCard({
  candidate: c,
  rank,
  tag,
  unit,
  flavor,
}: {
  candidate: Candidate
  rank: number
  tag?: string
  unit: Unit
  flavor?: string
}) {
  const { gpus } = useCalc()
  const [open, setOpen] = useState(false)
  const [copied, setCopied] = useState(false)
  const gpu = gpus.find((g) => g.id === c.gpu_id)
  const mem = memoryFit(c)
  const perf = performanceFit(c)
  const recommended = rank === 1
  const totalVram = c.kind === 'vgpu' ? c.vram_per_unit_gb : c.vram_per_unit_gb * c.count

  return (
    <article
      className={cx(
        'rounded-xl border border-l-4 bg-surface shadow-[0_1px_2px_rgba(15,23,42,0.04)] transition',
        TONE_BORDER[mem.tone],
        recommended ? 'border-brand/60 ring-1 ring-brand/30' : 'border-line',
      )}
    >
      <div className="p-4">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="flex min-w-0 items-center gap-2.5">
            <span className="tabular text-sm font-semibold text-ink-3">#{rank}</span>
            <VendorBadge vendor={c.gpu_vendor || 'GPU'} />
            <h3 className="truncate text-base font-semibold text-ink">{candidateTitle(c)}</h3>
          </div>
          <span className="flex items-center gap-1.5">
            {tag && <Badge>{tag}</Badge>}
            {recommended ? <Badge tone="brand">★ Recommended</Badge> : <Badge tone="good">✓ Compatible</Badge>}
          </span>
        </div>

        <div className="mt-3 grid grid-cols-2 gap-x-4 gap-y-3 sm:grid-cols-3 lg:grid-cols-6">
          <Stat label={c.kind === 'multi_gpu' ? 'Combined VRAM' : c.kind === 'vgpu' ? 'Memory slice' : 'VRAM'}>{fmtMem(totalVram, unit, 0)}</Stat>
          <Stat label="Memory fit">
            <span className={TONE_TEXT[mem.tone]}>
              {TONE_ICON[mem.tone]} {mem.label}
            </span>
          </Stat>
          <Stat label="GPU memory used">{Math.round(c.utilization_percent)}%</Stat>
          <Stat label="Headroom">{fmtMem(Math.max(0, c.headroom_gb) * (c.kind === 'multi_gpu' ? c.count : 1), unit, 0)}</Stat>
          <Stat label="AI compute">{classLabel(c.gpu_compute_class)}</Stat>
          <Stat label="GPU memory speed">{classLabel(c.gpu_bandwidth_class)}</Stat>
        </div>

        <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-1.5 text-xs">
          <span className="text-ink-2">
            <span className="text-ink-3">Mode </span>
            {modeLabel(c)}
          </span>
          <span className="text-ink-2">
            <span className="text-ink-3">GPU count </span>
            {c.kind === 'vgpu' ? `1 slice of ${Math.round(1 / c.gpu_units)}` : c.count}
          </span>
          <span className={TONE_TEXT[perf.tone]}>
            <span className="text-ink-3">Performance fit </span>
            {TONE_ICON[perf.tone]} {perf.label} <span className="text-ink-3">({perf.basis.toLowerCase()})</span>
          </span>
          {c.kind === 'multi_gpu' && <span className="text-warning-text">⚠ Needs model-parallel serving and a fast GPU link</span>}
        </div>

        {flavor && (
          <div className="mt-3 inline-flex items-center gap-2 rounded-lg bg-brand-soft py-1 pr-1 pl-3">
            <span className="text-xs text-brand-text">AI flavor</span>
            <span className="font-mono text-sm font-semibold text-brand-text">{flavor}</span>
            <button
              type="button"
              aria-label="Copy AI flavor"
              onClick={async () => {
                setCopied(await copyText(flavor))
                setTimeout(() => setCopied(false), 1200)
              }}
              className="rounded-md p-1 text-brand-text hover:bg-surface/60"
            >
              {copied ? <Check className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />}
            </button>
          </div>
        )}
      </div>

      <button
        type="button"
        aria-expanded={open}
        onClick={() => setOpen(!open)}
        className="flex w-full items-center justify-between border-t border-line px-4 py-2 text-left text-xs font-medium text-ink-3 hover:bg-surface-2 hover:text-ink"
      >
        Technical details
        <ChevronDown className={cx('h-3.5 w-3.5 transition', open && 'rotate-180')} aria-hidden />
      </button>
      {open && <TechnicalDetails candidate={c} gpu={gpu} unit={unit} />}
    </article>
  )
}

function Spec({ k, v }: { k: string; v: ReactNode }) {
  return (
    <div className="flex justify-between gap-3 py-1">
      <dt className="text-ink-3">{k}</dt>
      <dd className="tabular text-right text-ink">{v ?? '—'}</dd>
    </div>
  )
}

const yesNo = (v: boolean | null | undefined) => (v == null ? 'Unknown' : v ? 'Yes' : 'No')
const perf = (v: number | null | undefined, unit = 'TFLOPS') => (v == null ? '—' : `${fmtNum(v, v < 100 ? 1 : 0)} ${unit}`)

function TechnicalDetails({ candidate: c, gpu, unit }: { candidate: Candidate; gpu?: GPUSpec; unit: Unit }) {
  const m = c.memory_per_unit
  return (
    <div className="fade-up grid gap-6 border-t border-line bg-surface-2 p-4 text-xs md:grid-cols-2">
      <div>
        <div className="mb-1 font-semibold text-ink">GPU</div>
        {gpu ? (
          <dl className="divide-y divide-line">
            <Spec k="VRAM" v={`${gpu.vram_gb} GB`} />
            <Spec k="Memory bandwidth" v={gpu.memory_bandwidth_gbps ? `${fmtNum(gpu.memory_bandwidth_gbps)} GB/s` : '—'} />
            <Spec k="FP16 / BF16" v={`${perf(gpu.fp16_tflops)} / ${perf(gpu.bf16_tflops)}`} />
            <Spec k="FP8" v={perf(gpu.fp8_tflops)} />
            <Spec k="INT8 / INT4" v={`${perf(gpu.int8_tops, 'TOPS')} / ${perf(gpu.int4_tops, 'TOPS')}`} />
            <Spec k="GPU link" v={gpu.interconnect_type ?? '—'} />
            <Spec k="vGPU / partitioning" v={`${yesNo(gpu.supports_vgpu)} / ${yesNo(gpu.supports_mig)}`} />
            <Spec k="Power" v={gpu.power_watts ? `${gpu.power_watts} W` : '—'} />
            <Spec
              k="Source"
              v={
                gpu.metadata_source ? (
                  <a href={gpu.metadata_source} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1 text-brand-text hover:underline">
                    Datasheet <ExternalLink className="h-3 w-3" />
                  </a>
                ) : (
                  '—'
                )
              }
            />
          </dl>
        ) : (
          <p className="text-ink-3">GPU specifications unavailable.</p>
        )}
        <p className="mt-2 text-ink-3">Dense (non-sparse) datasheet peaks.</p>
      </div>
      <div>
        <div className="mb-1 font-semibold text-ink">Calculation {c.count > 1 ? `(per GPU, tensor parallel = ${c.tensor_parallel})` : ''}</div>
        <dl className="divide-y divide-line">
          <Spec k="Model weights" v={fmtMem(m.model_weights_gb, unit)} />
          <Spec k="KV cache" v={fmtMem(m.kv_cache_gb, unit)} />
          <Spec k="Runtime + workspace" v={fmtMem(m.runtime_overhead_gb + m.workspace_gb, unit)} />
          {m.communication_gb > 0 && <Spec k="Communication buffers" v={fmtMem(m.communication_gb, unit)} />}
          <Spec k={`Safety margin (${m.safety_margin_percent}%)`} v={fmtMem(m.headroom_gb, unit)} />
          <Spec k="Required / usable" v={`${fmtMem(c.required_per_unit_gb, unit)} / ${fmtMem(c.vram_per_unit_gb, unit)}`} />
          <Spec k="Bandwidth vs target" v={c.bandwidth_ratio != null ? `${fmtNum(c.bandwidth_ratio * 100)}%` : '—'} />
          <Spec k="Compute vs target" v={c.compute_ratio != null ? `${fmtNum(c.compute_ratio * 100)}%` : '—'} />
          <Spec k="Decode ceiling / request" v={c.est_decode_tokens_per_second_per_user ? `~${fmtNum(c.est_decode_tokens_per_second_per_user)} tok/s` : '—'} />
          <Spec k="Time to first token (idle)" v={c.est_ttft_ms ? `~${fmtNum(c.est_ttft_ms)} ms` : '—'} />
          <Spec k="Availability" v={c.availability.replace(/_/g, ' ')} />
          <Spec k="Cost" v={c.cost ? `$${c.cost.hourly.toFixed(2)}/h` : 'Not priced'} />
        </dl>
        {c.notes.length > 0 && (
          <ul className="mt-2 space-y-1 text-ink-3">
            {c.notes.map((n) => (
              <li key={n}>• {n}</li>
            ))}
          </ul>
        )}
      </div>
    </div>
  )
}
