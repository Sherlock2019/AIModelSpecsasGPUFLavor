import { Cable, Cpu, Gauge, HardDrive, Layers, Share2 } from 'lucide-react'
import type { ReactNode } from 'react'
import { fmtMem, type Unit } from '../../lib/format'
import { classLabel } from '../../lib/plain'
import type { CalculationResult, Candidate, MLSizingResult } from '../../types'
import { Badge, InfoTip, Spinner, cx } from '../ui'

const CLASS_TONE: Record<string, 'neutral' | 'brand' | 'warning' | 'critical'> = {
  low: 'neutral',
  medium: 'brand',
  high: 'brand',
  very_high: 'warning',
  extreme: 'critical',
}

/** What any GPU must provide - workload-agnostic (LLM or custom ML). */
export interface RequirementView {
  requiredGb: number
  breakdown: string
  computeClass: string
  computeTip: string
  bandwidthClass: string
  countLabel: string
  sharing: string
  sharingIsShared: boolean
  link: string
  benchmark: boolean
  extra?: ReactNode
}

function Row({ icon, label, tip, children }: { icon: ReactNode; label: string; tip?: string; children: ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-3 py-2.5">
      <span className="flex items-center gap-2.5 text-sm text-ink-2">
        <span className="text-ink-3" aria-hidden>
          {icon}
        </span>
        {label}
        {tip && <InfoTip>{tip}</InfoTip>}
      </span>
      <span className="text-right text-sm font-semibold text-ink">{children}</span>
    </div>
  )
}

function countFromMatches(rec: Candidate | undefined, matches: Candidate[]): string {
  if (rec?.kind === 'vgpu') return 'Part of 1 GPU'
  const counts = matches.filter((c) => c.kind !== 'vgpu').slice(0, 5).map((c) => c.count)
  if (!counts.length) return '—'
  const min = Math.min(...counts)
  const max = Math.max(...counts)
  if (min === 1) return '1 preferred'
  return min === max ? String(min) : `${min}–${max}`
}

export function llmRequirement(result: CalculationResult, unit: Unit): RequirementView {
  const rec = result.recommendation
  const m = result.memory
  return {
    requiredGb: m.required_vram_gb,
    breakdown: `${fmtMem(m.model_weights_gb, unit, 0)} model + ${fmtMem(m.kv_cache_gb, unit, 0)} context & users + ${fmtMem(
      m.runtime_overhead_gb + m.workspace_gb,
      unit,
      0,
    )} runtime + ${m.safety_margin_percent}% headroom`,
    computeClass: result.compute.classification,
    computeTip: 'Estimated compute class from active parameters × requested speed × concurrency. Not a benchmark.',
    bandwidthClass: result.compute.bandwidth_class,
    countLabel: countFromMatches(rec?.candidate, result.matches),
    sharing:
      rec?.mode === 'vgpu'
        ? 'Shared vGPU is enough'
        : result.matches.some((c) => c.kind === 'vgpu')
          ? 'Dedicated recommended (vGPU possible)'
          : 'Dedicated recommended',
    sharingIsShared: rec?.mode === 'vgpu',
    link: rec && rec.count > 1 ? 'NVLink / Infinity Fabric recommended' : 'Not required',
    benchmark: result.calculation_mode === 'benchmark',
  }
}

export function mlRequirement(result: MLSizingResult, unit: Unit): RequirementView {
  const m = result.memory
  const training = result.model.task !== 'inference'
  const parts = [
    `${fmtMem(m.model_weights_gb, unit, 0)} weights`,
    training ? `${fmtMem(m.gradients_gb + m.optimizer_gb, unit, 0)} gradients & optimizer` : null,
    `${fmtMem(m.activations_gb, unit, 0)} activations${result.activation_method === 'estimated' ? ' (est.)' : ''}`,
    `${fmtMem(m.runtime_overhead_gb + m.workspace_gb + m.communication_gb, unit, 0)} runtime`,
    `${m.safety_margin_percent}% headroom`,
  ].filter(Boolean)
  const r = result.requirement
  return {
    requiredGb: m.required_vram_gb,
    breakdown: parts.join(' + '),
    computeClass: result.compute.classification,
    computeTip: 'Estimated from parameters × input size × samples per second (and ×3 for training). Not a benchmark.',
    bandwidthClass: result.compute.bandwidth_class,
    countLabel: r.gpu_count_label,
    sharing: r.mode === 'Shared vGPU' ? 'Shared vGPU is enough' : r.mode === '—' ? '—' : `${r.mode} recommended`,
    sharingIsShared: r.mode === 'Shared vGPU',
    link: r.interconnect === 'Required' ? 'NVLink / Infinity Fabric required' : r.interconnect === 'Recommended' ? 'Fast GPU link recommended' : 'Not required',
    benchmark: false,
  }
}

export function RequirementPanel({ view, unit, updating, title = 'Your GPU requirement' }: { view: RequirementView; unit: Unit; updating?: boolean; title?: string }) {
  return (
    <section aria-live="polite" aria-busy={updating} className="rounded-2xl border border-line bg-surface shadow-[0_1px_3px_rgba(15,23,42,0.06)]">
      <header className="flex items-center justify-between gap-2 border-b border-line px-5 py-3">
        <h2 className="text-[15px] font-semibold text-ink">{title}</h2>
        <span className="flex items-center gap-2 text-xs text-ink-3">
          {updating && (
            <>
              <Spinner className="h-3 w-3" /> Updating
            </>
          )}
          <Badge tone={view.benchmark ? 'good' : 'neutral'}>{view.benchmark ? '◉ Benchmark-based' : '≈ Estimated'}</Badge>
        </span>
      </header>
      <div className={cx('grid gap-x-8 px-5 pb-2 transition-opacity md:grid-cols-[minmax(0,0.8fr)_minmax(0,1fr)]', updating && 'opacity-60')}>
        <div className="py-4">
          <div className="text-xs font-medium text-ink-3">GPU memory (VRAM)</div>
          <div className="mt-0.5 text-5xl font-semibold tracking-tight text-ink">
            <span className="mr-1 text-3xl font-normal text-ink-3">≥</span>
            {fmtMem(view.requiredGb, unit, 0)}
          </div>
          <p className="mt-2 text-xs text-ink-3">{view.breakdown}</p>
          {view.extra}
        </div>
        <div className="divide-y divide-line">
          <Row icon={<Cpu className="h-4 w-4" />} label="AI compute" tip={view.computeTip}>
            <Badge tone={CLASS_TONE[view.computeClass]}>{classLabel(view.computeClass)}</Badge>
          </Row>
          <Row
            icon={<Gauge className="h-4 w-4" />}
            label="GPU memory speed"
            tip="How fast the GPU must read and write its memory to keep up with the workload."
          >
            <Badge tone={CLASS_TONE[view.bandwidthClass]}>{classLabel(view.bandwidthClass)}</Badge>
          </Row>
          <Row icon={<Layers className="h-4 w-4" />} label="GPU count">
            {view.countLabel}
          </Row>
          <Row icon={<Share2 className="h-4 w-4" />} label="GPU sharing">
            <span className={view.sharingIsShared ? 'text-good-text' : ''}>{view.sharing}</span>
          </Row>
          <Row
            icon={<Cable className="h-4 w-4" />}
            label="Multi-GPU link"
            tip="When work is split across GPUs, a fast GPU-to-GPU link (NVLink or AMD Infinity Fabric) keeps it efficient."
          >
            {view.link}
          </Row>
        </div>
      </div>
      <div className="grid gap-px border-t border-line bg-line sm:grid-cols-2">
        <div className="flex items-start gap-2.5 bg-surface px-5 py-3">
          <HardDrive className="mt-0.5 h-4 w-4 text-ink-3" aria-hidden />
          <div>
            <div className="text-xs font-semibold text-ink">Memory fit — “Does it fit in GPU memory?”</div>
            <div className="text-xs text-ink-3">Calculated. Green, amber or red on each GPU below.</div>
          </div>
        </div>
        <div className="flex items-start gap-2.5 bg-surface px-5 py-3">
          <Gauge className="mt-0.5 h-4 w-4 text-ink-3" aria-hidden />
          <div>
            <div className="text-xs font-semibold text-ink">Performance fit — “Will it run fast enough?”</div>
            <div className="text-xs text-ink-3">
              {view.benchmark ? '✓ Benchmark verified.' : 'Estimated. Performance requires benchmark validation.'}
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
