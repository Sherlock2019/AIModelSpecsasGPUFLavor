import { Cable, Cpu, Gauge, HardDrive, Layers, Share2 } from 'lucide-react'
import type { ReactNode } from 'react'
import { fmtMem, type Unit } from '../../lib/format'
import { classLabel } from '../../lib/plain'
import type { CalculationResult } from '../../types'
import { Badge, InfoTip, Spinner, cx } from '../ui'

const CLASS_TONE: Record<string, 'neutral' | 'brand' | 'warning' | 'critical'> = {
  low: 'neutral',
  medium: 'brand',
  high: 'brand',
  very_high: 'warning',
  extreme: 'critical',
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

/** "Your GPU requirement": what any GPU must provide, before naming hardware. */
export function RequirementPanel({ result, unit, updating }: { result: CalculationResult; unit: Unit; updating?: boolean }) {
  const rec = result.recommendation
  const dedicated = result.matches.filter((c) => c.kind !== 'vgpu')
  const counts = dedicated.slice(0, 5).map((c) => c.count)
  const minCount = counts.length ? Math.min(...counts) : null
  const maxCount = counts.length ? Math.max(...counts) : null
  const countLabel =
    rec?.mode === 'vgpu'
      ? 'Part of 1 GPU'
      : minCount == null
        ? '—'
        : minCount === 1
          ? '1 preferred'
          : minCount === maxCount
            ? String(minCount)
            : `${minCount}–${maxCount}`
  const sharing =
    rec?.mode === 'vgpu'
      ? 'Shared vGPU is enough'
      : result.matches.some((c) => c.kind === 'vgpu')
        ? 'Dedicated recommended (vGPU possible)'
        : 'Dedicated recommended'
  const needsLink = rec ? rec.count > 1 : (minCount ?? 1) > 1

  return (
    <section
      aria-live="polite"
      aria-busy={updating}
      className="rounded-2xl border border-line bg-surface shadow-[0_1px_3px_rgba(15,23,42,0.06)]"
    >
      <header className="flex items-center justify-between gap-2 border-b border-line px-5 py-3">
        <h2 className="text-[15px] font-semibold text-ink">Your GPU requirement</h2>
        <span className="flex items-center gap-2 text-xs text-ink-3">
          {updating && (
            <>
              <Spinner className="h-3 w-3" /> Updating
            </>
          )}
          <Badge tone={result.calculation_mode === 'benchmark' ? 'good' : 'neutral'}>{result.confidence_label === 'ESTIMATED' ? '≈ Estimated' : '◉ Benchmark-based'}</Badge>
        </span>
      </header>
      <div className={cx('grid gap-x-8 px-5 pb-2 transition-opacity md:grid-cols-[minmax(0,0.8fr)_minmax(0,1fr)]', updating && 'opacity-60')}>
        <div className="py-4">
          <div className="text-xs font-medium text-ink-3">GPU memory (VRAM)</div>
          <div className="mt-0.5 text-5xl font-semibold tracking-tight text-ink">
            <span className="mr-1 text-3xl font-normal text-ink-3">≥</span>
            {fmtMem(result.memory.required_vram_gb, unit, 0)}
          </div>
          <p className="mt-2 text-xs text-ink-3">
            {fmtMem(result.memory.model_weights_gb, unit, 0)} model + {fmtMem(result.memory.kv_cache_gb, unit, 0)} context &amp; users +{' '}
            {fmtMem(result.memory.runtime_overhead_gb + result.memory.workspace_gb, unit, 0)} runtime + {result.memory.safety_margin_percent}% headroom
          </p>
        </div>
        <div className="divide-y divide-line">
          <Row
            icon={<Cpu className="h-4 w-4" />}
            label="AI compute"
            tip="Estimated compute class from active parameters × requested speed × concurrency. Not a benchmark."
          >
            <Badge tone={CLASS_TONE[result.compute.classification]}>{classLabel(result.compute.classification)}</Badge>
          </Row>
          <Row
            icon={<Gauge className="h-4 w-4" />}
            label="GPU memory speed"
            tip="How fast the GPU must read its memory to generate tokens at the target speed. LLM inference is usually limited by this."
          >
            <Badge tone={CLASS_TONE[result.compute.bandwidth_class]}>{classLabel(result.compute.bandwidth_class)}</Badge>
          </Row>
          <Row icon={<Layers className="h-4 w-4" />} label="GPU count">
            {countLabel}
          </Row>
          <Row icon={<Share2 className="h-4 w-4" />} label="GPU sharing">
            <span className={rec?.mode === 'vgpu' ? 'text-good-text' : ''}>{sharing}</span>
          </Row>
          <Row
            icon={<Cable className="h-4 w-4" />}
            label="Multi-GPU link"
            tip="When a model is split across GPUs, a fast GPU-to-GPU link (NVLink or AMD Infinity Fabric) keeps it responsive."
          >
            {needsLink ? 'NVLink / Infinity Fabric recommended' : 'Not required'}
          </Row>
        </div>
      </div>
      <div className="grid gap-px border-t border-line bg-line sm:grid-cols-2">
        <div className="flex items-start gap-2.5 bg-surface px-5 py-3">
          <HardDrive className="mt-0.5 h-4 w-4 text-ink-3" aria-hidden />
          <div>
            <div className="text-xs font-semibold text-ink">Memory fit — “Does it fit in GPU memory?”</div>
            <div className="text-xs text-ink-3">Calculated from the model architecture. Green, amber or red on each GPU below.</div>
          </div>
        </div>
        <div className="flex items-start gap-2.5 bg-surface px-5 py-3">
          <Gauge className="mt-0.5 h-4 w-4 text-ink-3" aria-hidden />
          <div>
            <div className="text-xs font-semibold text-ink">Performance fit — “Will it run fast enough?”</div>
            <div className="text-xs text-ink-3">
              {result.calculation_mode === 'benchmark'
                ? 'Based on measured benchmark data.'
                : 'Estimated. Performance requires benchmark validation.'}
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
