import { ChevronRight } from 'lucide-react'
import { Fragment } from 'react'
import { fmtContext, fmtMem, fmtNum, fmtParams, type Unit } from '../../lib/format'
import { candidateTitle, classLabel, classRank, whyNot } from '../../lib/plain'
import type { CalculationResult, Candidate, MLSizingResult, MemoryBreakdown, Priority, Recommendation } from '../../types'
import { PRIORITY_OPTIONS } from '../calculator/WorkloadPanel'
import { cx } from '../ui'

export interface FlowStep {
  top: string
  bottom: string
  strong?: boolean
}

export function FlowGraphic({ steps, title = 'From model to GPU' }: { steps: FlowStep[]; title?: string }) {
  return (
    <section aria-label="How the requirement is derived" className="rounded-2xl border border-line bg-surface p-5">
      <h2 className="mb-3 text-[15px] font-semibold text-ink">{title}</h2>
      <ol className="flex flex-wrap items-stretch gap-y-3">
        {steps.map((s, i) => (
          <Fragment key={i}>
            <li
              className={cx(
                'flex min-w-[112px] flex-col justify-center rounded-lg border px-3 py-2',
                s.strong ? 'border-brand bg-brand-soft' : 'border-line bg-surface-2',
              )}
            >
              <span className={cx('text-sm font-semibold', s.strong ? 'text-brand-text' : 'text-ink')}>{s.top}</span>
              <span className="text-[11px] leading-tight text-ink-3">{s.bottom}</span>
            </li>
            {i < steps.length - 1 && (
              <li aria-hidden className="flex items-center px-1 text-ink-3">
                <ChevronRight className="h-4 w-4" />
              </li>
            )}
          </Fragment>
        ))}
      </ol>
    </section>
  )
}

function gpuStep(rec: Recommendation | null | undefined, matches: Candidate[]): FlowStep {
  return {
    top: rec ? candidateTitle(rec.candidate) : 'No fit',
    bottom: matches[1] ? `or ${candidateTitle(matches[1])}` : 'best match',
    strong: true,
  }
}

/** Model → parameters → precision → weights → + context/users/runtime → VRAM → GPU. */
export function llmFlow(result: CalculationResult, unit: Unit, flavor?: string | null): FlowStep[] {
  const m = result.model
  const w = result.workload
  const mem = result.memory
  const steps: FlowStep[] = [
    { top: m.name.replace(/ Instruct$/, ''), bottom: m.architecture === 'moe' ? `MoE · ${fmtParams(m.active_parameters_b)} active` : 'Dense model' },
    { top: `${fmtParams(m.total_parameters_b)} parameters`, bottom: 'all stored in memory' },
    { top: w.precision.toUpperCase(), bottom: `${result.weights.bytes_per_parameter} byte / parameter` },
    { top: fmtMem(mem.model_weights_gb, unit, 0), bottom: 'model weights' },
    {
      top: `+ ${fmtMem(mem.kv_cache_gb + mem.runtime_overhead_gb + mem.workspace_gb + mem.headroom_gb, unit, 0)}`,
      bottom: `${fmtContext(w.context_length)} context · ${w.concurrent_sequences} requests · runtime · headroom`,
    },
    { top: `≈ ${fmtMem(mem.required_vram_gb, unit, 0)}`, bottom: 'GPU memory required', strong: true },
    gpuStep(result.recommendation, result.matches),
  ]
  if (flavor) steps.push({ top: flavor, bottom: 'AI flavor', strong: true })
  return steps
}

export function mlFlow(result: MLSizingResult, unit: Unit): FlowStep[] {
  const m = result.memory
  const training = result.model.task !== 'inference'
  return [
    { top: result.model.name, bottom: `${result.model.category_label} · ${result.model.task.replace('_', '-')}` },
    { top: `${fmtParams(result.model.parameters_b)} parameters`, bottom: result.model.precision.toUpperCase() },
    { top: fmtMem(m.model_weights_gb, unit, 0), bottom: 'weights' },
    ...(training ? [{ top: `+ ${fmtMem(m.gradients_gb + m.optimizer_gb, unit, 0)}`, bottom: 'gradients & optimizer' }] : []),
    {
      top: `+ ${fmtMem(m.activations_gb, unit, 0)}`,
      bottom: `activations (${result.activation_method === 'estimated' ? 'estimated' : 'yours'}) · ${result.workload.input_description}`,
    },
    { top: `+ ${fmtMem(m.runtime_overhead_gb + m.workspace_gb + m.communication_gb + m.headroom_gb, unit, 0)}`, bottom: 'runtime · headroom' },
    { top: `≈ ${fmtMem(m.required_vram_gb, unit, 0)}`, bottom: 'GPU memory required', strong: true },
    gpuStep(result.recommendation, result.matches),
    ...(result.recommendation ? [{ top: result.recommendation.ai_flavor, bottom: 'AI flavor', strong: true }] : []),
  ]
}

export function llmBreakdown(mem: MemoryBreakdown): [string, number][] {
  return [
    ['Model weights', mem.model_weights_gb],
    ['KV cache (context × requests)', mem.kv_cache_gb],
    ['Runtime + workspace', mem.runtime_overhead_gb + mem.workspace_gb + mem.communication_gb],
    [`Safety headroom (${mem.safety_margin_percent}%)`, mem.headroom_gb],
  ]
}

export function mlBreakdown(mem: MemoryBreakdown, activationEstimated: boolean): [string, number][] {
  const rows: [string, number][] = [['Weights', mem.model_weights_gb]]
  if (mem.gradients_gb > 0) rows.push(['Gradients', mem.gradients_gb])
  if (mem.optimizer_gb > 0) rows.push(['Optimizer states', mem.optimizer_gb])
  rows.push([`Activations & I/O${activationEstimated ? ' (estimated)' : ''}`, mem.activations_gb])
  rows.push(['Runtime + workspace', mem.runtime_overhead_gb + mem.workspace_gb + mem.communication_gb])
  rows.push([`Safety headroom (${mem.safety_margin_percent}%)`, mem.headroom_gb])
  return rows
}

/** "Why this recommendation?" plus "Why not …?" for runners-up and rejected GPUs. */
export function WhyPanel({
  rec,
  matches,
  rejected,
  breakdown,
  requiredGb,
  workloadBandwidthClass,
  priority,
  unit,
  noFitLines,
  speedNote,
  title,
}: {
  rec: Recommendation | null | undefined
  matches: Candidate[]
  rejected: Candidate[]
  breakdown: [string, number][]
  requiredGb: number
  workloadBandwidthClass: string
  priority: Priority
  unit: Unit
  noFitLines: string[]
  speedNote?: string | null
  title?: string
}) {
  if (!rec) {
    return (
      <section className="rounded-2xl border border-line bg-surface p-5">
        <h2 className="mb-2 text-[15px] font-semibold text-ink">Why no recommendation?</h2>
        <ul className="space-y-1.5 text-sm text-ink-2">
          {noFitLines.map((l, i) => (
            <li key={i}>{l}</li>
          ))}
        </ul>
      </section>
    )
  }
  const c = rec.candidate
  const name = candidateTitle(c)
  const priorityLabel = PRIORITY_OPTIONS.find((p) => p.id === priority)?.label ?? ''
  const runnersUp = matches.slice(1, 4)
  const shownRejected = rejected
    .filter((r) => r.kind !== 'vgpu' || priority !== 'maximum')
    .filter((r) => r.capacity_limited || r.vram_per_unit_gb * r.count >= requiredGb * 0.35)
    .slice(0, 3)
  const checks: { ok: boolean; text: string }[] = [
    { ok: true, text: 'Fits in GPU memory' },
    { ok: c.headroom_gb > 0, text: `~${fmtMem(Math.max(0, c.headroom_gb) * (c.kind === 'multi_gpu' ? c.count : 1), unit, 0)} memory left over` },
    c.kind === 'vgpu'
      ? { ok: true, text: 'A shared slice is enough; a dedicated GPU would waste capacity' }
      : c.count === 1
        ? { ok: true, text: 'Single-GPU deployment' }
        : { ok: false, text: `Split across ${c.count} GPUs (needs a fast GPU link)` },
    {
      ok: classRank(c.gpu_bandwidth_class) >= classRank(workloadBandwidthClass),
      text: `${classLabel(c.gpu_bandwidth_class)} GPU memory speed (workload needs ${classLabel(workloadBandwidthClass).toLowerCase()})`,
    },
    {
      ok: (c.compute_ratio ?? 1) >= 1,
      text: `${classLabel(c.gpu_compute_class)} AI compute — performance ${c.confidence === 'benchmark' ? 'measured' : 'estimated, needs benchmark validation'}`,
    },
  ]

  return (
    <section aria-labelledby="why-title" className="rounded-2xl border border-line bg-surface">
      <h2 id="why-title" className="border-b border-line px-5 py-3 text-[15px] font-semibold text-ink">
        {title ?? 'Why this recommendation?'}
      </h2>
      <div className="grid gap-6 p-5 lg:grid-cols-2">
        <div>
          <h3 className="mb-2 text-sm font-semibold text-ink">Why {name}?</h3>
          <p className="mb-2 text-sm text-ink-2">Your workload needs approximately:</p>
          <dl className="tabular divide-y divide-line text-sm">
            {breakdown.map(([k, v]) => (
              <div key={k} className="flex justify-between py-1">
                <dt className="text-ink-2">{k}</dt>
                <dd className="text-ink">{fmtMem(v, unit, 1)}</dd>
              </div>
            ))}
            <div className="flex justify-between py-1.5 font-semibold">
              <dt className="text-ink">Total</dt>
              <dd className="text-ink">{fmtMem(requiredGb, unit, 1)}</dd>
            </div>
          </dl>
          <p className="mt-3 text-sm text-ink-2">
            {name} provides <strong className="text-ink">{fmtMem(c.vram_per_unit_gb * (c.kind === 'multi_gpu' ? c.count : 1), unit, 0)}</strong> of GPU
            memory. Therefore:
          </p>
          <ul className="mt-2 space-y-1 text-sm">
            {checks.map((x) => (
              <li key={x.text} className={x.ok ? 'text-good-text' : 'text-warning-text'}>
                {x.ok ? '✓' : '⚠'} <span className="text-ink-2">{x.text}</span>
              </li>
            ))}
          </ul>
          {speedNote && <p className="mt-3 text-xs text-ink-3">{speedNote}</p>}
        </div>

        <div className="space-y-4">
          {runnersUp.map((alt) => {
            const reasons = whyNot(alt, c, unit)
            return (
              <div key={alt.key}>
                <h3 className="text-sm font-semibold text-ink">Why not {candidateTitle(alt)}?</h3>
                <ul className="mt-1 space-y-0.5 text-sm">
                  <li className="text-good-text">
                    ✓ <span className="text-ink-2">Fits ({Math.round(alt.utilization_percent)}% of GPU memory used)</span>
                  </li>
                  {reasons.map((r) => (
                    <li key={r.text} className={r.ok ? 'text-good-text' : 'text-warning-text'}>
                      {r.ok ? '✓' : '⚠'} <span className="text-ink-2">{r.text}</span>
                    </li>
                  ))}
                  {reasons.every((r) => r.ok) && <li className="text-ink-3">Ranked slightly lower overall.</li>}
                </ul>
              </div>
            )
          })}
          {shownRejected.map((r) => (
            <div key={r.key}>
              <h3 className="text-sm font-semibold text-ink">Why not {candidateTitle(r)}?</h3>
              <p className="mt-1 text-sm text-critical-text">
                ✗ <span className="text-ink-2">{r.rejected_reason}</span>
              </p>
            </div>
          ))}
          <p className="text-xs text-ink-3">
            Therefore {name} ranks first under <strong className="text-ink-2">{priorityLabel}</strong>.
          </p>
        </div>
      </div>
    </section>
  )
}

/** WhyPanel wired to an LLM calculation. */
export function LlmWhyPanel({ result, unit, title }: { result: CalculationResult; unit: Unit; title?: string }) {
  const c = result.recommendation?.candidate
  return (
    <WhyPanel
      rec={result.recommendation}
      matches={result.matches}
      rejected={result.rejected}
      breakdown={llmBreakdown(result.memory)}
      requiredGb={result.memory.required_vram_gb}
      workloadBandwidthClass={result.compute.bandwidth_class}
      priority={result.workload.performance_priority}
      unit={unit}
      noFitLines={result.explanation}
      title={title}
      speedNote={
        c?.est_decode_tokens_per_second_per_user != null
          ? `Estimated speed ceiling: ~${fmtNum(c.est_decode_tokens_per_second_per_user)} tokens/s per request at this concurrency (target ${result.workload.tokens_per_second_per_user}). This is a model estimate, not a measurement.`
          : null
      }
    />
  )
}
