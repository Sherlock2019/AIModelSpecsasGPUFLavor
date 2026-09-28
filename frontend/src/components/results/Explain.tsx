import { ChevronRight } from 'lucide-react'
import { Fragment } from 'react'
import { fmtContext, fmtMem, fmtNum, fmtParams, type Unit } from '../../lib/format'
import { candidateTitle, classLabel, classRank, whyNot } from '../../lib/plain'
import type { CalculationResult } from '../../types'
import { PRIORITY_OPTIONS } from '../calculator/WorkloadPanel'
import { cx } from '../ui'

/** Model → parameters → precision → weights → + context/users/runtime → VRAM → GPU. */
export function FlowGraphic({ result, unit }: { result: CalculationResult; unit: Unit }) {
  const m = result.model
  const w = result.workload
  const mem = result.memory
  const steps: { top: string; bottom: string; strong?: boolean }[] = [
    { top: m.name.replace(/ Instruct$/, ''), bottom: m.architecture === 'moe' ? `MoE · ${fmtParams(m.active_parameters_b)} active` : 'Dense model' },
    { top: `${fmtParams(m.total_parameters_b)} parameters`, bottom: 'all stored in memory' },
    { top: w.precision.toUpperCase(), bottom: `${result.weights.bytes_per_parameter} byte / parameter` },
    { top: `${fmtMem(mem.model_weights_gb, unit, 0)}`, bottom: 'model weights' },
    {
      top: `+ ${fmtMem(mem.kv_cache_gb + mem.runtime_overhead_gb + mem.workspace_gb + mem.headroom_gb, unit, 0)}`,
      bottom: `${fmtContext(w.context_length)} context · ${w.concurrent_sequences} requests · runtime · headroom`,
    },
    { top: `≈ ${fmtMem(mem.required_vram_gb, unit, 0)}`, bottom: 'GPU memory required', strong: true },
    {
      top: result.recommendation ? candidateTitle(result.recommendation.candidate) : 'No fit',
      bottom: result.matches[1] ? `or ${candidateTitle(result.matches[1])}` : 'best match',
      strong: true,
    },
  ]
  return (
    <section aria-label="How the requirement is derived" className="rounded-2xl border border-line bg-surface p-5">
      <h2 className="mb-3 text-[15px] font-semibold text-ink">From model to GPU</h2>
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

/** "Why this recommendation?" plus "Why not …?" for runners-up and rejected GPUs. */
export function WhyPanel({ result, unit }: { result: CalculationResult; unit: Unit }) {
  const rec = result.recommendation
  const mem = result.memory
  if (!rec) {
    return (
      <section className="rounded-2xl border border-line bg-surface p-5">
        <h2 className="mb-2 text-[15px] font-semibold text-ink">Why no recommendation?</h2>
        <ul className="space-y-1.5 text-sm text-ink-2">
          {result.explanation.map((l, i) => (
            <li key={i}>{l}</li>
          ))}
        </ul>
      </section>
    )
  }
  const c = rec.candidate
  const name = candidateTitle(c)
  const priority = PRIORITY_OPTIONS.find((p) => p.id === result.workload.performance_priority)?.label ?? ''
  const runnersUp = result.matches.slice(1, 4)
  const rejected = result.rejected
    .filter((r) => r.kind !== 'vgpu' || result.workload.performance_priority !== 'maximum')
    .filter((r) => r.vram_per_unit_gb * r.count >= mem.required_vram_gb * 0.35)
    .slice(0, 3)
  const breakdown: [string, number][] = [
    ['Model weights', mem.model_weights_gb],
    ['KV cache (context × requests)', mem.kv_cache_gb],
    ['Runtime + workspace', mem.runtime_overhead_gb + mem.workspace_gb + mem.communication_gb],
    [`Safety headroom (${mem.safety_margin_percent}%)`, mem.headroom_gb],
  ]
  const checks: { ok: boolean; text: string }[] = [
    { ok: true, text: 'Model fits in GPU memory' },
    { ok: c.headroom_gb > 0, text: `~${fmtMem(Math.max(0, c.headroom_gb) * (c.kind === 'multi_gpu' ? c.count : 1), unit, 0)} memory left over` },
    c.kind === 'vgpu'
      ? { ok: true, text: 'A shared slice is enough; a full GPU would waste capacity' }
      : c.count === 1
        ? { ok: true, text: 'Single-GPU deployment' }
        : { ok: false, text: `Model split across ${c.count} GPUs (needs a fast GPU link)` },
    {
      ok: classRank(c.gpu_bandwidth_class) >= classRank(result.compute.bandwidth_class),
      text: `${classLabel(c.gpu_bandwidth_class)} GPU memory speed (workload needs ${classLabel(result.compute.bandwidth_class).toLowerCase()})`,
    },
    {
      ok: (c.compute_ratio ?? 1) >= 1,
      text: `${classLabel(c.gpu_compute_class)} AI compute — performance ${c.confidence === 'benchmark' ? 'measured' : 'estimated, needs benchmark validation'}`,
    },
  ]

  return (
    <section aria-labelledby="why-title" className="rounded-2xl border border-line bg-surface">
      <h2 id="why-title" className="border-b border-line px-5 py-3 text-[15px] font-semibold text-ink">
        Why this recommendation?
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
              <dd className="text-ink">{fmtMem(mem.required_vram_gb, unit, 1)}</dd>
            </div>
          </dl>
          <p className="mt-3 text-sm text-ink-2">
            {name} provides <strong className="text-ink">{fmtMem(c.vram_per_unit_gb * (c.kind === 'multi_gpu' ? c.count : 1), unit, 0)}</strong> of GPU memory. Therefore:
          </p>
          <ul className="mt-2 space-y-1 text-sm">
            {checks.map((x) => (
              <li key={x.text} className={x.ok ? 'text-good-text' : 'text-warning-text'}>
                {x.ok ? '✓' : '⚠'} <span className="text-ink-2">{x.text}</span>
              </li>
            ))}
          </ul>
          {c.est_decode_tokens_per_second_per_user != null && (
            <p className="mt-3 text-xs text-ink-3">
              Estimated speed ceiling: ~{fmtNum(c.est_decode_tokens_per_second_per_user)} tokens/s per request at this concurrency
              (target {result.workload.tokens_per_second_per_user}). This is a model estimate, not a measurement.
            </p>
          )}
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
          {rejected.map((r) => (
            <div key={r.key}>
              <h3 className="text-sm font-semibold text-ink">Why not {candidateTitle(r)}?</h3>
              <p className="mt-1 text-sm text-critical-text">
                ✗ <span className="text-ink-2">{r.rejected_reason}</span>
              </p>
            </div>
          ))}
          <p className="text-xs text-ink-3">
            Therefore {name} ranks first under <strong className="text-ink-2">{priority}</strong>.
          </p>
        </div>
      </div>
    </section>
  )
}
