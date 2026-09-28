import { ArrowRight, Lightbulb } from 'lucide-react'
import { useState } from 'react'
import { fmtMem, type Unit } from '../../lib/format'
import type { CalcWarning, Suggestion } from '../../types'
import { Alert, Button } from '../ui'

/** One-click variants of the workload that need less hardware. */
export function WhatIfPanel<P extends Record<string, unknown>>({
  suggestions,
  unit,
  onApply,
}: {
  suggestions: Suggestion[]
  unit: Unit
  onApply: (patch: P) => void
}) {
  if (!suggestions.length) return null
  return (
    <section aria-labelledby="whatif-title">
      <h3 id="whatif-title" className="mb-2 flex items-center gap-2 text-sm font-semibold text-ink">
        <Lightbulb className="h-4 w-4 text-warning-text" aria-hidden />
        Ways to need less hardware
      </h3>
      <div className="grid gap-3 md:grid-cols-2 2xl:grid-cols-3">
        {suggestions.map((s) => (
          <div key={s.key} className="fade-up flex flex-col rounded-xl border border-line bg-surface p-4">
            <div className="text-sm font-semibold text-ink">{s.title}</div>
            <div className="mt-1.5 flex flex-wrap items-center gap-1.5 text-sm text-ink-2">
              <span className="tabular">{fmtMem(s.required_vram_gb, unit, 0)}</span>
              <ArrowRight className="h-3.5 w-3.5 text-ink-3" aria-hidden />
              <span className="font-medium text-ink">{s.recommendation.split(' — ')[0]}</span>
            </div>
            <p className="mt-1.5 flex-1 text-xs text-ink-3">{s.tradeoff}</p>
            <div className="mt-3 flex items-center justify-between gap-2">
              <span className="text-xs text-good-text">{s.saves_gb > 0 ? `Saves ${fmtMem(s.saves_gb, unit, 0)}` : ''}</span>
              <Button size="sm" onClick={() => onApply(s.patch as P)}>
                Apply
              </Button>
            </div>
          </div>
        ))}
      </div>
    </section>
  )
}

/** Critical and warning items always visible; informational notes behind a toggle. */
export function WarningsSummary({ warnings }: { warnings: CalcWarning[] }) {
  const [showNotes, setShowNotes] = useState(false)
  const important = warnings.filter((w) => w.severity !== 'info')
  const notes = warnings.filter((w) => w.severity === 'info')
  if (!warnings.length) return null
  const order: Record<CalcWarning['severity'], number> = { critical: 0, warning: 1, info: 2 }
  return (
    <div className="space-y-2">
      {[...important]
        .sort((a, b) => order[a.severity] - order[b.severity])
        .map((w) => (
          <Alert key={w.code} tone={w.severity === 'critical' ? 'critical' : 'warning'}>
            {w.message}
          </Alert>
        ))}
      {notes.length > 0 && (
        <div>
          <button type="button" onClick={() => setShowNotes(!showNotes)} className="text-xs font-medium text-ink-3 hover:text-ink">
            {showNotes ? 'Hide' : 'Show'} {notes.length} note{notes.length > 1 ? 's' : ''} about this estimate
          </button>
          {showNotes && (
            <div className="mt-2 space-y-2">
              {notes.map((w) => (
                <Alert key={w.code} tone="info">
                  {w.message}
                </Alert>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
