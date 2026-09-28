import { Bar, BarChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { fmtMem, type Unit } from '../../lib/format'
import type { MemoryBreakdown } from '../../types'

// Fixed categorical order (validated adjacent-pair order) + neutral hatch for headroom.
export const MEMORY_SEGMENTS = [
  { key: 'model_weights_gb', label: 'Model weights', color: 'var(--series-1)' },
  { key: 'kv_cache_gb', label: 'KV cache', color: 'var(--series-2)' },
  { key: 'runtime_overhead_gb', label: 'Runtime', color: 'var(--series-3)' },
  { key: 'workspace_gb', label: 'Workspace', color: 'var(--series-4)' },
  { key: 'communication_gb', label: 'Communication', color: 'var(--series-5)' },
  { key: 'headroom_gb', label: 'Safety headroom', color: 'url(#headroom-hatch)' },
] as const

type SegKey = (typeof MEMORY_SEGMENTS)[number]['key']

interface TooltipProps {
  active?: boolean
  payload?: { dataKey: string; value: number }[]
  unit: Unit
  total: number
}

function ChartTooltip({ active, payload, unit, total }: TooltipProps) {
  if (!active || !payload?.length) return null
  return (
    <div className="rounded-lg border border-line bg-surface px-3 py-2 text-xs shadow-lg">
      {payload
        .filter((p) => p.value > 0)
        .map((p) => {
          const seg = MEMORY_SEGMENTS.find((s) => s.key === p.dataKey)!
          return (
            <div key={p.dataKey} className="flex items-center justify-between gap-6 py-0.5">
              <span className="flex items-center gap-2 text-ink-2">
                <Swatch color={seg.color} />
                {seg.label}
              </span>
              <span className="tabular font-medium text-ink">
                {fmtMem(p.value, unit)} · {((p.value / total) * 100).toFixed(0)}%
              </span>
            </div>
          )
        })}
    </div>
  )
}

export function Swatch({ color }: { color: string }) {
  const hatch = color.startsWith('url')
  return (
    <span
      aria-hidden
      className="inline-block h-2.5 w-2.5 shrink-0 rounded-[3px]"
      style={
        hatch
          ? { background: 'repeating-linear-gradient(45deg, var(--series-headroom) 0 2px, transparent 2px 4px)', outline: '1px solid var(--series-headroom)' }
          : { background: color }
      }
    />
  )
}

export function MemoryChart({
  memory,
  unit,
  capacityGb,
  capacityLabel,
}: {
  memory: MemoryBreakdown
  unit: Unit
  capacityGb?: number | null
  capacityLabel?: string
}) {
  const present = MEMORY_SEGMENTS.filter((s) => memory[s.key as SegKey] > 0)
  const row = Object.fromEntries(present.map((s) => [s.key, memory[s.key as SegKey]]))
  const max = Math.max(memory.required_vram_gb, capacityGb ?? 0) * 1.04
  return (
    <div>
      <div className="h-[76px]" role="img" aria-label={`Memory breakdown, total ${fmtMem(memory.required_vram_gb, unit)}`}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={[row]} layout="vertical" margin={{ top: 18, right: 8, bottom: 0, left: 0 }} barCategoryGap={0}>
            <defs>
              <pattern id="headroom-hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
                <rect width="6" height="6" fill="var(--surface-2)" />
                <line x1="0" y1="0" x2="0" y2="6" stroke="var(--series-headroom)" strokeWidth="2.5" />
              </pattern>
            </defs>
            <XAxis type="number" hide domain={[0, max]} />
            <YAxis type="category" hide />
            <Tooltip
              cursor={false}
              content={<ChartTooltip unit={unit} total={memory.required_vram_gb} />}
              wrapperStyle={{ outline: 'none', zIndex: 10 }}
            />
            {present.map((s, i) => (
              <Bar
                key={s.key}
                dataKey={s.key}
                stackId="mem"
                fill={s.color}
                stroke="var(--surface)"
                strokeWidth={2}
                barSize={30}
                radius={
                  present.length === 1 ? 4 : i === 0 ? [4, 0, 0, 4] : i === present.length - 1 ? [0, 4, 4, 0] : 0
                }
                isAnimationActive={false}
              />
            ))}
            {capacityGb ? (
              <ReferenceLine
                x={capacityGb}
                stroke="var(--ink-2)"
                strokeDasharray="4 3"
                strokeWidth={1.5}
                label={({ viewBox }: { viewBox?: { x?: number; y?: number } }) => (
                  // Right-aligned to the line so the label never overflows the chart edge.
                  <text x={(viewBox?.x ?? 0) - 4} y={(viewBox?.y ?? 0) - 6} textAnchor="end" fill="var(--ink-2)" fontSize={11}>
                    {capacityLabel ?? 'GPU capacity'}
                  </text>
                )}
              />
            ) : null}
          </BarChart>
        </ResponsiveContainer>
      </div>
      <ul className="mt-3 grid grid-cols-2 gap-x-4 gap-y-1.5 text-sm sm:grid-cols-3">
        {MEMORY_SEGMENTS.map((s) => (
          <li key={s.key} className="flex items-center justify-between gap-2">
            <span className="flex items-center gap-2 text-ink-2">
              <Swatch color={s.color} />
              {s.label}
            </span>
            <span className="tabular font-medium text-ink">{fmtMem(memory[s.key as SegKey], unit)}</span>
          </li>
        ))}
      </ul>
    </div>
  )
}
