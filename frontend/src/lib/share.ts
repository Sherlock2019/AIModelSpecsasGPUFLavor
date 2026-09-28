import type { Priority, WorkloadInput } from '../types'

// Short query keys for shareable calculator links: ?m=llama-3.1-70b-instruct&p=int4&ctx=32768&n=20…
const NUM_KEYS = {
  ctx: 'context_length',
  n: 'concurrent_sequences',
  in: 'average_input_tokens',
  out: 'average_output_tokens',
} as const
const STR_KEYS = { p: 'precision', fw: 'framework', kv: 'kv_cache_precision' } as const
const PRIORITIES: Priority[] = ['economy', 'balanced', 'performance', 'maximum']

export function toQuery(modelId: string, w: WorkloadInput): string {
  const q = new URLSearchParams({ m: modelId })
  for (const [k, field] of Object.entries(STR_KEYS)) q.set(k, String(w[field]))
  for (const [k, field] of Object.entries(NUM_KEYS)) {
    const v = w[field]
    if (v != null) q.set(k, String(v))
  }
  q.set('pr', w.performance_priority)
  if (w.environment === 'development') q.set('env', 'dev')
  return q.toString()
}

/** Returns the model id and workload patch encoded in a link, or null when there is none. */
export function fromQuery(search: string): { modelId: string; patch: Partial<WorkloadInput> } | null {
  const q = new URLSearchParams(search)
  const modelId = q.get('m')
  if (!modelId) return null
  const patch: Partial<WorkloadInput> = {}
  for (const [k, field] of Object.entries(STR_KEYS)) {
    const v = q.get(k)
    if (v) (patch as Record<string, unknown>)[field] = v
  }
  for (const [k, field] of Object.entries(NUM_KEYS)) {
    const v = Number(q.get(k))
    if (q.has(k) && Number.isFinite(v) && v > 0) (patch as Record<string, unknown>)[field] = Math.round(v)
  }
  const pr = q.get('pr') as Priority | null
  if (pr && PRIORITIES.includes(pr)) patch.performance_priority = pr
  if (q.get('env') === 'dev') patch.environment = 'development'
  return { modelId, patch }
}
