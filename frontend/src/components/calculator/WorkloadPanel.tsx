import { ChevronDown, Gauge, Leaf, Minus, Plus, Rocket, Scale, type LucideIcon } from 'lucide-react'
import { useState } from 'react'
import { fmtContext, fmtNum } from '../../lib/format'
import { useCalc, useSelectedModel } from '../../state'
import type { Priority } from '../../types'
import { Field, InfoTip, NumberInput, Segmented, Select, Toggle, cx } from '../ui'

function Label({ id, children, tip }: { id?: string; children: string; tip?: string }) {
  return (
    <div className="mb-1.5 flex items-center gap-1.5 text-[13px] font-medium text-ink-2">
      <span id={id}>{children}</span>
      {tip && <InfoTip>{tip}</InfoTip>}
    </div>
  )
}

// ----------------------------------------------------------------------------- precision

const PRECISION_ORDER = ['int4', 'mxfp4', 'int8', 'fp8', 'fp16', 'bf16', 'fp32']
const PRECISION_NOTE: Record<string, string> = {
  int4: 'Lowest memory · optimized inference',
  mxfp4: '4-bit microscaling · native for gpt-oss',
  int8: 'Half the memory of 16-bit',
  fp8: 'Fast modern AI inference (Hopper, Blackwell, MI300)',
  fp16: 'Higher precision · larger memory requirement',
  bf16: 'Higher precision · larger memory requirement',
  fp32: 'Full precision · rarely used for serving',
}

export function PrecisionSelect() {
  const { form, setWorkload, meta } = useCalc()
  const model = useSelectedModel()
  const supported = model?.supported_serving_precisions ?? []
  const options = (meta?.precisions ?? [])
    .filter((p) => (supported.length ? supported.includes(p.id) : !['fp32', 'mxfp4'].includes(p.id)) || p.id === form.workload.precision)
    .sort((a, b) => PRECISION_ORDER.indexOf(a.id) - PRECISION_ORDER.indexOf(b.id))
  const current = options.find((p) => p.id === form.workload.precision)
  const weights = (bytes: number, overhead: number) => (model ? model.total_parameters_b * bytes * overhead : null)
  const w = current ? weights(current.bytes_per_parameter, current.default_overhead_factor) : null
  return (
    <div>
      <Label id="precision-label" tip="How many bytes store each model weight. Fewer bytes means less GPU memory, with a small quality trade-off for 4-bit.">
        Precision
      </Label>
      <div className="flex rounded-lg border border-line-strong bg-surface-2 p-0.5" role="radiogroup" aria-labelledby="precision-label">
        {options.map((p) => {
          const on = p.id === form.workload.precision
          const native = model?.native_precision.includes(p.id)
          return (
            <button
              key={p.id}
              type="button"
              role="radio"
              aria-checked={on}
              onClick={() => setWorkload({ precision: p.id })}
              title={PRECISION_NOTE[p.id]}
              className={cx(
                'relative h-8 flex-1 rounded-md text-[13px] font-semibold transition',
                on ? 'bg-surface text-brand-text shadow-sm' : 'text-ink-2 hover:text-ink',
              )}
            >
              {p.label}
              {native && <span className="absolute top-0.5 right-1 h-1.5 w-1.5 rounded-full bg-good" title="Native precision" />}
            </button>
          )
        })}
      </div>
      <p className="mt-1.5 flex justify-between gap-2 text-xs text-ink-3">
        <span>{current ? PRECISION_NOTE[current.id] : ''}</span>
        {w != null && <span className="tabular shrink-0 font-medium text-ink-2">~{fmtNum(w, w < 10 ? 1 : 0)} GB weights</span>}
      </p>
    </div>
  )
}

// ----------------------------------------------------------------------------- context

const CONTEXT_CHIPS = [4096, 8192, 16384, 32768, 65536, 131072]

export function ContextSelect() {
  const { form, setWorkload } = useCalc()
  const model = useSelectedModel()
  const max = model?.max_context_length ?? null
  const value = form.workload.context_length
  const chips = CONTEXT_CHIPS.filter((c) => !max || c <= max)
  const isMax = max != null && value === max && !chips.includes(value)
  const isCustom = !chips.includes(value) && !isMax
  const [custom, setCustom] = useState(isCustom)
  const chip = (on: boolean) =>
    cx(
      'tabular h-8 rounded-lg border px-2.5 text-[13px] font-medium transition',
      on ? 'border-brand bg-brand text-brand-ink' : 'border-line text-ink-2 hover:border-line-strong hover:text-ink',
    )
  return (
    <div>
      <Label
        id="context-label"
        tip="The longest single request (prompt + answer) you must support. Memory is sized for your typical requests plus one request at this full length."
      >
        Context length
      </Label>
      <div className="flex flex-wrap gap-1.5" role="radiogroup" aria-labelledby="context-label">
        {chips.map((c) => (
          <button
            key={c}
            type="button"
            role="radio"
            aria-checked={value === c && !custom}
            onClick={() => {
              setCustom(false)
              setWorkload({ context_length: c })
            }}
            className={chip(value === c && !custom)}
          >
            {fmtContext(c)}
          </button>
        ))}
        {max != null && !chips.includes(max) && (
          <button
            type="button"
            role="radio"
            aria-checked={isMax && !custom}
            onClick={() => {
              setCustom(false)
              setWorkload({ context_length: max })
            }}
            className={chip(isMax && !custom)}
          >
            Max {fmtContext(max)}
          </button>
        )}
        <button type="button" role="radio" aria-checked={custom} onClick={() => setCustom(true)} className={chip(custom)}>
          Custom
        </button>
      </div>
      {custom && (
        <div className="mt-2 flex items-center gap-2">
          <NumberInput
            aria-label="Custom context length"
            min={256}
            max={max ?? undefined}
            className="w-32"
            value={value}
            onChange={(v) => v && setWorkload({ context_length: Math.min(max ?? v, Math.max(256, Math.round(v))) })}
          />
          <span className="text-xs text-ink-3">tokens{max ? ` (model maximum ${max.toLocaleString()})` : ''}</span>
        </div>
      )}
    </div>
  )
}

// ----------------------------------------------------------------------------- concurrency

const CONCURRENCY_PRESETS = [
  { label: 'Development', value: 1 },
  { label: 'Small', value: 5 },
  { label: 'Medium', value: 20 },
  { label: 'Large', value: 100 },
]

export function ConcurrencyStepper() {
  const { form, setWorkload } = useCalc()
  const n = form.workload.concurrent_sequences
  const set = (v: number) => setWorkload({ concurrent_sequences: Math.max(1, Math.min(100000, Math.round(v))) })
  const step = n >= 100 ? 10 : n >= 20 ? 5 : 1
  return (
    <div>
      <Label tip="Requests being generated at the same moment, not total users. A common rule of thumb is 5–10% of active users. Each one needs its own KV cache memory.">
        Concurrent requests
      </Label>
      <div className="flex flex-wrap items-center gap-3">
        <div className="flex items-center rounded-lg border border-line-strong">
          <button type="button" aria-label="Fewer requests" onClick={() => set(n - step)} className="flex h-9 w-9 items-center justify-center text-ink-2 hover:text-ink disabled:opacity-40" disabled={n <= 1}>
            <Minus className="h-4 w-4" />
          </button>
          <NumberInput
            aria-label="Concurrent requests"
            min={1}
            max={100000}
            value={n}
            onChange={(v) => set(v ?? 1)}
            className="h-9 w-16 rounded-none border-0 border-x border-line-strong text-center focus:ring-0"
          />
          <button type="button" aria-label="More requests" onClick={() => set(n + step)} className="flex h-9 w-9 items-center justify-center text-ink-2 hover:text-ink">
            <Plus className="h-4 w-4" />
          </button>
        </div>
        <div className="flex flex-wrap gap-1.5">
          {CONCURRENCY_PRESETS.map((p) => (
            <button
              key={p.label}
              type="button"
              onClick={() => set(p.value)}
              aria-pressed={n === p.value}
              className={cx(
                'h-7 rounded-full border px-2.5 text-xs font-medium transition',
                n === p.value ? 'border-brand bg-brand-soft text-brand-text' : 'border-line text-ink-3 hover:border-line-strong hover:text-ink',
              )}
            >
              {p.label} · {p.value}
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}

// ----------------------------------------------------------------------------- priority

export const PRIORITY_OPTIONS: { id: Priority; label: string; icon: LucideIcon; blurb: string }[] = [
  { id: 'economy', label: 'Lowest cost', icon: Leaf, blurb: 'Smallest GPU or vGPU that safely fits' },
  { id: 'balanced', label: 'Balanced', icon: Scale, blurb: 'Reasonable headroom, one GPU where possible' },
  { id: 'performance', label: 'Performance', icon: Gauge, blurb: 'Faster GPU, more memory speed' },
  { id: 'maximum', label: 'Max performance', icon: Rocket, blurb: 'High-end dedicated GPUs' },
]

export function PrioritySelect() {
  const { form, setWorkload } = useCalc()
  return <PriorityPicker value={form.workload.performance_priority} onChange={(v) => setWorkload({ performance_priority: v })} />
}

/** State-agnostic priority selector (used by the LLM, AI flavor and custom ML pages). */
export function PriorityPicker({ value, onChange }: { value: Priority; onChange: (v: Priority) => void }) {
  const current = PRIORITY_OPTIONS.find((p) => p.id === value)
  return (
    <div>
      <Label id="priority-label">Performance priority</Label>
      <div className="grid grid-cols-2 gap-1.5 sm:grid-cols-4" role="radiogroup" aria-labelledby="priority-label">
        {PRIORITY_OPTIONS.map((p) => {
          const on = p.id === value
          const Icon = p.icon
          return (
            <button
              key={p.id}
              type="button"
              role="radio"
              aria-checked={on}
              onClick={() => onChange(p.id)}
              className={cx(
                'flex flex-col items-center gap-1 rounded-lg border px-2 py-2 text-center text-xs font-semibold transition',
                on ? 'border-brand bg-brand-soft text-brand-text ring-1 ring-brand' : 'border-line text-ink-2 hover:border-line-strong hover:text-ink',
              )}
            >
              <Icon className="h-4 w-4" aria-hidden />
              {p.label}
            </button>
          )
        })}
      </div>
      {current && <p className="mt-1.5 text-xs text-ink-3">{current.blurb}</p>}
    </div>
  )
}

// ----------------------------------------------------------------------------- advanced options

export function AdvancedOptions() {
  const { form, setWorkload, meta } = useCalc()
  const [open, setOpen] = useState(false)
  const w = form.workload
  const autoMargin =
    w.environment === 'development'
      ? meta?.environments.find((e) => e.id === 'development')?.default_safety_margin_percent
      : meta?.priorities.find((p) => p.id === w.performance_priority)?.safety_margin_percent
  const changed = [
    w.framework !== 'vllm' && (meta?.frameworks.find((f) => f.id === w.framework)?.label ?? w.framework),
    !w.allow_vgpu && 'no vGPU',
    !w.allow_multi_gpu && 'single GPU only',
    w.kv_cache_precision !== 'auto' && `KV ${w.kv_cache_precision.toUpperCase()}`,
    w.context_mode === 'max' && 'worst-case context',
    w.safety_margin_percent != null && `${w.safety_margin_percent}% margin`,
    !w.add_safety_headroom && 'no headroom',
    w.environment === 'development' && 'development',
    w.dedicated_gpu_required && 'dedicated GPU',
    w.tensor_parallel_size && `split across ${w.tensor_parallel_size}`,
  ].filter(Boolean) as string[]

  return (
    <div className="border-t border-line pt-3">
      <button
        type="button"
        aria-expanded={open}
        onClick={() => setOpen(!open)}
        className="flex w-full items-center justify-between gap-3 text-left"
      >
        <span>
          <span className="block text-[13px] font-medium text-ink-2">Advanced options</span>
          <span className="block text-xs text-ink-3">
            {changed.length ? changed.join(' · ') : 'vLLM · typical request 4,000 in / 1,000 out · 15% headroom'}
          </span>
        </span>
        <ChevronDown className={cx('h-4 w-4 text-ink-3 transition', open && 'rotate-180')} aria-hidden />
      </button>
      {open && (
        <div className="fade-up mt-4 space-y-5">
          <div className="grid grid-cols-2 gap-3">
            <Field label="Serving framework">
              <Select value={w.framework} onChange={(e) => setWorkload({ framework: e.target.value })}>
                {(meta?.frameworks ?? [{ id: 'vllm', label: 'vLLM' }]).map((f) => (
                  <option key={f.id} value={f.id}>
                    {f.label}
                  </option>
                ))}
              </Select>
            </Field>
            <Field label="KV cache precision">
              <Select value={w.kv_cache_precision} onChange={(e) => setWorkload({ kv_cache_precision: e.target.value })}>
                <option value="auto">Auto (FP16/BF16)</option>
                {(meta?.kv_precisions ?? []).map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.label}
                  </option>
                ))}
              </Select>
            </Field>
            <Field label="Typical input tokens">
              <NumberInput min={0} value={w.average_input_tokens} disabled={w.context_mode === 'max'} onChange={(v) => setWorkload({ average_input_tokens: v })} />
            </Field>
            <Field label="Typical output tokens">
              <NumberInput min={0} value={w.average_output_tokens} disabled={w.context_mode === 'max'} onChange={(v) => setWorkload({ average_output_tokens: v })} />
            </Field>
          </div>
          <Field label="Size memory for">
            <Segmented
              value={w.context_mode}
              options={[
                { value: 'realistic', label: 'Typical requests' },
                { value: 'max', label: 'Every request at full context' },
              ]}
              onChange={(v) => setWorkload({ context_mode: v })}
            />
          </Field>

          <div className="space-y-1">
            <Toggle checked={w.allow_vgpu} onChange={(v) => setWorkload({ allow_vgpu: v })} label="Allow shared GPU (vGPU)" hint="Use a slice of a GPU when the model is small enough" />
            <Toggle checked={w.allow_multi_gpu} onChange={(v) => setWorkload({ allow_multi_gpu: v })} label="Allow multiple GPUs" hint="Split large models across GPUs" />
            <Toggle checked={w.prefer_lowest_cost} onChange={(v) => setWorkload({ prefer_lowest_cost: v })} label="Prefer lowest cost" hint="Uses GPU prices when set in the catalog" />
            <Toggle checked={w.dedicated_gpu_required} onChange={(v) => setWorkload({ dedicated_gpu_required: v })} label="Dedicated GPU required" hint="Never share a GPU" />
            <Toggle checked={w.add_safety_headroom} onChange={(v) => setWorkload({ add_safety_headroom: v })} label="Add safety headroom" />
          </div>

          <div className="grid grid-cols-2 gap-3">
            <Field label="Safety margin %">
              <NumberInput
                min={0}
                max={100}
                disabled={!w.add_safety_headroom}
                value={w.safety_margin_percent}
                placeholder={autoMargin != null ? `Auto (${autoMargin}%)` : 'Auto'}
                onChange={(v) => setWorkload({ safety_margin_percent: v })}
              />
            </Field>
            <Field label="Environment">
              <Select value={w.environment} onChange={(e) => setWorkload({ environment: e.target.value as 'production' | 'development' })}>
                <option value="production">Production</option>
                <option value="development">Development</option>
              </Select>
            </Field>
            <Field label="Max GPU count">
              <Select value={w.max_gpu_count} onChange={(e) => setWorkload({ max_gpu_count: Number(e.target.value) })}>
                {[1, 2, 4, 8, 16].map((n) => (
                  <option key={n} value={n}>
                    {n}
                  </option>
                ))}
              </Select>
            </Field>
            <Field label="Split model across">
              <Select value={w.tensor_parallel_size ?? ''} onChange={(e) => setWorkload({ tensor_parallel_size: e.target.value ? Number(e.target.value) : null })}>
                <option value="">Automatic</option>
                {[1, 2, 4, 8, 16].map((n) => (
                  <option key={n} value={n}>
                    {n} GPU{n > 1 ? 's' : ''}
                  </option>
                ))}
              </Select>
            </Field>
            <Field label="Target speed per request (tokens/s)">
              <NumberInput min={0} step="any" value={w.target_tokens_per_second_per_user} placeholder="By priority" onChange={(v) => setWorkload({ target_tokens_per_second_per_user: v || null })} />
            </Field>
            <Field label="Target total tokens/s">
              <NumberInput min={0} step="any" value={w.target_aggregate_tokens_per_second} placeholder="—" onChange={(v) => setWorkload({ target_aggregate_tokens_per_second: v || null })} />
            </Field>
            <Field label="Time to first token target (ms)">
              <NumberInput min={0} value={w.ttft_target_ms} placeholder="—" onChange={(v) => setWorkload({ ttft_target_ms: v || null })} />
            </Field>
            <Field label="Time per output token target (ms)">
              <NumberInput min={0} value={w.tpot_target_ms} placeholder="—" onChange={(v) => setWorkload({ tpot_target_ms: v || null })} />
            </Field>
            <Field label="Runtime overhead (GB)">
              <NumberInput min={0} step="any" value={w.runtime_overhead_override_gb} placeholder="Framework default" onChange={(v) => setWorkload({ runtime_overhead_override_gb: v })} />
            </Field>
            <Field label="Workspace %">
              <NumberInput min={0} max={100} step="any" value={w.workspace_percent_override} placeholder="Framework default" onChange={(v) => setWorkload({ workspace_percent_override: v })} />
            </Field>
            <Field label="Quantization overhead ×">
              <NumberInput min={1} max={3} step="0.01" value={w.quantization_overhead_factor} placeholder="Precision default" onChange={(v) => setWorkload({ quantization_overhead_factor: v })} />
            </Field>
            <Field label="Calculation mode">
              <Select value={w.calculation_mode} onChange={(e) => setWorkload({ calculation_mode: e.target.value as 'auto' | 'estimate' | 'benchmark' })}>
                <option value="auto">Auto (benchmarks when available)</option>
                <option value="estimate">Estimate only</option>
                <option value="benchmark">Benchmark preferred</option>
              </Select>
            </Field>
          </div>
        </div>
      )}
    </div>
  )
}
