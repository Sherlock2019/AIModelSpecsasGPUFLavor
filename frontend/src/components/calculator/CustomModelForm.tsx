import { ChevronDown } from 'lucide-react'
import { useState } from 'react'
import { useCalc } from '../../state'
import type { ModelSpec } from '../../types'
import { Field, Input, NumberInput, Segmented, Select, cx } from '../ui'

const PRECISIONS = ['bf16', 'fp16', 'fp8', 'int8', 'int4']

/**
 * Model metadata editor. Basics first (name, architecture, parameters, precision, context);
 * the architecture fields that make the KV cache exact sit under "Advanced".
 */
export function ModelSpecFields({
  value,
  onChange,
  showIdentity = true,
  showSupported = false,
  advancedOpen = false,
}: {
  value: ModelSpec
  onChange: (m: ModelSpec) => void
  showIdentity?: boolean
  showSupported?: boolean
  advancedOpen?: boolean
}) {
  const [advanced, setAdvanced] = useState(advancedOpen)
  const set = (patch: Partial<ModelSpec>) => onChange({ ...value, ...patch })
  const isMoe = value.architecture === 'moe'
  const derivedHead =
    !value.head_dim && value.hidden_size && value.attention_heads ? Math.floor(value.hidden_size / value.attention_heads) : null
  const exactKv = !!(value.num_layers && value.kv_heads && (value.head_dim || derivedHead)) || value.attention_type === 'mla'
  return (
    <div className="space-y-3">
      {showIdentity && (
        <Field label="Model name">
          <Input value={value.name} onChange={(e) => set({ name: e.target.value })} />
        </Field>
      )}
      <Field label="Architecture">
        <Segmented
          value={value.architecture}
          options={[
            { value: 'dense', label: 'Dense' },
            { value: 'moe', label: 'MoE' },
          ]}
          onChange={(a) => set({ architecture: a, active_parameters_b: a === 'dense' ? null : value.active_parameters_b })}
        />
      </Field>
      <div className="grid grid-cols-2 gap-3">
        <Field label="Total parameters (B)">
          <NumberInput min={0.01} step="any" value={value.total_parameters_b} onChange={(v) => set({ total_parameters_b: v ?? 0 })} />
        </Field>
        {isMoe ? (
          <Field label="Active parameters (B)" hint="Required for MoE">
            <NumberInput min={0.01} step="any" value={value.active_parameters_b} onChange={(v) => set({ active_parameters_b: v })} />
          </Field>
        ) : (
          <div />
        )}
        <Field label="Native precision">
          <Select value={value.native_precision[0] ?? ''} onChange={(e) => set({ native_precision: e.target.value ? [e.target.value] : [] })}>
            <option value="">Unknown</option>
            {[...PRECISIONS, 'mxfp4'].map((p) => (
              <option key={p} value={p}>
                {p.toUpperCase()}
              </option>
            ))}
          </Select>
        </Field>
        <Field label="Max context (tokens)">
          <NumberInput min={1} value={value.max_context_length} onChange={(v) => set({ max_context_length: v })} />
        </Field>
      </div>
      {showSupported && (
        <Field label="Supported serving precisions">
          <div className="flex flex-wrap gap-1.5">
            {PRECISIONS.map((p) => {
              const on = value.supported_serving_precisions.includes(p)
              return (
                <button
                  key={p}
                  type="button"
                  aria-pressed={on}
                  onClick={() =>
                    set({
                      supported_serving_precisions: on
                        ? value.supported_serving_precisions.filter((x) => x !== p)
                        : [...value.supported_serving_precisions, p],
                    })
                  }
                  className={cx(
                    'h-7 rounded-md border px-2.5 text-xs font-semibold uppercase transition',
                    on ? 'border-brand bg-brand-soft text-brand-text' : 'border-line-strong text-ink-2 hover:bg-surface-2',
                  )}
                >
                  {p}
                </button>
              )
            })}
          </div>
        </Field>
      )}

      <div className="rounded-lg border border-line">
        <button
          type="button"
          aria-expanded={advanced}
          onClick={() => setAdvanced(!advanced)}
          className="flex w-full items-center justify-between px-3 py-2 text-left text-[13px] font-medium text-ink-2 hover:text-ink"
        >
          <span>
            Advanced · architecture
            <span className={cx('ml-2 text-xs font-normal', exactKv ? 'text-good-text' : 'text-warning-text')}>
              {exactKv ? '✓ exact KV cache' : '⚠ KV cache estimated'}
            </span>
          </span>
          <ChevronDown className={cx('h-4 w-4 transition', advanced && 'rotate-180')} aria-hidden />
        </button>
        {advanced && (
          <div className="grid grid-cols-2 gap-3 border-t border-line p-3">
            <Field label="Layers">
              <NumberInput min={1} value={value.num_layers} onChange={(v) => set({ num_layers: v })} />
            </Field>
            <Field label="Hidden size">
              <NumberInput min={1} value={value.hidden_size} onChange={(v) => set({ hidden_size: v })} />
            </Field>
            <Field label="Attention heads">
              <NumberInput min={1} value={value.attention_heads} onChange={(v) => set({ attention_heads: v })} />
            </Field>
            <Field label="KV heads" hint="Fewer than attention heads for GQA/MQA">
              <NumberInput min={1} value={value.kv_heads} onChange={(v) => set({ kv_heads: v })} />
            </Field>
            <Field label="Head dimension" hint={derivedHead ? `Derived: ${derivedHead}` : undefined}>
              <NumberInput
                min={1}
                value={value.head_dim}
                placeholder={derivedHead ? String(derivedHead) : ''}
                onChange={(v) => set({ head_dim: v })}
              />
            </Field>
          </div>
        )}
      </div>
    </div>
  )
}

export function CustomModelForm() {
  const { form, setForm } = useCalc()
  return <ModelSpecFields value={form.custom} onChange={(custom) => setForm((f) => ({ ...f, custom }))} />
}
