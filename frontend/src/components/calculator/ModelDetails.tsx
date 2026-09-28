import { ChevronDown, ExternalLink, Pencil, RotateCcw } from 'lucide-react'
import { useState } from 'react'
import { fmtContext, fmtParams } from '../../lib/format'
import { CUSTOM_MODEL_ID, useCalc, useCatalogModel, useSelectedModel } from '../../state'
import { Badge, Button, StatusBadge, cx } from '../ui'
import { CustomModelForm, ModelSpecFields } from './CustomModelForm'

/** Collapsible model metadata with Edit (creates a USER OVERRIDE of the catalog values). */
export function ModelDetails() {
  const { form, setForm } = useCalc()
  const model = useSelectedModel()
  const official = useCatalogModel()
  const isCustom = form.modelId === CUSTOM_MODEL_ID
  const overridden = !!form.override
  const [open, setOpen] = useState(isCustom)
  const [editing, setEditing] = useState(false)

  if (isCustom) {
    return (
      <div className="mt-3 rounded-lg border border-line bg-surface-2 p-3">
        <div className="mb-3 flex items-center justify-between">
          <span className="text-[13px] font-semibold text-ink">Custom model</span>
          <StatusBadge status="user_defined" />
        </div>
        <CustomModelForm />
      </div>
    )
  }
  if (!model) return null

  const headDim = model.head_dim ?? (model.hidden_size && model.attention_heads ? Math.floor(model.hidden_size / model.attention_heads) : null)
  const rows: [string, string][] = [
    ['Architecture', model.architecture === 'moe' ? 'Mixture of Experts' : 'Dense'],
    ['Total parameters', fmtParams(model.total_parameters_b)],
    ['Active parameters', fmtParams(model.active_parameters_b ?? model.total_parameters_b)],
    ['Native precision', model.native_precision.map((p) => p.toUpperCase()).join(', ') || 'Unknown'],
    ['Max context', fmtContext(model.max_context_length)],
    ['Layers', model.num_layers ? String(model.num_layers) : '—'],
    ['Hidden size', model.hidden_size ? model.hidden_size.toLocaleString() : '—'],
    ['Attention heads', model.attention_heads ? String(model.attention_heads) : '—'],
    ['KV heads', model.attention_type === 'mla' ? 'MLA (latent)' : model.kv_heads ? String(model.kv_heads) : '—'],
    ['Head dimension', headDim ? String(headDim) : '—'],
  ]

  return (
    <div className="mt-3">
      <button
        type="button"
        aria-expanded={open}
        onClick={() => setOpen(!open)}
        className="flex w-full items-center justify-between gap-2 text-left text-[13px] font-medium text-ink-2 hover:text-ink"
      >
        <span className="flex items-center gap-2">
          Model details
          {overridden ? <Badge tone="warning">User override</Badge> : <StatusBadge status={model.metadata_status} />}
        </span>
        <ChevronDown className={cx('h-4 w-4 transition', open && 'rotate-180')} aria-hidden />
      </button>

      {open && (
        <div className="fade-up mt-2 rounded-lg border border-line bg-surface-2 p-3">
          {editing ? (
            <>
              <ModelSpecFields
                value={form.override ?? official ?? model}
                showIdentity={false}
                advancedOpen
                onChange={(m) => setForm((f) => ({ ...f, override: { ...m, id: f.modelId } }))}
              />
              <div className="mt-3 flex justify-end gap-2">
                <Button size="sm" onClick={() => setEditing(false)}>
                  Done
                </Button>
              </div>
            </>
          ) : (
            <>
              <dl className="grid grid-cols-2 gap-x-4 gap-y-2 text-sm">
                {rows.map(([k, v]) => (
                  <div key={k}>
                    <dt className="text-xs text-ink-3">{k}</dt>
                    <dd className="font-medium text-ink">{v}</dd>
                  </div>
                ))}
              </dl>
              {model.notes && <p className="mt-3 text-xs text-ink-3">{model.notes}</p>}
              <div className="mt-3 flex flex-wrap items-center gap-2">
                <Button size="sm" onClick={() => setEditing(true)}>
                  <Pencil className="h-3.5 w-3.5" /> Edit
                </Button>
                {overridden && (
                  <Button size="sm" variant="ghost" onClick={() => setForm((f) => ({ ...f, override: null }))}>
                    <RotateCcw className="h-3.5 w-3.5" /> Reset to official values
                  </Button>
                )}
                {model.source_url && (
                  <a
                    href={model.source_url}
                    target="_blank"
                    rel="noreferrer"
                    className="ml-auto inline-flex items-center gap-1 text-xs text-brand-text hover:underline"
                  >
                    Model card <ExternalLink className="h-3 w-3" aria-hidden />
                  </a>
                )}
              </div>
              {overridden && (
                <p className="mt-2 text-xs text-warning-text">
                  ⚠ You changed the official values. Results use your values and are marked user-defined.
                </p>
              )}
            </>
          )}
        </div>
      )}
    </div>
  )
}
