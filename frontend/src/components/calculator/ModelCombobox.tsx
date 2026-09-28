import { Check, ChevronsUpDown, PenLine, Search } from 'lucide-react'
import { useEffect, useMemo, useRef, useState, type KeyboardEvent } from 'react'
import { fmtContext, fmtParams } from '../../lib/format'
import { CUSTOM_MODEL_ID, useCalc, useSelectedModel } from '../../state'
import type { ModelSpec } from '../../types'
import { Badge, cx } from '../ui'

// Shown first in the dropdown (the most commonly requested sizes of each family).
const POPULAR = [
  'llama-3.1-8b-instruct',
  'llama-3.1-70b-instruct',
  'llama-3.1-405b-instruct',
  'qwen3-8b',
  'qwen3-14b',
  'qwen3-32b',
  'qwen2.5-72b-instruct',
  'qwen3-30b-a3b',
  'qwen3-235b-a22b',
  'deepseek-r1-distill-qwen-7b',
  'deepseek-r1-distill-qwen-14b',
  'deepseek-r1-distill-qwen-32b',
  'deepseek-r1-distill-llama-70b',
  'deepseek-r1',
  'mistral-7b-instruct-v0.3',
  'mixtral-8x7b-instruct-v0.1',
  'mixtral-8x22b-instruct-v0.1',
  'gemma-3-4b-it',
  'gemma-3-12b-it',
  'gemma-3-27b-it',
  'phi-4',
]

function matches(m: ModelSpec, q: string) {
  const hay = `${m.name} ${m.family} ${m.vendor} ${m.id}`.toLowerCase()
  return q
    .toLowerCase()
    .split(/\s+/)
    .filter(Boolean)
    .every((t) => hay.includes(t))
}

export function modelSummary(m: ModelSpec) {
  const params =
    m.architecture === 'moe'
      ? `${fmtParams(m.total_parameters_b)} MoE · ${fmtParams(m.active_parameters_b)} active`
      : `${fmtParams(m.total_parameters_b)} dense`
  return `${params} · ${fmtContext(m.max_context_length)} context`
}

interface Group {
  label: string
  items: ModelSpec[]
}

export function ModelCombobox({ onModelChange }: { onModelChange: (m: ModelSpec | null) => void }) {
  const { form, setForm, models } = useCalc()
  const model = useSelectedModel()
  const isCustom = form.modelId === CUSTOM_MODEL_ID
  const [open, setOpen] = useState(false)
  const [query, setQuery] = useState('')
  const [active, setActive] = useState(0)
  const rootRef = useRef<HTMLDivElement>(null)
  const listRef = useRef<HTMLUListElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  const groups = useMemo<Group[]>(() => {
    if (query) {
      const hits = models.filter((m) => matches(m, query))
      return hits.length ? [{ label: `${hits.length} result${hits.length > 1 ? 's' : ''}`, items: hits }] : []
    }
    const popular = POPULAR.map((id) => models.find((m) => m.id === id)).filter(Boolean) as ModelSpec[]
    const byFamily = new Map<string, ModelSpec[]>()
    for (const m of models) byFamily.set(m.family, [...(byFamily.get(m.family) ?? []), m])
    return [{ label: 'Popular', items: popular }, ...[...byFamily].map(([label, items]) => ({ label, items }))]
  }, [models, query])

  const optionIds = useMemo(() => [...groups.flatMap((g) => g.items.map((m) => m.id)), CUSTOM_MODEL_ID], [groups])

  useEffect(() => {
    if (!open) return
    const onDown = (e: MouseEvent) => {
      if (!rootRef.current?.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', onDown)
    return () => document.removeEventListener('mousedown', onDown)
  }, [open])

  useEffect(() => {
    listRef.current?.querySelector(`[data-index="${active}"]`)?.scrollIntoView({ block: 'nearest' })
  }, [active])

  const pick = (id: string) => {
    setForm((f) => ({ ...f, modelId: id, override: null }))
    onModelChange(id === CUSTOM_MODEL_ID ? form.custom : (models.find((m) => m.id === id) ?? null))
    setOpen(false)
    setQuery('')
  }

  const openList = () => {
    setOpen(true)
    setActive(Math.max(0, optionIds.indexOf(form.modelId)))
    requestAnimationFrame(() => inputRef.current?.focus())
  }

  const onKey = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault()
      setActive((a) => Math.min(optionIds.length - 1, a + 1))
    } else if (e.key === 'ArrowUp') {
      e.preventDefault()
      setActive((a) => Math.max(0, a - 1))
    } else if (e.key === 'Enter') {
      e.preventDefault()
      if (optionIds[active]) pick(optionIds[active])
    } else if (e.key === 'Escape') {
      setOpen(false)
    }
  }

  let index = -1
  const customIndex = optionIds.length - 1

  return (
    <div ref={rootRef} className="relative">
      <button
        type="button"
        onClick={() => (open ? setOpen(false) : openList())}
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-label="Model"
        data-model-id={form.modelId}
        className="flex w-full items-center gap-3 rounded-xl border border-line-strong bg-surface px-3.5 py-2.5 text-left transition hover:border-ink-3 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brand"
      >
        <span className="min-w-0 flex-1">
          <span className="block truncate text-[15px] font-semibold text-ink">
            {isCustom ? form.custom.name || 'Custom model' : (model?.name ?? 'Select a model')}
          </span>
          <span className="block truncate text-xs text-ink-3">
            {model ? modelSummary(model) : 'Loading catalog…'}
          </span>
        </span>
        <ChevronsUpDown className="h-4 w-4 shrink-0 text-ink-3" aria-hidden />
      </button>

      {open && (
        <div className="fade-up absolute inset-x-0 top-full z-40 mt-1.5 overflow-hidden rounded-xl border border-line bg-surface shadow-xl">
          <div className="flex items-center gap-2 border-b border-line px-3">
            <Search className="h-4 w-4 text-ink-3" aria-hidden />
            <input
              ref={inputRef}
              role="combobox"
              aria-label="Search model"
              aria-expanded
              aria-controls="model-listbox"
              aria-activedescendant={`model-opt-${active}`}
              placeholder="Search model…"
              value={query}
              onChange={(e) => {
                setQuery(e.target.value)
                setActive(0)
              }}
              onKeyDown={onKey}
              className="h-11 w-full bg-transparent text-sm text-ink placeholder:text-ink-3 focus:outline-none"
            />
          </div>
          <ul ref={listRef} id="model-listbox" role="listbox" className="max-h-96 overflow-y-auto py-1">
            {groups.map((g) => (
              <li key={g.label} role="presentation">
                <div className="px-3 pt-2 pb-1 text-[11px] font-semibold tracking-wider text-ink-3 uppercase">{g.label}</div>
                <ul role="presentation">
                  {g.items.map((m) => {
                    index += 1
                    const i = index
                    return (
                      <li
                        key={`${g.label}-${m.id}`}
                        id={`model-opt-${i}`}
                        data-index={i}
                        role="option"
                        aria-selected={m.id === form.modelId}
                        onMouseEnter={() => setActive(i)}
                        onMouseDown={(e) => {
                          e.preventDefault()
                          pick(m.id)
                        }}
                        className={cx('mx-1 flex cursor-pointer items-center gap-3 rounded-lg px-2.5 py-2', i === active && 'bg-surface-2')}
                      >
                        <span className="min-w-0 flex-1">
                          <span className="block truncate text-sm font-medium text-ink">{m.name}</span>
                          <span className="block truncate text-xs text-ink-3">{modelSummary(m)}</span>
                        </span>
                        {m.architecture === 'moe' && <Badge tone="brand">MoE</Badge>}
                        {m.id === form.modelId && <Check className="h-4 w-4 text-brand-text" aria-hidden />}
                      </li>
                    )
                  })}
                </ul>
              </li>
            ))}
            {groups.length === 0 && <li className="px-3 py-3 text-sm text-ink-3">No catalog match. Try a custom model.</li>}
            <li
              id={`model-opt-${customIndex}`}
              data-index={customIndex}
              role="option"
              aria-selected={isCustom}
              onMouseEnter={() => setActive(customIndex)}
              onMouseDown={(e) => {
                e.preventDefault()
                pick(CUSTOM_MODEL_ID)
              }}
              className={cx(
                'mx-1 mt-1 flex cursor-pointer items-center gap-2 rounded-lg border-t border-line px-2.5 py-2.5 text-sm text-ink',
                active === customIndex && 'bg-surface-2',
              )}
            >
              <PenLine className="h-4 w-4 text-ink-3" aria-hidden />
              Custom model — enter your own
            </li>
          </ul>
        </div>
      )}
    </div>
  )
}
