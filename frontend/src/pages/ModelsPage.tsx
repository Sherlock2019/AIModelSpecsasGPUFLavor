import { useEffect, useMemo, useState } from 'react'
import { api } from '../api'
import { ModelSpecFields } from '../components/calculator/CustomModelForm'
import { Modal } from '../components/Modal'
import { Alert, Badge, Button, Field, Input, PageHeader, Select, StatusBadge, Table, Toggle } from '../components/ui'
import { downloadText } from '../lib/download'
import { fmtContext, fmtParams } from '../lib/format'
import { EMPTY_CUSTOM_MODEL, useCalc } from '../state'
import type { MetadataStatus, ModelSpec } from '../types'

export function ModelsPage() {
  const { reloadModels } = useCalc()
  const [models, setModels] = useState<ModelSpec[]>([])
  const [query, setQuery] = useState('')
  const [family, setFamily] = useState('')
  const [editing, setEditing] = useState<{ model: ModelSpec; isNew: boolean } | null>(null)
  const [importing, setImporting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const load = () =>
    api
      .models(true)
      .then(setModels)
      .catch((e: Error) => setError(e.message))
  useEffect(() => {
    void load()
  }, [])

  const refresh = () => {
    void load()
    reloadModels()
  }

  const families = useMemo(() => [...new Set(models.map((m) => m.family))].sort(), [models])
  const visible = models.filter(
    (m) =>
      (!family || m.family === family) &&
      (!query || `${m.name} ${m.id} ${m.vendor}`.toLowerCase().includes(query.toLowerCase())),
  )

  const toggle = async (m: ModelSpec) => {
    try {
      await api.patchModel(m.id, { enabled: !m.enabled })
      refresh()
    } catch (e) {
      setError((e as Error).message)
    }
  }

  return (
    <div>
      <PageHeader
        title="Model catalog"
        description="Curated open-weight models. Metadata status shows whether values were verified against the official config or imported, or were entered by a user."
        actions={
          <>
            <Button onClick={async () => downloadText('models.yaml', await api.exportModels(), 'application/yaml')}>Export YAML</Button>
            <Button onClick={() => setImporting(true)}>Import YAML</Button>
            <Button variant="primary" onClick={() => setEditing({ model: { ...EMPTY_CUSTOM_MODEL, id: '', metadata_status: 'imported' }, isNew: true })}>
              Add model
            </Button>
          </>
        }
      />
      {error && (
        <div className="mb-4">
          <Alert tone="critical">{error}</Alert>
        </div>
      )}
      <div className="mb-4 flex flex-wrap gap-3">
        <Input className="max-w-xs" placeholder="Search models" value={query} onChange={(e) => setQuery(e.target.value)} />
        <Select className="max-w-[200px]" value={family} onChange={(e) => setFamily(e.target.value)}>
          <option value="">All families</option>
          {families.map((f) => (
            <option key={f}>{f}</option>
          ))}
        </Select>
        <span className="self-center text-sm text-ink-3">{visible.length} models</span>
      </div>
      <Table>
        <thead>
          <tr>
            <th>Model</th>
            <th>Family</th>
            <th>Architecture</th>
            <th className="text-right">Total</th>
            <th className="text-right">Active</th>
            <th className="text-right">Max context</th>
            <th>KV metadata</th>
            <th>Status</th>
            <th>Enabled</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {visible.map((m) => (
            <tr key={m.id} className={m.enabled ? '' : 'opacity-50'}>
              <td>
                <div className="font-medium text-ink">{m.name}</div>
                <div className="font-mono text-xs text-ink-3">{m.id}</div>
              </td>
              <td>{m.family}</td>
              <td>{m.architecture === 'moe' ? <Badge tone="brand">MoE</Badge> : 'Dense'}</td>
              <td className="text-right">{fmtParams(m.total_parameters_b)}</td>
              <td className="text-right">{fmtParams(m.active_parameters_b)}</td>
              <td className="text-right">{fmtContext(m.max_context_length)}</td>
              <td className="text-xs">
                {m.attention_type === 'mla' ? 'MLA' : m.num_layers && m.kv_heads ? `${m.num_layers}L · ${m.kv_heads} KV` : <span className="text-warning-text">Estimate mode</span>}
              </td>
              <td>
                <StatusBadge status={m.metadata_status} />
              </td>
              <td>
                <Toggle checked={m.enabled} onChange={() => toggle(m)} label={<span className="sr-only">Enabled</span>} />
              </td>
              <td className="text-right whitespace-nowrap">
                {m.source_url && (
                  <a href={m.source_url} target="_blank" rel="noreferrer" className="mr-2 text-xs text-brand-text hover:underline">
                    Source
                  </a>
                )}
                <Button size="sm" variant="ghost" onClick={() => setEditing({ model: m, isNew: false })}>
                  Edit
                </Button>
              </td>
            </tr>
          ))}
        </tbody>
      </Table>

      {editing && (
        <ModelEditor
          initial={editing.model}
          isNew={editing.isNew}
          onClose={() => setEditing(null)}
          onSaved={() => {
            setEditing(null)
            refresh()
          }}
        />
      )}
      {importing && (
        <ImportDialog
          onClose={() => setImporting(false)}
          onDone={() => {
            refresh()
          }}
        />
      )}
    </div>
  )
}

function ModelEditor({ initial, isNew, onClose, onSaved }: { initial: ModelSpec; isNew: boolean; onClose: () => void; onSaved: () => void }) {
  const [m, setM] = useState<ModelSpec>(initial)
  const [error, setError] = useState<string | null>(null)
  const set = (patch: Partial<ModelSpec>) => setM({ ...m, ...patch })
  const save = async () => {
    setError(null)
    try {
      await (isNew ? api.createModel(m) : api.updateModel(m))
      onSaved()
    } catch (e) {
      setError((e as Error).message)
    }
  }
  return (
    <Modal
      title={isNew ? 'Add model' : `Edit ${initial.name}`}
      onClose={onClose}
      footer={
        <>
          <Button onClick={onClose}>Cancel</Button>
          <Button variant="primary" onClick={save}>
            Save model
          </Button>
        </>
      }
    >
      <div className="space-y-3">
        {error && <Alert tone="critical">{error}</Alert>}
        <div className="grid grid-cols-2 gap-3">
          <Field label="ID" hint="lowercase, e.g. llama-3.1-8b-instruct">
            <Input value={m.id} disabled={!isNew} onChange={(e) => set({ id: e.target.value })} />
          </Field>
          <Field label="Name">
            <Input value={m.name} onChange={(e) => set({ name: e.target.value })} />
          </Field>
          <Field label="Vendor">
            <Input value={m.vendor} onChange={(e) => set({ vendor: e.target.value })} />
          </Field>
          <Field label="Family">
            <Input value={m.family} onChange={(e) => set({ family: e.target.value })} />
          </Field>
          <Field label="Source URL">
            <Input value={m.source_url ?? ''} onChange={(e) => set({ source_url: e.target.value || null })} />
          </Field>
          <Field label="Metadata status">
            <Select value={m.metadata_status} onChange={(e) => set({ metadata_status: e.target.value as MetadataStatus })}>
              <option value="verified">Verified</option>
              <option value="imported">Imported</option>
              <option value="user_defined">User-defined</option>
            </Select>
          </Field>
        </div>
        <ModelSpecFields value={m} onChange={setM} showIdentity={false} showSupported advancedOpen />
        <Field label="Notes">
          <Input value={m.notes ?? ''} onChange={(e) => set({ notes: e.target.value || null })} />
        </Field>
      </div>
    </Modal>
  )
}

function ImportDialog({ onClose, onDone }: { onClose: () => void; onDone: () => void }) {
  const [text, setText] = useState('')
  const [overwrite, setOverwrite] = useState(false)
  const [result, setResult] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const run = async () => {
    setError(null)
    try {
      const r = await api.importModels(text, overwrite)
      setResult(`${r.created} created · ${r.updated} updated · ${r.skipped} skipped${r.errors.length ? ` · ${r.errors.length} errors` : ''}`)
      if (r.errors.length) setError(r.errors.slice(0, 5).join('\n'))
      onDone()
    } catch (e) {
      setError((e as Error).message)
    }
  }
  return (
    <Modal
      title="Import models from YAML"
      onClose={onClose}
      footer={
        <>
          <Button onClick={onClose}>Close</Button>
          <Button variant="primary" onClick={run} disabled={!text.trim()}>
            Import
          </Button>
        </>
      }
    >
      <div className="space-y-3">
        <p className="text-sm text-ink-2">Metadata only: the YAML is parsed as data and validated per model. Nothing is executed.</p>
        <input
          type="file"
          accept=".yaml,.yml"
          className="text-sm"
          onChange={async (e) => {
            const f = e.target.files?.[0]
            if (f) setText(await f.text())
          }}
        />
        <textarea
          className="h-64 w-full rounded-lg border border-line-strong bg-surface p-3 font-mono text-xs text-ink"
          placeholder={'models:\n  - id: my-model\n    name: My Model\n    architecture: dense\n    total_parameters_b: 8'}
          value={text}
          onChange={(e) => setText(e.target.value)}
        />
        <Toggle checked={overwrite} onChange={setOverwrite} label="Overwrite existing models with the same id" />
        {result && <Alert tone="good">{result}</Alert>}
        {error && (
          <Alert tone="critical">
            <pre className="text-xs whitespace-pre-wrap">{error}</pre>
          </Alert>
        )}
      </div>
    </Modal>
  )
}
