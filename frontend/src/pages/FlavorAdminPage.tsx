import { useEffect, useState } from 'react'
import { api } from '../api'
import { Modal } from '../components/Modal'
import { Alert, Badge, Button, Field, Input, NumberInput, PageHeader, Select, Table, Toggle } from '../components/ui'
import type { AIFlavor, Priority } from '../types'

const NEW_FLAVOR: AIFlavor = {
  id: '',
  display_name: 'AI-',
  description: '',
  group: 'medium',
  parameter_floor_b: 0,
  parameter_ceiling_b: 10,
  baseline_precision: 'int4',
  default_vram_min_gb: 8,
  default_vram_max_gb: 16,
  recommended_use_cases: [],
  example_model_ids: [],
  allow_vgpu: true,
  allow_full_gpu: true,
  allow_multi_gpu: false,
  performance_tier: 'standard',
  default_context_length: 32768,
  default_concurrency: 20,
  default_performance_priority: 'balanced',
  commercial_sku: null,
  enabled: true,
}

export function FlavorAdminPage() {
  const [flavors, setFlavors] = useState<AIFlavor[]>([])
  const [editing, setEditing] = useState<{ f: AIFlavor; isNew: boolean } | null>(null)
  const [error, setError] = useState<string | null>(null)
  const load = () =>
    api
      .aiFlavors(true)
      .then(setFlavors)
      .catch((e: Error) => setError(e.message))
  useEffect(() => {
    void load()
  }, [])

  const toggle = async (f: AIFlavor) => {
    try {
      const { model_ids: _m, ...rest } = f
      await api.updateAIFlavor({ ...rest, enabled: !f.enabled })
      void load()
    } catch (e) {
      setError((e as Error).message)
    }
  }

  return (
    <div>
      <PageHeader
        title="AI flavor catalog"
        description="Customer-facing AI VM tiers. These are product definitions, not hardware: the sizing engine decides vGPU / full GPU / multi-GPU for every request, within the policy set here."
        actions={
          <Button variant="primary" onClick={() => setEditing({ f: NEW_FLAVOR, isNew: true })}>
            Add flavor tier
          </Button>
        }
      />
      {error && (
        <div className="mb-4">
          <Alert tone="critical">{error}</Alert>
        </div>
      )}
      <Table>
        <thead>
          <tr>
            <th>Flavor</th>
            <th>Group</th>
            <th className="text-right">Parameters</th>
            <th>Baseline</th>
            <th className="text-right">Typical VRAM</th>
            <th>Allowed modes</th>
            <th>Defaults</th>
            <th className="text-right">Models</th>
            <th>SKU</th>
            <th>Enabled</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {flavors.map((f) => (
            <tr key={f.id} className={f.enabled ? '' : 'opacity-50'}>
              <td>
                <div className="font-mono font-semibold">{f.display_name}</div>
                <div className="text-xs text-ink-3">{f.description}</div>
              </td>
              <td className="capitalize">{f.group}</td>
              <td className="text-right">
                {f.parameter_floor_b}–{f.parameter_ceiling_b}B
              </td>
              <td>{f.baseline_precision.toUpperCase()}</td>
              <td className="text-right">
                {f.default_vram_min_gb}–{f.default_vram_max_gb} GB
              </td>
              <td className="space-x-1 whitespace-nowrap">
                {f.allow_vgpu && <Badge tone="good">vGPU</Badge>}
                {f.allow_full_gpu && <Badge tone="brand">Full</Badge>}
                {f.allow_multi_gpu && <Badge tone="warning">Multi</Badge>}
              </td>
              <td className="text-xs text-ink-2">
                {Math.round(f.default_context_length / 1024)}K · {f.default_concurrency} req · {f.default_performance_priority}
              </td>
              <td className="text-right">{f.model_ids?.length ?? 0}</td>
              <td className="font-mono text-xs">{f.commercial_sku ?? '—'}</td>
              <td>
                <Toggle checked={f.enabled} onChange={() => toggle(f)} label={<span className="sr-only">Enabled</span>} />
              </td>
              <td>
                <Button size="sm" variant="ghost" onClick={() => setEditing({ f, isNew: false })}>
                  Edit
                </Button>
              </td>
            </tr>
          ))}
        </tbody>
      </Table>
      <p className="mt-3 text-xs text-ink-3">
        A model belongs to the smallest enabled tier whose parameter ceiling (plus the tolerance in Settings, default 10%) covers its total parameters.
      </p>
      {editing && (
        <FlavorEditor
          initial={editing.f}
          isNew={editing.isNew}
          onClose={() => setEditing(null)}
          onSaved={() => {
            setEditing(null)
            void load()
          }}
        />
      )}
    </div>
  )
}

function FlavorEditor({ initial, isNew, onClose, onSaved }: { initial: AIFlavor; isNew: boolean; onClose: () => void; onSaved: () => void }) {
  const [f, setF] = useState<AIFlavor>(initial)
  const [error, setError] = useState<string | null>(null)
  const set = (patch: Partial<AIFlavor>) => setF({ ...f, ...patch })
  const num = (key: keyof AIFlavor, label: string) => (
    <Field label={label}>
      <NumberInput min={0} step="any" value={f[key] as number} onChange={(v) => set({ [key]: v ?? 0 } as Partial<AIFlavor>)} />
    </Field>
  )
  const save = async () => {
    setError(null)
    const { model_ids: _m, ...body } = f
    try {
      await (isNew ? api.createAIFlavor(body) : api.updateAIFlavor(body))
      onSaved()
    } catch (e) {
      setError((e as Error).message)
    }
  }
  return (
    <Modal
      title={isNew ? 'Add AI flavor tier' : `Edit ${initial.display_name}`}
      onClose={onClose}
      footer={
        <>
          <Button onClick={onClose}>Cancel</Button>
          <Button variant="primary" onClick={save}>
            Save flavor
          </Button>
        </>
      }
    >
      {error && (
        <div className="mb-3">
          <Alert tone="critical">{error}</Alert>
        </div>
      )}
      <div className="grid grid-cols-2 gap-3">
        <Field label="ID" hint="e.g. ai-24b">
          <Input value={f.id} disabled={!isNew} onChange={(e) => set({ id: e.target.value })} />
        </Field>
        <Field label="Display name">
          <Input value={f.display_name} onChange={(e) => set({ display_name: e.target.value })} />
        </Field>
        <Field label="Description" className="col-span-2">
          <Input value={f.description} onChange={(e) => set({ description: e.target.value })} />
        </Field>
        <Field label="Group">
          <Select value={f.group} onChange={(e) => set({ group: e.target.value as AIFlavor['group'] })}>
            {['small', 'medium', 'large', 'ultra'].map((g) => (
              <option key={g}>{g}</option>
            ))}
          </Select>
        </Field>
        <Field label="Commercial SKU">
          <Input value={f.commercial_sku ?? ''} onChange={(e) => set({ commercial_sku: e.target.value || null })} />
        </Field>
        {num('parameter_floor_b', 'Parameter floor (B)')}
        {num('parameter_ceiling_b', 'Parameter ceiling (B)')}
        {num('default_vram_min_gb', 'Typical VRAM min (GB)')}
        {num('default_vram_max_gb', 'Typical VRAM max (GB)')}
        <Field label="Baseline precision">
          <Select value={f.baseline_precision} onChange={(e) => set({ baseline_precision: e.target.value })}>
            {['int4', 'int8', 'fp8', 'bf16', 'fp16', 'mxfp4'].map((p) => (
              <option key={p} value={p}>
                {p.toUpperCase()}
              </option>
            ))}
          </Select>
        </Field>
        <Field label="Performance tier">
          <Input value={f.performance_tier} onChange={(e) => set({ performance_tier: e.target.value })} />
        </Field>
        {num('default_context_length', 'Default context (tokens)')}
        {num('default_concurrency', 'Default concurrency')}
        <Field label="Default priority">
          <Select value={f.default_performance_priority} onChange={(e) => set({ default_performance_priority: e.target.value as Priority })}>
            <option value="economy">Lowest cost</option>
            <option value="balanced">Balanced</option>
            <option value="performance">Performance</option>
            <option value="maximum">Max performance</option>
          </Select>
        </Field>
        <Field label="Use cases" hint="Comma separated">
          <Input value={f.recommended_use_cases.join(', ')} onChange={(e) => set({ recommended_use_cases: e.target.value.split(',').map((s) => s.trim()).filter(Boolean) })} />
        </Field>
        <Field label="Example model ids" hint="Comma separated catalog ids" className="col-span-2">
          <Input value={f.example_model_ids.join(', ')} onChange={(e) => set({ example_model_ids: e.target.value.split(',').map((s) => s.trim()).filter(Boolean) })} />
        </Field>
      </div>
      <div className="mt-3 space-y-1">
        <Toggle checked={f.allow_vgpu} onChange={(v) => set({ allow_vgpu: v })} label="Allow shared vGPU (-SHARED)" />
        <Toggle checked={f.allow_full_gpu} onChange={(v) => set({ allow_full_gpu: v })} label="Allow one dedicated GPU (-PRO)" />
        <Toggle checked={f.allow_multi_gpu} onChange={(v) => set({ allow_multi_gpu: v })} label="Allow multi-GPU (-MULTI)" />
        <Toggle checked={f.enabled} onChange={(v) => set({ enabled: v })} label="Show in the AI VM catalog" />
      </div>
    </Modal>
  )
}
