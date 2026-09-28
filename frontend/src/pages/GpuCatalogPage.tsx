import { useEffect, useState } from 'react'
import { api } from '../api'
import { Modal } from '../components/Modal'
import { Alert, Badge, Button, Field, Input, NumberInput, PageHeader, Select, StatusBadge, Table, Tabs, Toggle } from '../components/ui'
import { fmtNum } from '../lib/format'
import type { GPUSpec, InventoryRecord, VGPUProfile } from '../types'

type Tab = 'gpus' | 'vgpu' | 'inventory'

const yesNo = (v: boolean | null | undefined) => (v == null ? <span className="text-ink-3">?</span> : v ? '✓' : '—')

export function GpuCatalogPage() {
  const [tab, setTab] = useState<Tab>('gpus')
  const [gpus, setGpus] = useState<GPUSpec[]>([])
  const [profiles, setProfiles] = useState<VGPUProfile[]>([])
  const [inventory, setInventory] = useState<InventoryRecord[]>([])
  const [error, setError] = useState<string | null>(null)

  const load = () => {
    Promise.all([api.gpus(), api.vgpuProfiles(), api.inventory()])
      .then(([g, p, i]) => {
        setGpus(g)
        setProfiles(p)
        setInventory(i)
      })
      .catch((e: Error) => setError(e.message))
  }
  useEffect(load, [])

  return (
    <div>
      <PageHeader
        title="GPU catalog"
        description="Physical GPUs, vGPU profiles and site inventory used by the recommendation engine. Unknown datasheet values stay empty and are never estimated."
      />
      {error && (
        <div className="mb-4">
          <Alert tone="critical">{error}</Alert>
        </div>
      )}
      <div className="mb-4">
        <Tabs<Tab>
          value={tab}
          onChange={setTab}
          tabs={[
            { value: 'gpus', label: `GPUs (${gpus.length})` },
            { value: 'vgpu', label: `vGPU profiles (${profiles.length})` },
            { value: 'inventory', label: 'Inventory' },
          ]}
        />
      </div>
      {tab === 'gpus' && <GpuTable gpus={gpus} onChanged={load} onError={setError} />}
      {tab === 'vgpu' && <ProfileTable profiles={profiles} gpus={gpus} onChanged={load} onError={setError} />}
      {tab === 'inventory' && <InventoryEditor rows={inventory} gpus={gpus} onSaved={load} />}
    </div>
  )
}

function GpuTable({ gpus, onChanged, onError }: { gpus: GPUSpec[]; onChanged: () => void; onError: (e: string) => void }) {
  const [editing, setEditing] = useState<{ gpu: GPUSpec; isNew: boolean } | null>(null)
  return (
    <>
      <div className="mb-3 flex justify-end">
        <Button
          variant="primary"
          onClick={() =>
            setEditing({
              isNew: true,
              gpu: { id: '', vendor: 'NVIDIA', model: '', vram_gb: 24, availability: 'unknown', metadata_status: 'user_defined', enabled: true },
            })
          }
        >
          Add GPU
        </Button>
      </div>
      <Table>
        <thead>
          <tr>
            <th>GPU</th>
            <th className="text-right">VRAM</th>
            <th className="text-right">Bandwidth</th>
            <th className="text-right">BF16 / FP8 TFLOPS</th>
            <th>vGPU</th>
            <th>MIG</th>
            <th>NVLink</th>
            <th className="text-right">Cost / h</th>
            <th>Availability</th>
            <th>Status</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {gpus.map((g) => (
            <tr key={g.id} className={g.enabled ? '' : 'opacity-50'}>
              <td>
                <div className="font-medium">{g.vendor} {g.model}</div>
                <div className="text-xs text-ink-3">
                  {g.architecture_generation} · {g.form_factor}
                </div>
              </td>
              <td className="text-right">{g.vram_gb} GB</td>
              <td className="text-right">{g.memory_bandwidth_gbps ? `${fmtNum(g.memory_bandwidth_gbps)} GB/s` : '—'}</td>
              <td className="text-right">
                {g.bf16_tflops ?? '—'} / {g.fp8_tflops ?? '—'}
              </td>
              <td>{yesNo(g.supports_vgpu)}</td>
              <td>{yesNo(g.supports_mig)}</td>
              <td>{yesNo(g.supports_nvlink)}</td>
              <td className="text-right">{g.cost_per_hour != null ? `$${g.cost_per_hour.toFixed(2)}` : <span className="text-ink-3">Not set</span>}</td>
              <td>
                <Badge tone={g.availability === 'available' ? 'good' : g.availability === 'unavailable' ? 'critical' : g.availability === 'limited' ? 'warning' : 'neutral'}>
                  {g.availability}
                </Badge>
              </td>
              <td>
                <StatusBadge status={g.metadata_status} />
              </td>
              <td>
                <Button size="sm" variant="ghost" onClick={() => setEditing({ gpu: g, isNew: false })}>
                  Edit
                </Button>
              </td>
            </tr>
          ))}
        </tbody>
      </Table>
      {editing && (
        <GpuEditor
          initial={editing.gpu}
          isNew={editing.isNew}
          onClose={() => setEditing(null)}
          onSaved={() => {
            setEditing(null)
            onChanged()
          }}
          onError={onError}
        />
      )}
    </>
  )
}

function GpuEditor({ initial, isNew, onClose, onSaved }: { initial: GPUSpec; isNew: boolean; onClose: () => void; onSaved: () => void; onError: (e: string) => void }) {
  const [g, setG] = useState<GPUSpec>(initial)
  const [error, setError] = useState<string | null>(null)
  const set = (patch: Partial<GPUSpec>) => setG({ ...g, ...patch })
  const num = (key: keyof GPUSpec, label: string, hint?: string) => (
    <Field label={label} hint={hint}>
      <NumberInput min={0} step="any" value={g[key] as number | null} onChange={(v) => set({ [key]: v } as Partial<GPUSpec>)} />
    </Field>
  )
  const tri = (key: 'supports_vgpu' | 'supports_mig' | 'supports_nvlink', label: string) => (
    <Field label={label}>
      <Select value={g[key] == null ? '' : String(g[key])} onChange={(e) => set({ [key]: e.target.value === '' ? null : e.target.value === 'true' })}>
        <option value="">Unknown</option>
        <option value="true">Yes</option>
        <option value="false">No</option>
      </Select>
    </Field>
  )
  const save = async () => {
    try {
      await (isNew ? api.createGpu(g) : api.updateGpu(g))
      onSaved()
    } catch (e) {
      setError((e as Error).message)
    }
  }
  return (
    <Modal
      title={isNew ? 'Add GPU' : `Edit ${initial.model}`}
      onClose={onClose}
      footer={
        <>
          <Button onClick={onClose}>Cancel</Button>
          <Button variant="primary" onClick={save}>
            Save GPU
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
        <Field label="ID">
          <Input value={g.id} disabled={!isNew} onChange={(e) => set({ id: e.target.value })} />
        </Field>
        <Field label="Model">
          <Input value={g.model} onChange={(e) => set({ model: e.target.value })} />
        </Field>
        <Field label="Vendor">
          <Input value={g.vendor} onChange={(e) => set({ vendor: e.target.value })} />
        </Field>
        <Field label="Architecture">
          <Input value={g.architecture_generation ?? ''} onChange={(e) => set({ architecture_generation: e.target.value || null })} />
        </Field>
        {num('vram_gb', 'VRAM (GB)')}
        {num('usable_vram_mib', 'Usable VRAM (MiB)', 'nvidia-smi total; optional')}
        {num('memory_bandwidth_gbps', 'Memory bandwidth (GB/s)')}
        {num('fp16_tflops', 'FP16 TFLOPS (dense)')}
        {num('bf16_tflops', 'BF16 TFLOPS (dense)')}
        {num('fp8_tflops', 'FP8 TFLOPS (dense)')}
        {num('int8_tops', 'INT8 TOPS (dense)')}
        {num('power_watts', 'Power (W)')}
        {tri('supports_vgpu', 'vGPU support')}
        {tri('supports_mig', 'MIG support')}
        {tri('supports_nvlink', 'NVLink')}
        {num('cost_per_hour', 'Cost per hour ($)', 'Leave empty if not priced')}
        <Field label="Availability">
          <Select value={g.availability} onChange={(e) => set({ availability: e.target.value as GPUSpec['availability'] })}>
            {['available', 'limited', 'unavailable', 'unknown'].map((a) => (
              <option key={a}>{a}</option>
            ))}
          </Select>
        </Field>
        <Field label="OpenStack PCI alias" hint="Must match [pci] alias in nova.conf">
          <Input value={g.openstack_pci_alias ?? ''} onChange={(e) => set({ openstack_pci_alias: e.target.value || null })} />
        </Field>
        <Field label="GPU link">
          <Input value={g.interconnect_type ?? ''} placeholder="PCIe, NVLink, Infinity Fabric…" onChange={(e) => set({ interconnect_type: e.target.value || null })} />
        </Field>
      </div>
      <div className="mt-3">
        <Toggle checked={g.enabled} onChange={(v) => set({ enabled: v })} label="Offer this GPU in recommendations" />
      </div>
    </Modal>
  )
}

function ProfileTable({ profiles, gpus, onChanged }: { profiles: VGPUProfile[]; gpus: GPUSpec[]; onChanged: () => void; onError: (e: string) => void }) {
  const [editing, setEditing] = useState<VGPUProfile | null>(null)
  const name = (id: string) => gpus.find((g) => g.id === id)?.model ?? id
  return (
    <>
      <div className="mb-3 flex items-center justify-between gap-3">
        <p className="text-sm text-ink-3">
          Time-sliced profiles share compute (worst case 1/N). MIG profiles get dedicated slices. Equal VRAM does not mean equal performance.
        </p>
        <Button
          variant="primary"
          onClick={() =>
            setEditing({
              profile_name: '',
              physical_gpu: gpus[0]?.id ?? '',
              vram_gb: 24,
              max_instances_per_gpu: 2,
              compute_mode: 'compute',
              sharing_mode: 'time_sliced',
              supports_cuda: true,
              license_required: true,
              metadata_status: 'user_defined',
              enabled: true,
            })
          }
        >
          Add vGPU profile
        </Button>
      </div>
      <Table>
        <thead>
          <tr>
            <th>Profile</th>
            <th>Physical GPU</th>
            <th className="text-right">VRAM (GiB)</th>
            <th className="text-right">Max / GPU</th>
            <th>Sharing</th>
            <th className="text-right">License $/h</th>
            <th className="text-right">Free</th>
            <th>OpenStack trait</th>
            <th>Status</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {profiles.map((p) => (
            <tr key={p.profile_name} className={p.enabled ? '' : 'opacity-50'}>
              <td className="font-mono text-xs font-semibold">{p.profile_name}</td>
              <td>{name(p.physical_gpu)}</td>
              <td className="text-right">{p.vram_gb}</td>
              <td className="text-right">{p.max_instances_per_gpu}</td>
              <td>{p.sharing_mode === 'mig' ? <Badge tone="brand">MIG {p.mig_compute_slices}/7</Badge> : 'Time-sliced'}</td>
              <td className="text-right">{p.license_cost_per_hour ?? '—'}</td>
              <td className="text-right">
                {p.available_instances == null ? <span className="text-ink-3">?</span> : p.available_instances === 0 ? <Badge tone="critical">0</Badge> : p.available_instances}
              </td>
              <td className="font-mono text-xs">{p.openstack_trait ?? <span className="text-ink-3">not set</span>}</td>
              <td>
                <StatusBadge status={p.metadata_status} />
              </td>
              <td>
                <Button size="sm" variant="ghost" onClick={() => setEditing(p)}>
                  Edit
                </Button>
              </td>
            </tr>
          ))}
        </tbody>
      </Table>
      {editing && (
        <ProfileEditor
          initial={editing}
          gpus={gpus}
          onClose={() => setEditing(null)}
          onSaved={() => {
            setEditing(null)
            onChanged()
          }}
        />
      )}
    </>
  )
}

function ProfileEditor({ initial, gpus, onClose, onSaved }: { initial: VGPUProfile; gpus: GPUSpec[]; onClose: () => void; onSaved: () => void }) {
  const [p, setP] = useState(initial)
  const [error, setError] = useState<string | null>(null)
  const set = (patch: Partial<VGPUProfile>) => setP({ ...p, ...patch })
  const save = async () => {
    try {
      await api.upsertVgpuProfile(p)
      onSaved()
    } catch (e) {
      setError((e as Error).message)
    }
  }
  return (
    <Modal
      title={initial.profile_name ? `Edit ${initial.profile_name}` : 'Add vGPU profile'}
      onClose={onClose}
      footer={
        <>
          <Button onClick={onClose}>Cancel</Button>
          <Button variant="primary" onClick={save}>
            Save profile
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
        <Field label="Profile name">
          <Input value={p.profile_name} disabled={!!initial.profile_name} onChange={(e) => set({ profile_name: e.target.value })} />
        </Field>
        <Field label="Physical GPU">
          <Select value={p.physical_gpu} onChange={(e) => set({ physical_gpu: e.target.value })}>
            {gpus.map((g) => (
              <option key={g.id} value={g.id}>
                {g.model}
              </option>
            ))}
          </Select>
        </Field>
        <Field label="Framebuffer (GiB)">
          <NumberInput min={1} step="any" value={p.vram_gb} onChange={(v) => set({ vram_gb: v ?? 1 })} />
        </Field>
        <Field label="Max instances per GPU">
          <NumberInput min={1} value={p.max_instances_per_gpu} onChange={(v) => set({ max_instances_per_gpu: v ?? 1 })} />
        </Field>
        <Field label="Sharing mode">
          <Select value={p.sharing_mode} onChange={(e) => set({ sharing_mode: e.target.value as VGPUProfile['sharing_mode'] })}>
            <option value="time_sliced">Time-sliced</option>
            <option value="mig">MIG-backed</option>
          </Select>
        </Field>
        <Field label="MIG compute slices (of 7)">
          <NumberInput min={1} max={7} disabled={p.sharing_mode !== 'mig'} value={p.mig_compute_slices} onChange={(v) => set({ mig_compute_slices: v })} />
        </Field>
        <Field label="License cost per hour ($)">
          <NumberInput min={0} step="any" value={p.license_cost_per_hour} onChange={(v) => set({ license_cost_per_hour: v })} />
        </Field>
        <Field label="Free instances now" hint="Empty = unknown. 0 = exhausted (skipped with a reason)">
          <NumberInput min={0} value={p.available_instances} onChange={(v) => set({ available_instances: v })} />
        </Field>
        <Field label="OpenStack Placement trait" hint="e.g. CUSTOM_VGPU_L40S_24C (from your cloud config)" className="col-span-2">
          <Input value={p.openstack_trait ?? ''} onChange={(e) => set({ openstack_trait: e.target.value || null })} />
        </Field>
      </div>
      <div className="mt-3">
        <Toggle checked={p.enabled} onChange={(v) => set({ enabled: v })} label="Offer this profile (policy)" />
      </div>
    </Modal>
  )
}

function InventoryEditor({ rows, gpus, onSaved }: { rows: InventoryRecord[]; gpus: GPUSpec[]; onSaved: () => void }) {
  const [draft, setDraft] = useState<InventoryRecord[]>(rows)
  const [msg, setMsg] = useState<{ tone: 'good' | 'critical'; text: string } | null>(null)
  useEffect(() => setDraft(rows), [rows])
  const update = (i: number, patch: Partial<InventoryRecord>) => setDraft(draft.map((r, j) => (j === i ? { ...r, ...patch } : r)))
  const save = async () => {
    try {
      await api.putInventory(draft.map(({ id: _id, ...r }) => r))
      setMsg({ tone: 'good', text: 'Inventory saved' })
      onSaved()
    } catch (e) {
      setMsg({ tone: 'critical', text: (e as Error).message })
    }
  }
  return (
    <div className="space-y-3">
      <p className="text-sm text-ink-3">
        Static inventory used for availability ranking. GPUs with too few free units are penalised, not excluded. A future OpenStack adapter can read this from Placement.
      </p>
      <Table>
        <thead>
          <tr>
            <th>GPU</th>
            <th>Site / location</th>
            <th className="text-right">Installed</th>
            <th className="text-right">Available</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {draft.map((r, i) => (
            <tr key={i}>
              <td>
                <Select value={r.gpu_id} onChange={(e) => update(i, { gpu_id: e.target.value })}>
                  {gpus.map((g) => (
                    <option key={g.id} value={g.id}>
                      {g.model}
                    </option>
                  ))}
                </Select>
              </td>
              <td>
                <Input value={r.site} onChange={(e) => update(i, { site: e.target.value })} />
              </td>
              <td>
                <NumberInput min={0} className="text-right" value={r.installed} onChange={(v) => update(i, { installed: v ?? 0 })} />
              </td>
              <td>
                <NumberInput min={0} className="text-right" value={r.available} onChange={(v) => update(i, { available: v ?? 0 })} />
              </td>
              <td>
                <Button size="sm" variant="ghost" onClick={() => setDraft(draft.filter((_, j) => j !== i))}>
                  Remove
                </Button>
              </td>
            </tr>
          ))}
        </tbody>
      </Table>
      <div className="flex items-center gap-2">
        <Button onClick={() => setDraft([...draft, { gpu_id: gpus[0]?.id ?? '', site: 'default', installed: 1, available: 1 }])}>Add row</Button>
        <Button variant="primary" onClick={save}>
          Save inventory
        </Button>
      </div>
      {msg && <Alert tone={msg.tone}>{msg.text}</Alert>}
    </div>
  )
}
