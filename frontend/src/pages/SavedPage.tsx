import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '../api'
import { Modal } from '../components/Modal'
import { Alert, Badge, Button, Empty, PageHeader, Table } from '../components/ui'
import { downloadText } from '../lib/download'
import { fmtMem } from '../lib/format'
import { CUSTOM_MODEL_ID, DEFAULT_WORKLOAD, EMPTY_CUSTOM_MODEL, useCalc } from '../state'
import type { DeploymentSpecRecord, SavedCalculationSummary, WorkloadInput } from '../types'

export function SavedPage() {
  const { setForm, setResult } = useCalc()
  const navigate = useNavigate()
  const [rows, setRows] = useState<SavedCalculationSummary[]>([])
  const [error, setError] = useState<string | null>(null)
  const load = () =>
    api
      .calculations()
      .then(setRows)
      .catch((e: Error) => setError(e.message))
  useEffect(() => {
    void load()
  }, [])

  const open = async (id: number) => {
    try {
      const saved = await api.calculation(id)
      const { model_id, custom_model, ...workload } = saved.request
      setForm({
        modelId: custom_model ? CUSTOM_MODEL_ID : (model_id ?? ''),
        custom: custom_model ?? EMPTY_CUSTOM_MODEL,
        workload: { ...DEFAULT_WORKLOAD, ...(workload as Partial<WorkloadInput>) },
      })
      setResult(saved.result, saved.request)
      navigate('/calculator')
    } catch (e) {
      setError((e as Error).message)
    }
  }

  return (
    <div>
      <PageHeader title="Saved calculations" description="Snapshots of past calculations. Open one to reload its inputs and results." />
      {error && (
        <div className="mb-4">
          <Alert tone="critical">{error}</Alert>
        </div>
      )}
      {rows.length === 0 ? (
        <Empty>No saved calculations yet. Run a calculation and use Save at the bottom of the results.</Empty>
      ) : (
        <Table>
          <thead>
            <tr>
              <th>Name</th>
              <th>Model</th>
              <th>Precision</th>
              <th className="text-right">Required VRAM</th>
              <th>Recommendation</th>
              <th>AI flavor</th>
              <th>Saved</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.id}>
                <td className="font-medium">{r.name}</td>
                <td>{r.model}</td>
                <td>{r.precision?.toUpperCase()}</td>
                <td className="text-right">{fmtMem(r.required_vram_gb)}</td>
                <td>{r.recommended ?? '—'}</td>
                <td className="font-mono text-xs">{r.ai_flavor ?? '—'}</td>
                <td className="text-xs text-ink-3">{new Date(r.created_at).toLocaleString()}</td>
                <td className="text-right whitespace-nowrap">
                  <Button size="sm" onClick={() => open(r.id)}>
                    Open
                  </Button>{' '}
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={async () => {
                      await api.deleteCalculation(r.id)
                      void load()
                    }}
                  >
                    Delete
                  </Button>
                </td>
              </tr>
            ))}
          </tbody>
        </Table>
      )}
      <DeploymentSpecs />
    </div>
  )
}

/** AI VM / ML deployment specs created with "Deploy" (nothing is provisioned). */
function DeploymentSpecs() {
  const [specs, setSpecs] = useState<DeploymentSpecRecord[]>([])
  const [open, setOpen] = useState<DeploymentSpecRecord | null>(null)
  const [error, setError] = useState<string | null>(null)
  const load = () =>
    api
      .deploymentSpecs()
      .then(setSpecs)
      .catch((e: Error) => setError(e.message))
  useEffect(() => {
    void load()
  }, [])
  return (
    <section className="mt-10">
      <h2 className="mb-1 text-xl font-semibold text-ink">Deployment specs</h2>
      <p className="mb-4 text-sm text-ink-2">Created by “Deploy” on an AI flavor or custom ML result. Each includes the compiled OpenStack flavor recommendation; nothing is provisioned.</p>
      {error && <Alert tone="critical">{error}</Alert>}
      {specs.length === 0 ? (
        <Empty>No deployment specs yet.</Empty>
      ) : (
        <Table>
          <thead>
            <tr>
              <th>#</th>
              <th>AI flavor</th>
              <th>Workload</th>
              <th>GPU</th>
              <th>OpenStack flavor</th>
              <th>Needs config</th>
              <th>Created</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {specs.map((s) => (
              <tr key={s.id}>
                <td>{s.id}</td>
                <td>
                  <div className="font-mono text-xs font-semibold">{s.ai_flavor}</div>
                  <div className="text-xs text-ink-3">{s.name}</div>
                </td>
                <td>{s.workload_type === 'custom_ml' ? 'Custom ML' : 'LLM'}</td>
                <td className="text-xs">
                  {String(s.spec.gpu.count)} × {s.spec.gpu.physical_gpu}
                  {s.spec.gpu.vgpu_profile ? ` (${s.spec.gpu.vgpu_profile})` : ''}
                </td>
                <td className="font-mono text-xs">{s.openstack.name}</td>
                <td>{s.openstack.requires_configuration.length ? <Badge tone="warning">{s.openstack.requires_configuration.length}</Badge> : <Badge tone="good">None</Badge>}</td>
                <td className="text-xs text-ink-3">{new Date(s.created_at).toLocaleString()}</td>
                <td className="text-right whitespace-nowrap">
                  <Button size="sm" onClick={() => setOpen(s)}>
                    View
                  </Button>{' '}
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={async () => {
                      await api.deleteDeploymentSpec(s.id)
                      void load()
                    }}
                  >
                    Delete
                  </Button>
                </td>
              </tr>
            ))}
          </tbody>
        </Table>
      )}
      {open && (
        <Modal
          title={`Deployment spec #${open.id} · ${open.ai_flavor}`}
          onClose={() => setOpen(null)}
          footer={
            <Button onClick={() => downloadText(`${open.openstack.name}.json`, JSON.stringify(open, null, 2), 'application/json')}>Download JSON</Button>
          }
        >
          <div className="space-y-3">
            {open.openstack.requires_configuration.length > 0 && (
              <Alert tone="warning">
                Configure before creating the Nova flavor:
                <ul className="list-disc pl-5">
                  {open.openstack.requires_configuration.map((n) => (
                    <li key={n}>{n}</li>
                  ))}
                </ul>
              </Alert>
            )}
            <div className="text-xs font-semibold text-ink-3 uppercase">OpenStack flavor</div>
            <pre className="overflow-x-auto rounded-lg bg-surface-2 p-3 font-mono text-xs text-ink">{JSON.stringify(open.openstack, null, 2)}</pre>
            <div className="text-xs font-semibold text-ink-3 uppercase">Deployment spec</div>
            <pre className="overflow-x-auto rounded-lg bg-surface-2 p-3 font-mono text-xs text-ink">{JSON.stringify(open.spec, null, 2)}</pre>
          </div>
        </Modal>
      )}
    </section>
  )
}
