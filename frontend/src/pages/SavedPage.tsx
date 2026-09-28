import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '../api'
import { Alert, Button, Empty, PageHeader, Table } from '../components/ui'
import { fmtMem } from '../lib/format'
import { CUSTOM_MODEL_ID, DEFAULT_WORKLOAD, EMPTY_CUSTOM_MODEL, useCalc } from '../state'
import type { SavedCalculationSummary, WorkloadInput } from '../types'

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
      navigate('/')
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
    </div>
  )
}
