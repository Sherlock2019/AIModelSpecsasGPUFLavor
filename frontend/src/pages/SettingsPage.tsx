import { useEffect, useState } from 'react'
import { api, getAdminToken, setAdminToken } from '../api'
import { Alert, Button, Card, Field, Input, NumberInput, PageHeader, Table } from '../components/ui'
import { useCalc } from '../state'

type Obj = Record<string, Record<string, unknown>>
type Settings = Record<string, unknown> & { precision: Obj; frameworks: Obj; priorities: Obj }

export function SettingsPage() {
  const { reloadModels } = useCalc()
  const [settings, setSettings] = useState<Settings | null>(null)
  const [json, setJson] = useState('')
  const [token, setToken] = useState(getAdminToken())
  const [msg, setMsg] = useState<{ tone: 'good' | 'critical'; text: string } | null>(null)

  const apply = (s: Record<string, unknown>) => {
    setSettings(s as Settings)
    setJson(JSON.stringify(s, null, 2))
  }
  useEffect(() => {
    api.settings().then(apply).catch((e: Error) => setMsg({ tone: 'critical', text: e.message }))
  }, [])

  const setCell = (section: 'precision' | 'frameworks' | 'priorities', key: string, field: string, value: number | null) => {
    if (!settings || value == null) return
    const next = { ...settings, [section]: { ...settings[section], [key]: { ...settings[section][key], [field]: value } } }
    apply(next)
  }

  const save = async (value: Record<string, unknown>) => {
    try {
      apply(await api.putSettings(value))
      setMsg({ tone: 'good', text: 'Settings saved. New calculations use these values.' })
      reloadModels()
    } catch (e) {
      setMsg({ tone: 'critical', text: (e as Error).message })
    }
  }

  const saveJson = () => {
    try {
      void save(JSON.parse(json))
    } catch {
      setMsg({ tone: 'critical', text: 'Invalid JSON' })
    }
  }

  const numCell = (section: 'precision' | 'frameworks' | 'priorities', key: string, field: string, step = '0.01') => (
    <NumberInput
      className="h-8 w-24 text-right"
      step={step}
      value={(settings?.[section][key]?.[field] as number) ?? null}
      onChange={(v) => setCell(section, key, field, v)}
    />
  )

  return (
    <div className="space-y-5">
      <PageHeader
        title="Settings"
        description="Every engine constant is a configurable estimate. Changes apply to new calculations for everyone using this server."
        actions={
          <>
            <Button
              onClick={async () => {
                try {
                  apply(await api.resetSettings())
                  setMsg({ tone: 'good', text: 'Settings reset to defaults.' })
                } catch (e) {
                  setMsg({ tone: 'critical', text: (e as Error).message })
                }
              }}
            >
              Reset to defaults
            </Button>
            <Button variant="primary" onClick={() => settings && save(settings)}>
              Save changes
            </Button>
          </>
        }
      />
      {msg && <Alert tone={msg.tone}>{msg.text}</Alert>}

      <Card title="Admin access">
        <div className="flex flex-wrap items-end gap-3">
          <Field label="Admin token" hint="Only required when the server sets ADMIN_TOKEN. Stored in this browser only." className="w-80">
            <Input type="password" value={token} onChange={(e) => setToken(e.target.value)} />
          </Field>
          <Button
            onClick={() => {
              setAdminToken(token)
              setMsg({ tone: 'good', text: 'Admin token saved in this browser.' })
            }}
          >
            Save token
          </Button>
        </div>
      </Card>

      {settings && (
        <>
          <Card title="Precision" eyebrow="Weight memory = parameters × bytes/param × overhead">
            <Table>
              <thead>
                <tr>
                  <th>Precision</th>
                  <th className="text-right">Bytes / parameter</th>
                  <th className="text-right">Quantization overhead ×</th>
                  <th>Compute dtype</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(settings.precision).map(([k, v]) => (
                  <tr key={k}>
                    <td className="font-medium">{String(v.label)}</td>
                    <td className="text-right">{numCell('precision', k, 'bytes_per_parameter')}</td>
                    <td className="text-right">{numCell('precision', k, 'default_overhead_factor')}</td>
                    <td>{String(v.compute).toUpperCase()}</td>
                  </tr>
                ))}
              </tbody>
            </Table>
          </Card>

          <Card title="Framework overhead" eyebrow="Configurable estimates">
            <Table>
              <thead>
                <tr>
                  <th>Framework</th>
                  <th className="text-right">Base overhead (GB / GPU)</th>
                  <th className="text-right">Workspace (% of weights + KV)</th>
                  <th className="text-right">Communication (GB / GPU, multi-GPU)</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(settings.frameworks).map(([k, v]) => (
                  <tr key={k}>
                    <td className="font-medium">{String(v.label)}</td>
                    <td className="text-right">{numCell('frameworks', k, 'base_overhead_gb', '0.1')}</td>
                    <td className="text-right">{numCell('frameworks', k, 'workspace_percent', '0.5')}</td>
                    <td className="text-right">{numCell('frameworks', k, 'communication_overhead_gb_per_gpu', '0.1')}</td>
                  </tr>
                ))}
              </tbody>
            </Table>
          </Card>

          <Card title="Performance priorities" eyebrow="Recommendation policy">
            <Table>
              <thead>
                <tr>
                  <th>Priority</th>
                  <th className="text-right">Safety margin %</th>
                  <th className="text-right">Tokens/s per request</th>
                  <th className="text-right">Comfortable utilization</th>
                  <th>vGPU sharing allowed</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(settings.priorities).map(([k, v]) => (
                  <tr key={k}>
                    <td className="font-medium">{String(v.label)}</td>
                    <td className="text-right">{numCell('priorities', k, 'safety_margin_percent', '1')}</td>
                    <td className="text-right">{numCell('priorities', k, 'tokens_per_second_per_user', '1')}</td>
                    <td className="text-right">{numCell('priorities', k, 'comfortable_utilization')}</td>
                    <td className="text-xs">{(v.allowed_vgpu_sharing as string[]).join(', ') || 'none'}</td>
                  </tr>
                ))}
              </tbody>
            </Table>
          </Card>

          <Card title="All settings (JSON)" eyebrow="Advanced" actions={<Button size="sm" onClick={saveJson}>Validate &amp; save JSON</Button>}>
            <textarea
              className="h-96 w-full rounded-lg border border-line-strong bg-surface-2 p-3 font-mono text-xs text-ink"
              spellCheck={false}
              value={json}
              onChange={(e) => setJson(e.target.value)}
            />
          </Card>
        </>
      )}
    </div>
  )
}
