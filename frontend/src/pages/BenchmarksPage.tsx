import { useEffect, useState } from 'react'
import { api } from '../api'
import { Modal } from '../components/Modal'
import { Alert, Badge, Button, Empty, Field, Input, NumberInput, PageHeader, Select, Table, Toggle } from '../components/ui'
import { fmtNum } from '../lib/format'
import { useCalc } from '../state'
import type { BenchmarkRecord, GPUSpec } from '../types'

const EMPTY: BenchmarkRecord = {
  model_id: '',
  gpu_id: '',
  gpu_count: 1,
  precision: 'int4',
  framework: 'vllm',
  tensor_parallel: 1,
  context_length: 8192,
  concurrency: 8,
  source: '',
  verified: false,
}

export function BenchmarksPage() {
  const { models, meta } = useCalc()
  const [rows, setRows] = useState<BenchmarkRecord[]>([])
  const [gpus, setGpus] = useState<GPUSpec[]>([])
  const [adding, setAdding] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const load = () =>
    Promise.all([api.benchmarks(), api.gpus()])
      .then(([b, g]) => {
        setRows(b)
        setGpus(g)
      })
      .catch((e: Error) => setError(e.message))
  useEffect(() => {
    void load()
  }, [])

  const modelName = (id: string) => models.find((m) => m.id === id)?.name ?? id
  const gpuName = (id: string) => gpus.find((g) => g.id === id)?.model ?? id

  return (
    <div>
      <PageHeader
        title="Benchmarks"
        description="Measured results. When a benchmark matches the model, GPU, GPU count and precision, recommendations use it and are labelled BENCHMARK-BASED. Otherwise they fall back to estimate mode."
        actions={
          <Button variant="primary" onClick={() => setAdding(true)}>
            Add benchmark
          </Button>
        }
      />
      {error && (
        <div className="mb-4">
          <Alert tone="critical">{error}</Alert>
        </div>
      )}
      {rows.length === 0 ? (
        <Empty>
          No benchmark data yet. All recommendations are currently <strong>estimated</strong>. Add measured results from your own
          load tests (for example vLLM <code>benchmark_serving</code>) to enable benchmark mode.
        </Empty>
      ) : (
        <Table>
          <thead>
            <tr>
              <th>Model</th>
              <th>GPU</th>
              <th>Precision</th>
              <th>Framework</th>
              <th className="text-right">Context</th>
              <th className="text-right">Concurrency</th>
              <th className="text-right">Decode tok/s</th>
              <th className="text-right">TTFT ms</th>
              <th className="text-right">TPOT ms</th>
              <th className="text-right">Peak VRAM</th>
              <th>Source</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {rows.map((b) => (
              <tr key={b.id}>
                <td className="font-medium">{modelName(b.model_id)}</td>
                <td>
                  {b.gpu_count} × {gpuName(b.gpu_id)}
                </td>
                <td>{b.precision.toUpperCase()}</td>
                <td>{b.framework}</td>
                <td className="text-right">{fmtNum(b.context_length)}</td>
                <td className="text-right">{b.concurrency}</td>
                <td className="text-right">{fmtNum(b.decode_tokens_per_second ?? b.aggregate_tokens_per_second)}</td>
                <td className="text-right">{fmtNum(b.ttft_ms)}</td>
                <td className="text-right">{fmtNum(b.tpot_ms, 1)}</td>
                <td className="text-right">{b.peak_vram_gb != null ? `${b.peak_vram_gb} GB` : '—'}</td>
                <td>
                  {b.source} {b.verified && <Badge tone="good">✓ Verified</Badge>}
                  {b.date && <div className="text-xs text-ink-3">{b.date}</div>}
                </td>
                <td>
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={async () => {
                      try {
                        await api.deleteBenchmark(b.id!)
                        void load()
                      } catch (e) {
                        setError((e as Error).message)
                      }
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
      {adding && (
        <BenchmarkForm
          gpus={gpus}
          models={models.map((m) => ({ id: m.id, name: m.name }))}
          precisions={meta?.precisions.map((p) => p.id) ?? ['int4', 'int8', 'fp8', 'bf16', 'fp16']}
          frameworks={meta?.frameworks ?? [{ id: 'vllm', label: 'vLLM' }]}
          onClose={() => setAdding(false)}
          onSaved={() => {
            setAdding(false)
            void load()
          }}
        />
      )}
    </div>
  )
}

function BenchmarkForm({
  gpus,
  models,
  precisions,
  frameworks,
  onClose,
  onSaved,
}: {
  gpus: GPUSpec[]
  models: { id: string; name: string }[]
  precisions: string[]
  frameworks: { id: string; label: string }[]
  onClose: () => void
  onSaved: () => void
}) {
  const [b, setB] = useState<BenchmarkRecord>({ ...EMPTY, model_id: models[0]?.id ?? '', gpu_id: gpus[0]?.id ?? '' })
  const [error, setError] = useState<string | null>(null)
  const set = (patch: Partial<BenchmarkRecord>) => setB({ ...b, ...patch })
  const num = (key: keyof BenchmarkRecord, label: string) => (
    <Field label={label}>
      <NumberInput min={0} step="any" value={b[key] as number | null} onChange={(v) => set({ [key]: v } as Partial<BenchmarkRecord>)} />
    </Field>
  )
  const save = async () => {
    try {
      await api.createBenchmark(b)
      onSaved()
    } catch (e) {
      setError((e as Error).message)
    }
  }
  return (
    <Modal
      title="Add benchmark result"
      onClose={onClose}
      footer={
        <>
          <Button onClick={onClose}>Cancel</Button>
          <Button variant="primary" onClick={save} disabled={!b.source.trim()}>
            Save benchmark
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
        <Field label="Model">
          <Select value={b.model_id} onChange={(e) => set({ model_id: e.target.value })}>
            {models.map((m) => (
              <option key={m.id} value={m.id}>
                {m.name}
              </option>
            ))}
          </Select>
        </Field>
        <Field label="GPU">
          <Select value={b.gpu_id} onChange={(e) => set({ gpu_id: e.target.value })}>
            {gpus.map((g) => (
              <option key={g.id} value={g.id}>
                {g.model}
              </option>
            ))}
          </Select>
        </Field>
        {num('gpu_count', 'GPU count')}
        {num('tensor_parallel', 'Tensor parallel')}
        <Field label="Precision">
          <Select value={b.precision} onChange={(e) => set({ precision: e.target.value })}>
            {precisions.map((p) => (
              <option key={p} value={p}>
                {p.toUpperCase()}
              </option>
            ))}
          </Select>
        </Field>
        <Field label="Framework">
          <Select value={b.framework} onChange={(e) => set({ framework: e.target.value })}>
            {frameworks.map((f) => (
              <option key={f.id} value={f.id}>
                {f.label}
              </option>
            ))}
          </Select>
        </Field>
        {num('context_length', 'Context length')}
        {num('concurrency', 'Concurrency')}
        {num('batch_size', 'Batch size')}
        {num('prefill_tokens_per_second', 'Prefill tokens/s')}
        {num('decode_tokens_per_second', 'Decode tokens/s (aggregate)')}
        {num('aggregate_tokens_per_second', 'Total tokens/s')}
        {num('ttft_ms', 'TTFT (ms)')}
        {num('tpot_ms', 'TPOT (ms)')}
        {num('peak_vram_gb', 'Peak VRAM (GB)')}
        {num('average_gpu_utilization', 'Avg GPU utilization (%)')}
        {num('power_watts', 'Power (W)')}
        <Field label="Date">
          <Input type="date" value={b.date ?? ''} onChange={(e) => set({ date: e.target.value || null })} />
        </Field>
        <Field label="Source" className="col-span-2">
          <Input placeholder="e.g. internal load test 2026-09, vLLM 0.11 benchmark_serving" value={b.source} onChange={(e) => set({ source: e.target.value })} />
        </Field>
      </div>
      <div className="mt-3">
        <Toggle checked={b.verified} onChange={(v) => set({ verified: v })} label="Verified measurement" />
      </div>
    </Modal>
  )
}
