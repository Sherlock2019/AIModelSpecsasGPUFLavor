import { describe, expect, it } from 'vitest'
import type { Candidate } from '../types'
import { candidateTitle, memoryFit, modeLabel, performanceFit, whyNot } from './plain'

const base: Candidate = {
  key: 'full_gpu:nvidia-h100-80gb:1',
  kind: 'full_gpu',
  gpu_id: 'nvidia-h100-80gb',
  gpu_name: 'NVIDIA H100 80GB',
  gpu_vendor: 'NVIDIA',
  gpu_compute_class: 'very_high',
  gpu_bandwidth_class: 'very_high',
  count: 1,
  tensor_parallel: 1,
  vram_per_unit_gb: 85.5,
  required_per_unit_gb: 68.4,
  memory_per_unit: {
    model_weights_gb: 38.8,
    kv_cache_gb: 16.4,
    runtime_overhead_gb: 1.5,
    workspace_gb: 2.8,
    communication_gb: 0,
    subtotal_gb: 59.5,
    headroom_gb: 8.9,
    required_vram_gb: 68.4,
    safety_margin_percent: 15,
    activations_gb: 0,
    gradients_gb: 0,
    optimizer_gb: 0,
  },
  utilization_percent: 80,
  headroom_gb: 17.1,
  fits: true,
  fits_weights_only: true,
  gpu_units: 1,
  bandwidth_ratio: 2.1,
  compute_ratio: 1.3,
  availability: 'unknown',
  confidence: 'estimated',
  notes: [],
  label: '1 × NVIDIA H100 80GB',
}

describe('plain-language result helpers', () => {
  it('separates memory fit from performance fit', () => {
    expect(memoryFit(base)).toEqual({ tone: 'good', label: 'Fits' })
    expect(memoryFit({ ...base, utilization_percent: 96 }).tone).toBe('warning')
    expect(memoryFit({ ...base, fits: false }).tone).toBe('critical')
    expect(performanceFit(base)).toMatchObject({ tone: 'good', basis: 'Estimated' })
    expect(performanceFit({ ...base, bandwidth_ratio: 0.7 }).label).toBe('May be slower than target')
    expect(performanceFit({ ...base, bandwidth_ratio: null, compute_ratio: null }).label).toBe('Unknown')
    expect(performanceFit({ ...base, confidence: 'benchmark' }).basis).toBe('Measured')
  })

  it('uses product language for titles and modes', () => {
    const multi = { ...base, kind: 'multi_gpu' as const, count: 2, gpu_name: 'NVIDIA L40S' }
    expect(candidateTitle(multi)).toBe('2 × NVIDIA L40S')
    expect(modeLabel(multi)).toBe('Model split across 2 GPUs')
    expect(candidateTitle({ ...base, kind: 'vgpu', vgpu_profile: 'L40S-16C', gpu_name: 'NVIDIA L40S' })).toBe('NVIDIA L40S · L40S-16C shared slice')
  })

  it('explains why a bigger GPU ranked lower', () => {
    const h200 = { ...base, gpu_name: 'NVIDIA H200 141GB', vram_per_unit_gb: 150.8, utilization_percent: 45, gpu_compute_class: 'very_high' }
    const reasons = whyNot(h200, base, 'GB').map((r) => r.text)
    expect(reasons.some((t) => t.startsWith('Significant unused memory'))).toBe(true)
    const twoL40s = { ...base, kind: 'multi_gpu' as const, count: 2, gpu_bandwidth_class: 'medium' }
    expect(whyNot(twoL40s, base, 'GB').map((r) => r.text)).toContain('Needs 2 GPUs and a model split')
  })
})
