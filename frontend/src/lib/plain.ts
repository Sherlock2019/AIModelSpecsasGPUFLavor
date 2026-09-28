// Plain-language wording for results, so non-GPU experts can read them.
import type { Candidate } from '../types'
import { fmtMem, type Unit } from './format'

export type Tone = 'good' | 'warning' | 'critical' | 'neutral'

const CLASS_ORDER = ['low', 'medium', 'high', 'very_high', 'extreme']

export function classLabel(c: string | null | undefined): string {
  if (!c) return 'Unknown'
  return c.replace('_', ' ').replace(/^\w/, (x) => x.toUpperCase())
}

export function classRank(c: string | null | undefined): number {
  return c ? CLASS_ORDER.indexOf(c) : -1
}

/** "Does it fit in GPU memory?" */
export function memoryFit(c: Candidate): { tone: Tone; label: string } {
  if (!c.fits) return { tone: 'critical', label: 'Does not fit' }
  if (c.utilization_percent > 92) return { tone: 'warning', label: 'Tight fit' }
  return { tone: 'good', label: 'Fits' }
}

/** "Will it run fast enough?" Estimated unless a benchmark matched. */
export function performanceFit(c: Candidate): { tone: Tone; label: string; basis: string } {
  const basis = c.confidence === 'benchmark' ? 'Measured' : 'Estimated'
  const ratios = [c.bandwidth_ratio, c.compute_ratio].filter((r): r is number => r != null)
  if (!ratios.length) return { tone: 'neutral', label: 'Unknown', basis }
  const worst = Math.min(...ratios)
  if (worst >= 1) return { tone: 'good', label: 'Likely fast enough', basis }
  if (worst >= 0.6) return { tone: 'warning', label: 'May be slower than target', basis }
  return { tone: 'critical', label: 'Likely too slow', basis }
}

/** Card title in product language: "NVIDIA H100 80GB", "2 × NVIDIA L40S", "NVIDIA L40S · 16 GB shared slice". */
export function candidateTitle(c: Candidate): string {
  if (c.kind === 'vgpu') return `${c.gpu_name} · ${c.vgpu_profile} shared slice`
  return c.count > 1 ? `${c.count} × ${c.gpu_name}` : c.gpu_name
}

export function modeLabel(c: Candidate): string {
  if (c.kind === 'vgpu') return c.sharing_mode === 'mig' ? 'Shared GPU (hardware partition)' : 'Shared GPU (vGPU)'
  if (c.kind === 'multi_gpu') return `Model split across ${c.count} GPUs`
  return 'Full GPU'
}

/** Reasons an option ranked below the recommendation (for "Why not …?"). */
export function whyNot(alt: Candidate, rec: Candidate, unit: Unit): { ok: boolean; text: string }[] {
  const out: { ok: boolean; text: string }[] = []
  if (alt.kind === 'vgpu' && rec.kind !== 'vgpu') out.push({ ok: false, text: 'Shared GPU: performance is not guaranteed' })
  if (alt.count > rec.count && alt.kind !== 'vgpu') {
    out.push({ ok: false, text: `Needs ${alt.count} GPUs and a model split` })
  }
  if (alt.utilization_percent > 92) out.push({ ok: false, text: 'Tight fit: little memory headroom' })
  const unused = alt.vram_per_unit_gb * alt.count - alt.required_per_unit_gb * alt.count
  if (alt.utilization_percent < rec.utilization_percent - 15 && unused > 10) {
    out.push({ ok: false, text: `Significant unused memory (~${fmtMem(unused, unit, 0)} idle)` })
  }
  if (classRank(alt.gpu_bandwidth_class) > classRank(rec.gpu_bandwidth_class)) out.push({ ok: true, text: 'Faster GPU memory' })
  if (classRank(alt.gpu_compute_class) > classRank(rec.gpu_compute_class)) out.push({ ok: true, text: 'More AI compute' })
  if ((alt.bandwidth_ratio ?? 1) < 1 && (rec.bandwidth_ratio ?? 1) >= 1) out.push({ ok: false, text: 'GPU memory speed may limit response speed' })
  if (alt.cost && rec.cost && alt.cost.hourly > rec.cost.hourly) out.push({ ok: false, text: 'Higher expected cost' })
  if (!alt.cost && !rec.cost && classRank(alt.gpu_compute_class) > classRank(rec.gpu_compute_class)) {
    out.push({ ok: false, text: 'Likely higher cost (higher-end GPU)' })
  }
  if (['insufficient', 'not_in_inventory', 'unavailable'].includes(alt.availability) && rec.availability === 'available') {
    out.push({ ok: false, text: 'Not available in current inventory' })
  }
  return out
}
