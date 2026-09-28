export type Unit = 'GB' | 'GiB'

const GIB_PER_GB = 1e9 / 1024 ** 3

/** Values arrive from the API in decimal GB; convert on display only. */
export function fmtMem(gb: number | null | undefined, unit: Unit = 'GB', digits = 1): string {
  if (gb == null || Number.isNaN(gb)) return '—'
  const v = unit === 'GiB' ? gb * GIB_PER_GB : gb
  return `${v.toLocaleString(undefined, { minimumFractionDigits: digits, maximumFractionDigits: digits })} ${unit}`
}

export function fmtParams(b: number | null | undefined): string {
  if (b == null) return '—'
  if (b >= 1) return `${Number(b.toFixed(b >= 100 ? 0 : 1)).toLocaleString()}B`
  return `${Math.round(b * 1000)}M`
}

/** 32768 -> "32K", 131072 -> "128K", 1048576 -> "1M", 5000 -> "5,000". */
export function fmtContext(tokens: number | null | undefined): string {
  if (tokens == null) return '—'
  const M = 1024 * 1024
  if (tokens >= M && tokens % M === 0) return `${tokens / M}M`
  if (tokens >= 1024 && tokens % 1024 === 0) return `${tokens / 1024}K`
  return tokens.toLocaleString()
}

export function fmtNum(n: number | null | undefined, digits = 0): string {
  if (n == null || Number.isNaN(n)) return '—'
  return n.toLocaleString(undefined, { minimumFractionDigits: digits, maximumFractionDigits: digits })
}

export function fmtPct(n: number | null | undefined, digits = 0): string {
  return n == null ? '—' : `${fmtNum(n, digits)}%`
}

export function fmtMoney(n: number | null | undefined, digits = 2): string {
  return n == null ? '—' : `$${fmtNum(n, digits)}`
}

export function titleCase(s: string): string {
  return s.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())
}

export const CONTEXT_OPTIONS = [2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 1048576]

/** Context choices available for a model (always includes its maximum). */
export function contextOptionsFor(max: number | null | undefined): number[] {
  if (!max) return CONTEXT_OPTIONS
  const opts = CONTEXT_OPTIONS.filter((c) => c <= max)
  if (!opts.includes(max)) opts.push(max)
  return opts
}

/** Keep the chosen context if valid, otherwise the largest option <= preferred. */
export function clampContext(current: number, max: number | null | undefined): number {
  if (!max || current <= max) return current
  const opts = contextOptionsFor(max)
  return opts[opts.length - 1]
}
