import { describe, expect, it } from 'vitest'
import { clampContext, contextOptionsFor, fmtContext, fmtMem, fmtParams } from './format'

describe('format', () => {
  it('formats context lengths', () => {
    expect(fmtContext(32768)).toBe('32K')
    expect(fmtContext(131072)).toBe('128K')
    expect(fmtContext(1048576)).toBe('1M')
    expect(fmtContext(5000)).toBe('5,000')
  })

  it('converts GB to GiB for display', () => {
    expect(fmtMem(80, 'GB')).toBe('80.0 GB')
    expect(fmtMem(1.073741824, 'GiB', 2)).toBe('1.00 GiB')
    expect(fmtMem(null)).toBe('—')
  })

  it('formats parameter counts', () => {
    expect(fmtParams(70.55)).toBe('70.5B')
    expect(fmtParams(405.85)).toBe('406B')
    expect(fmtParams(0.6)).toBe('600M')
  })

  it('limits context options to the model maximum', () => {
    expect(contextOptionsFor(32768).at(-1)).toBe(32768)
    expect(contextOptionsFor(202752)).toContain(202752)
    expect(clampContext(131072, 32768)).toBe(32768)
    expect(clampContext(8192, 32768)).toBe(8192)
  })
})
