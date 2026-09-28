import { describe, expect, it } from 'vitest'
import { DEFAULT_WORKLOAD } from '../state'
import { fromQuery, toQuery } from './share'

describe('share links', () => {
  it('round-trips the model and main workload fields', () => {
    const w = { ...DEFAULT_WORKLOAD, precision: 'fp8', concurrent_sequences: 7, performance_priority: 'performance' as const }
    const parsed = fromQuery(toQuery('qwen3-32b', w))
    expect(parsed?.modelId).toBe('qwen3-32b')
    expect(parsed?.patch).toMatchObject({ precision: 'fp8', concurrent_sequences: 7, performance_priority: 'performance', context_length: 32768 })
  })

  it('ignores links without a model and invalid values', () => {
    expect(fromQuery('?n=5')).toBeNull()
    expect(fromQuery('?m=x&n=-3&pr=turbo')?.patch).toEqual({})
  })
})
