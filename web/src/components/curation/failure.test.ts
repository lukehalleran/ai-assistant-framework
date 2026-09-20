import { afterEach, describe, expect, it, vi } from 'vitest'
import { api, HttpStatusError } from '../../api/client'
import { describeCurationFailure } from './failure'

// BC-80 (docs/BUG_CLASSES.md): the Curation Center's apply/dismiss/undo/scan
// run in a server worker that outlives the HTTP request — a dropped
// connection must never be reported as "<verb> failed" (the same
// false-failure shape fixed for notes sync on 2026-09-16). describeCurationFailure
// is the classifier; these tests drive it directly and, for the HTTP-status
// leg, through the real client.ts curation call.

describe('describeCurationFailure', () => {
  it('an HttpStatusError with a server detail is a real failure carrying that detail', () => {
    const err = new HttpStatusError(409, 'Apply', 'Proposal already applied')
    const result = describeCurationFailure(err, 'Apply')
    expect(result).toEqual({
      lost: false,
      title: 'Apply failed',
      message: 'Proposal already applied',
      color: 'red',
    })
  })

  it('an HttpStatusError with no server detail falls back to its own message', () => {
    const err = new HttpStatusError(500, 'Dismiss')
    const result = describeCurationFailure(err, 'Dismiss')
    expect(result.lost).toBe(false)
    expect(result.title).toBe('Dismiss failed')
    expect(result.message).toBe(err.message)
    expect(result.color).toBe('red')
  })

  it('a fetch TypeError (dropped connection) is a lost response, never "<verb> failed"', () => {
    const result = describeCurationFailure(new TypeError('Failed to fetch'), 'Undo')
    expect(result.lost).toBe(true)
    expect(result.title).toBe('Connection lost')
    expect(result.title).not.toContain('failed')
    expect(result.color).toBe('yellow')
    expect(result.message).toContain('Undo')
  })

  it('an AbortError is a lost response', () => {
    const result = describeCurationFailure(new DOMException('Aborted', 'AbortError'), 'Scan')
    expect(result.lost).toBe(true)
    expect(result.title).toBe('Connection lost')
    expect(result.color).toBe('yellow')
  })

  it('a plain Error (no HTTP status attached) is treated as lost, not a reported failure', () => {
    const result = describeCurationFailure(new Error('boom'), 'Apply')
    expect(result.lost).toBe(true)
    expect(result.title).toBe('Connection lost')
  })
})

describe('client.ts curation calls (BC-80 detail plumbing)', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('a 409 apply response rejects with HttpStatusError carrying the server detail', async () => {
    const fetchMock = vi.fn(async () => ({
      ok: false,
      status: 409,
      statusText: 'Conflict',
      json: async () => ({ detail: 'Proposal already applied' }),
    }))
    vi.stubGlobal('fetch', fetchMock)

    const err: unknown = await api.applyCurationProposal('p1').catch((e) => e)
    expect(err).toBeInstanceOf(HttpStatusError)
    expect((err as HttpStatusError).status).toBe(409)
    expect((err as HttpStatusError).detail).toBe('Proposal already applied')
    expect((err as HttpStatusError).message).toBe('Proposal already applied')

    const described = describeCurationFailure(err, 'Apply')
    expect(described).toEqual({
      lost: false,
      title: 'Apply failed',
      message: 'Proposal already applied',
      color: 'red',
    })
  })

  it('a rejected fetch (connection dropped) on undo is NOT an HttpStatusError', async () => {
    const fetchMock = vi.fn(async () => {
      throw new TypeError('Failed to fetch')
    })
    vi.stubGlobal('fetch', fetchMock)

    const err: unknown = await api.undoCurationProposal('p1').catch((e) => e)
    expect(err).not.toBeInstanceOf(HttpStatusError)
    expect(describeCurationFailure(err, 'Undo').lost).toBe(true)
  })
})
