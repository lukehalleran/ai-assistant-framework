import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import {
  HttpStatusError,
  NotesSyncNotStartedError,
  pollNotesSync,
  startNotesSyncAndPoll,
  type NotesSyncStatus,
} from './client'
import { __resetLaunchTokenForTests, LAUNCH_TOKEN_HEADER } from './launchAuth'

// BC-80 (docs/BUG_CLASSES.md): on 2026-09-14 and again on 2026-09-16 the
// server finished a notes sync while the SPA showed "Notes sync failed",
// because the outcome was delivered only on the POST whose connection had
// dropped. These tests drive the REAL client helpers against a scripted
// fetch: a dropped connection must never become a failure, the POST must
// never be re-sent, and only an HTTP status ends polling.

const TOKEN = 'tok-notes-sync'

function setMetaToken(value: string | null) {
  document.querySelectorAll('meta[name="daemon-launch-token"]').forEach((el) => el.remove())
  if (value !== null) {
    const meta = document.createElement('meta')
    meta.setAttribute('name', 'daemon-launch-token')
    meta.setAttribute('content', value)
    document.head.appendChild(meta)
  }
  __resetLaunchTokenForTests()
}

const NOW = 1_700_000_000
function status(over: Partial<NotesSyncStatus>): NotesSyncStatus {
  return {
    status: 'idle',
    message: null,
    error: null,
    task_id: null,
    started_at: null,
    finished_at: null,
    server_time: NOW,
    last_result: null,
    ...over,
  }
}

type Step = { body: unknown; status?: number } | { reject: Error }

/** Script fetch responses in call order and record every request. */
function scriptFetch(steps: Step[]) {
  const calls: { url: string; method: string; headers: Headers }[] = []
  const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    calls.push({
      url: String(input),
      method: init?.method ?? 'GET',
      headers: new Headers(init?.headers),
    })
    const step = steps.shift()
    if (!step) throw new Error(`unscripted fetch #${calls.length}: ${String(input)}`)
    if ('reject' in step) throw step.reject
    const code = step.status ?? 200
    return {
      ok: code < 400,
      status: code,
      statusText: 'x',
      json: async () => step.body,
    } as unknown as Response
  })
  vi.stubGlobal('fetch', fetchMock)
  return { calls, remaining: steps }
}

const RUNNING = status({ status: 'running', task_id: 't1', started_at: NOW - 1, message: 'Notes sync started' })
const DONE = status({
  status: 'succeeded',
  task_id: 't1',
  started_at: NOW - 3,
  finished_at: NOW,
  message: '✅ Synced 2 new, 1 updated notes',
})

describe('notes sync client (BC-80)', () => {
  beforeEach(() => setMetaToken(TOKEN))
  afterEach(() => {
    vi.unstubAllGlobals()
    setMetaToken(null)
  })

  it('starts once, polls the status endpoint with the launch token, returns the terminal outcome', async () => {
    const { calls, remaining } = scriptFetch([{ body: RUNNING }, { body: RUNNING }, { body: DONE }])
    const seen: string[] = []
    const outcome = await startNotesSyncAndPoll({ intervalMs: 0, onStatus: (s) => seen.push(s.status) })
    expect(outcome.status).toBe('succeeded')
    expect(outcome.message).toContain('2 new')
    expect(seen).toEqual(['running', 'running', 'succeeded'])
    expect(calls.map((c) => `${c.method} ${c.url}`)).toEqual([
      'POST /api/sync-notes',
      'GET /api/sync-notes/status',
      'GET /api/sync-notes/status',
    ])
    for (const c of calls) expect(c.headers.get(LAUNCH_TOKEN_HEADER)).toBe(TOKEN)
    expect(remaining).toEqual([])
  })

  it('a lost POST response is never re-sent: the retained state is polled and the real outcome returned', async () => {
    const { calls } = scriptFetch([
      { reject: new TypeError('Failed to fetch') },
      { body: RUNNING },
      { body: DONE },
    ])
    const transport = vi.fn()
    const outcome = await startNotesSyncAndPoll({ intervalMs: 0, onTransportError: transport })
    expect(outcome.status).toBe('succeeded')
    expect(transport).toHaveBeenCalledTimes(1)
    expect(calls.filter((c) => c.method === 'POST')).toHaveLength(1)
  })

  it('a lost POST whose retained outcome predates the request is "not started", not this sync\'s result', async () => {
    const stale = status({
      status: 'succeeded',
      task_id: 'old',
      started_at: NOW - 3600,
      finished_at: NOW - 3500,
      message: 'old run',
    })
    scriptFetch([{ reject: new TypeError('Failed to fetch') }, { body: stale }])
    await expect(startNotesSyncAndPoll({ intervalMs: 0 })).rejects.toBeInstanceOf(NotesSyncNotStartedError)
  })

  it('a server-reported failure is returned as failed (the only source of a failed outcome)', async () => {
    scriptFetch([
      { body: RUNNING },
      { body: status({ status: 'failed', task_id: 't1', started_at: NOW - 2, finished_at: NOW, message: '❌ Sync failed: boom', error: 'boom' }) },
    ])
    const outcome = await startNotesSyncAndPoll({ intervalMs: 0 })
    expect(outcome.status).toBe('failed')
    expect(outcome.error).toBe('boom')
  })

  it('a dropped status read keeps polling; the next read completes it', async () => {
    scriptFetch([{ reject: new TypeError('Failed to fetch') }, { body: DONE }])
    const transport = vi.fn()
    const outcome = await pollNotesSync({ intervalMs: 0, onTransportError: transport })
    expect(outcome.status).toBe('succeeded')
    expect(transport).toHaveBeenCalledTimes(1)
  })

  it('an HTTP status (401 after a restart) stops polling and carries the shared not-authorized wording', async () => {
    const { calls } = scriptFetch([{ body: null, status: 401 }, { body: DONE }])
    const err = await pollNotesSync({ intervalMs: 0 }).catch((e) => e)
    expect(err).toBeInstanceOf(HttpStatusError)
    expect(err.status).toBe(401)
    expect(err.message).toContain('Not authorized (401)')
    expect(calls).toHaveLength(1)
  })

  it('an "already running" start joins the running job instead of erroring', async () => {
    scriptFetch([
      { body: status({ status: 'running', task_id: 't1', started_at: NOW - 10, message: 'Notes sync already running' }) },
      { body: DONE },
    ])
    const outcome = await startNotesSyncAndPoll({ intervalMs: 0 })
    expect(outcome.status).toBe('succeeded')
  })

  it('abort stops the loop with an AbortError', async () => {
    scriptFetch([{ body: RUNNING }, { body: RUNNING }, { body: RUNNING }, { body: RUNNING }])
    const controller = new AbortController()
    const pending = pollNotesSync({ intervalMs: 5, signal: controller.signal })
    setTimeout(() => controller.abort(), 12)
    const err = await pending.catch((e) => e)
    expect(err?.name).toBe('AbortError')
  })
})
