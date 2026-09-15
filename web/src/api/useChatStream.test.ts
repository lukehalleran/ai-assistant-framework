import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { act, renderHook, waitFor } from '@testing-library/react'
import { fetchEventSource } from '@microsoft/fetch-event-source'
import { useChatStream } from './useChatStream'
import { __resetLaunchTokenForTests, LAUNCH_TOKEN_HEADER } from './launchAuth'

// F01/G06-T02 (A02): mock the library, not the hook, to capture a chat POST
// without hanging on an open stream. EventStreamContentType (R2) is a real
// value the hook imports for its onopen content-type check.
vi.mock('@microsoft/fetch-event-source', () => ({
  fetchEventSource: vi.fn(),
  EventStreamContentType: 'text/event-stream',
}))

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

// R2 test helper: a minimal fake Response for the onopen check below.
type FesInit = { onopen?: (r: Response) => Promise<void>; onerror?: (err: unknown) => void }
function fakeResponse(status: number, contentType: string): Response {
  return { ok: status < 400, status, headers: new Headers({ 'content-type': contentType }) } as unknown as Response
}

// Audit F07 (docs/HANDOFF_20260909_independent_bug_audit.md): App used to
// unconditionally clear `pendingActionId` after every decision. These tests
// drive the REAL hook/reducer (not a re-implementation) to prove the new
// `setPendingAction` transition — used by App to chain to the backend's
// `outcome.next_action_id` — actually updates state, and that clearing on a
// final item (`next_action_id: null`) still works.

describe('useChatStream pending-action chaining', () => {
  beforeEach(() => {
    // The hook fetches /api/session on mount to restore a refreshed page;
    // keep it inert so the initial state stays deterministic for this test.
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({ ok: false, status: 404, json: async () => ({}) }),
    )
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('starts with no pending action', async () => {
    const { result } = renderHook(() => useChatStream())
    await waitFor(() => expect(result.current.pendingActionId).toBeNull())
  })

  it('setPendingAction hands the card the next chained action id', async () => {
    const { result } = renderHook(() => useChatStream())
    await waitFor(() => expect(result.current.pendingActionId).toBeNull())

    act(() => {
      result.current.setPendingAction('second')
    })

    expect(result.current.pendingActionId).toBe('second')
  })

  it('setPendingAction(null) clears it, as on the final item of a chain', async () => {
    const { result } = renderHook(() => useChatStream())
    await waitFor(() => expect(result.current.pendingActionId).toBeNull())

    act(() => {
      result.current.setPendingAction('second')
    })
    expect(result.current.pendingActionId).toBe('second')

    act(() => {
      result.current.setPendingAction(null)
    })
    expect(result.current.pendingActionId).toBeNull()
  })

  it('clearPendingAction remains a working alias for clearing (back-compat)', async () => {
    const { result } = renderHook(() => useChatStream())
    await waitFor(() => expect(result.current.pendingActionId).toBeNull())

    act(() => {
      result.current.setPendingAction('x')
    })
    act(() => {
      result.current.clearPendingAction()
    })
    expect(result.current.pendingActionId).toBeNull()
  })
})

// F01/G06-T02 (A02): every /api/* request needs `X-Daemon-Launch-Token`.
// Drives the real hook against a stubbed `fetch` and mocked
// `@microsoft/fetch-event-source` — never a re-implementation of the hook.
describe('useChatStream launch-token transport (F01/G06-T02, A02)', () => {
  beforeEach(() => {
    vi.mocked(fetchEventSource).mockReset()
    // Default for the SSE tests; tests that care about session-restore override both.
    setMetaToken('sse-token')
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({ ok: false, status: 404, json: async () => ({}) }),
    )
  })

  afterEach(() => {
    vi.unstubAllGlobals()
    setMetaToken(null)
  })

  it('mount-time session restore carries the launch token header', async () => {
    setMetaToken('restore-token')
    const fetchMock = vi.fn().mockResolvedValue({ ok: false, status: 404, json: async () => ({}) })
    vi.stubGlobal('fetch', fetchMock)

    renderHook(() => useChatStream())

    await waitFor(() => expect(fetchMock).toHaveBeenCalled())
    const [url, init] = fetchMock.mock.calls[0]
    expect(url).toBe('/api/session')
    const headers = new Headers(init?.headers)
    expect(headers.get(LAUNCH_TOKEN_HEADER)).toBe('restore-token')
  })

  it('missing meta tag: mount restore sends no token header, and a 401 dispatches a visible error', async () => {
    setMetaToken(null)
    const fetchMock = vi.fn().mockResolvedValue({
      ok: false,
      status: 401,
      json: async () => ({ detail: 'nope' }),
    })
    vi.stubGlobal('fetch', fetchMock)

    const { result } = renderHook(() => useChatStream())

    await waitFor(() => expect(result.current.error).not.toBeNull())
    const [, init] = fetchMock.mock.calls[0]
    const headers = new Headers(init?.headers)
    expect(headers.has(LAUNCH_TOKEN_HEADER)).toBe(false)
    // R2: same shared, token-free "reload" wording as the SSE chat 401 below.
    expect(result.current.error).toContain('401')
    expect(result.current.error?.toLowerCase()).toContain('reload')
  })

  it('clearAll carries the launch token header on DELETE /api/session', async () => {
    setMetaToken('clear-token')
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce({ ok: false, status: 404, json: async () => ({}) }) // mount restore
      .mockResolvedValueOnce({ ok: true, status: 200, json: async () => ({}) }) // clearAll
    vi.stubGlobal('fetch', fetchMock)

    const { result } = renderHook(() => useChatStream())
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1))

    await act(async () => {
      await result.current.clearAll()
    })

    expect(fetchMock).toHaveBeenCalledTimes(2)
    const [url, init] = fetchMock.mock.calls[1]
    expect(url).toBe('/api/session')
    expect(init?.method).toBe('DELETE')
    const headers = new Headers(init?.headers)
    expect(headers.get(LAUNCH_TOKEN_HEADER)).toBe('clear-token')
  })

  // Also R2's control case: a real 200 text/event-stream onopen must not throw.
  it('send() attaches the launch token header to the /api/chat SSE POST; onopen accepts a valid stream', async () => {
    let onopenError: unknown
    vi.mocked(fetchEventSource).mockImplementation(async (_url, init) => {
      try {
        await (init as FesInit).onopen?.(fakeResponse(200, 'text/event-stream'))
      } catch (e) {
        onopenError = e
      }
    })

    const { result } = renderHook(() => useChatStream())
    await waitFor(() => expect(result.current.pendingActionId).toBeNull())

    await act(async () => {
      await result.current.send({ text: 'hi' })
    })

    expect(fetchEventSource).toHaveBeenCalledTimes(1)
    const [url, init] = vi.mocked(fetchEventSource).mock.calls[0]
    expect(url).toBe('/api/chat')
    const headers = new Headers((init as { headers?: Record<string, string> }).headers)
    expect(headers.get(LAUNCH_TOKEN_HEADER)).toBe('sse-token')
    expect(onopenError).toBeUndefined()
  })

  it('abort() aborts the in-flight SSE request via its AbortController', async () => {
    let capturedSignal: AbortSignal | undefined
    vi.mocked(fetchEventSource).mockImplementation(async (_url, init) => {
      capturedSignal = (init as { signal?: AbortSignal }).signal
      return new Promise<void>(() => {}) // simulates an open stream that never resolves
    })

    const { result } = renderHook(() => useChatStream())
    await waitFor(() => expect(result.current.pendingActionId).toBeNull())

    act(() => {
      void result.current.send({ text: 'hi' })
    })
    await waitFor(() => expect(capturedSignal).toBeDefined())

    act(() => {
      result.current.abort()
    })

    expect(capturedSignal?.aborted).toBe(true)
  })

  it('the onerror handler rethrows, so fetch-event-source never auto-retries', async () => {
    let onerrorHandler: ((err: unknown) => void) | undefined
    vi.mocked(fetchEventSource).mockImplementation(async (_url, init) => {
      onerrorHandler = (init as { onerror?: (err: unknown) => void }).onerror
    })

    const { result } = renderHook(() => useChatStream())
    await waitFor(() => expect(result.current.pendingActionId).toBeNull())

    await act(async () => {
      await result.current.send({ text: 'hi' })
    })

    expect(onerrorHandler).toBeDefined()
    expect(() => onerrorHandler!(new Error('stream broke'))).toThrow('stream broke')
    // A second send() would be a new call; onerror itself must not trigger one.
    expect(fetchEventSource).toHaveBeenCalledTimes(1)
  })

  // R2: a stale token makes /api/chat answer 401 JSON; without a custom
  // onopen the library's own check says only "Expected content-type to be
  // text/event-stream, Actual: application/json" — true but useless.
  it('R2: a 401 on /api/chat surfaces a clear, token-free, reload message', async () => {
    vi.mocked(fetchEventSource).mockImplementation(async (_url, init) => {
      try {
        await (init as FesInit).onopen?.(fakeResponse(401, 'application/json'))
      } catch (e) {
        ;(init as FesInit).onerror?.(e)
        throw e
      }
    })

    const { result } = renderHook(() => useChatStream())
    await waitFor(() => expect(result.current.pendingActionId).toBeNull())

    await act(async () => {
      await result.current.send({ text: 'hi' })
    })

    expect(fetchEventSource).toHaveBeenCalledTimes(1)
    expect(result.current.error).toContain('401')
    expect(result.current.error?.toLowerCase()).toContain('reload')
    expect(result.current.error?.includes('sse-token')).toBe(false)
  })
})

// F13c-2a: owner decision 4 (PARENT_STATE.md) — a failed memory save is a
// status-bar notice, never chat text. The signal rides the `complete`
// event's debug record (F13b's `storage_failed`); a background failure that
// lands just after stream close is caught by one 2500ms follow-up read of
// GET /api/debug (F13c-1), never a poll loop. Drives the real hook/reducer;
// `fetchEventSource` and `fetch` stay mocked per the file's existing pattern.
describe('useChatStream storage-failure notice (F13c-2a)', () => {
  type FesFullInit = {
    onopen?: (r: Response) => Promise<void>
    onmessage?: (ev: { event?: string; data: string }) => void
    onerror?: (err: unknown) => void
  }

  // debug === undefined: no `complete` event at all (e.g. the stream errored
  // before one arrived). debug === null: a `complete` event with no record.
  function mockComplete(debug: Record<string, unknown> | null | undefined) {
    vi.mocked(fetchEventSource).mockImplementation(async (_url, init) => {
      const i = init as FesFullInit
      if (debug !== undefined) {
        i.onmessage?.({
          event: 'complete',
          data: JSON.stringify({ content: 'hi', pending_action_id: null, debug, turn_index: 0 }),
        })
      }
    })
  }

  async function sendAndFinish(
    result: { current: ReturnType<typeof useChatStream> },
    debug: Record<string, unknown> | null | undefined,
  ) {
    mockComplete(debug)
    await act(async () => {
      await result.current.send({ text: 'hi' })
    })
  }

  beforeEach(() => {
    vi.mocked(fetchEventSource).mockReset()
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({ ok: false, status: 404, json: async () => ({}) }),
    )
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('1. storage_failed on complete shows the notice immediately; no /api/debug fetch', async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: false, status: 404, json: async () => ({}) })
    vi.stubGlobal('fetch', fetchMock)
    const { result } = renderHook(() => useChatStream())
    await waitFor(() => expect(result.current.pendingActionId).toBeNull())

    vi.useFakeTimers()
    try {
      await sendAndFinish(result, { storage_failed: 'chroma write failed' })
      expect(result.current.storageNotice).toBe(true)
      expect(fetchMock).toHaveBeenCalledTimes(1) // only the mount-time /api/session
    } finally {
      vi.useRealTimers()
    }
  })

  it('2. the notice clears after 4000 ms', async () => {
    const { result } = renderHook(() => useChatStream())
    await waitFor(() => expect(result.current.pendingActionId).toBeNull())

    vi.useFakeTimers()
    try {
      await sendAndFinish(result, { storage_failed: 'x' })
      expect(result.current.storageNotice).toBe(true)

      await act(async () => {
        await vi.advanceTimersByTimeAsync(4000)
      })
      expect(result.current.storageNotice).toBe(false)
    } finally {
      vi.useRealTimers()
    }
  })

  it('3. no key on complete: exactly one follow-up fetch after 2500 ms; a keyless last record stays false', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce({ ok: false, status: 404, json: async () => ({}) }) // mount /api/session
      .mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: async () => ({ records: [{ mode: 'a' }, { mode: 'b' }], count: 2 }),
      }) // follow-up /api/debug
    vi.stubGlobal('fetch', fetchMock)

    const { result } = renderHook(() => useChatStream())
    await waitFor(() => expect(result.current.pendingActionId).toBeNull())

    vi.useFakeTimers()
    try {
      await sendAndFinish(result, {})
      expect(result.current.storageNotice).toBe(false)
      expect(fetchMock).toHaveBeenCalledTimes(1)

      await act(async () => {
        await vi.advanceTimersByTimeAsync(2500)
      })
      expect(fetchMock).toHaveBeenCalledTimes(2)
      expect(result.current.storageNotice).toBe(false)

      await act(async () => {
        await vi.advanceTimersByTimeAsync(5000)
      })
      expect(fetchMock).toHaveBeenCalledTimes(2) // no repeat poll
    } finally {
      vi.useRealTimers()
    }
  })

  it("4. the follow-up's last record carries the key: the notice becomes true", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce({ ok: false, status: 404, json: async () => ({}) })
      .mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: async () => ({
          records: [{ mode: 'a' }, { mode: 'b', storage_failed: 'disk full' }],
          count: 2,
        }),
      })
    vi.stubGlobal('fetch', fetchMock)

    const { result } = renderHook(() => useChatStream())
    await waitFor(() => expect(result.current.pendingActionId).toBeNull())

    vi.useFakeTimers()
    try {
      await sendAndFinish(result, {})
      await act(async () => {
        await vi.advanceTimersByTimeAsync(2500)
      })
      expect(result.current.storageNotice).toBe(true)
    } finally {
      vi.useRealTimers()
    }
  })

  it('5. the follow-up rejects or 500s: storageNotice stays false, error stays null, nothing thrown', async () => {
    for (const behavior of ['reject', '500'] as const) {
      const fetchMock =
        behavior === 'reject'
          ? vi
              .fn()
              .mockResolvedValueOnce({ ok: false, status: 404, json: async () => ({}) })
              .mockRejectedValueOnce(new Error('network down'))
          : vi
              .fn()
              .mockResolvedValueOnce({ ok: false, status: 404, json: async () => ({}) })
              .mockResolvedValueOnce({ ok: false, status: 500, json: async () => ({}) })
      vi.stubGlobal('fetch', fetchMock)

      const { result } = renderHook(() => useChatStream())
      await waitFor(() => expect(result.current.pendingActionId).toBeNull())

      vi.useFakeTimers()
      try {
        await sendAndFinish(result, {})
        await act(async () => {
          await vi.advanceTimersByTimeAsync(2500)
        })
        expect(result.current.storageNotice).toBe(false)
        expect(result.current.error).toBeNull()
      } finally {
        vi.useRealTimers()
      }
      vi.unstubAllGlobals()
    }
  })

  it('6. the notice resets immediately on stream_started of the next send', async () => {
    const { result } = renderHook(() => useChatStream())
    await waitFor(() => expect(result.current.pendingActionId).toBeNull())

    vi.useFakeTimers()
    try {
      await sendAndFinish(result, { storage_failed: 'x' })
      expect(result.current.storageNotice).toBe(true)

      let releaseStream: (() => void) | undefined
      vi.mocked(fetchEventSource).mockImplementation(async () => {
        await new Promise<void>((resolve) => {
          releaseStream = resolve
        })
      })
      act(() => {
        void result.current.send({ text: 'again' })
      })
      // The mock's executor runs synchronously up to its own await, so
      // releaseStream is already assigned once the sync act() above returns.
      expect(releaseStream).toBeDefined()
      expect(result.current.storageNotice).toBe(false)

      await act(async () => {
        releaseStream?.()
      })
    } finally {
      vi.useRealTimers()
    }
  })

  it('7. complete.debug === null: no follow-up fetch, notice stays false', async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: false, status: 404, json: async () => ({}) })
    vi.stubGlobal('fetch', fetchMock)
    const { result } = renderHook(() => useChatStream())
    await waitFor(() => expect(result.current.pendingActionId).toBeNull())

    vi.useFakeTimers()
    try {
      await sendAndFinish(result, null)
      await act(async () => {
        await vi.advanceTimersByTimeAsync(5000)
      })
      expect(result.current.storageNotice).toBe(false)
      expect(fetchMock).toHaveBeenCalledTimes(1) // only the mount-time /api/session
    } finally {
      vi.useRealTimers()
    }
  })

  it('8. a new send during the pending follow-up makes it stale (no notice, no stray fetch)', async () => {
    const fetchMock = vi.fn().mockResolvedValueOnce({ ok: false, status: 404, json: async () => ({}) })
    vi.stubGlobal('fetch', fetchMock)

    const { result } = renderHook(() => useChatStream())
    await waitFor(() => expect(result.current.pendingActionId).toBeNull())

    vi.useFakeTimers()
    try {
      await sendAndFinish(result, {}) // schedules the follow-up
      await sendAndFinish(result, undefined) // a fresh send bumps the counter, clears timers

      await act(async () => {
        await vi.advanceTimersByTimeAsync(5000)
      })
      expect(result.current.storageNotice).toBe(false)
      expect(fetchMock).toHaveBeenCalledTimes(1) // the stale follow-up never fires
    } finally {
      vi.useRealTimers()
    }
  })

  it('8b. clearAll during the pending follow-up makes it stale (no notice, no stray fetch)', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce({ ok: false, status: 404, json: async () => ({}) }) // mount
      .mockResolvedValueOnce({ ok: true, status: 200, json: async () => ({}) }) // clearAll DELETE
    vi.stubGlobal('fetch', fetchMock)

    const { result } = renderHook(() => useChatStream())
    await waitFor(() => expect(result.current.pendingActionId).toBeNull())

    vi.useFakeTimers()
    try {
      await sendAndFinish(result, {}) // schedules the follow-up
      await act(async () => {
        await result.current.clearAll()
      })

      await act(async () => {
        await vi.advanceTimersByTimeAsync(5000)
      })
      expect(result.current.storageNotice).toBe(false)
      expect(fetchMock).toHaveBeenCalledTimes(2) // mount + clearAll DELETE only
    } finally {
      vi.useRealTimers()
    }
  })

  it('9. an aborted stream never shows or schedules the notice, even with storage_failed pending', async () => {
    let resolveStream: (() => void) | undefined
    vi.mocked(fetchEventSource).mockImplementation(async (_url, init) => {
      const i = init as FesFullInit
      i.onmessage?.({
        event: 'complete',
        data: JSON.stringify({
          content: 'hi',
          pending_action_id: null,
          debug: { storage_failed: 'yep' },
          turn_index: 0,
        }),
      })
      await new Promise<void>((resolve) => {
        resolveStream = resolve
      })
    })

    const { result } = renderHook(() => useChatStream())
    await waitFor(() => expect(result.current.pendingActionId).toBeNull())

    vi.useFakeTimers()
    try {
      let sendPromise!: Promise<void>
      act(() => {
        sendPromise = result.current.send({ text: 'hi' })
      })
      // Synchronous prefix of the mock (incl. the onmessage 'complete' call
      // and the resolveStream assignment) has already run by this point.
      expect(resolveStream).toBeDefined()

      act(() => {
        result.current.abort()
      })

      await act(async () => {
        resolveStream?.()
        await sendPromise
      })
      expect(result.current.storageNotice).toBe(false)

      await act(async () => {
        await vi.advanceTimersByTimeAsync(5000)
      })
      expect(result.current.storageNotice).toBe(false)
    } finally {
      vi.useRealTimers()
    }
  })
})
