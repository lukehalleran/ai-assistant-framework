import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { act, renderHook, waitFor } from '@testing-library/react'
import { useChatStream } from './useChatStream'

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
