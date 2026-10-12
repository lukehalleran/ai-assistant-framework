import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { act, renderHook } from '@testing-library/react'
import { api } from './client'
import { HEARTBEAT_INTERVAL_MS, useHeartbeat } from './useHeartbeat'

describe('useHeartbeat', () => {
  beforeEach(() => vi.useFakeTimers())
  afterEach(() => vi.useRealTimers())

  it('pings on mount, every 30 s, and on visibilitychange->visible', () => {
    const send = vi.fn().mockResolvedValue(undefined)
    renderHook(() => useHeartbeat(true, send))
    expect(send).toHaveBeenCalledTimes(1)
    vi.advanceTimersByTime(HEARTBEAT_INTERVAL_MS)
    expect(send).toHaveBeenCalledTimes(2)
    vi.advanceTimersByTime(HEARTBEAT_INTERVAL_MS * 2)
    expect(send).toHaveBeenCalledTimes(4)
    document.dispatchEvent(new Event('visibilitychange'))
    expect(send).toHaveBeenCalledTimes(5)
  })

  it('stops when disabled or unmounted, and swallows send failures', () => {
    const send = vi.fn().mockRejectedValue(new Error('down'))
    const { rerender, unmount } = renderHook(({ on }) => useHeartbeat(on, send), {
      initialProps: { on: true },
    })
    expect(send).toHaveBeenCalledTimes(1)
    rerender({ on: false })
    vi.advanceTimersByTime(HEARTBEAT_INTERVAL_MS * 3)
    expect(send).toHaveBeenCalledTimes(1)
    rerender({ on: true })
    expect(send).toHaveBeenCalledTimes(2)
    unmount()
    vi.advanceTimersByTime(HEARTBEAT_INTERVAL_MS * 3)
    expect(send).toHaveBeenCalledTimes(2)
  })

  it('does not re-ping on re-render with the default sender', () => {
    const spy = vi.spyOn(api, 'heartbeat').mockResolvedValue(false)
    const { rerender } = renderHook(() => useHeartbeat(true))
    for (let i = 0; i < 5; i++) rerender()
    expect(spy).toHaveBeenCalledTimes(1)
    spy.mockRestore()
  })

  it('exposes the latest quit_allowed: false until a heartbeat says true', async () => {
    const send = vi.fn().mockResolvedValueOnce(false).mockResolvedValue(true)
    const { result } = renderHook(() => useHeartbeat(true, send))
    await act(async () => {})
    expect(result.current).toBe(false)
    await act(async () => {
      vi.advanceTimersByTime(HEARTBEAT_INTERVAL_MS)
    })
    expect(result.current).toBe(true)
  })
})
