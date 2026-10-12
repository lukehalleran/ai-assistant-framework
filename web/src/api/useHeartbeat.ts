import { useEffect } from 'react'
import { api } from './client'

export const HEARTBEAT_INTERVAL_MS = 30_000

// Module-level so the effect's deps stay stable: an inline default would be a
// new function every render and re-ping on every App re-render (each streamed token).
const defaultSend = () => api.heartbeat()

/** App lifecycle (B3): tell the server this tab is alive. Sends immediately,
 * every 30 s, and when the tab becomes visible again (a throttled background
 * timer may have lagged). Failures are ignored -- the server decides what a
 * missing heartbeat means. `enabled=false` stops it (Daemon has shut down). */
export function useHeartbeat(enabled: boolean, send: () => Promise<void> = defaultSend): void {
  useEffect(() => {
    if (!enabled) return
    const ping = () => {
      try {
        send().catch(() => {})
      } catch {
        // a synchronous throw (e.g. a stubbed client) must not break the app
      }
    }
    ping()
    const id = setInterval(ping, HEARTBEAT_INTERVAL_MS)
    const onVisible = () => {
      if (document.visibilityState === 'visible') ping()
    }
    document.addEventListener('visibilitychange', onVisible)
    return () => {
      clearInterval(id)
      document.removeEventListener('visibilitychange', onVisible)
    }
  }, [enabled, send])
}
