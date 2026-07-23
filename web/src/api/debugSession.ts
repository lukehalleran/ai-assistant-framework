import { api } from './client'

// Debug records are server-held (they accumulate until DELETE /api/session),
// but the Gradio debug tab starts empty on every page load (per-session
// gr.State). This baseline scopes the SPA's Debug/Provenance views the same
// way: only turns from the ongoing UI session are shown. Captured once at app
// mount; idempotent, so StrictMode double-effects are harmless.
let baseline: Promise<number> | null = null

export function captureDebugBaseline(): Promise<number> {
  if (!baseline) {
    baseline = api
      .getDebugRecords()
      .then((r) => r.count)
      .catch(() => 0)
  }
  return baseline
}

/** Called when the user clears the session — server records reset to 0. */
export function resetDebugBaseline() {
  baseline = Promise.resolve(0)
}

/** Baseline adjusted for a just-fetched record count: a count BELOW the
 * baseline means the server-side records were cleared out-of-band (server
 * restart) — everything present belongs to a newer session, start from 0. */
export async function debugBaselineFor(count: number): Promise<number> {
  const b = await captureDebugBaseline()
  if (count < b) {
    resetDebugBaseline()
    return 0
  }
  return b
}
