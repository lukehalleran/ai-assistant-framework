// F01 / G06-T02 client transport authorization (A02). The A01 server injects
// `<meta name="daemon-launch-token">` after `<head>`; every /api/* request
// must carry it back as `X-Daemon-Launch-Token` via this single chokepoint.
// Token: module memory only — never a cookie, storage, URL, or log.

export const LAUNCH_TOKEN_HEADER = 'X-Daemon-Launch-Token'

const LAUNCH_TOKEN_META_NAME = 'daemon-launch-token'

// undefined = not yet read; null = absent. Read lazily so a test can set the meta tag after import.
let cachedToken: string | null | undefined

function readLaunchTokenFromDocument(): string | null {
  if (typeof document === 'undefined') return null
  const meta = document.querySelector(`meta[name="${LAUNCH_TOKEN_META_NAME}"]`)
  const content = meta?.getAttribute('content')
  return content ? content : null
}

export function getLaunchToken(): string | null {
  if (cachedToken === undefined) {
    cachedToken = readLaunchTokenFromDocument()
  }
  return cachedToken
}

/** Test-only: force the next getLaunchToken() call to re-read the meta tag.
 * Production code never calls this — the token cannot change mid-launch. */
export function __resetLaunchTokenForTests(): void {
  cachedToken = undefined
}

// R3: scope the chokepoint to same-origin /api/ paths so a future absolute/
// foreign URL can't leak the token — reject BEFORE fetch runs, token-free.
export function assertApiPath(path: string): void {
  const resolved = new URL(path, window.location.origin)
  if (resolved.origin !== window.location.origin || !resolved.pathname.startsWith('/api/')) {
    throw new Error('Refused to attach the launch token to a non-API request.')
  }
}

// R2 (round 2 review): one shared, token-free "not authorized" wording, used
// by both the SSE chat onopen check and mount-restore, so a stale/rotated
// token (e.g. a Daemon restart) reads the same way everywhere.
export function nonOkRequestMessage(status: number, label: string): string {
  if (status === 401) {
    return 'Not authorized (401) — Daemon may have restarted; reload the page.'
  }
  return `${label} failed (${status})`
}

/** Merges the launch token into caller-supplied headers. Never sets
 * Content-Type (FormData owns that). Missing token: omit the header rather
 * than invent one — the server's own 401 is what surfaces the failure. */
function withLaunchToken(extra?: HeadersInit): Headers {
  const headers = new Headers(extra)
  const token = getLaunchToken()
  if (token) headers.set(LAUNCH_TOKEN_HEADER, token)
  return headers
}

/** The single chokepoint for ordinary (non-SSE) /api/* requests: client.ts
 * and useChatStream's session-restore/clearAll go through this. `input` is
 * `string` (every caller passes one) so `assertApiPath` can reject a
 * non-same-origin/non-/api/ path before `fetch` ever runs. */
export async function authorizedFetch(input: string, init?: RequestInit): Promise<Response> {
  assertApiPath(input)
  return fetch(input, { ...init, headers: withLaunchToken(init?.headers) })
}

/** For fetchEventSource, which wants a plain Record<string, string> rather
 * than performing the fetch itself. */
export function authorizedHeaders(extra?: HeadersInit): Record<string, string> {
  return Object.fromEntries(withLaunchToken(extra).entries())
}

function parseContentDispositionFilename(value: string | null): string | null {
  if (!value) return null
  const match = /filename\*?=(?:UTF-8''|")?([^";]+)"?/i.exec(value)
  if (!match) return null
  try {
    return decodeURIComponent(match[1])
  } catch {
    return match[1]
  }
}

/** Authorized fetch -> Blob -> object URL -> anchor click -> revoke. Replaces
 * a native `<a href="/api/...">` download, which can't carry the header.
 * Filename from Content-Disposition, else `fallbackFilename`. Throws a
 * status-only Error on failure — never the token or full URL. */
export async function downloadViaAuthorizedFetch(
  url: string,
  fallbackFilename: string,
): Promise<void> {
  const resp = await authorizedFetch(url)
  if (!resp.ok) {
    throw new Error(`Download failed (${resp.status})`)
  }
  const blob = await resp.blob()
  const filename = parseContentDispositionFilename(resp.headers.get('content-disposition')) || fallbackFilename
  const objectUrl = URL.createObjectURL(blob)
  try {
    const a = document.createElement('a')
    a.href = objectUrl
    a.download = filename
    a.style.display = 'none'
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
  } finally {
    // R1 (round 2 review): Firefox/Safari start a blob download
    // asynchronously; revoking synchronously right after click() can cancel
    // it. Defer instead — `finally` still schedules it even if
    // creating/clicking the anchor itself throws.
    setTimeout(() => URL.revokeObjectURL(objectUrl), 1000)
  }
}
