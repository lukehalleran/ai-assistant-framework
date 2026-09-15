import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { api } from './client'
import { __resetLaunchTokenForTests, authorizedFetch, LAUNCH_TOKEN_HEADER } from './launchAuth'

// F01 / G06-T02 (A02): every /api/* request needs `X-Daemon-Launch-Token`,
// read from `<meta name="daemon-launch-token">` via one chokepoint (never a
// cookie/storage/URL). Enumerates the real `api` object (Object.entries) so
// a new function is covered by construction; failing-before is in A02.md.

const TOKEN = 'test-launch-token-abc123'

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

interface RecordedRequest {
  input: RequestInfo | URL
  init?: RequestInit
}

function stubFetch(responseFactory?: (req: RecordedRequest) => Partial<Response>) {
  const calls: RecordedRequest[] = []
  const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const req = { input, init }
    calls.push(req)
    const base: Partial<Response> = {
      ok: true,
      status: 200,
      statusText: 'OK',
      headers: new Headers(),
      json: async () => ({}),
      blob: async () => new Blob(['x'], { type: 'text/plain' }),
      text: async () => '',
    }
    const overrides = responseFactory ? responseFactory(req) : {}
    return { ...base, ...overrides } as Response
  })
  vi.stubGlobal('fetch', fetchMock)
  return { calls, fetchMock }
}

// One representative call per api function; none is intentionally skipped.
const REPRESENTATIVE_ARGS: Record<string, unknown[]> = {
  getSession: [],
  clearSession: [],
  getModels: [],
  setActiveModel: ['gpt-x'],
  approveAction: ['act-1'],
  rejectAction: ['act-1'],
  syncNotes: [],
  getDebugRecords: [],
  getProvenance: [3],
  downloadPromptExport: [3],
  getSettings: [],
  putSettings: ['temperature', { temperature: 0.5 }],
  uploadFiles: [[new File(['hi'], 'a.txt', { type: 'text/plain' })]],
  getCurationQueue: [],
  runCurationScan: [],
  applyCurationProposal: ['p-1'],
  dismissCurationProposal: ['p-1', 'because'],
  undoCurationProposal: ['p-1'],
  getCurationActivity: [50],
}

describe('api client transport authorization (F01/G06-T02, A02)', () => {
  let createObjectURLSpy: ReturnType<typeof vi.fn>
  let revokeObjectURLSpy: ReturnType<typeof vi.fn>
  let anchorClickSpy: ReturnType<typeof vi.spyOn>

  beforeEach(() => {
    createObjectURLSpy = vi.fn(() => 'blob:mock-url')
    revokeObjectURLSpy = vi.fn()
    // jsdom does not implement these; downloadPromptExport needs them.
    // @ts-expect-error -- test shim, not a spec-complete URL
    window.URL.createObjectURL = createObjectURLSpy
    // @ts-expect-error -- test shim, not a spec-complete URL
    window.URL.revokeObjectURL = revokeObjectURLSpy
    // jsdom logs (not throws) "Not implemented: navigation" on an <a> click;
    // silence it so the test only observes our download-trigger call.
    anchorClickSpy = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {})
  })

  afterEach(() => {
    vi.unstubAllGlobals()
    anchorClickSpy.mockRestore()
    setMetaToken(null)
  })

  it('every exported api function is enumerated and covered (fails on empty enumeration)', () => {
    const names = Object.keys(api)
    expect(names.length).toBeGreaterThan(0)
    const uncovered = names.filter((n) => !(n in REPRESENTATIVE_ARGS))
    // No function is intentionally skipped from this contract.
    expect(uncovered).toEqual([])
  })

  // R4: attribute requests per-function so one that made ZERO requests can't
  // hide behind another that made two (the old aggregate `calls.length >=
  // functionCount` check would not have caught that).
  it('every function produces at least one /api request carrying the header (per-function attribution)', async () => {
    setMetaToken(TOKEN)
    for (const [name, args] of Object.entries(REPRESENTATIVE_ARGS)) {
      const { calls } = stubFetch()
      const fn = (api as unknown as Record<string, (...a: unknown[]) => unknown>)[name]
      expect(fn, `api.${name} must exist`).toBeTypeOf('function')
      await (fn(...args) as Promise<unknown>).catch(() => {})

      expect(calls.length, `api.${name} made zero /api requests`).toBeGreaterThanOrEqual(1)
      for (const { input, init } of calls) {
        const url = String(input)
        expect(url.startsWith('/api/'), `api.${name}: unexpected non-/api/ request: ${url}`).toBe(true)
        const headers = new Headers(init?.headers)
        expect(headers.get(LAUNCH_TOKEN_HEADER), `api.${name}: missing header for ${url}`).toBe(TOKEN)
        // The token must never appear in a requested URL or query string.
        expect(url.includes(TOKEN)).toBe(false)
      }
    }
  })

  it('upload sends FormData with the header and never sets Content-Type manually', async () => {
    setMetaToken(TOKEN)
    const { calls } = stubFetch()
    await api.uploadFiles([new File(['hi'], 'a.txt', { type: 'text/plain' })])
    const uploadCall = calls.find((c) => String(c.input).includes('/api/uploads'))
    expect(uploadCall).toBeDefined()
    expect(uploadCall!.init?.body).toBeInstanceOf(FormData)
    const headers = new Headers(uploadCall!.init?.headers)
    expect(headers.get(LAUNCH_TOKEN_HEADER)).toBe(TOKEN)
    expect(headers.has('Content-Type')).toBe(false)
  })

  // R1 (round 2 review): Firefox/Safari start a blob download asynchronously;
  // revoking the object URL synchronously right after click() can cancel it,
  // so the revoke must be deferred (scheduled), not immediate.
  it('download: header present; object URL created, anchor clicked, revoke deferred (R1)', async () => {
    vi.useFakeTimers()
    try {
      setMetaToken(TOKEN)
      stubFetch(() => ({
        headers: new Headers({ 'content-disposition': 'attachment; filename="daemon_prompt_x.txt"' }),
      }))
      await api.downloadPromptExport(2)
      expect(createObjectURLSpy).toHaveBeenCalledTimes(1)
      expect(anchorClickSpy).toHaveBeenCalledTimes(1)
      expect(document.querySelectorAll('a[download]').length).toBe(0)
      expect(revokeObjectURLSpy).not.toHaveBeenCalled()
      vi.advanceTimersByTime(1000)
      expect(revokeObjectURLSpy).toHaveBeenCalledTimes(1)
    } finally {
      vi.useRealTimers()
    }
  })

  it('missing meta tag: no token header is sent, and a 401 surfaces as an error, not empty success', async () => {
    setMetaToken(null)
    const { calls } = stubFetch(() => ({
      ok: false,
      status: 401,
      statusText: 'Unauthorized',
      json: async () => ({ detail: 'no token' }),
    }))

    await expect(api.getSession()).rejects.toThrow()
    await api.getModels().catch(() => {})

    for (const { init } of calls) {
      const headers = new Headers(init?.headers)
      expect(headers.has(LAUNCH_TOKEN_HEADER)).toBe(false)
    }
  })

  it('the token never appears in a thrown error message (json helper and download helper)', async () => {
    setMetaToken(TOKEN)
    stubFetch(() => ({
      ok: false,
      status: 500,
      statusText: 'Internal Server Error',
      json: async () => ({ detail: 'nope' }),
    }))
    for (const call of [() => api.getSettings(), () => api.downloadPromptExport(2)]) {
      let threw = false
      try {
        await call()
      } catch (err) {
        threw = true
        const message = err instanceof Error ? err.message : String(err)
        expect(message.includes(TOKEN)).toBe(false)
      }
      expect(threw, 'expected the call to reject on a 500').toBe(true)
    }
  })

  // "unit test the download helper plus a check DebugPage uses it" — a full
  // render harness is out of this batch's file ownership; a lightweight
  // static check via Vite's `?raw` import instead (no @types/node needed).
  it('DebugPage no longer builds a native /api download href', async () => {
    // @ts-expect-error -- Vite/Vitest's `?raw` suffix has no ambient module
    // declaration in this repo's tsconfig; the runtime import still works
    // via the shared Vite transform pipeline (see vitest.config.ts).
    const debugPageSource = (await import('../components/debug/DebugPage.tsx?raw')).default
    expect(debugPageSource.includes('promptExportUrl')).toBe(false)
    expect(debugPageSource.includes('href={api.')).toBe(false)
    expect(debugPageSource.includes('downloadPromptExport')).toBe(true)
  })

  // R3: reject a non-same-origin/non-/api/ path before fetch ever runs.
  it.each([
    ['https://evil.example/api/x', 'foreign absolute origin'],
    ['//evil.example/api/x', 'protocol-relative foreign origin'],
    ['/other', 'non-/api/ path'],
    ['/api/../x', 'path traversal that escapes /api/'],
  ])('R3: rejects %s (%s) with zero fetch calls', async (path) => {
    setMetaToken(TOKEN)
    const { fetchMock } = stubFetch()
    await expect(authorizedFetch(path)).rejects.toThrow()
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it('R3 control: a genuine same-origin /api/ path is allowed through', async () => {
    setMetaToken(TOKEN)
    const { fetchMock } = stubFetch()
    await authorizedFetch('/api/session')
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })
})
