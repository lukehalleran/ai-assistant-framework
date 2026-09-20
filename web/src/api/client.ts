import type {
  ActionDecisionResponse,
  CurationProposal,
  CurationQueueResponse,
  CurationScanReport,
  DebugRecordsResponse,
  DuelSettings,
  ModelListResponse,
  ProposalsSettings,
  SessionState,
  SettingsApplyResult,
  SettingsSnapshot,
  StreamingSettings,
  SynthesisSettings,
  TokenSettings,
  UploadedFileInfo,
  WebSearchSettings,
} from './types'
import { authorizedFetch, downloadViaAuthorizedFetch, nonOkRequestMessage } from './launchAuth'

async function json<T>(resp: Response): Promise<T> {
  if (!resp.ok) throw new Error(`${resp.status} ${resp.statusText}`)
  return resp.json() as Promise<T>
}

// ---- Notes sync (BC-80: outcome retained on the server, read by polling) ----

export interface NotesSyncResult {
  embedded_files: number
  updated_files: number
  skipped_files: number
  processed_files: number
  total_files: number
  total_chunks: number
  errors: string[]
  duration_seconds: number
}

export interface NotesSyncStatus {
  status: 'idle' | 'running' | 'succeeded' | 'failed'
  message: string | null
  error: string | null
  task_id: string | null
  started_at: number | null
  finished_at: number | null
  /** Server clock (epoch seconds) at the time of the read. */
  server_time: number
  last_result: {
    task_id: string
    status: 'succeeded' | 'failed'
    message: string
    error: string | null
    started_at: number | null
    finished_at: number
    result: NotesSyncResult | null
  } | null
}

/** A non-ok HTTP status. Distinguishable from a dropped connection (a bare
 * fetch rejection), which is the case BC-80 must never report as a failure.
 * `detail` is the server's own error text (e.g. a FastAPI `{"detail": …}`
 * body) when the caller could read one; existing call sites that omit it
 * keep the generic `nonOkRequestMessage` wording unchanged. */
export class HttpStatusError extends Error {
  readonly status: number
  readonly detail?: string
  constructor(status: number, label: string, detail?: string) {
    super(detail || nonOkRequestMessage(status, label))
    this.name = 'HttpStatusError'
    this.status = status
    this.detail = detail
  }
}

/** The POST's response was lost AND the server's retained state predates the
 * request: nothing started. Surfaced as "unknown / not started", not "failed". */
export class NotesSyncNotStartedError extends Error {
  constructor() {
    super('The sync request was interrupted before the server received it; nothing started. Try again.')
    this.name = 'NotesSyncNotStartedError'
  }
}

async function statusJson<T>(resp: Response, label: string): Promise<T> {
  if (!resp.ok) throw new HttpStatusError(resp.status, label)
  return resp.json() as Promise<T>
}

function abortableDelay(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    const abortError = () => new DOMException('Aborted', 'AbortError')
    if (signal?.aborted) {
      reject(abortError())
      return
    }
    const timer = setTimeout(() => {
      signal?.removeEventListener('abort', onAbort)
      resolve()
    }, ms)
    function onAbort() {
      clearTimeout(timer)
      reject(abortError())
    }
    signal?.addEventListener('abort', onAbort, { once: true })
  })
}

export interface NotesSyncPollOptions {
  signal?: AbortSignal
  intervalMs?: number
  onStatus?: (status: NotesSyncStatus) => void
  /** A fetch rejection (connection dropped) — recoverable; polling continues. */
  onTransportError?: (error: unknown) => void
}

/** Poll the retained server state until it is no longer `running`.
 * A dropped connection is retried; an HTTP status (401 after a restart, 5xx)
 * or an abort is thrown. */
export async function pollNotesSync(options: NotesSyncPollOptions = {}): Promise<NotesSyncStatus> {
  const intervalMs = options.intervalMs ?? 1000
  for (;;) {
    let status: NotesSyncStatus | null = null
    try {
      status = await api.getNotesSyncStatus(options.signal)
    } catch (error) {
      if (options.signal?.aborted || error instanceof HttpStatusError) throw error
      options.onTransportError?.(error)
    }
    if (status) {
      options.onStatus?.(status)
      if (status.status !== 'running') return status
    }
    await abortableDelay(intervalMs, options.signal)
  }
}

/** Start ONE sync, then poll for its outcome. If the POST's response is lost
 * the request is never re-sent (the server may have accepted it — a second
 * POST would either double-run or read as "already running"); the retained
 * state is polled instead, and an outcome that predates the request is
 * reported as NotesSyncNotStartedError rather than as this sync's result. */
export async function startNotesSyncAndPoll(
  options: NotesSyncPollOptions = {},
): Promise<NotesSyncStatus> {
  const postedAt = Date.now()
  let requestLost = false
  try {
    const started = await api.syncNotes()
    options.onStatus?.(started)
    if (started.status !== 'running') return started
  } catch (error) {
    if (options.signal?.aborted || error instanceof HttpStatusError) throw error
    requestLost = true
    options.onTransportError?.(error)
  }
  const terminal = await pollNotesSync(options)
  if (requestLost && !startedSince(terminal, postedAt)) throw new NotesSyncNotStartedError()
  return terminal
}

/** True when the server's current/retained job started no earlier than our
 * request (compared in server-clock seconds with 5 s of slack). */
function startedSince(status: NotesSyncStatus, postedAtMs: number): boolean {
  if (status.started_at === null) return false
  const ageS = status.server_time - status.started_at
  const sinceRequestS = (Date.now() - postedAtMs) / 1000
  return ageS <= sinceRequestS + 5
}

/** Reads a Curation Center endpoint's response. A non-ok status is a real
 * failure (BC-80): thrown as HttpStatusError carrying the server's `detail`
 * text when the body has one, so the caller can tell it apart from a lost
 * response (any other rejection — see failure.ts). */
async function curationResult<T>(resp: Response, label: string): Promise<T> {
  if (!resp.ok) {
    const detail = await resp
      .json()
      .then((b: unknown) =>
        b && typeof b === 'object' && typeof (b as { detail?: unknown }).detail === 'string'
          ? (b as { detail: string }).detail
          : null,
      )
      .catch(() => null)
    throw new HttpStatusError(resp.status, label, detail ?? undefined)
  }
  return resp.json() as Promise<T>
}


export const api = {
  getSession: () => authorizedFetch('/api/session').then((r) => json<SessionState>(r)),

  // F01/G06-T02 (A02): every /api/* request needs the launch token, including
  // this one — throw on a non-ok response instead of resolving with it, so a
  // rejected/unauthorized clear never looks like silent success.
  clearSession: () =>
    authorizedFetch('/api/session', { method: 'DELETE' }).then((r) => {
      if (!r.ok) throw new Error(`${r.status} ${r.statusText}`)
      return r
    }),

  getModels: () => authorizedFetch('/api/models').then((r) => json<ModelListResponse>(r)),

  setActiveModel: (name: string) =>
    authorizedFetch('/api/models/active', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name }),
    }).then((r) => json<ModelListResponse>(r)),

  approveAction: (actionId: string) =>
    authorizedFetch(`/api/actions/${actionId}/approve`, { method: 'POST' }).then((r) =>
      json<ActionDecisionResponse>(r),
    ),

  rejectAction: (actionId: string) =>
    authorizedFetch(`/api/actions/${actionId}/reject`, { method: 'POST' }).then((r) =>
      json<ActionDecisionResponse>(r),
    ),

  // Starts the retained background job; the outcome is read from
  // getNotesSyncStatus / pollNotesSync. `message` is kept for the toast.
  syncNotes: () =>
    authorizedFetch('/api/sync-notes', { method: 'POST' }).then((r) =>
      statusJson<NotesSyncStatus>(r, 'Notes sync'),
    ),

  getNotesSyncStatus: (signal?: AbortSignal) =>
    authorizedFetch('/api/sync-notes/status', { signal }).then((r) =>
      statusJson<NotesSyncStatus>(r, 'Notes sync status'),
    ),

  // ---- Debug / Provenance (server-held per-turn records) ----

  getDebugRecords: () => authorizedFetch('/api/debug').then((r) => json<DebugRecordsResponse>(r)),

  getProvenance: (index = -1) =>
    authorizedFetch(`/api/provenance?index=${index}`).then((r) =>
      json<Record<string, unknown>>(r),
    ),

  // Full-prompt TXT export. A native `<a href>` cannot carry the launch-token
  // header, so this drives an authorized fetch -> Blob -> object URL ->
  // temporary anchor click -> revoke instead (see launchAuth.ts). Renamed
  // from `promptExportUrl`; DebugPage.tsx is its only consumer.
  downloadPromptExport: (index = -1) =>
    downloadViaAuthorizedFetch(`/api/debug/prompt?index=${index}`, `daemon_prompt_${index}.txt`),

  // ---- Settings ----

  getSettings: () => authorizedFetch('/api/settings').then((r) => json<SettingsSnapshot>(r)),

  putSettings: (
    section:
      | 'streaming'
      | 'web-search'
      | 'duel'
      | 'tokens'
      | 'temperature'
      | 'summary-cadence'
      | 'synthesis'
      | 'proposals',
    body:
      | StreamingSettings
      | WebSearchSettings
      | DuelSettings
      | TokenSettings
      | { temperature: number }
      | { every_n: number }
      | SynthesisSettings
      | ProposalsSettings,
  ) =>
    authorizedFetch(`/api/settings/${section}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }).then(async (r) => {
      if (!r.ok) {
        // 400s carry an actionable detail message (e.g. duel validation)
        const detail = await r.json().then((b) => b.detail).catch(() => null)
        throw new Error(detail || `${r.status} ${r.statusText}`)
      }
      return r.json() as Promise<SettingsApplyResult>
    }),

  uploadFiles: async (files: File[]): Promise<UploadedFileInfo[]> => {
    const form = new FormData()
    files.forEach((f) => form.append('files', f))
    const resp = await authorizedFetch('/api/uploads', { method: 'POST', body: form })
    const body = await json<{ files: UploadedFileInfo[] }>(resp)
    return body.files
  },

  // ---- Curation Center (docs/AUTONOMOUS_CURATION_DESIGN.md) ----
  //
  // BC-80: apply/dismiss/undo/scan run in a server worker that outlives the
  // HTTP request (api/routes/curation.py `_run_operation`) — the queue +
  // journal are the retained outcome. A non-ok response is a real failure
  // (HttpStatusError, carrying the server's `detail` when it sent one); any
  // other rejection (network reset, aborted fetch) means the response was
  // lost, not that the operation failed — see failure.ts describeCurationFailure.

  getCurationQueue: () =>
    authorizedFetch('/api/curation/queue').then((r) =>
      curationResult<CurationQueueResponse>(r, 'Curation queue'),
    ),

  runCurationScan: () =>
    authorizedFetch('/api/curation/scan', { method: 'POST' }).then((r) =>
      curationResult<CurationScanReport>(r, 'Scan'),
    ),

  applyCurationProposal: (id: string) =>
    authorizedFetch(`/api/curation/${id}/apply`, { method: 'POST' }).then((r) =>
      curationResult<CurationProposal>(r, 'Apply'),
    ),

  dismissCurationProposal: (id: string, reason = '') =>
    authorizedFetch(`/api/curation/${id}/dismiss`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ reason }),
    }).then((r) => curationResult<CurationProposal>(r, 'Dismiss')),

  undoCurationProposal: (id: string) =>
    authorizedFetch(`/api/curation/${id}/undo`, { method: 'POST' }).then((r) =>
      curationResult<CurationProposal>(r, 'Undo'),
    ),

  getCurationActivity: (limit = 100) =>
    authorizedFetch(`/api/curation/activity?limit=${limit}`).then((r) =>
      curationResult<{ events: Record<string, unknown>[] }>(r, 'Curation activity'),
    ),
}
