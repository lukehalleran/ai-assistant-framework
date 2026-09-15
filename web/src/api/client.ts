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
import { authorizedFetch, downloadViaAuthorizedFetch } from './launchAuth'

async function json<T>(resp: Response): Promise<T> {
  if (!resp.ok) throw new Error(`${resp.status} ${resp.statusText}`)
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

  syncNotes: () =>
    authorizedFetch('/api/sync-notes', { method: 'POST' }).then((r) =>
      json<{ message: string }>(r),
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

  getCurationQueue: () =>
    authorizedFetch('/api/curation/queue').then((r) => json<CurationQueueResponse>(r)),

  runCurationScan: () =>
    authorizedFetch('/api/curation/scan', { method: 'POST' }).then((r) =>
      json<CurationScanReport>(r),
    ),

  applyCurationProposal: (id: string) =>
    authorizedFetch(`/api/curation/${id}/apply`, { method: 'POST' }).then(async (r) => {
      if (!r.ok) {
        const detail = await r.json().then((b) => b.detail).catch(() => null)
        throw new Error(detail || `${r.status} ${r.statusText}`)
      }
      return r.json() as Promise<CurationProposal>
    }),

  dismissCurationProposal: (id: string, reason = '') =>
    authorizedFetch(`/api/curation/${id}/dismiss`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ reason }),
    }).then((r) => json<CurationProposal>(r)),

  undoCurationProposal: (id: string) =>
    authorizedFetch(`/api/curation/${id}/undo`, { method: 'POST' }).then(async (r) => {
      if (!r.ok) {
        const detail = await r.json().then((b) => b.detail).catch(() => null)
        throw new Error(detail || `${r.status} ${r.statusText}`)
      }
      return r.json() as Promise<CurationProposal>
    }),

  getCurationActivity: (limit = 100) =>
    authorizedFetch(`/api/curation/activity?limit=${limit}`).then((r) =>
      json<{ events: Record<string, unknown>[] }>(r),
    ),
}
