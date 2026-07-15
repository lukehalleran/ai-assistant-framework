import type {
  ActionDecisionResponse,
  DebugRecordsResponse,
  DuelSettings,
  ModelListResponse,
  SessionState,
  SettingsApplyResult,
  SettingsSnapshot,
  StreamingSettings,
  TokenSettings,
  UploadedFileInfo,
  WebSearchSettings,
} from './types'

async function json<T>(resp: Response): Promise<T> {
  if (!resp.ok) throw new Error(`${resp.status} ${resp.statusText}`)
  return resp.json() as Promise<T>
}

export const api = {
  getSession: () => fetch('/api/session').then((r) => json<SessionState>(r)),

  clearSession: () => fetch('/api/session', { method: 'DELETE' }),

  getModels: () => fetch('/api/models').then((r) => json<ModelListResponse>(r)),

  setActiveModel: (name: string) =>
    fetch('/api/models/active', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name }),
    }).then((r) => json<ModelListResponse>(r)),

  approveAction: (actionId: string) =>
    fetch(`/api/actions/${actionId}/approve`, { method: 'POST' }).then((r) =>
      json<ActionDecisionResponse>(r),
    ),

  rejectAction: (actionId: string) =>
    fetch(`/api/actions/${actionId}/reject`, { method: 'POST' }).then((r) =>
      json<ActionDecisionResponse>(r),
    ),

  syncNotes: () =>
    fetch('/api/sync-notes', { method: 'POST' }).then((r) =>
      json<{ message: string }>(r),
    ),

  // ---- Debug / Provenance (server-held per-turn records) ----

  getDebugRecords: () => fetch('/api/debug').then((r) => json<DebugRecordsResponse>(r)),

  getProvenance: (index = -1) =>
    fetch(`/api/provenance?index=${index}`).then((r) =>
      json<Record<string, unknown>>(r),
    ),

  // Full-prompt TXT export — served with Content-Disposition: attachment,
  // so navigating to it triggers a native download.
  promptExportUrl: (index = -1) => `/api/debug/prompt?index=${index}`,

  // ---- Settings ----

  getSettings: () => fetch('/api/settings').then((r) => json<SettingsSnapshot>(r)),

  putSettings: (
    section:
      | 'streaming'
      | 'web-search'
      | 'duel'
      | 'tokens'
      | 'temperature'
      | 'summary-cadence',
    body:
      | StreamingSettings
      | WebSearchSettings
      | DuelSettings
      | TokenSettings
      | { temperature: number }
      | { every_n: number },
  ) =>
    fetch(`/api/settings/${section}`, {
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
    const resp = await fetch('/api/uploads', { method: 'POST', body: form })
    const body = await json<{ files: UploadedFileInfo[] }>(resp)
    return body.files
  },
}
