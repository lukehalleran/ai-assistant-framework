// Mirrors api/schemas.py — keep the two in sync by hand (single source of
// truth is the Python side; this file is the only place the shapes repeat).

export type Role = 'user' | 'assistant'

export interface ChatMessage {
  role: Role
  content: string
}

export interface ChatRequest {
  text: string
  raw_mode?: boolean
  fast_mode?: boolean
  enable_citations?: boolean
  file_ids?: string[]
}

export interface DuelThinking {
  thinking_a?: string
  thinking_b?: string
  model_a?: string
  model_b?: string
  winner?: string
  scores?: Record<string, number>
}

export interface SessionState {
  history: ChatMessage[]
  pending_action_id: string | null
  personality: string
}

export interface DebugRecord {
  mode?: string
  model?: string
  query?: string
  prompt?: string
  system_prompt?: string
  response?: string
  prompt_tokens?: number
  system_tokens?: number
  total_tokens?: number
  citations_enabled?: boolean
  citations?: unknown[]
  provenance?: Record<string, unknown>
  phase_timings?: Record<string, number>
  task_timings?: Record<string, number>
  gather_elapsed?: number
  [key: string]: unknown
}

export interface DebugRecordsResponse {
  records: DebugRecord[]
  count: number
}

// ---- Settings (mirrors api/schemas.py settings models) ----

export interface StreamingSettings {
  disable_best_of: boolean
  disable_query_rewrite: boolean
  disable_llm_summaries: boolean
  best_of_latency_budget_s: number
}

export interface WebSearchSettings {
  enabled: boolean
  daily_credit_limit: number
}

export interface DuelSettings {
  enabled: boolean
  model_1: string | null
  model_2: string | null
}

export interface TokenSettings {
  best_of_max_tokens: number
  judge_max_tokens: number
  streaming_max_tokens: number
}

export interface SynthesisSettings {
  enabled: boolean
  candidates_per_session: number
}

export interface ProposalsSettings {
  enabled: boolean
  max_per_session: number
}

export interface SettingsSnapshot {
  streaming: StreamingSettings
  web_search: WebSearchSettings
  duel: DuelSettings
  tokens: TokenSettings
  temperature: number
  summary_every_n: number
  synthesis: SynthesisSettings
  proposals: ProposalsSettings
  model_choices: string[]
}

export interface SettingsApplyResult {
  ok: boolean
  persisted: boolean
  message: string
}

export interface CompletePayload {
  content: string
  pending_action_id: string | null
  debug: DebugRecord | null
  turn_index: number
}

export interface UploadedFileInfo {
  file_id: string
  name: string
  size: number
}

export interface ModelListResponse {
  models: string[]
  active: string | null
}

export interface ActionOutcome {
  status: 'executed' | 'failed' | 'rejected' | 'not_found'
  message: string
  action_type: string | null
  summary: string | null
}

export interface ActionDecisionResponse {
  outcome: ActionOutcome
  message: ChatMessage
}
