import { beforeEach, describe, expect, it, vi } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import { MantineProvider } from '@mantine/core'
import App from './App'
import SettingsPage from './components/settings/SettingsPage'
import { api } from './api/client'
import type { Availability, SettingsSnapshot } from './api/types'

// Subgoal D4 (2026-09-30): a generic install hides what cannot work. The
// server's GET /api/settings `availability` drives it; while loading every
// flag is false (hidden). Drives the real App and SettingsPage.

vi.mock('./api/client', () => ({
  api: {
    getModels: vi.fn(),
    getSettings: vi.fn(),
    getNotesSyncStatus: vi.fn(),
  },
  NotesSyncNotStartedError: class extends Error {},
  pollNotesSync: vi.fn(),
  startNotesSyncAndPoll: vi.fn(),
}))
vi.mock('./api/debugSession', () => ({ captureDebugBaseline: vi.fn() }))
vi.mock('./api/useChatStream', () => ({
  useChatStream: () => ({
    messages: [],
    debugRecords: [],
    progressLog: [],
    streaming: false,
    progressText: '',
    thinkingText: '',
    startedAt: null,
    storageNotice: false,
    pendingActionId: null,
    duelThinking: false,
    error: null,
    send: vi.fn(),
    abort: vi.fn(),
    clearAll: vi.fn(),
    appendAssistant: vi.fn(),
    setPendingAction: vi.fn(),
  }),
}))
vi.mock('./components/chat/ActivityLog', () => ({ default: () => null }))
vi.mock('./components/chat/ChatInput', () => ({ default: () => null }))
vi.mock('./components/chat/MessageList', () => ({ default: () => null }))
vi.mock('./components/chat/ProgressIndicator', () => ({ default: () => null }))
vi.mock('./components/showcase/MemoryPanel', () => ({ default: () => null }))

const mockedApi = vi.mocked(api)

const ALL_FALSE: Availability = { web_search_key: false, vault: false, dev_mode: false }
const ALL_TRUE: Availability = { web_search_key: true, vault: true, dev_mode: true }

function snapshot(availability: Availability): SettingsSnapshot {
  return {
    streaming: {
      disable_best_of: false,
      disable_query_rewrite: false,
      disable_llm_summaries: false,
      best_of_latency_budget_s: 0,
    },
    web_search: { enabled: true, daily_credit_limit: 100 },
    duel: { enabled: false, model_1: null, model_2: null },
    tokens: { best_of_max_tokens: 128, judge_max_tokens: 64, streaming_max_tokens: 2048 },
    temperature: 0.7,
    summary_every_n: 10,
    synthesis: { enabled: false, candidates_per_session: 8 },
    proposals: { enabled: false, max_per_session: 5 },
    model_choices: [],
    availability,
  } as unknown as SettingsSnapshot
}

const wrap = (ui: React.ReactElement) => render(<MantineProvider>{ui}</MantineProvider>)

beforeEach(() => {
  vi.clearAllMocks()
  mockedApi.getModels.mockResolvedValue({ models: [], active: null })
  mockedApi.getNotesSyncStatus.mockResolvedValue({
    status: 'idle',
    finished_at: null,
    server_time: 0,
  } as never)
})

describe('SettingsPage hides (D4)', () => {
  it('all-false: Tavily hint, no web controls, no duel/synthesis/proposals', async () => {
    mockedApi.getSettings.mockResolvedValue(snapshot(ALL_FALSE))
    wrap(<SettingsPage availability={ALL_FALSE} />)
    await screen.findByText(/Runtime Settings/)
    expect(
      screen.getByText('Web search needs a Tavily API key (TAVILY_API_KEY in .env).'),
    ).toBeInTheDocument()
    expect(screen.queryByText('Enable web search')).not.toBeInTheDocument()
    expect(screen.queryByText(/Best-of \/ duel mode/)).not.toBeInTheDocument()
    expect(screen.queryByText(/Synthesis dreaming/)).not.toBeInTheDocument()
    expect(screen.queryByText(/Code proposals/)).not.toBeInTheDocument()
  })

  it('all-true: web controls and the three dev cards present', async () => {
    mockedApi.getSettings.mockResolvedValue(snapshot(ALL_TRUE))
    wrap(<SettingsPage availability={ALL_TRUE} />)
    await screen.findByText(/Runtime Settings/)
    expect(screen.getByText('Enable web search')).toBeInTheDocument()
    expect(screen.queryByText(/Web search needs a Tavily/)).not.toBeInTheDocument()
    expect(screen.getByText(/Best-of \/ duel mode/)).toBeInTheDocument()
    expect(screen.getByText(/Synthesis dreaming/)).toBeInTheDocument()
    expect(screen.getByText(/Code proposals/)).toBeInTheDocument()
  })
})

describe('App hides (D4)', () => {
  it('all-false: no Sync notes, no Curation, no /admin link, Raw mode label, no sync poll', async () => {
    mockedApi.getSettings.mockResolvedValue(snapshot(ALL_FALSE))
    wrap(<App />)
    await waitFor(() => expect(mockedApi.getSettings).toHaveBeenCalledTimes(1))
    expect(screen.getByText('Raw mode (bypass memory)')).toBeInTheDocument()
    expect(screen.queryByText(/Raw GPT/)).not.toBeInTheDocument()
    expect(screen.queryByText(/Sync notes/)).not.toBeInTheDocument()
    expect(screen.queryByRole('radio', { name: '🧹' })).not.toBeInTheDocument()
    expect(screen.queryByText(/Dev tabs live at/)).not.toBeInTheDocument()
    expect(mockedApi.getNotesSyncStatus).not.toHaveBeenCalled()
  })

  it('all-true: Sync notes, Curation and the /admin link appear; sync status polled', async () => {
    mockedApi.getSettings.mockResolvedValue(snapshot(ALL_TRUE))
    wrap(<App />)
    expect(await screen.findByText(/Sync notes/)).toBeInTheDocument()
    expect(screen.getByRole('radio', { name: '🧹' })).toBeInTheDocument()
    expect(screen.getByText(/Dev tabs live at/)).toBeInTheDocument()
    await waitFor(() => expect(mockedApi.getNotesSyncStatus).toHaveBeenCalled())
  })

  it('settings fetch failure keeps everything hidden', async () => {
    mockedApi.getSettings.mockRejectedValue(new Error('down'))
    wrap(<App />)
    await waitFor(() => expect(mockedApi.getSettings).toHaveBeenCalled())
    expect(screen.queryByText(/Sync notes/)).not.toBeInTheDocument()
    expect(screen.queryByRole('radio', { name: '🧹' })).not.toBeInTheDocument()
  })
})
