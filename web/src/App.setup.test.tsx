import { beforeEach, describe, expect, it, vi } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import { MantineProvider } from '@mantine/core'
import App from './App'
import { api, type SetupStatus } from './api/client'

// B2: the App gate. complete===false -> Setup page; Finish -> complete ->
// re-fetch -> chat; a status-fetch failure shows the normal app.

vi.mock('./api/client', () => ({
  api: {
    getModels: vi.fn(),
    getSettings: vi.fn(),
    getNotesSyncStatus: vi.fn(),
    getSetupStatus: vi.fn(),
    postSetupComplete: vi.fn(),
    postSetupOpenrouter: vi.fn(),
    postSetupKey: vi.fn(),
    postSetupProfile: vi.fn(),
    postSetupObsidian: vi.fn(),
    shutdown: vi.fn(),
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
vi.mock('./components/chat/ChatInput', () => ({ default: () => <div>CHAT-INPUT</div> }))
vi.mock('./components/chat/MessageList', () => ({ default: () => null }))
vi.mock('./components/chat/ProgressIndicator', () => ({ default: () => null }))
vi.mock('./components/showcase/MemoryPanel', () => ({ default: () => null }))

const mocked = vi.mocked(api)

function status(complete: boolean): SetupStatus {
  return {
    complete,
    steps: {
      openrouter: { state: 'found', hint: '...abcd' },
      profile: { state: 'found', name: 'Sam', pronouns: null },
      style: { state: 'missing', value: null },
      tavily: { state: 'missing', hint: null },
      wolfram: { state: 'missing', hint: null },
      e2b: { state: 'missing', hint: null },
      obsidian: { state: 'missing', path: null },
    },
  }
}

function renderApp() {
  render(
    <MantineProvider>
      <App />
    </MantineProvider>,
  )
}

describe('App setup gate', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocked.getModels.mockResolvedValue({ models: [], active: null } as never)
    mocked.getSettings.mockResolvedValue({} as never)
    mocked.getNotesSyncStatus.mockResolvedValue({ status: 'idle', finished_at: null } as never)
  })

  it('incomplete setup shows the Setup page, not chat, with Quit reachable', async () => {
    mocked.getSetupStatus.mockResolvedValue(status(false))
    renderApp()
    expect(await screen.findByText('Welcome to Daemon')).toBeInTheDocument()
    expect(screen.queryByText('CHAT-INPUT')).toBeNull()
    expect(screen.getByText(/Quit Daemon/)).toBeInTheDocument()
  })

  it('Finish calls complete, re-fetches status, then shows chat', async () => {
    mocked.getSetupStatus.mockResolvedValueOnce(status(false)).mockResolvedValue(status(true))
    mocked.postSetupComplete.mockResolvedValue(status(true))
    renderApp()
    await screen.findByText('Welcome to Daemon')
    const { fireEvent } = await import('@testing-library/react')
    // required steps are found, so setup resumes at the optional Style step
    for (let i = 0; i < 3; i++) {
      fireEvent.click(screen.getAllByRole('button', { name: 'Skip' })[0])
    }
    fireEvent.click(screen.getByRole('button', { name: 'Start chatting' }))
    await waitFor(() => expect(screen.getByText('CHAT-INPUT')).toBeInTheDocument())
    expect(mocked.postSetupComplete).toHaveBeenCalledTimes(1)
    expect(mocked.getSetupStatus).toHaveBeenCalledTimes(2)
    expect(screen.queryByText('Welcome to Daemon')).toBeNull()
  })

  it('a status-fetch failure shows the normal app (no lock-out)', async () => {
    mocked.getSetupStatus.mockRejectedValue(new Error('500 Internal Server Error'))
    renderApp()
    expect(await screen.findByText('CHAT-INPUT')).toBeInTheDocument()
    expect(screen.queryByText('Welcome to Daemon')).toBeNull()
  })

  it('a complete setup shows chat', async () => {
    mocked.getSetupStatus.mockResolvedValue(status(true))
    renderApp()
    expect(await screen.findByText('CHAT-INPUT')).toBeInTheDocument()
  })
})
