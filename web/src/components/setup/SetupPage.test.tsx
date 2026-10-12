import { describe, expect, it, vi } from 'vitest'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MantineProvider } from '@mantine/core'
import SetupPage, { initialStepIndex } from './SetupPage'
import type { SetupStatus } from '../../api/client'
import type { SetupClient } from './types'

const SECRET = 'sk-or-v1-SUPERSECRETVALUE1234'

function makeStatus(over: Partial<SetupStatus['steps']> = {}, complete = false): SetupStatus {
  return {
    complete,
    steps: {
      openrouter: { state: 'missing', hint: null },
      profile: { state: 'missing', name: null, pronouns: null },
      style: { state: 'missing', value: null },
      tavily: { state: 'missing', hint: null },
      wolfram: { state: 'missing', hint: null },
      e2b: { state: 'missing', hint: null },
      obsidian: { state: 'missing', path: null },
      ...over,
    },
  }
}

function makeClient(over: Partial<SetupClient> = {}): SetupClient {
  return {
    getSetupStatus: vi.fn().mockResolvedValue(makeStatus()),
    postSetupOpenrouter: vi.fn().mockResolvedValue({ ok: true, error: null, hint: '1234' }),
    postSetupKey: vi.fn().mockResolvedValue({ ok: true, error: null }),
    postSetupProfile: vi.fn().mockResolvedValue({ ok: true, error: null }),
    postSetupObsidian: vi.fn().mockResolvedValue({ ok: true, error: null }),
    postSetupComplete: vi.fn().mockResolvedValue(makeStatus({}, true)),
    ...over,
  } as SetupClient
}

function renderPage(status: SetupStatus, client: SetupClient, onFinished = vi.fn()) {
  render(
    <MantineProvider>
      <SetupPage initialStatus={status} client={client} onFinished={onFinished} onShutDown={vi.fn()} />
    </MantineProvider>,
  )
  return onFinished
}

const FOUND = { state: 'found' as const, hint: '...abcd' }
const PROFILE_FOUND = { state: 'found' as const, name: 'Sam', pronouns: 'they/them' }

describe('SetupPage', () => {
  it('resumes at the first missing REQUIRED step', () => {
    expect(initialStepIndex(makeStatus())).toBe(0)
    expect(initialStepIndex(makeStatus({ openrouter: FOUND }))).toBe(1)
    expect(initialStepIndex(makeStatus({ openrouter: FOUND, profile: PROFILE_FOUND }))).toBe(2)

    renderPage(makeStatus({ openrouter: FOUND }), makeClient())
    expect(screen.getByText(/Step 2 of 6: About you/)).toBeInTheDocument()
    expect(screen.getByLabelText(/Your name/)).toBeInTheDocument()
  })

  it('found badge hides the key input until Replace is clicked', () => {
    renderPage(makeStatus({ openrouter: FOUND }), makeClient())
    fireEvent.click(screen.getByRole('button', { name: 'Back' })) // back to step 1
    expect(screen.getByText(/found \.\.\.abcd/)).toBeInTheDocument()
    expect(screen.queryByLabelText(/OpenRouter API key/)).toBeNull()
    fireEvent.click(screen.getByRole('button', { name: 'Replace' }))
    expect(screen.getByLabelText(/OpenRouter API key/)).toBeInTheDocument()
  })

  it('an API error renders inline and does not advance', async () => {
    const client = makeClient({
      postSetupOpenrouter: vi.fn().mockResolvedValue({ ok: false, error: 'OpenRouter rejected this key (401).' }),
    })
    renderPage(makeStatus(), client)
    fireEvent.change(screen.getByLabelText(/OpenRouter API key/), { target: { value: SECRET } })
    fireEvent.click(screen.getByRole('button', { name: 'Test & save' }))
    expect(await screen.findByText(/rejected this key \(401\)/)).toBeInTheDocument()
    expect(screen.getByText(/Step 1 of 6/)).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Save anyway' })).toBeNull()
  })

  it('an unreachable OpenRouter offers Save anyway, sent as save_unverified', async () => {
    const post = vi
      .fn()
      .mockResolvedValueOnce({ ok: false, error: 'Could not reach OpenRouter.' })
      .mockResolvedValueOnce({ ok: true, error: null, hint: '1234' })
    renderPage(makeStatus(), makeClient({ postSetupOpenrouter: post }))
    fireEvent.change(screen.getByLabelText(/OpenRouter API key/), { target: { value: SECRET } })
    fireEvent.click(screen.getByRole('button', { name: 'Test & save' }))
    fireEvent.click(await screen.findByRole('button', { name: 'Save anyway' }))
    await waitFor(() => expect(post).toHaveBeenLastCalledWith(SECRET, true))
    expect(await screen.findByText(/Step 2 of 6/)).toBeInTheDocument()
  })

  it('the explicit unreachable flag offers Save anyway whatever the error wording', async () => {
    const post = vi.fn().mockResolvedValue({ ok: false, error: 'Network timeout.', unreachable: true })
    renderPage(makeStatus(), makeClient({ postSetupOpenrouter: post }))
    fireEvent.change(screen.getByLabelText(/OpenRouter API key/), { target: { value: SECRET } })
    fireEvent.click(screen.getByRole('button', { name: 'Test & save' }))
    expect(await screen.findByRole('button', { name: 'Save anyway' })).toBeInTheDocument()
  })

  it('no typed key remains in the DOM after a successful save', async () => {
    const client = makeClient()
    const { container } = render(
      <MantineProvider>
        <SetupPage initialStatus={makeStatus()} client={client} onFinished={vi.fn()} onShutDown={vi.fn()} />
      </MantineProvider>,
    )
    fireEvent.change(screen.getByLabelText(/OpenRouter API key/), { target: { value: SECRET } })
    fireEvent.click(screen.getByRole('button', { name: 'Test & save' }))
    expect(await screen.findByText(/Step 2 of 6/)).toBeInTheDocument()
    expect(container.innerHTML).not.toContain(SECRET)
    expect(document.body.innerHTML).not.toContain(SECRET)
    for (const el of Array.from(document.querySelectorAll('input'))) {
      expect((el as HTMLInputElement).value).not.toContain(SECRET)
    }
  })

  it('Skip advances on every optional step without saving', () => {
    const client = makeClient()
    renderPage(makeStatus({ openrouter: FOUND, profile: PROFILE_FOUND }), client)
    expect(screen.getByText(/Response style \(optional\)/)).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Skip' }))
    expect(screen.getByText(/Optional keys \(optional\)/)).toBeInTheDocument()
    fireEvent.click(screen.getAllByRole('button', { name: 'Skip' })[0])
    expect(screen.getByText(/Obsidian vault \(optional\)/)).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Skip' }))
    expect(screen.getByRole('button', { name: 'Start chatting' })).toBeInTheDocument()
    expect(client.postSetupProfile).not.toHaveBeenCalled()
    expect(client.postSetupKey).not.toHaveBeenCalled()
    expect(client.postSetupObsidian).not.toHaveBeenCalled()
  })

  it('Finish calls complete, then hands back to the caller', async () => {
    const client = makeClient()
    const onFinished = renderPage(makeStatus({ openrouter: FOUND, profile: PROFILE_FOUND }), client)
    for (let i = 0; i < 3; i++) {
      fireEvent.click(screen.getAllByRole('button', { name: 'Skip' })[0])
    }
    fireEvent.click(screen.getByRole('button', { name: 'Start chatting' }))
    await waitFor(() => expect(onFinished).toHaveBeenCalledTimes(1))
    expect(client.postSetupComplete).toHaveBeenCalledTimes(1)
  })

  it('a failed complete shows the error and does not hand off', async () => {
    const client = makeClient({ postSetupComplete: vi.fn().mockRejectedValue(new Error('500 Server Error')) })
    const onFinished = renderPage(makeStatus({ openrouter: FOUND, profile: PROFILE_FOUND }), client)
    for (let i = 0; i < 3; i++) {
      fireEvent.click(screen.getAllByRole('button', { name: 'Skip' })[0])
    }
    fireEvent.click(screen.getByRole('button', { name: 'Start chatting' }))
    expect(await screen.findByText(/500 Server Error/)).toBeInTheDocument()
    expect(onFinished).not.toHaveBeenCalled()
  })

  it('the Quit control stays reachable', () => {
    renderPage(makeStatus(), makeClient())
    expect(screen.getByText(/Quit Daemon/)).toBeInTheDocument()
  })
})

describe('optional keys by mode', () => {
  const atOptionalKeys = (mode?: 'user' | 'dev') => {
    const st = makeStatus({
      openrouter: { state: 'found', hint: '...abcd' },
      profile: { state: 'found', name: 'Sam', pronouns: 'they/them' },
      style: { state: 'found', value: 'balanced' },
    })
    renderPage(mode ? { ...st, mode } : st, makeClient())
    fireEvent.click(screen.getByRole('button', { name: 'Skip' })) // Style -> Optional keys
  }

  it('user mode (and an absent mode) hides the dev-only E2B key', async () => {
    atOptionalKeys('user')
    expect(await screen.findByLabelText(/Tavily/)).toBeInTheDocument()
    expect(screen.queryByLabelText(/E2B/)).toBeNull()
  })

  it('dev mode shows the E2B key', async () => {
    atOptionalKeys('dev')
    expect(await screen.findByLabelText(/E2B/)).toBeInTheDocument()
  })
})
