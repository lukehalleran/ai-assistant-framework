import { beforeEach, describe, expect, it, vi } from 'vitest'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MantineProvider } from '@mantine/core'
import ActionApprovalCard from './ActionApprovalCard'
import { api } from '../../api/client'
import type { ActionDecisionResponse, ActionOutcome } from '../../api/types'

// Audit F07 (docs/HANDOFF_20260909_independent_bug_audit.md): the backend
// chains approvals via `outcome.next_action_id`, but the card only forwarded
// `resp.message.content` — the browser had no way to learn about a second
// pending proposal from the same turn. These tests drive the real card
// against a mocked API client shaped exactly like the backend response.

vi.mock('../../api/client', () => ({
  api: {
    approveAction: vi.fn(),
    rejectAction: vi.fn(),
  },
}))

const mockedApi = vi.mocked(api)

function outcomeResponse(overrides: Partial<ActionOutcome> = {}): ActionDecisionResponse {
  const outcome: ActionOutcome = {
    status: 'executed',
    message: 'Done.',
    action_type: 'calendar_create_event',
    summary: 'Created event',
    next_action_id: null,
    next_summary: null,
    ...overrides,
  }
  return { outcome, message: { role: 'assistant', content: outcome.message } }
}

function renderCard(actionId: string, onDecided: (outcome: ActionOutcome, chatLine: string) => void) {
  return render(
    <MantineProvider>
      <ActionApprovalCard actionId={actionId} onDecided={onDecided} />
    </MantineProvider>,
  )
}

describe('ActionApprovalCard', () => {
  beforeEach(() => {
    mockedApi.approveAction.mockReset()
    mockedApi.rejectAction.mockReset()
  })

  it('forwards the full outcome (incl. next_action_id) to onDecided on approve', async () => {
    mockedApi.approveAction.mockResolvedValue(
      outcomeResponse({ next_action_id: 'second', next_summary: 'Delete stale event' }),
    )
    const onDecided = vi.fn()
    renderCard('first', onDecided)

    fireEvent.click(screen.getByRole('button', { name: /approve/i }))

    await waitFor(() => expect(onDecided).toHaveBeenCalledTimes(1))
    const [outcome, chatLine] = onDecided.mock.calls[0]
    expect(outcome.next_action_id).toBe('second')
    expect(outcome.next_summary).toBe('Delete stale event')
    expect(chatLine).toBe('Done.')
    expect(mockedApi.approveAction).toHaveBeenCalledWith('first')
  })

  it('forwards next_action_id: null on the final item in the chain', async () => {
    mockedApi.approveAction.mockResolvedValue(outcomeResponse({ next_action_id: null }))
    const onDecided = vi.fn()
    renderCard('only', onDecided)

    fireEvent.click(screen.getByRole('button', { name: /approve/i }))

    await waitFor(() => expect(onDecided).toHaveBeenCalledTimes(1))
    expect(onDecided.mock.calls[0][0].next_action_id).toBeNull()
  })

  it('a reject decision forwards the outcome the same way', async () => {
    mockedApi.rejectAction.mockResolvedValue(
      outcomeResponse({ status: 'rejected', message: 'Rejected.', next_action_id: 'second' }),
    )
    const onDecided = vi.fn()
    renderCard('first', onDecided)

    fireEvent.click(screen.getByRole('button', { name: /reject/i }))

    await waitFor(() => expect(onDecided).toHaveBeenCalledTimes(1))
    expect(onDecided.mock.calls[0][0].status).toBe('rejected')
    expect(onDecided.mock.calls[0][0].next_action_id).toBe('second')
    expect(mockedApi.rejectAction).toHaveBeenCalledWith('first')
  })

  it('resets busy state when the parent hands the same mounted card a new actionId', async () => {
    mockedApi.approveAction.mockResolvedValue(
      outcomeResponse({ next_action_id: 'second', next_summary: 'Delete stale event' }),
    )
    const onDecided = vi.fn()
    const { rerender } = renderCard('first', onDecided)

    const approveButton = () => screen.getByRole('button', { name: /approve/i })
    fireEvent.click(approveButton())

    // While the approve call is in flight (and immediately after, before the
    // parent re-renders with a new id), the button stays disabled.
    expect(approveButton()).toBeDisabled()
    await waitFor(() => expect(onDecided).toHaveBeenCalledTimes(1))

    // The parent hands the SAME component instance the next proposal's id —
    // no unmount. Before the fix this left `busy` set forever.
    rerender(
      <MantineProvider>
        <ActionApprovalCard actionId="second" onDecided={onDecided} />
      </MantineProvider>,
    )

    await waitFor(() => expect(approveButton()).not.toBeDisabled())
  })

  it('surfaces an error notification and re-enables the buttons on failure', async () => {
    mockedApi.approveAction.mockRejectedValue(new Error('network down'))
    const onDecided = vi.fn()
    renderCard('first', onDecided)

    fireEvent.click(screen.getByRole('button', { name: /approve/i }))

    await waitFor(() => expect(screen.getByRole('button', { name: /approve/i })).not.toBeDisabled())
    expect(onDecided).not.toHaveBeenCalled()
  })
})
