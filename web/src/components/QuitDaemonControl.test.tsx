import { describe, expect, it, vi } from 'vitest'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MantineProvider } from '@mantine/core'
import QuitDaemonControl, { ShutdownScreen } from './QuitDaemonControl'

// B3: Quit asks for confirmation, only then POSTs shutdown, and only a
// successful POST hands off to the shutdown screen.

function renderControl(shutdown: () => Promise<void>, onShutDown = vi.fn()) {
  render(
    <MantineProvider>
      <QuitDaemonControl shutdown={shutdown} onShutDown={onShutDown} />
    </MantineProvider>,
  )
  return onShutDown
}

describe('QuitDaemonControl', () => {
  it('does not call shutdown until the confirm modal is accepted', async () => {
    const shutdown = vi.fn().mockResolvedValue(undefined)
    const onShutDown = renderControl(shutdown)
    fireEvent.click(screen.getByText(/Quit Daemon/))
    expect(await screen.findByText('Quit Daemon?')).toBeInTheDocument()
    expect(shutdown).not.toHaveBeenCalled()
    fireEvent.click(screen.getByRole('button', { name: 'Quit' }))
    await waitFor(() => expect(onShutDown).toHaveBeenCalledTimes(1))
    expect(shutdown).toHaveBeenCalledTimes(1)
  })

  it('a failed POST shows the error and does not shut the UI down', async () => {
    const shutdown = vi.fn().mockRejectedValue(new Error('401 Unauthorized'))
    const onShutDown = renderControl(shutdown)
    fireEvent.click(screen.getByText(/Quit Daemon/))
    fireEvent.click(await screen.findByRole('button', { name: 'Quit' }))
    expect(await screen.findByText(/401 Unauthorized/)).toBeInTheDocument()
    expect(onShutDown).not.toHaveBeenCalled()
  })

  it('cancel does nothing', async () => {
    const shutdown = vi.fn().mockResolvedValue(undefined)
    renderControl(shutdown)
    fireEvent.click(screen.getByText(/Quit Daemon/))
    fireEvent.click(await screen.findByRole('button', { name: 'Cancel' }))
    expect(shutdown).not.toHaveBeenCalled()
  })

  it('shutdown screen tells the user to close the tab', () => {
    render(
      <MantineProvider>
        <ShutdownScreen />
      </MantineProvider>,
    )
    expect(screen.getByText(/Daemon has shut down/)).toBeInTheDocument()
    expect(screen.getByText(/close this tab/)).toBeInTheDocument()
  })
})
