import { useState } from 'react'
import { describe, expect, it, vi } from 'vitest'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MantineProvider } from '@mantine/core'
import KeyField from './KeyField'

function mount(save: () => Promise<{ ok: boolean; error: string | null; restart_required?: boolean; disabled_in_config?: boolean; notice?: string }>) {
  // `found` flips to true after the save, as the real status refresh does.
  const Wrapper = () => {
    const [found, setFound] = useState(false)
    return (
      <MantineProvider>
        <KeyField
          label="Wolfram"
          found={found}
          hint={found ? '…abcd' : null}
          link="https://example.invalid"
          linkLabel="get a key"
          save={save}
          onSaved={() => setFound(true)}
        />
      </MantineProvider>
    )
  }
  return render(<Wrapper />)
}

describe('KeyField restart notice', () => {
  it('shows the restart notice when the server says restart_required', async () => {
    mount(vi.fn().mockResolvedValue({ ok: true, error: null, restart_required: true }))
    fireEvent.change(screen.getByLabelText('Wolfram'), { target: { value: 'WOLF-SYNTH1' } })
    fireEvent.click(screen.getByText('Save'))
    await waitFor(() => expect(screen.getByRole('status').textContent).toMatch(/Restart Daemon/))
  })

  it('says the tool is disabled in config, not that a restart will enable it', async () => {
    mount(vi.fn().mockResolvedValue({
      ok: true, error: null, disabled_in_config: true,
      notice: 'Saved, but this tool is disabled in your config. Restarting alone will not enable it.',
    }))
    fireEvent.change(screen.getByLabelText('Wolfram'), { target: { value: 'WOLF-SYNTH1' } })
    fireEvent.click(screen.getByText('Save'))
    await waitFor(() => expect(screen.getByRole('status').textContent).toMatch(/disabled in your config/))
    expect(screen.getByRole('status').textContent).not.toMatch(/Restart Daemon/)
  })

  it('shows no notice on a live save', async () => {
    mount(vi.fn().mockResolvedValue({ ok: true, error: null }))
    fireEvent.change(screen.getByLabelText('Wolfram'), { target: { value: 'WOLF-SYNTH1' } })
    fireEvent.click(screen.getByText('Save'))
    await waitFor(() => expect(screen.getByText(/found/)).toBeTruthy())
    expect(screen.queryByRole('status')).toBeNull()
  })
})
