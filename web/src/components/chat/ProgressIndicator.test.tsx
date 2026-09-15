import type { ComponentProps } from 'react'
import { describe, expect, it } from 'vitest'
import { render, screen } from '@testing-library/react'
import { MantineProvider } from '@mantine/core'
import ProgressIndicator from './ProgressIndicator'

// F13c-2a: owner decision 4 (PARENT_STATE.md, REVISED) — a failed memory
// save is a status-bar notice here, never chat text. The fixed string is
// "Memory save failed"; the server's own `storage_failed` label must never
// render. Drives the real component (MantineProvider pattern per
// ActionApprovalCard.test.tsx).

function renderIndicator(props: Partial<ComponentProps<typeof ProgressIndicator>> = {}) {
  return render(
    <MantineProvider>
      <ProgressIndicator
        streaming={false}
        progressText=""
        thinkingText=""
        startedAt={null}
        {...props}
      />
    </MantineProvider>,
  )
}

describe('ProgressIndicator storage notice (F13c-2a)', () => {
  it('not streaming with storageNotice: shows the fixed "Memory save failed" text with role=status', () => {
    renderIndicator({ storageNotice: true })
    const status = screen.getByRole('status')
    expect(status).toHaveTextContent('Memory save failed')
  })

  it('not streaming without storageNotice: renders nothing (paired control)', () => {
    // Not `toBeEmptyDOMElement()`: MantineProvider injects its own
    // <style data-mantine-styles> elements regardless of what the component
    // renders, so the container is never truly empty. Assert the component's
    // own output instead: no status role and no notice text.
    renderIndicator({ storageNotice: false })
    expect(screen.queryByRole('status')).not.toBeInTheDocument()
    expect(screen.queryByText('Memory save failed')).not.toBeInTheDocument()
  })

  it('streaming with storageNotice: shows the typing label, never the notice', () => {
    renderIndicator({ streaming: true, storageNotice: true })
    expect(screen.getByText('Assistant is typing…')).toBeInTheDocument()
    expect(screen.queryByText('Memory save failed')).not.toBeInTheDocument()
    expect(screen.queryByRole('status')).not.toBeInTheDocument()
  })

  it('never renders a server-provided label in place of the fixed string', () => {
    renderIndicator({ storageNotice: true })
    expect(screen.queryByText('storage_failed')).not.toBeInTheDocument()
    expect(screen.queryByText(/chroma|disk full|write failed/i)).not.toBeInTheDocument()
  })
})
