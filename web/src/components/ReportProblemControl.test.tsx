import { afterEach, describe, expect, it, vi, type Mock } from 'vitest'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MantineProvider } from '@mantine/core'
import ReportProblemControl, { reportFilename } from './ReportProblemControl'

const TURNS = [
  { index: -1, ts: '2026-10-12T10:00:00', mode: 'enhanced', preview: 'newest message' },
  { index: -2, ts: '2026-10-12T09:00:00', mode: 'basic', preview: 'older message' },
]
const BUNDLE = {
  schema_version: 1,
  description: 'it forgot',
  app: { version: '1.0', mode: 'user' },
  turn: { ts: 't', mode: 'enhanced', intent: 'general' },
  full: { prompt: 'SECRET PROMPT', response: 'SECRET REPLY' },
}

type Build = (i: number, d: string, f: boolean) => Promise<Record<string, unknown>>
function setup(over: { build?: Mock<Build> } = {}) {
  const getTurns = vi.fn(async () => ({ turns: TURNS }))
  const build = over.build ?? vi.fn<Build>(async () => BUNDLE)
  render(
    <MantineProvider>
      <ReportProblemControl getTurns={getTurns} build={build} />
    </MantineProvider>,
  )
  fireEvent.click(screen.getByRole('button', { name: 'Report a problem' }))
  return { getTurns, build }
}

afterEach(() => vi.restoreAllMocks())

describe('ReportProblemControl', () => {
  it('selects the newest turn by default and sends include_full=false', async () => {
    const { build } = setup()
    await screen.findByText(/newest message/)
    await waitFor(() => expect(screen.getByLabelText(/newest message/)).toBeChecked())
    expect(screen.getByLabelText(/Include the full prompt/)).not.toBeChecked()
    fireEvent.click(screen.getByRole('button', { name: 'Preview' }))
    await waitFor(() => expect(build).toHaveBeenCalledWith(-1, '', false))
  })

  it('Preview renders the server JSON', async () => {
    setup()
    await screen.findByText(/newest message/)
    fireEvent.click(screen.getByRole('button', { name: 'Preview' }))
    const pre = await screen.findByTestId('report-preview')
    expect(pre.textContent).toContain('"schema_version": 1')
    expect(pre.textContent).toContain('it forgot')
  })

  it('Download creates a Blob with the right filename', async () => {
    const create = vi.fn(() => 'blob:x')
    const revoke = vi.fn()
    Object.assign(URL, { createObjectURL: create, revokeObjectURL: revoke })
    let name = ''
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(function (this: HTMLAnchorElement) {
      name = this.download
    })
    setup()
    await screen.findByText(/newest message/)
    fireEvent.click(screen.getByRole('button', { name: 'Download' }))
    await waitFor(() => expect(create).toHaveBeenCalledTimes(1))
    const blob = (create.mock.calls[0] as unknown as [Blob])[0]
    expect(blob).toBeInstanceOf(Blob)
    expect(blob.type).toBe('application/json')
    expect(name).toMatch(/^daemon_report_\d{8}_\d{6}\.json$/)
    expect(reportFilename(new Date(2026, 9, 12, 8, 5, 3))).toBe('daemon_report_20261012_080503.json')
    expect(revoke).toHaveBeenCalledWith('blob:x')
  })

  it('Copy summary writes text without the full part', async () => {
    const writeText = vi.fn().mockResolvedValue(undefined)
    Object.defineProperty(navigator, 'clipboard', { value: { writeText }, configurable: true })
    setup()
    await screen.findByText(/newest message/)
    fireEvent.click(screen.getByRole('button', { name: 'Copy summary' }))
    await waitFor(() => expect(writeText).toHaveBeenCalledTimes(1))
    const text = writeText.mock.calls[0][0] as string
    expect(text).toContain('intent: general')
    expect(text).not.toContain('SECRET')
  })

  it('an API error renders inline', async () => {
    const build = vi.fn<Build>().mockRejectedValue(new Error('Send a message first.'))
    setup({ build })
    await screen.findByText(/newest message/)
    fireEvent.click(screen.getByRole('button', { name: 'Preview' }))
    expect(await screen.findByText('Send a message first.')).toBeInTheDocument()
  })

  it('checking include-full sends true', async () => {
    const { build } = setup()
    await screen.findByText(/newest message/)
    fireEvent.click(screen.getByLabelText(/Include the full prompt/))
    fireEvent.click(screen.getByRole('button', { name: 'Preview' }))
    await waitFor(() => expect(build).toHaveBeenCalledWith(-1, '', true))
  })
})
