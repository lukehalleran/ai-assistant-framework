import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { act, render, screen } from '@testing-library/react'
import { MantineProvider } from '@mantine/core'
import StaleSessionBanner from '../components/StaleSessionBanner'
import {
  __resetStaleSessionForTests,
  authorizedFetch,
  isStaleSession,
  onStaleSession,
} from './launchAuth'

// BC-80: a tab holding a pre-restart launch token gets the LaunchAuthMiddleware's
// rejection on every /api/* call. Only THAT rejection (status + detail from
// api/launch_auth.py) raises the stale flag; a route's own 403 must not.

function stubStatus(status: number, detail: string) {
  vi.stubGlobal(
    'fetch',
    vi.fn(async () => new Response(JSON.stringify({ detail }), { status })),
  )
}

describe('stale launch token', () => {
  beforeEach(() => __resetStaleSessionForTests())
  afterEach(() => vi.unstubAllGlobals())

  it('a middleware-shaped 401 sets the flag, notifies, and still returns the response', async () => {
    stubStatus(401, 'unauthorized')
    const cb = vi.fn()
    onStaleSession(cb)
    const resp = await authorizedFetch('/api/models')
    expect(resp.status).toBe(401)
    expect(isStaleSession()).toBe(true)
    expect(cb).toHaveBeenCalledTimes(1)
  })

  it('a middleware origin rejection (403) does not claim a restart', async () => {
    stubStatus(403, 'origin not allowed')
    await authorizedFetch('/api/session', { method: 'DELETE' })
    expect(isStaleSession()).toBe(false)
  })

  it("a route's own 403 (/api/shutdown loopback-only) does not", async () => {
    stubStatus(403, 'shutdown is loopback-only')
    await authorizedFetch('/api/shutdown', { method: 'POST' })
    expect(isStaleSession()).toBe(false)
  })

  it('a 401 with a different detail does not', async () => {
    stubStatus(401, 'bad api key')
    await authorizedFetch('/api/setup/openrouter', { method: 'POST' })
    expect(isStaleSession()).toBe(false)
  })

  it('the banner appears when set, with a Reload button', async () => {
    render(
      <MantineProvider>
        <StaleSessionBanner />
      </MantineProvider>,
    )
    expect(screen.queryByText('This page is out of date')).not.toBeInTheDocument()
    stubStatus(401, 'unauthorized')
    await act(async () => {
      await authorizedFetch('/api/models')
    })
    expect(await screen.findByText('This page is out of date')).toBeInTheDocument()
    expect(screen.getByText(/Daemon restarted since this tab was opened/)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Reload' })).toBeInTheDocument()
  })

  it('the banner stays absent for a route 403', async () => {
    render(
      <MantineProvider>
        <StaleSessionBanner />
      </MantineProvider>,
    )
    stubStatus(403, 'shutdown is loopback-only')
    await act(async () => {
      await authorizedFetch('/api/shutdown', { method: 'POST' })
    })
    expect(screen.queryByText('This page is out of date')).not.toBeInTheDocument()
  })
})
