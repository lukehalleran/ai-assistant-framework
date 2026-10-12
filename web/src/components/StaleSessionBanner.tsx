import { useSyncExternalStore } from 'react'
import { Alert, Button, Group } from '@mantine/core'
import { isStaleSession, onStaleSession } from '../api/launchAuth'

/** BC-80: shown when the launch-auth middleware rejected this tab's token —
 * i.e. Daemon restarted since the page was opened. Renders nothing otherwise. */
export default function StaleSessionBanner() {
  const stale = useSyncExternalStore(onStaleSession, isStaleSession, () => false)
  if (!stale) return null
  return (
    <div style={{ position: 'fixed', top: 0, left: 0, right: 0, zIndex: 10000 }}>
    <Alert color="red" variant="filled" radius={0} title="This page is out of date" role="alert">
      <Group justify="space-between" wrap="nowrap">
        <span>Daemon restarted since this tab was opened. Reload the page to reconnect.</span>
        <Button size="xs" variant="white" color="red" onClick={() => window.location.reload()}>
          Reload
        </Button>
      </Group>
    </Alert>
    </div>
  )
}
