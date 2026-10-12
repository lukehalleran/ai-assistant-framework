import { useState } from 'react'
import { Button, Center, Group, Modal, Stack, Text, Title } from '@mantine/core'
import { useDisclosure } from '@mantine/hooks'
import { api } from '../api/client'

/** App lifecycle (B3): the navbar "Quit Daemon" control. Confirms, asks the
 * server for its normal clean shutdown, then hands control to the caller so
 * it can replace the UI with <ShutdownScreen/>. */
export default function QuitDaemonControl({
  onShutDown,
  quitAllowed,
  shutdown = api.shutdown,
}: {
  onShutDown: () => void
  /** From the heartbeat: false/unknown hides the control (POST /api/shutdown is loopback-only). */
  quitAllowed: boolean
  shutdown?: () => Promise<void>
}) {
  const [opened, { open, close }] = useDisclosure(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const confirm = async () => {
    setBusy(true)
    setError(null)
    try {
      await shutdown()
      close()
      onShutDown()
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }

  if (!quitAllowed) return null

  return (
    <>
      <Button variant="outline" color="red" size="xs" onClick={open}>
        ⏻ Quit Daemon
      </Button>
      <Modal opened={opened} onClose={close} title="Quit Daemon?" centered>
        <Stack gap="md">
          <Text size="sm">
            Daemon will save this session and shut down. You will need to start it again to keep
            chatting.
          </Text>
          {error && (
            <Text size="sm" c="red">
              Could not quit: {error}
            </Text>
          )}
          <Group justify="flex-end">
            <Button variant="default" onClick={close} disabled={busy}>
              Cancel
            </Button>
            <Button color="red" onClick={confirm} loading={busy}>
              Quit
            </Button>
          </Group>
        </Stack>
      </Modal>
    </>
  )
}

export function ShutdownScreen() {
  return (
    <Center h="100dvh">
      <Stack align="center" gap="xs">
        <Title order={3}>Daemon has shut down</Title>
        <Text c="dimmed">You can close this tab.</Text>
      </Stack>
    </Center>
  )
}
