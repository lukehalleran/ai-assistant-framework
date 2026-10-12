import { useState } from 'react'
import { Button, Stack, Text } from '@mantine/core'
import { errText } from './types'

export default function FinishStep({ onFinish }: { onFinish: () => Promise<void> }) {
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const go = async () => {
    setBusy(true)
    setError(null)
    try {
      await onFinish()
    } catch (err) {
      setError(errText(err))
    } finally {
      setBusy(false)
    }
  }
  return (
    <Stack gap="sm">
      <Text size="sm">All set. You can change any of this later in Settings.</Text>
      {error && (
        <Text size="sm" c="red">
          {error}
        </Text>
      )}
      <Button loading={busy} onClick={go} style={{ alignSelf: 'flex-start' }}>
        Start chatting
      </Button>
    </Stack>
  )
}
