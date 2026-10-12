import { useState } from 'react'
import { Button, Group, Stack, Text, TextInput } from '@mantine/core'
import { errText, type StepProps } from './types'

export default function ObsidianStep({ client, status, refresh, next }: StepProps) {
  const [path, setPath] = useState(status.steps.obsidian.path ?? '')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const save = async () => {
    setBusy(true)
    setError(null)
    try {
      const res = await client.postSetupObsidian(path.trim() || null)
      if (!res.ok) {
        setError(res.error ?? 'Could not save this folder.')
        return
      }
      await refresh()
      next()
    } catch (err) {
      setError(errText(err))
    } finally {
      setBusy(false)
    }
  }

  return (
    <Stack gap="sm">
      <Text size="sm" c="dimmed">
        Optional. Point Daemon at an Obsidian vault folder to remember your notes. Leave empty for no
        vault.
      </Text>
      <TextInput
        label="Vault folder path"
        placeholder="/home/you/Documents/Vault"
        value={path}
        onChange={(e) => setPath(e.currentTarget.value)}
        error={error}
      />
      <Group gap="xs">
        <Button loading={busy} onClick={save}>
          Save
        </Button>
        <Button variant="subtle" onClick={next}>
          Skip
        </Button>
      </Group>
    </Stack>
  )
}
