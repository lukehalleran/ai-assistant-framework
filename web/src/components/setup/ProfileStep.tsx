import { useState } from 'react'
import { Button, Stack, Text, TextInput } from '@mantine/core'
import { errText, type StepProps } from './types'

export default function ProfileStep({ client, status, refresh, next }: StepProps) {
  const prof = status.steps.profile
  const [name, setName] = useState(prof.name ?? '')
  const [pronouns, setPronouns] = useState(prof.pronouns ?? 'they/them')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const save = async () => {
    setBusy(true)
    setError(null)
    try {
      const res = await client.postSetupProfile({
        name: name.trim(),
        pronouns: pronouns.trim() || null,
        style: status.steps.style.value,
      })
      if (!res.ok) {
        setError(res.error ?? 'Could not save your profile.')
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
        How should Daemon address you?
      </Text>
      <TextInput
        label="Your name"
        required
        value={name}
        onChange={(e) => setName(e.currentTarget.value)}
        error={error}
      />
      <TextInput
        label="Pronouns (optional)"
        value={pronouns}
        onChange={(e) => setPronouns(e.currentTarget.value)}
      />
      <Button loading={busy} disabled={!name.trim()} onClick={save} style={{ alignSelf: 'flex-start' }}>
        Save
      </Button>
    </Stack>
  )
}
