import { useState } from 'react'
import { Anchor, Badge, Button, Group, PasswordInput, Stack, Text } from '@mantine/core'
import type { SetupResult } from '../../api/client'
import { errText } from './types'

/** One key: a "found" badge (masked hint) hiding the input until Replace, or
 * a PasswordInput + save button. The typed key lives in state only until the
 * server accepts it, then is cleared; it is never logged. */
export default function KeyField({
  label,
  found,
  hint,
  link,
  linkLabel,
  saveLabel = 'Save',
  save,
  onSaved,
  offerSaveAnyway,
}: {
  label: string
  found: boolean
  hint: string | null
  link: string
  linkLabel: string
  saveLabel?: string
  save: (key: string, unverified: boolean) => Promise<SetupResult>
  onSaved: () => Promise<void> | void
  offerSaveAnyway?: boolean
}) {
  const [replacing, setReplacing] = useState(false)
  const [key, setKey] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [unreachable, setUnreachable] = useState(false)
  const [restartNeeded, setRestartNeeded] = useState(false)

  const run = async (unverified: boolean) => {
    setBusy(true)
    setError(null)
    try {
      const res = await save(key.trim(), unverified)
      if (res.ok) {
        setKey('')
        setReplacing(false)
        setUnreachable(false)
        setRestartNeeded(res.restart_required === true)
        await onSaved()
      } else {
        setError(res.error ?? 'Could not save this key.')
        setUnreachable(
          !!offerSaveAnyway && (res.unreachable === true || /could not reach/i.test(res.error ?? '')),
        )
      }
    } catch (err) {
      setError(errText(err))
    } finally {
      setBusy(false)
    }
  }

  if (found && !replacing) {
    return (
      <Group justify="space-between">
        <Group gap="xs">
          <Text size="sm">{label}</Text>
          <Badge color="green" variant="light">
            found{hint ? ` ${hint}` : ''}
          </Badge>
          {restartNeeded && (
            <Text size="xs" c="orange" role="status">
              Saved. Restart Daemon for this feature to start using it.
            </Text>
          )}
        </Group>
        <Button variant="subtle" size="xs" onClick={() => setReplacing(true)}>
          Replace
        </Button>
      </Group>
    )
  }

  return (
    <Stack gap="xs">
      <PasswordInput
        label={label}
        value={key}
        onChange={(e) => setKey(e.currentTarget.value)}
        error={error}
        autoComplete="off"
        description={
          <Anchor href={link} target="_blank" rel="noreferrer" size="xs">
            {linkLabel}
          </Anchor>
        }
      />
      <Group gap="xs">
        <Button size="xs" loading={busy} disabled={!key.trim()} onClick={() => run(false)}>
          {saveLabel}
        </Button>
        {unreachable && (
          <Button size="xs" variant="outline" loading={busy} onClick={() => run(true)}>
            Save anyway
          </Button>
        )}
        {found && (
          <Button size="xs" variant="subtle" onClick={() => setReplacing(false)}>
            Cancel
          </Button>
        )}
      </Group>
    </Stack>
  )
}
