import { useState } from 'react'
import { Button, Group, Radio, Stack, Text } from '@mantine/core'
import { errText, type StepProps } from './types'

// Values are gui/wizard.py parse_style_preference's outputs.
const STYLES = [
  { value: 'warm', label: 'Warm & supportive', hint: 'Longer, empathetic replies' },
  { value: 'balanced', label: 'Balanced', hint: 'Adapts to the moment (default)' },
  { value: 'direct', label: 'Direct & concise', hint: 'Short and to the point' },
]

export default function StyleStep({ client, status, refresh, next }: StepProps) {
  const [value, setValue] = useState(status.steps.style.value ?? 'balanced')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const save = async () => {
    setBusy(true)
    setError(null)
    try {
      const prof = status.steps.profile
      const res = await client.postSetupProfile({
        name: prof.name ?? '',
        pronouns: prof.pronouns,
        style: value,
      })
      if (!res.ok) {
        setError(res.error ?? 'Could not save your style.')
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
        Optional. You can change this later in Settings.
      </Text>
      <Radio.Group value={value} onChange={setValue} error={error}>
        <Stack gap="xs">
          {STYLES.map((s) => (
            <Radio key={s.value} value={s.value} label={s.label} description={s.hint} />
          ))}
        </Stack>
      </Radio.Group>
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
