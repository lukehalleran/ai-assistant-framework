import { Button, Divider, Group, Stack, Text } from '@mantine/core'
import KeyField from './KeyField'
import type { StepProps } from './types'

const KEYS = [
  { name: 'tavily', label: 'Tavily (web search)', link: 'https://app.tavily.com/', linkLabel: 'Get a Tavily key' },
  {
    name: 'wolfram',
    label: 'Wolfram Alpha',
    link: 'https://developer.wolframalpha.com/',
    linkLabel: 'Get a Wolfram Alpha key',
  },
  {
    name: 'e2b',
    label: 'E2B (code sandbox)',
    link: 'https://e2b.dev/dashboard',
    linkLabel: 'Get an E2B key',
    devOnly: true,
  },
] as const

export default function OptionalKeysStep({ client, status, refresh, next }: StepProps) {
  return (
    <Stack gap="sm">
      <Text size="sm" c="dimmed">
        Optional. Each key only unlocks its own feature; keys are format-checked, not tested.
      </Text>
      {KEYS.filter((k) => !('devOnly' in k) || status.mode === 'dev').map((k, i) => (
        <Stack key={k.name} gap="xs">
          {i > 0 && <Divider />}
          <KeyField
            label={k.label}
            found={status.steps[k.name].state === 'found'}
            hint={status.steps[k.name].hint}
            link={k.link}
            linkLabel={k.linkLabel}
            save={(key) => client.postSetupKey(k.name, key)}
            onSaved={refresh}
          />
        </Stack>
      ))}
      <Group gap="xs">
        <Button onClick={next}>Continue</Button>
        <Button variant="subtle" onClick={next}>
          Skip
        </Button>
      </Group>
    </Stack>
  )
}
