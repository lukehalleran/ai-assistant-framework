import { Button, Stack, Text } from '@mantine/core'
import KeyField from './KeyField'
import type { StepProps } from './types'

export default function OpenRouterStep({ client, status, refresh, next }: StepProps) {
  const step = status.steps.openrouter
  return (
    <Stack gap="sm">
      <Text size="sm" c="dimmed">
        Daemon talks to models through OpenRouter. Your key is checked with OpenRouter (no charge) and
        stored only on this machine.
      </Text>
      <KeyField
        label="OpenRouter API key"
        found={step.state === 'found'}
        hint={step.hint}
        link="https://openrouter.ai/keys"
        linkLabel="Get a key at openrouter.ai/keys"
        saveLabel="Test & save"
        offerSaveAnyway
        save={(key, unverified) => client.postSetupOpenrouter(key, unverified)}
        onSaved={async () => {
          await refresh()
          next()
        }}
      />
      {step.state === 'found' && (
        <Button onClick={next} style={{ alignSelf: 'flex-start' }}>
          Continue
        </Button>
      )}
    </Stack>
  )
}
