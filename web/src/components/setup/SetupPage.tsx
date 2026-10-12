import { useState } from 'react'
import { Button, Card, Container, Group, Progress, Stack, Text, Title } from '@mantine/core'
import { api, type SetupStatus } from '../../api/client'
import QuitDaemonControl from '../QuitDaemonControl'
import FinishStep from './FinishStep'
import ObsidianStep from './ObsidianStep'
import OpenRouterStep from './OpenRouterStep'
import OptionalKeysStep from './OptionalKeysStep'
import ProfileStep from './ProfileStep'
import StyleStep from './StyleStep'
import type { SetupClient, StepProps } from './types'

const STEPS = [
  { id: 'openrouter', title: 'OpenRouter API key', required: true },
  { id: 'profile', title: 'About you', required: true },
  { id: 'style', title: 'Response style', required: false },
  { id: 'keys', title: 'Optional keys', required: false },
  { id: 'obsidian', title: 'Obsidian vault', required: false },
  { id: 'finish', title: 'Finish', required: false },
] as const

/** Resumable: the first REQUIRED step still missing, else the first optional one. */
export function initialStepIndex(status: SetupStatus): number {
  if (status.steps.openrouter.state !== 'found') return 0
  if (status.steps.profile.state !== 'found') return 1
  return 2
}

export default function SetupPage({
  initialStatus,
  onFinished,
  onShutDown,
  quitAllowed = false,
  client = api,
}: {
  initialStatus: SetupStatus
  /** Called after POST /api/setup/complete succeeds; the caller re-reads status. */
  onFinished: () => void | Promise<void>
  onShutDown: () => void
  quitAllowed?: boolean
  client?: SetupClient
}) {
  const [status, setStatus] = useState(initialStatus)
  const [index, setIndex] = useState(() => initialStepIndex(initialStatus))

  const refresh = async () => {
    try {
      setStatus(await client.getSetupStatus())
    } catch {
      // keep the last known status; the save itself already succeeded
    }
  }
  const next = () => setIndex((i) => Math.min(i + 1, STEPS.length - 1))
  const stepProps: StepProps = { client, status, refresh, next }
  const step = STEPS[index]

  const finish = async () => {
    await client.postSetupComplete()
    await onFinished()
  }

  return (
    <Container size="sm" py="xl">
      <Stack gap="md">
        <Group justify="space-between">
          <Title order={2}>Welcome to Daemon</Title>
          <QuitDaemonControl onShutDown={onShutDown} quitAllowed={quitAllowed} />
        </Group>
        <Progress value={((index + 1) / STEPS.length) * 100} size="xs" />
        <Card withBorder padding="lg">
          <Stack gap="md">
            <Text fw={600}>
              Step {index + 1} of {STEPS.length}: {step.title}
              {step.required ? '' : ' (optional)'}
            </Text>
            {step.id === 'openrouter' && <OpenRouterStep {...stepProps} />}
            {step.id === 'profile' && <ProfileStep {...stepProps} />}
            {step.id === 'style' && <StyleStep {...stepProps} />}
            {step.id === 'keys' && <OptionalKeysStep {...stepProps} />}
            {step.id === 'obsidian' && <ObsidianStep {...stepProps} />}
            {step.id === 'finish' && <FinishStep onFinish={finish} />}
            {index > 0 && (
              <Button
                variant="subtle"
                size="xs"
                style={{ alignSelf: 'flex-start' }}
                onClick={() => setIndex((i) => Math.max(i - 1, 0))}
              >
                Back
              </Button>
            )}
          </Stack>
        </Card>
      </Stack>
    </Container>
  )
}
