import { Accordion, Badge, Group, Text } from '@mantine/core'
import type { DuelThinking } from '../../api/types'
import MarkdownMessage from './MarkdownMessage'

// Duel mode: two models raced, a judge picked the winner — show both chains.
export default function ThinkingBlock({ duel }: { duel: DuelThinking }) {
  const { thinking_a, thinking_b, model_a, model_b, winner, scores } = duel
  if (!thinking_a && !thinking_b) return null

  return (
    <Accordion variant="contained" radius="md" maw="85%" style={{ alignSelf: 'flex-start' }}>
      <Accordion.Item value="duel">
        <Accordion.Control>
          <Group gap="xs">
            <Text size="sm">💭 Thinking (duel mode)</Text>
            {winner && (
              <Badge size="xs" variant="light">
                🏆 Model {winner}
                {scores ? ` (A=${scores.A ?? '–'}, B=${scores.B ?? '–'})` : ''}
              </Badge>
            )}
          </Group>
        </Accordion.Control>
        <Accordion.Panel>
          {thinking_a && (
            <>
              <Text size="xs" fw={700} c="dimmed">
                {model_a || 'Model A'}
              </Text>
              <MarkdownMessage content={thinking_a} />
            </>
          )}
          {thinking_b && (
            <>
              <Text size="xs" fw={700} c="dimmed" mt="sm">
                {model_b || 'Model B'}
              </Text>
              <MarkdownMessage content={thinking_b} />
            </>
          )}
        </Accordion.Panel>
      </Accordion.Item>
    </Accordion>
  )
}
