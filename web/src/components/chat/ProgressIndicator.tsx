import { useEffect, useState } from 'react'
import { Group, Loader, Text } from '@mantine/core'

interface Props {
  streaming: boolean
  progressText: string
  thinkingText: string
  startedAt: number | null
}

export default function ProgressIndicator({ streaming, progressText, thinkingText, startedAt }: Props) {
  const [elapsed, setElapsed] = useState(0)

  useEffect(() => {
    if (!streaming || !startedAt) return
    const id = setInterval(() => setElapsed((Date.now() - startedAt) / 1000), 100)
    return () => clearInterval(id)
  }, [streaming, startedAt])

  if (!streaming) return null

  const label = thinkingText ? `💭 ${thinkingText}` : progressText || 'Assistant is typing…'

  return (
    // wrap="nowrap" + truncate + miw:0 so a long status line shrinks/ellipsizes
    // instead of widening the page (horizontal-overflow "uncentered" bug).
    <Group gap="xs" px="md" pb={4} justify="flex-end" wrap="nowrap" maw="100%">
      <Loader size="xs" type="dots" style={{ flexShrink: 0 }} />
      <Text size="xs" c="dimmed" truncate style={{ minWidth: 0 }}>
        {label}
      </Text>
      <Text size="xs" c="dimmed" style={{ flexShrink: 0, whiteSpace: 'nowrap' }}>
        ⏱️ {elapsed.toFixed(1)}s
      </Text>
    </Group>
  )
}
