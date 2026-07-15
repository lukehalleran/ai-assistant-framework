import { useEffect, useRef } from 'react'
import { Box, Paper, Text } from '@mantine/core'

// Live agent-activity feed during long turns: every tool step the pipeline
// reports ("🔍 Searching for: …", "🧠 Searching conversations: …", "📄 Found
// 12 results") instead of a single opaque spinner for 90 seconds.
export default function ActivityLog({ log, streaming }: { log: string[]; streaming: boolean }) {
  const viewport = useRef<HTMLDivElement>(null)

  useEffect(() => {
    viewport.current?.scrollTo({ top: viewport.current.scrollHeight })
  }, [log.length])

  if (!streaming || log.length === 0) return null

  return (
    <Paper mx="md" mb={4} p="xs" radius="md" withBorder>
      <Box ref={viewport} mah={120} style={{ overflowY: 'auto' }}>
        {log.map((line, i) => (
          <Text
            key={i}
            size="xs"
            c={i === log.length - 1 ? undefined : 'dimmed'}
            style={{ whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}
          >
            {line}
          </Text>
        ))}
      </Box>
    </Paper>
  )
}
