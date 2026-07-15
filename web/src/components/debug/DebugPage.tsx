import { useCallback, useEffect, useState } from 'react'
import {
  Accordion,
  Badge,
  Box,
  Button,
  Code,
  Group,
  ScrollArea,
  Spoiler,
  Stack,
  Text,
} from '@mantine/core'
import { notifications } from '@mantine/notifications'
import { api } from '../../api/client'
import type { DebugRecord } from '../../api/types'

// Debug Trace (Gradio dev tab → SPA, 2026-07-14): per-turn Query → Prompt →
// Response with token counts, timing waterfall, and full-prompt TXT export.
// Reads the server-held records (survive page reloads), refreshed on demand.

function seconds(v: unknown): string {
  const n = typeof v === 'number' ? v : parseFloat(String(v))
  if (!Number.isFinite(n)) return '—'
  return n >= 1 ? `${n.toFixed(2)}s` : `${Math.round(n * 1000)}ms`
}

function TimingBars({ title, timings, max: maxOverride }: {
  title: string
  timings: Record<string, number>
  max?: number
}) {
  const entries = Object.entries(timings || {})
    .filter(([k, v]) => k !== 'total_wall' && Number.isFinite(v) && v > 0.01)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 8)
  if (!entries.length) return null
  const max = maxOverride || entries[0][1]

  return (
    <Box>
      <Text size="xs" fw={700} c="dimmed" mb={4}>
        {title}
      </Text>
      <Stack gap={4}>
        {entries.map(([name, v]) => (
          <Box key={name}>
            <Group justify="space-between" gap="xs">
              <Text size="xs" truncate maw="60%">
                {name}
              </Text>
              <Text size="xs" c="dimmed">
                {seconds(v)}
              </Text>
            </Group>
            <Box
              h={6}
              w={`${Math.max(3, (v / max) * 100)}%`}
              bg="daemonBlue.5"
              style={{ borderRadius: 3 }}
            />
          </Box>
        ))}
      </Stack>
    </Box>
  )
}

function PromptBlock({ label, text }: { label: string; text: string }) {
  if (!text) return null
  return (
    <Box>
      <Text size="xs" fw={700} c="dimmed" mb={4}>
        {label}
      </Text>
      <Spoiler maxHeight={160} showLabel="Show full text" hideLabel="Collapse">
        <Code block style={{ whiteSpace: 'pre-wrap', fontSize: 11 }}>
          {text}
        </Code>
      </Spoiler>
    </Box>
  )
}

export default function DebugPage() {
  const [records, setRecords] = useState<DebugRecord[]>([])
  const [loading, setLoading] = useState(false)

  const refresh = useCallback(() => {
    setLoading(true)
    api
      .getDebugRecords()
      .then((r) => setRecords(r.records))
      .catch((err) =>
        notifications.show({
          color: 'red',
          title: 'Debug records failed',
          message: err instanceof Error ? err.message : String(err),
        }),
      )
      .finally(() => setLoading(false))
  }, [])

  useEffect(refresh, [refresh])

  return (
    <ScrollArea h="calc(100dvh - 56px)" offsetScrollbars style={{ flex: 1, minWidth: 0 }}>
      <Stack gap="md" p="md" maw={960} mx="auto">
        <Group justify="space-between">
          <Text fw={700}>🔎 Query → Prompt → Response</Text>
          <Button size="xs" variant="outline" color="gray" loading={loading} onClick={refresh}>
            Refresh
          </Button>
        </Group>

        {!records.length && (
          <Text size="sm" c="dimmed">
            No debug entries yet this session. Send a message in Chat, then refresh.
          </Text>
        )}

        <Accordion
          multiple
          variant="separated"
          defaultValue={records.length ? [String(records.length - 1)] : []}
        >
          {records.map((rec, i) => {
            const totalWall = rec.phase_timings?.total_wall
            return (
              <Accordion.Item key={i} value={String(i)}>
                <Accordion.Control>
                  <Group gap="xs" wrap="nowrap">
                    <Badge size="sm" variant="light">
                      #{i + 1}
                    </Badge>
                    <Text size="sm" truncate style={{ flex: 1 }}>
                      {rec.query || '(no query)'}
                    </Text>
                    <Text size="xs" c="dimmed" style={{ whiteSpace: 'nowrap' }}>
                      {rec.mode || 'enhanced'} · {rec.model || '?'}
                      {rec.total_tokens ? ` · ${rec.total_tokens} tok` : ''}
                      {totalWall ? ` · ${seconds(totalWall)}` : ''}
                    </Text>
                  </Group>
                </Accordion.Control>
                <Accordion.Panel>
                  <Stack gap="md">
                    {rec.prompt_tokens != null && (
                      <Text size="xs" c="dimmed">
                        Tokens — prompt: {rec.prompt_tokens} · system: {rec.system_tokens ?? 0} ·
                        total: {rec.total_tokens ?? rec.prompt_tokens}
                      </Text>
                    )}
                    {rec.phase_timings && (
                      <TimingBars title="Pipeline phases" timings={rec.phase_timings} />
                    )}
                    {rec.task_timings && (
                      <TimingBars
                        title={`Retrieval tasks${rec.gather_elapsed ? ` (gather ${seconds(rec.gather_elapsed)})` : ''}`}
                        timings={rec.task_timings}
                      />
                    )}
                    <PromptBlock label="Query" text={rec.query || ''} />
                    <PromptBlock label="Prompt" text={rec.prompt || ''} />
                    {rec.system_prompt && (
                      <PromptBlock label="System prompt (dev mode)" text={rec.system_prompt} />
                    )}
                    <PromptBlock label="Response" text={rec.response || ''} />
                    <Group>
                      <Button
                        size="xs"
                        variant="outline"
                        component="a"
                        href={api.promptExportUrl(i)}
                      >
                        📥 Download full prompt as TXT
                      </Button>
                    </Group>
                  </Stack>
                </Accordion.Panel>
              </Accordion.Item>
            )
          })}
        </Accordion>
      </Stack>
    </ScrollArea>
  )
}
