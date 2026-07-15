import { useEffect, useState } from 'react'
import { Badge, Box, Divider, Group, ScrollArea, Select, Stack, Text } from '@mantine/core'
import type { DebugRecord } from '../../api/types'

// "What Daemon remembered and why" — renders the per-turn debug record the
// pipeline already produces (mode, model, tokens, phase waterfall, retrieval
// task timings, citations). Pure data rendering; no extra core plumbing.

function toMs(v: unknown): number {
  const n = typeof v === 'number' ? v : parseFloat(String(v))
  return Number.isFinite(n) ? n * 1000 : 0
}

function Waterfall({ title, timings }: { title: string; timings: Record<string, unknown> }) {
  const entries = Object.entries(timings || {})
    .map(([k, v]) => [k, toMs(v)] as const)
    .filter(([, ms]) => ms > 0)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 8)
  if (!entries.length) return null
  const max = entries[0][1]

  return (
    <Box>
      <Text size="xs" fw={700} c="dimmed" mb={4}>
        {title}
      </Text>
      <Stack gap={4}>
        {entries.map(([name, ms]) => (
          <Box key={name}>
            <Group justify="space-between" gap="xs">
              <Text size="xs" truncate maw="60%">
                {name}
              </Text>
              <Text size="xs" c="dimmed">
                {ms >= 1000 ? `${(ms / 1000).toFixed(2)}s` : `${Math.round(ms)}ms`}
              </Text>
            </Group>
            <Box
              h={6}
              w={`${Math.max(3, (ms / max) * 100)}%`}
              bg="daemonBlue.5"
              style={{ borderRadius: 3 }}
            />
          </Box>
        ))}
      </Stack>
    </Box>
  )
}

export default function MemoryPanel({ records }: { records: DebugRecord[] }) {
  const [selected, setSelected] = useState<string | null>(null)

  // Follow the latest turn as new records arrive
  useEffect(() => {
    if (records.length) setSelected(String(records.length - 1))
  }, [records.length])

  if (!records.length) {
    return (
      <Text size="sm" c="dimmed" p="md">
        Send a message — this panel shows what Daemon retrieved, how long each
        pipeline phase took, and what it cost in tokens.
      </Text>
    )
  }

  const idx = Math.min(parseInt(selected ?? '0', 10) || 0, records.length - 1)
  const rec = records[idx] as Record<string, any>
  const citations: unknown[] = Array.isArray(rec.citations) ? rec.citations : []

  return (
    <ScrollArea h="100%" offsetScrollbars>
      <Stack gap="md" p="md">
        <Select
          size="xs"
          label="Turn"
          data={records.map((r: any, i) => ({
            value: String(i),
            label: `Turn ${i + 1} — ${r.mode ?? '?'}`,
          }))}
          value={String(idx)}
          onChange={setSelected}
          allowDeselect={false}
        />

        <Group gap="xs">
          {rec.mode && <Badge variant="light">{rec.mode}</Badge>}
          {rec.model && (
            <Badge variant="light" color="gray">
              {rec.model}
            </Badge>
          )}
          {rec.total_tokens != null && (
            <Badge variant="light" color="teal">
              {rec.total_tokens} tok
            </Badge>
          )}
        </Group>

        <Waterfall title="Pipeline phases" timings={rec.phase_timings || {}} />
        <Waterfall title="Retrieval tasks" timings={rec.task_timings || {}} />

        {Array.isArray(rec.activity_log) && rec.activity_log.length > 0 && (
          <Box>
            <Divider mb="xs" />
            <Text size="xs" fw={700} c="dimmed" mb={4}>
              Agent activity ({rec.activity_log.length} steps)
            </Text>
            <Stack gap={2}>
              {rec.activity_log.map((line: string, i: number) => (
                <Text key={i} size="xs" c="dimmed" lineClamp={1}>
                  {line}
                </Text>
              ))}
            </Stack>
          </Box>
        )}

        {citations.length > 0 && (
          <Box>
            <Divider mb="xs" />
            <Text size="xs" fw={700} c="dimmed" mb={4}>
              Memories referenced ({citations.length})
            </Text>
            <Stack gap={4}>
              {citations.slice(0, 10).map((c: any, i) => (
                <Text key={i} size="xs" c="dimmed" lineClamp={2}>
                  • {typeof c === 'string' ? c : c?.content ?? c?.id ?? JSON.stringify(c)}
                </Text>
              ))}
            </Stack>
          </Box>
        )}

        {rec.provenance && (
          <Box>
            <Divider mb="xs" />
            <Text size="xs" fw={700} c="dimmed" mb={4}>
              Provenance
            </Text>
            <Text size="xs" c="dimmed" style={{ wordBreak: 'break-all' }}>
              session {String(rec.provenance.session_id ?? '?')} ·{' '}
              {String(rec.provenance.response_mode ?? '?')}
              {Array.isArray(rec.provenance.cited_ids) && rec.provenance.cited_ids.length
                ? ` · ${rec.provenance.cited_ids.length} cited memories`
                : ''}
            </Text>
          </Box>
        )}
      </Stack>
    </ScrollArea>
  )
}
