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
import { debugBaselineFor } from '../../api/debugSession'
import type { DebugRecord } from '../../api/types'

// Debug Trace (Gradio dev tab → SPA, 2026-07-14): per-turn Query → Prompt →
// Response with token counts, timing waterfall, and full-prompt TXT export.
// Like the Gradio tab, only the ongoing UI session's turns are shown — the
// server-held backlog before this page load is hidden via debugSession's
// baseline (absolute indices are kept for the prompt-export links).

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

// Assemble one turn's full debug output as plain text with section headers —
// the single-click copy target (replaces the old per-block copy buttons).
function recordToText(rec: DebugRecord, index: number): string {
  const lines: string[] = []
  const totalWall = rec.phase_timings?.total_wall
  lines.push(
    `=== TURN #${index + 1} — ${rec.mode || 'enhanced'} · ${rec.model || '?'}` +
      `${totalWall ? ` · ${seconds(totalWall)}` : ''} ===`,
  )
  if (rec.prompt_tokens != null) {
    lines.push(
      `Tokens — prompt: ${rec.prompt_tokens} · system: ${rec.system_tokens ?? 0} · ` +
        `total: ${rec.total_tokens ?? rec.prompt_tokens}`,
    )
  }
  const timingBlock = (title: string, timings?: Record<string, number> | null) => {
    const entries = Object.entries(timings || {})
      .filter(([, v]) => Number.isFinite(v) && v > 0.01)
      .sort((a, b) => b[1] - a[1])
    if (!entries.length) return
    lines.push('', `--- ${title} ---`)
    for (const [name, v] of entries) lines.push(`${name}: ${seconds(v)}`)
  }
  timingBlock('PIPELINE PHASES', rec.phase_timings)
  timingBlock('RETRIEVAL TASKS', rec.task_timings)
  const textBlock = (title: string, text?: string | null) => {
    if (!text) return
    lines.push('', `--- ${title} ---`, text)
  }
  textBlock('QUERY', rec.query)
  textBlock('PROMPT', rec.prompt)
  textBlock('SYSTEM PROMPT', rec.system_prompt)
  textBlock('RESPONSE', rec.response)
  return lines.join('\n')
}

// Clipboard write that also works in INSECURE contexts (the SPA is usually
// reached over plain HTTP via Tailscale — on Android navigator.clipboard is
// undefined there, which made Mantine's CopyButton silently no-op).
async function copyText(text: string): Promise<boolean> {
  try {
    if (window.isSecureContext && navigator.clipboard) {
      await navigator.clipboard.writeText(text)
      return true
    }
  } catch {
    // fall through to the legacy path
  }
  try {
    const ta = document.createElement('textarea')
    ta.value = text
    ta.style.position = 'fixed'
    ta.style.opacity = '0'
    document.body.appendChild(ta)
    ta.focus()
    ta.select()
    const ok = document.execCommand('copy')
    document.body.removeChild(ta)
    return ok
  } catch {
    return false
  }
}

function CopyAllButton({ value, label }: { value: string; label: string }) {
  const [copied, setCopied] = useState(false)
  const onClick = async () => {
    const ok = await copyText(value)
    if (ok) {
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    } else {
      notifications.show({
        color: 'red',
        title: 'Copy failed',
        message: 'Clipboard is unavailable in this browser context.',
      })
    }
  }
  return (
    <Button
      size="compact-xs"
      variant={copied ? 'light' : 'outline'}
      color={copied ? 'teal' : 'gray'}
      onClick={onClick}
    >
      {copied ? '✓ Copied' : `📋 ${label}`}
    </Button>
  )
}

export default function DebugPage() {
  const [records, setRecords] = useState<DebugRecord[]>([])
  // Absolute index of this session's first record (for prompt-export links)
  const [baseline, setBaseline] = useState(0)
  const [loading, setLoading] = useState(false)

  const refresh = useCallback(() => {
    setLoading(true)
    api
      .getDebugRecords()
      .then(async (r) => {
        const base = await debugBaselineFor(r.count)
        setBaseline(base)
        setRecords(r.records.slice(base))
      })
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
          <Group gap="xs">
            {records.length > 0 && (
              <CopyAllButton
                label="Copy session"
                value={records.map((r, i) => recordToText(r, i)).join('\n\n\n')}
              />
            )}
            <Button size="xs" variant="outline" color="gray" loading={loading} onClick={refresh}>
              Refresh
            </Button>
          </Group>
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
                    <Group justify="space-between">
                      {rec.prompt_tokens != null ? (
                        <Text size="xs" c="dimmed">
                          Tokens — prompt: {rec.prompt_tokens} · system: {rec.system_tokens ?? 0} ·
                          total: {rec.total_tokens ?? rec.prompt_tokens}
                        </Text>
                      ) : (
                        <span />
                      )}
                      <CopyAllButton label="Copy all" value={recordToText(rec, i)} />
                    </Group>
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
                        href={api.promptExportUrl(baseline + i)}
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
