import { useCallback, useEffect, useState } from 'react'
import {
  Box,
  Button,
  Code,
  CopyButton,
  Group,
  ScrollArea,
  Select,
  Stack,
  Switch,
  Text,
} from '@mantine/core'
import { notifications } from '@mantine/notifications'
import { api } from '../../api/client'
import { debugBaselineFor } from '../../api/debugSession'

// Provenance (Gradio tab → SPA, 2026-07-14): the per-turn provenance object —
// response mode, session id, cited memory ids, duel thinking/winner, agentic
// rounds — rendered as JSON, with a turn selector (Gradio showed latest only).
//
// 2026-09-04: same baseline bug as DebugPage — a page refresh re-captured
// the debugSession baseline AFTER the turn the owner just ran, hiding it
// from the turn selector even though GET /api/debug still returned it. The
// selector now defaults to EVERY record the server has (newest first);
// "only since this page load" is an explicit opt-in toggle (default OFF).
// Selector values stay ABSOLUTE indices for the API either way.

export default function ProvenancePage() {
  const [baseline, setBaseline] = useState(0)
  const [totalCount, setTotalCount] = useState(0)
  const [sinceLoadOnly, setSinceLoadOnly] = useState(false)
  const [selected, setSelected] = useState<string>('-1')
  const [prov, setProv] = useState<Record<string, unknown> | null>(null)
  const [loading, setLoading] = useState(false)

  const refresh = useCallback((index: string) => {
    setLoading(true)
    api
      .getDebugRecords()
      .then(async (records) => {
        const base = await debugBaselineFor(records.count)
        setBaseline(base)
        setTotalCount(records.count)
        const availableTurns = sinceLoadOnly ? records.count - base : records.count
        if (availableTurns <= 0) {
          // "Latest" would resolve to a nonexistent/pre-session record.
          setProv(null)
          return
        }
        setProv(await api.getProvenance(parseInt(index, 10)))
      })
      .catch((err) => {
        setProv(null)
        const msg = err instanceof Error ? err.message : String(err)
        if (!msg.includes('404')) {
          notifications.show({ color: 'red', title: 'Provenance failed', message: msg })
        }
      })
      .finally(() => setLoading(false))
  }, [sinceLoadOnly])

  useEffect(() => refresh(selected), [refresh, selected])

  const firstIndex = sinceLoadOnly ? baseline : 0
  const availableTurns = Math.max(0, totalCount - firstIndex)
  // Newest first: turn #1 in the dropdown is always the most recent record.
  const turnOptions = [
    { value: '-1', label: `Latest turn` },
    ...Array.from({ length: availableTurns }, (_, i) => {
      const absIndex = totalCount - 1 - i
      return { value: String(absIndex), label: `Turn #${absIndex + 1}` }
    }),
  ]

  const provText = prov ? JSON.stringify(prov, null, 2) : ''

  return (
    <ScrollArea h="calc(100dvh - 56px)" offsetScrollbars style={{ flex: 1, minWidth: 0 }}>
      <Stack gap="md" p="md" maw={960} mx="auto">
        <Group justify="space-between" wrap="wrap">
          <Text fw={700}>Provenance</Text>
          <Group gap="xs">
            <Switch
              size="xs"
              label="Since this page load only"
              checked={sinceLoadOnly}
              onChange={(e) => {
                setSinceLoadOnly(e.currentTarget.checked)
                setSelected('-1')
              }}
            />
            <Select
              size="xs"
              w={140}
              data={turnOptions}
              value={selected}
              onChange={(v) => setSelected(v ?? '-1')}
              allowDeselect={false}
            />
            <Button
              size="xs"
              variant="outline"
              color="gray"
              loading={loading}
              onClick={() => refresh(selected)}
            >
              Refresh
            </Button>
          </Group>
        </Group>

        {prov ? (
          <Box>
            <Group justify="flex-end" mb={4}>
              <CopyButton value={provText} timeout={1500}>
                {({ copied, copy }) => (
                  <Button
                    size="compact-xs"
                    variant="subtle"
                    color={copied ? 'teal' : 'gray'}
                    onClick={copy}
                  >
                    {copied ? '✓ Copied' : '📋 Copy'}
                  </Button>
                )}
              </CopyButton>
            </Group>
            <Code block style={{ whiteSpace: 'pre-wrap', fontSize: 12 }}>
              {provText}
            </Code>
          </Box>
        ) : (
          <Text size="sm" c="dimmed">
            {sinceLoadOnly
              ? 'No provenance yet this session. Send a message in Chat, then refresh.'
              : 'No provenance yet. Send a message in Chat, then refresh.'}
          </Text>
        )}
      </Stack>
    </ScrollArea>
  )
}
