import { useCallback, useEffect, useState } from 'react'
import { Box, Button, Code, CopyButton, Group, ScrollArea, Select, Stack, Text } from '@mantine/core'
import { notifications } from '@mantine/notifications'
import { api } from '../../api/client'
import { debugBaselineFor } from '../../api/debugSession'

// Provenance (Gradio tab → SPA, 2026-07-14): the per-turn provenance object —
// response mode, session id, cited memory ids, duel thinking/winner, agentic
// rounds — rendered as JSON, with a turn selector (Gradio showed latest only).
// Like the Gradio tab, only turns from the ongoing UI session are offered —
// the server-held backlog before this page load is hidden (debugSession
// baseline); selector values stay ABSOLUTE indices for the API.

export default function ProvenancePage() {
  const [baseline, setBaseline] = useState(0)
  const [sessionTurns, setSessionTurns] = useState(0)
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
        const turns = records.count - base
        setSessionTurns(turns)
        if (turns <= 0) {
          // "Latest" would resolve to a pre-session record — show nothing
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
  }, [])

  useEffect(() => refresh(selected), [refresh, selected])

  const turnOptions = [
    { value: '-1', label: `Latest turn` },
    ...Array.from({ length: sessionTurns }, (_, i) => ({
      value: String(baseline + i),
      label: `Turn #${i + 1}`,
    })),
  ]

  const provText = prov ? JSON.stringify(prov, null, 2) : ''

  return (
    <ScrollArea h="calc(100dvh - 56px)" offsetScrollbars style={{ flex: 1, minWidth: 0 }}>
      <Stack gap="md" p="md" maw={960} mx="auto">
        <Group justify="space-between">
          <Text fw={700}>Provenance</Text>
          <Group gap="xs">
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
            No provenance yet this session. Send a message in Chat, then refresh.
          </Text>
        )}
      </Stack>
    </ScrollArea>
  )
}
