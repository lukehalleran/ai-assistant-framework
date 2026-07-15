import { useCallback, useEffect, useState } from 'react'
import { Button, Code, Group, ScrollArea, Select, Stack, Text } from '@mantine/core'
import { notifications } from '@mantine/notifications'
import { api } from '../../api/client'

// Provenance (Gradio tab → SPA, 2026-07-14): the per-turn provenance object —
// response mode, session id, cited memory ids, duel thinking/winner, agentic
// rounds — rendered as JSON, with a turn selector (Gradio showed latest only).

export default function ProvenancePage() {
  const [turnCount, setTurnCount] = useState(0)
  const [selected, setSelected] = useState<string>('-1')
  const [prov, setProv] = useState<Record<string, unknown> | null>(null)
  const [loading, setLoading] = useState(false)

  const refresh = useCallback((index: string) => {
    setLoading(true)
    Promise.all([api.getDebugRecords(), api.getProvenance(parseInt(index, 10))])
      .then(([records, p]) => {
        setTurnCount(records.count)
        setProv(p)
      })
      .catch((err) => {
        setProv(null)
        const msg = err instanceof Error ? err.message : String(err)
        if (!msg.includes('404')) {
          notifications.show({ color: 'red', title: 'Provenance failed', message: msg })
        }
        // still learn the turn count so the selector populates
        api.getDebugRecords().then((r) => setTurnCount(r.count)).catch(() => {})
      })
      .finally(() => setLoading(false))
  }, [])

  useEffect(() => refresh(selected), [refresh, selected])

  const turnOptions = [
    { value: '-1', label: `Latest turn` },
    ...Array.from({ length: turnCount }, (_, i) => ({
      value: String(i),
      label: `Turn #${i + 1}`,
    })),
  ]

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
          <Code block style={{ whiteSpace: 'pre-wrap', fontSize: 12 }}>
            {JSON.stringify(prov, null, 2)}
          </Code>
        ) : (
          <Text size="sm" c="dimmed">
            No provenance yet this session. Send a message in Chat, then refresh.
          </Text>
        )}
      </Stack>
    </ScrollArea>
  )
}
