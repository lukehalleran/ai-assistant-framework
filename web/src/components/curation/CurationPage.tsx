import { useCallback, useEffect, useState } from 'react'
import {
  Accordion,
  Badge,
  Button,
  Code,
  Group,
  ScrollArea,
  Stack,
  Text,
} from '@mantine/core'
import { notifications } from '@mantine/notifications'
import { api } from '../../api/client'
import type { CurationProposal } from '../../api/types'

// Curation Center (docs/AUTONOMOUS_CURATION_DESIGN.md): the one-click queue
// that replaces terminal candidate files + --apply scripts. Proposals carry
// their evidence and exact changes; Apply/Dismiss act through the in-process
// engine (pre-images + journal + undo). Deletion is never proposed here —
// quarantine flips a metadata flag retrieval already respects.

const CONFIDENCE_COLOR: Record<string, string> = {
  deterministic: 'teal',
  dual_llm: 'blue',
  single_llm: 'yellow',
}

function ProposalCard({
  p,
  onResolve,
}: {
  p: CurationProposal
  onResolve: () => void
}) {
  const [busy, setBusy] = useState(false)

  const act = (fn: () => Promise<unknown>, verb: string) => {
    setBusy(true)
    fn()
      .then(() => {
        notifications.show({ color: 'teal', title: verb, message: p.title })
        onResolve()
      })
      .catch((err) =>
        notifications.show({
          color: 'red',
          title: `${verb} failed`,
          message: err instanceof Error ? err.message : String(err),
        }),
      )
      .finally(() => setBusy(false))
  }

  return (
    <Accordion.Item value={p.proposal_id}>
      <Accordion.Control>
        <Group gap="xs" wrap="nowrap">
          <Badge size="xs" color={CONFIDENCE_COLOR[p.confidence] ?? 'gray'}>
            {p.confidence}
          </Badge>
          <Badge size="xs" variant="outline" color="gray">
            {p.curator}
          </Badge>
          {p.batch && (
            <Badge size="xs" variant="light" color="grape">
              batch ×{p.items.length}
            </Badge>
          )}
          <Text size="sm" truncate>
            {p.title}
          </Text>
        </Group>
      </Accordion.Control>
      <Accordion.Panel>
        <Stack gap="xs">
          <Text size="xs">{p.evidence}</Text>
          {!p.batch && p.items[0] && (
            <Code block>
              {p.items[0].store} · {p.items[0].doc_id} ·{' '}
              {p.items[0].change_type}
              {Object.keys(p.items[0].after).length > 0 &&
                `\n${JSON.stringify(p.items[0].after, null, 1)}`}
            </Code>
          )}
          <Group gap="xs">
            <Button
              size="xs"
              color="teal"
              loading={busy}
              onClick={() => act(() => api.applyCurationProposal(p.proposal_id), 'Applied')}
            >
              Apply
            </Button>
            <Button
              size="xs"
              variant="outline"
              color="gray"
              loading={busy}
              onClick={() =>
                act(() => api.dismissCurationProposal(p.proposal_id), 'Dismissed')
              }
            >
              Dismiss
            </Button>
          </Group>
        </Stack>
      </Accordion.Panel>
    </Accordion.Item>
  )
}

export default function CurationPage() {
  const [proposals, setProposals] = useState<CurationProposal[]>([])
  const [activity, setActivity] = useState<Record<string, unknown>[]>([])
  const [scanning, setScanning] = useState(false)

  const refresh = useCallback(() => {
    api
      .getCurationQueue()
      .then((q) => setProposals(q.proposals))
      .catch(() => setProposals([]))
    api
      .getCurationActivity(50)
      .then((a) => setActivity(a.events))
      .catch(() => setActivity([]))
  }, [])

  useEffect(() => refresh(), [refresh])

  const scan = () => {
    setScanning(true)
    api
      .runCurationScan()
      .then((rep) => {
        notifications.show({
          color: 'teal',
          title: 'Scan finished',
          message: `${rep.proposals_queued} proposal(s) queued`,
        })
        refresh()
      })
      .catch((err) =>
        notifications.show({
          color: 'red',
          title: 'Scan failed',
          message: err instanceof Error ? err.message : String(err),
        }),
      )
      .finally(() => setScanning(false))
  }

  const appliedRecently = activity.filter((e) => e.event === 'applied')

  return (
    <Stack p="md" flex={1} style={{ minWidth: 0 }}>
      <Group justify="space-between">
        <Text fw={600}>🧹 Curation</Text>
        <Button size="xs" variant="outline" loading={scanning} onClick={scan}>
          Scan now
        </Button>
      </Group>
      <Text size="xs" c="dimmed">
        Proposed data-hygiene actions. Everything here is reversible — applied
        items keep their pre-image and can be undone from Activity. Nothing is
        ever deleted.
      </Text>
      {proposals.length === 0 ? (
        <Text size="sm" c="dimmed">
          Queue is empty — nothing needs attention.
        </Text>
      ) : (
        <Accordion multiple variant="separated">
          {proposals.map((p) => (
            <ProposalCard key={p.proposal_id} p={p} onResolve={refresh} />
          ))}
        </Accordion>
      )}

      <Text fw={600} size="sm" mt="md">
        Recent activity
      </Text>
      <ScrollArea.Autosize mah={280}>
        <Stack gap={4}>
          {activity.length === 0 && (
            <Text size="xs" c="dimmed">
              No curation activity yet.
            </Text>
          )}
          {activity.map((e, i) => (
            <Group key={i} gap="xs" wrap="nowrap">
              <Text size="xs" c="dimmed" style={{ whiteSpace: 'nowrap' }}>
                {String(e.ts ?? '').slice(0, 19).replace('T', ' ')}
              </Text>
              <Badge size="xs" variant="light" color="gray">
                {String(e.event ?? '')}
              </Badge>
              <Text size="xs" truncate>
                {String(e.title ?? e.curator ?? '')}
              </Text>
              {e.event === 'applied' && typeof e.proposal_id === 'string' && (
                <Button
                  size="compact-xs"
                  variant="subtle"
                  color="red"
                  onClick={() =>
                    api
                      .undoCurationProposal(e.proposal_id as string)
                      .then(() => {
                        notifications.show({
                          color: 'teal',
                          title: 'Undone',
                          message: String(e.title ?? ''),
                        })
                        refresh()
                      })
                      .catch((err) =>
                        notifications.show({
                          color: 'red',
                          title: 'Undo failed',
                          message:
                            err instanceof Error ? err.message : String(err),
                        }),
                      )
                  }
                >
                  Undo
                </Button>
              )}
            </Group>
          ))}
          {appliedRecently.length === 0 && null}
        </Stack>
      </ScrollArea.Autosize>
    </Stack>
  )
}
