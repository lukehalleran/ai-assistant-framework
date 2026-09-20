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
import { describeCurationFailure } from './failure'

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
      .catch((err) => {
        const f = describeCurationFailure(err, verb)
        notifications.show({ color: f.color, title: f.title, message: f.message })
        // A failed save can leave an interrupted operation requiring Undo.
        // A lost response may have completed on the server — re-read the
        // real state either way.
        onResolve()
      })
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
          {p.status === 'interrupted' && (
            <Text size="sm" c="orange">
              {p.status_detail || 'This operation was interrupted. Undo to restore its previous values.'}
            </Text>
          )}
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
              onClick={() => p.status === 'interrupted'
                ? act(() => api.undoCurationProposal(p.proposal_id), 'Undone')
                : act(() => api.applyCurationProposal(p.proposal_id), 'Applied')}
            >
              {p.status === 'interrupted' ? 'Undo interrupted operation' : 'Apply'}
            </Button>
            <Button
              size="xs"
              variant="outline"
              color="gray"
              loading={busy}
              disabled={p.status === 'interrupted'}
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
  const [queueError, setQueueError] = useState<string | null>(null)

  const refresh = useCallback(() => {
    api
      .getCurationQueue()
      .then((q) => {
        setProposals(q.proposals)
        setQueueError(null)
      })
      .catch((err) => setQueueError(err instanceof Error ? err.message : String(err)))
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
      .catch((err) => {
        const f = describeCurationFailure(err, 'Scan')
        notifications.show({ color: f.color, title: f.title, message: f.message })
        // A lost response may have completed on the server; re-read the queue.
        if (f.lost) refresh()
      })
      .finally(() => setScanning(false))
  }

  const appliedRecently = activity.filter((e) => e.event === 'applied')

  return (
    <Stack p="md" flex={1} style={{ minWidth: 0 }}>
      <Group justify="space-between">
        <Text fw={600}>🧹 Curation</Text>
        <Group gap="xs">
          <Button size="xs" variant="subtle" onClick={refresh}>Refresh</Button>
          <Button size="xs" variant="outline" loading={scanning} onClick={scan}>
            Scan now
          </Button>
        </Group>
      </Group>
      <Text size="xs" c="dimmed">
        Proposed data-hygiene actions. Everything here is reversible — applied
        items keep their pre-image and can be undone from Activity. Nothing is
        ever deleted.
      </Text>
      {queueError && <Text size="sm" c="orange">Queue unavailable: {queueError}</Text>}
      {proposals.length === 0 ? (
        !queueError && <Text size="sm" c="dimmed">
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
                      .catch((err) => {
                        const f = describeCurationFailure(err, 'Undo')
                        notifications.show({ color: f.color, title: f.title, message: f.message })
                        // A lost response may have completed on the server;
                        // re-read the activity feed either way.
                        if (f.lost) refresh()
                      })
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
