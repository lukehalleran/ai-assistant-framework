import { useEffect, useState } from 'react'
import { Button, Card, Group, Text } from '@mantine/core'
import { notifications } from '@mantine/notifications'
import { api } from '../../api/client'
import type { ActionOutcome } from '../../api/types'

interface Props {
  actionId: string
  onDecided: (outcome: ActionOutcome, chatLine: string) => void
}

// Human-in-the-loop gate: Daemon proposed an external write action (email,
// calendar, …) and waits for explicit approval. A demo highlight.
export default function ActionApprovalCard({ actionId, onDecided }: Props) {
  const [busy, setBusy] = useState<'approve' | 'reject' | null>(null)

  // Approval chaining (F07): the parent can hand this same mounted card the
  // NEXT proposal's id (rather than unmounting it) when a turn had more than
  // one pending action. Without this, `busy` from the previous decision
  // would leave the card permanently disabled for the next one.
  useEffect(() => {
    setBusy(null)
  }, [actionId])

  const decide = async (kind: 'approve' | 'reject') => {
    setBusy(kind)
    try {
      const resp =
        kind === 'approve' ? await api.approveAction(actionId) : await api.rejectAction(actionId)
      onDecided(resp.outcome, resp.message.content)
    } catch (err) {
      notifications.show({
        color: 'red',
        title: 'Action decision failed',
        message: err instanceof Error ? err.message : String(err),
      })
      setBusy(null)
    }
  }

  return (
    <Card withBorder radius="md" p="md" style={{ alignSelf: 'flex-start' }} maw="85%">
      <Text size="sm" fw={600} mb="xs">
        ⚡ Daemon wants to perform an internet action
      </Text>
      <Text size="xs" c="dimmed" mb="sm">
        Nothing is sent until you approve.
      </Text>
      <Group gap="sm">
        <Button size="xs" loading={busy === 'approve'} disabled={busy !== null} onClick={() => decide('approve')}>
          Approve
        </Button>
        <Button
          size="xs"
          color="red"
          variant="outline"
          loading={busy === 'reject'}
          disabled={busy !== null}
          onClick={() => decide('reject')}
        >
          Reject
        </Button>
      </Group>
    </Card>
  )
}
