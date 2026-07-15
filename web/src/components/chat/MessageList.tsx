import { useEffect, useRef } from 'react'
import { ScrollArea, Stack, Text } from '@mantine/core'
import type { ChatMessage, DuelThinking } from '../../api/types'
import MessageBubble from './MessageBubble'
import ActionApprovalCard from './ActionApprovalCard'
import ThinkingBlock from './ThinkingBlock'

interface Props {
  messages: ChatMessage[]
  pendingActionId: string | null
  duelThinking: DuelThinking | null
  onActionDecided: (chatLine: string) => void
}

export default function MessageList({ messages, pendingActionId, duelThinking, onActionDecided }: Props) {
  const viewport = useRef<HTMLDivElement>(null)
  const lastLen = messages.length
    ? messages[messages.length - 1].content.length
    : 0

  // Auto-scroll on new content; simple version — always follow the stream.
  useEffect(() => {
    viewport.current?.scrollTo({ top: viewport.current.scrollHeight, behavior: 'smooth' })
  }, [messages.length, lastLen, pendingActionId])

  return (
    <ScrollArea flex={1} viewportRef={viewport} offsetScrollbars>
      <Stack gap="sm" p="md">
        {messages.length === 0 && (
          <Text c="dimmed" ta="center" mt="xl">
            No messages yet.
          </Text>
        )}
        {messages.map((m, i) => (
          <MessageBubble key={i} message={m} />
        ))}
        {duelThinking && <ThinkingBlock duel={duelThinking} />}
        {pendingActionId && (
          <ActionApprovalCard actionId={pendingActionId} onDecided={onActionDecided} />
        )}
      </Stack>
    </ScrollArea>
  )
}
