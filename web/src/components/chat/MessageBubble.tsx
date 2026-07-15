import { Paper } from '@mantine/core'
import type { ChatMessage } from '../../api/types'
import { BOT_BUBBLE, USER_BUBBLE } from '../../theme'
import MarkdownMessage from './MarkdownMessage'

export default function MessageBubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === 'user'
  return (
    <Paper
      p="sm"
      radius="md"
      maw="85%"
      style={{
        alignSelf: isUser ? 'flex-end' : 'flex-start',
        backgroundColor: isUser ? USER_BUBBLE : BOT_BUBBLE,
        overflowWrap: 'anywhere',
      }}
    >
      {isUser ? (
        <span style={{ whiteSpace: 'pre-wrap' }}>{message.content}</span>
      ) : (
        <MarkdownMessage content={message.content} />
      )}
    </Paper>
  )
}
