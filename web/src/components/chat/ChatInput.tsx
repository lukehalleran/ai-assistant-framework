import { useState } from 'react'
import { ActionIcon, Chip, FileButton, Group, Textarea, Tooltip } from '@mantine/core'
import { notifications } from '@mantine/notifications'
import { api } from '../../api/client'
import type { UploadedFileInfo } from '../../api/types'

interface Props {
  streaming: boolean
  onSend: (text: string, fileIds: string[]) => void
  onStop: () => void
}

const ACCEPT =
  '.txt,.md,.json,.yaml,.yml,.log,.html,.xml,.csv,.py,.docx,.xlsx,.pdf,.png,.jpg,.jpeg,.gif,.webp'

export default function ChatInput({ streaming, onSend, onStop }: Props) {
  const [text, setText] = useState('')
  const [attached, setAttached] = useState<UploadedFileInfo[]>([])
  const [uploading, setUploading] = useState(false)

  const attach = async (files: File[]) => {
    if (!files.length) return
    setUploading(true)
    try {
      const infos = await api.uploadFiles(files)
      setAttached((prev) => [...prev, ...infos])
    } catch (err) {
      notifications.show({
        color: 'red',
        title: 'Upload failed',
        message: err instanceof Error ? err.message : String(err),
      })
    } finally {
      setUploading(false)
    }
  }

  const submit = () => {
    const t = text.trim()
    if (!t || streaming || uploading) return
    onSend(t, attached.map((f) => f.file_id))
    setText('')
    setAttached([])
  }

  return (
    <div>
      {attached.length > 0 && (
        <Group gap="xs" px="md" pb={4}>
          {attached.map((f) => (
            <Chip
              key={f.file_id}
              checked
              size="xs"
              onClick={() => setAttached((prev) => prev.filter((x) => x.file_id !== f.file_id))}
            >
              📄 {f.name} ✕
            </Chip>
          ))}
        </Group>
      )}
      <Group align="flex-end" gap="sm" p="md" pt={4}>
        <FileButton onChange={attach} accept={ACCEPT} multiple>
          {(props) => (
            <Tooltip label="Attach files">
              <ActionIcon
                {...props}
                size="xl"
                variant="outline"
                color="gray"
                loading={uploading}
                disabled={streaming}
              >
                📎
              </ActionIcon>
            </Tooltip>
          )}
        </FileButton>
        <Textarea
          flex={1}
          autosize
          minRows={2}
          maxRows={8}
          placeholder="Ask Daemon something…  (Enter to send, Shift+Enter for newline)"
          value={text}
          onChange={(e) => setText(e.currentTarget.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault()
              submit()
            }
          }}
        />
        {streaming ? (
          <Tooltip label="Stop generating">
            <ActionIcon size="xl" color="red" variant="outline" onClick={onStop}>
              ■
            </ActionIcon>
          </Tooltip>
        ) : (
          <Tooltip label="Send">
            <ActionIcon size="xl" onClick={submit} disabled={!text.trim() || uploading}>
              ➤
            </ActionIcon>
          </Tooltip>
        )}
      </Group>
    </div>
  )
}
