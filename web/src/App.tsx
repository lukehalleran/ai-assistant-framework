import { useEffect, useState } from 'react'
import {
  ActionIcon,
  AppShell,
  Burger,
  Button,
  Group,
  SegmentedControl,
  Select,
  Stack,
  Switch,
  Text,
  Title,
  Tooltip,
} from '@mantine/core'
import { useDisclosure } from '@mantine/hooks'
import { notifications } from '@mantine/notifications'
import { api } from './api/client'
import { useChatStream } from './api/useChatStream'
import ActivityLog from './components/chat/ActivityLog'
import ChatInput from './components/chat/ChatInput'
import MessageList from './components/chat/MessageList'
import ProgressIndicator from './components/chat/ProgressIndicator'
import MemoryPanel from './components/showcase/MemoryPanel'
import DebugPage from './components/debug/DebugPage'
import ProvenancePage from './components/debug/ProvenancePage'
import SettingsPage from './components/settings/SettingsPage'

type View = 'chat' | 'debug' | 'provenance' | 'settings'

export default function App() {
  const chat = useChatStream()
  const [view, setView] = useState<View>('chat')
  const [fastMode, setFastMode] = useState(false)
  const [rawMode, setRawMode] = useState(false)
  const [citations, setCitations] = useState(false)
  const [syncing, setSyncing] = useState(false)
  const [models, setModels] = useState<string[]>([])
  const [activeModel, setActiveModel] = useState<string | null>(null)
  // Mobile: sidebar collapses into a burger-toggled drawer so chat gets the screen
  const [navOpened, { toggle: toggleNav, close: closeNav }] = useDisclosure(false)
  // Memory Transparency panel — closed by default, toggled from the header
  const [asideOpened, { toggle: toggleAside }] = useDisclosure(false)

  useEffect(() => {
    api
      .getModels()
      .then((m) => {
        setModels(m.models)
        setActiveModel(m.active)
      })
      .catch(() => {})
  }, [])

  useEffect(() => {
    if (chat.error) {
      notifications.show({ color: 'red', title: 'Stream error', message: chat.error })
    }
  }, [chat.error])

  const switchModel = async (name: string | null) => {
    if (!name) return
    try {
      const m = await api.setActiveModel(name)
      setActiveModel(m.active)
      notifications.show({ message: `Model: ${m.active}` })
    } catch (err) {
      notifications.show({
        color: 'red',
        title: 'Model switch failed',
        message: err instanceof Error ? err.message : String(err),
      })
    }
  }

  const syncNotes = async () => {
    setSyncing(true)
    try {
      const { message } = await api.syncNotes()
      notifications.show({ title: 'Notes sync', message, autoClose: 8000 })
    } catch (err) {
      notifications.show({
        color: 'red',
        title: 'Notes sync failed',
        message: err instanceof Error ? err.message : String(err),
      })
    } finally {
      setSyncing(false)
    }
  }

  return (
    <AppShell
      header={{ height: 56 }}
      navbar={{
        width: 260,
        breakpoint: 'sm',
        collapsed: { mobile: !navOpened, desktop: false },
      }}
      aside={{
        width: 320,
        breakpoint: 'md',
        collapsed: { mobile: !asideOpened, desktop: !asideOpened },
      }}
      padding={0}
    >
      <AppShell.Header>
        <Group h="100%" px="md" justify="space-between">
          <Group gap="sm">
            <Burger opened={navOpened} onClick={toggleNav} hiddenFrom="sm" size="sm" />
            <Title order={3}>Daemon</Title>
          </Group>
          <Group gap="sm">
            <Text size="xs" c="dimmed" visibleFrom="sm">
              local · private · yours
            </Text>
            <Tooltip label="Memory transparency — what Daemon remembered and why">
              <ActionIcon
                variant={asideOpened ? 'filled' : 'outline'}
                onClick={toggleAside}
                aria-label="Toggle memory panel"
              >
                🧠
              </ActionIcon>
            </Tooltip>
          </Group>
        </Group>
      </AppShell.Header>

      <AppShell.Navbar p="md">
        <Stack gap="md">
          <SegmentedControl
            fullWidth
            size="xs"
            value={view}
            onChange={(v) => {
              setView(v as View)
              closeNav()
            }}
            data={[
              { value: 'chat', label: '💬 Chat' },
              { value: 'debug', label: '🔎 Debug' },
              { value: 'provenance', label: '🧾 Prov.' },
              { value: 'settings', label: '⚙️' },
            ]}
          />
          <Select
            label="Model"
            data={models}
            value={activeModel}
            onChange={(name) => {
              switchModel(name)
              closeNav()
            }}
            searchable
            allowDeselect={false}
          />
          <Switch
            label="Fast mode ⚡"
            checked={fastMode}
            onChange={(e) => setFastMode(e.currentTarget.checked)}
          />
          <Switch
            label="Raw GPT (bypass memory)"
            checked={rawMode}
            onChange={(e) => setRawMode(e.currentTarget.checked)}
          />
          <Switch
            label="Memory citations"
            checked={citations}
            onChange={(e) => setCitations(e.currentTarget.checked)}
          />
          <Button
            variant="outline"
            color="gray"
            size="xs"
            loading={syncing}
            onClick={() => {
              syncNotes()
              closeNav()
            }}
          >
            📝 Sync notes
          </Button>
          <Button
            variant="outline"
            color="gray"
            size="xs"
            onClick={() => {
              chat.clearAll()
              closeNav()
            }}
          >
            🧹 Clear chat
          </Button>
          <Text size="xs" c="dimmed">
            Dev tabs live at <a href="/admin">/admin</a>
          </Text>
        </Stack>
      </AppShell.Navbar>

      <AppShell.Aside>
        <MemoryPanel records={chat.debugRecords} />
      </AppShell.Aside>

      <AppShell.Main style={{ display: 'flex' }}>
        {/* Non-chat views mount on demand; the chat column stays MOUNTED but
            hidden so an in-flight stream keeps rendering into it. */}
        {view === 'debug' && <DebugPage />}
        {view === 'provenance' && <ProvenancePage />}
        {view === 'settings' && <SettingsPage />}
        {/* minWidth/overflowX clamp: a long unbroken line anywhere in the chat
            column must shrink within the viewport, never widen the page */}
        <Stack
          gap={0}
          flex={1}
          style={{
            height: 'calc(100dvh - 56px)',
            minWidth: 0,
            overflowX: 'hidden',
            display: view === 'chat' ? 'flex' : 'none',
          }}
        >
          <MessageList
            messages={chat.messages}
            pendingActionId={chat.pendingActionId}
            duelThinking={chat.duelThinking}
            onActionDecided={(line) => {
              chat.appendAssistant(line)
              chat.clearPendingAction()
            }}
          />
          <ActivityLog log={chat.progressLog} streaming={chat.streaming} />
          <ProgressIndicator
            streaming={chat.streaming}
            progressText={chat.progressText}
            thinkingText={chat.thinkingText}
            startedAt={chat.startedAt}
          />
          <ChatInput
            streaming={chat.streaming}
            onSend={(text, fileIds) =>
              chat.send({
                text,
                fast_mode: fastMode,
                raw_mode: rawMode,
                enable_citations: citations,
                file_ids: fileIds,
              })
            }
            onStop={chat.abort}
          />
        </Stack>
      </AppShell.Main>
    </AppShell>
  )
}
