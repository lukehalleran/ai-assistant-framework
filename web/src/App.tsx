import { Suspense, lazy, useEffect, useRef, useState } from 'react'
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
import {
  api,
  NotesSyncNotStartedError,
  pollNotesSync,
  startNotesSyncAndPoll,
  type NotesSyncStatus,
} from './api/client'
import { captureDebugBaseline } from './api/debugSession'
import { useChatStream } from './api/useChatStream'
import ActivityLog from './components/chat/ActivityLog'
import ChatInput from './components/chat/ChatInput'
import MessageList from './components/chat/MessageList'
import ProgressIndicator from './components/chat/ProgressIndicator'
import MemoryPanel from './components/showcase/MemoryPanel'
// Non-chat views are lazy-loaded (2026-09-03): they are opened rarely and
// carried a large share of the main bundle.
const DebugPage = lazy(() => import('./components/debug/DebugPage'))
const ProvenancePage = lazy(() => import('./components/debug/ProvenancePage'))
const SettingsPage = lazy(() => import('./components/settings/SettingsPage'))
const CurationPage = lazy(() => import('./components/curation/CurationPage'))

type View = 'chat' | 'debug' | 'provenance' | 'settings' | 'curation'

export default function App() {
  const chat = useChatStream()
  const [view, setView] = useState<View>('chat')
  const [fastMode, setFastMode] = useState(false)
  const [rawMode, setRawMode] = useState(false)
  const [citations, setCitations] = useState(false)
  const [syncing, setSyncing] = useState(false)
  // One poll loop at a time; the newest controller owns the `syncing` flag.
  const syncPollRef = useRef<AbortController | null>(null)
  const [models, setModels] = useState<string[]>([])
  const [activeModel, setActiveModel] = useState<string | null>(null)
  // Mobile: sidebar collapses into a burger-toggled drawer so chat gets the screen
  const [navOpened, { toggle: toggleNav, close: closeNav }] = useDisclosure(false)
  // Memory Transparency panel — closed by default, toggled from the header
  const [asideOpened, { toggle: toggleAside }] = useDisclosure(false)

  useEffect(() => {
    // Debug/Provenance show only turns from this page load (Gradio parity) —
    // snapshot the server-held record count before any chatting happens.
    captureDebugBaseline()
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

  // BC-80: the sync runs as a retained server job. A red "failed" card means
  // the SERVER reported failure; a dropped connection is reported as unknown
  // and the retained outcome is re-read, never invented from the rejection.
  const showSyncOutcome = (status: NotesSyncStatus) => {
    if (status.status === 'failed') {
      notifications.show({
        color: 'red',
        title: 'Notes sync failed',
        message: status.message ?? status.error ?? 'The server reported a failure.',
      })
    } else if (status.status === 'succeeded') {
      notifications.show({
        title: 'Notes sync',
        message: status.message ?? 'Notes sync complete',
        autoClose: 8000,
      })
    }
  }

  const beginSyncWatch = () => {
    syncPollRef.current?.abort()
    const controller = new AbortController()
    syncPollRef.current = controller
    setSyncing(true)
    const done = () => {
      if (syncPollRef.current === controller) setSyncing(false)
    }
    return { controller, done }
  }

  // On mount, read the retained state: a reload after a dropped request
  // sees the in-flight job (and polls it) or an outcome finished within the
  // last two minutes, instead of nothing.
  useEffect(() => {
    const { controller, done } = beginSyncWatch()
    api
      .getNotesSyncStatus(controller.signal)
      .then(async (status) => {
        if (status.status === 'running') {
          showSyncOutcome(await pollNotesSync({ signal: controller.signal }))
        } else if (status.finished_at !== null && status.server_time - status.finished_at <= 120) {
          showSyncOutcome(status)
        }
      })
      .catch(() => {})
      .finally(done)
    return () => controller.abort()
  }, [])

  const syncNotes = async () => {
    const { controller, done } = beginSyncWatch()
    try {
      const outcome = await startNotesSyncAndPoll({
        signal: controller.signal,
        onTransportError: () =>
          notifications.show({
            id: 'notes-sync-transport',
            color: 'yellow',
            title: 'Notes sync',
            message: 'Connection interrupted — still checking the server for the outcome.',
          }),
      })
      showSyncOutcome(outcome)
    } catch (err) {
      if (controller.signal.aborted) return
      if (err instanceof NotesSyncNotStartedError) {
        notifications.show({ color: 'yellow', title: 'Notes sync not started', message: err.message })
        return
      }
      // Only a real HTTP status reaches here (401 after a restart, 5xx).
      notifications.show({
        color: 'red',
        title: 'Notes sync failed',
        message: err instanceof Error ? err.message : String(err),
      })
    } finally {
      done()
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
              { value: 'curation', label: '🧹' },
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
        <Suspense fallback={null}>
          {view === 'debug' && <DebugPage />}
          {view === 'provenance' && <ProvenancePage />}
          {view === 'settings' && <SettingsPage />}
          {view === 'curation' && <CurationPage />}
        </Suspense>
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
            onActionDecided={(outcome, line) => {
              chat.appendAssistant(line)
              // Approval chaining (F07): hand the card to the next proposal
              // from the same turn when the server returned one; otherwise
              // clear it like before.
              chat.setPendingAction(outcome.next_action_id ?? null)
            }}
          />
          <ActivityLog log={chat.progressLog} streaming={chat.streaming} />
          <ProgressIndicator
            streaming={chat.streaming}
            progressText={chat.progressText}
            thinkingText={chat.thinkingText}
            startedAt={chat.startedAt}
            storageNotice={chat.storageNotice}
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
