import { useState } from 'react'
import {
  Button,
  Checkbox,
  Code,
  Group,
  Modal,
  Radio,
  ScrollArea,
  Stack,
  Text,
  Textarea,
} from '@mantine/core'
import { useDisclosure } from '@mantine/hooks'
import { api, type ReportBundle, type ReportTurn } from '../api/client'

const MAX_DESCRIPTION = 2000

function pad(n: number) {
  return String(n).padStart(2, '0')
}

export function reportFilename(d = new Date()): string {
  return (
    `daemon_report_${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}` +
    `_${pad(d.getHours())}${pad(d.getMinutes())}${pad(d.getSeconds())}.json`
  )
}

/** Short plain-text rendering of the summary tier. Never includes `full`. */
export function summaryText(bundle: ReportBundle): string {
  const lines: string[] = ['Daemon problem report']
  const emit = (title: string, obj: unknown) => {
    if (!obj || typeof obj !== 'object' || Array.isArray(obj)) return
    lines.push(`${title}:`)
    for (const [k, v] of Object.entries(obj as Record<string, unknown>)) {
      if (v === null || v === undefined) continue
      lines.push(`  ${k}: ${typeof v === 'object' ? JSON.stringify(v) : String(v)}`)
    }
  }
  if (typeof bundle.description === 'string' && bundle.description) {
    lines.push(`Description: ${bundle.description}`)
  }
  emit('App', bundle.app)
  emit('Turn', bundle.turn)
  return lines.join('\n')
}

/** R1: "Report a problem" — builds a diagnostic bundle on the server, shows
 * exactly what it contains, and lets the user download it or copy a summary.
 * Nothing is ever sent automatically. */
export default function ReportProblemControl({
  getTurns = api.getReportTurns,
  build = api.buildReport,
}: {
  getTurns?: () => Promise<{ turns: ReportTurn[] }>
  build?: (index: number, description: string, includeFull: boolean) => Promise<ReportBundle>
}) {
  const [opened, { open, close }] = useDisclosure(false)
  const [turns, setTurns] = useState<ReportTurn[]>([])
  const [selected, setSelected] = useState<string>('')
  const [description, setDescription] = useState('')
  const [includeFull, setIncludeFull] = useState(false)
  const [bundle, setBundle] = useState<ReportBundle | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [notice, setNotice] = useState<string | null>(null)

  const onOpen = async () => {
    setBundle(null)
    setError(null)
    setNotice(null)
    setIncludeFull(false)
    open()
    try {
      const { turns: t } = await getTurns()
      setTurns(t)
      setSelected(t.length ? String(t[0].index) : '')
    } catch (err) {
      setTurns([])
      setSelected('')
      setError(err instanceof Error ? err.message : String(err))
    }
  }

  const doBuild = async (): Promise<ReportBundle | null> => {
    setBusy(true)
    setError(null)
    setNotice(null)
    try {
      const b = await build(selected === '' ? -1 : Number(selected), description, includeFull)
      setBundle(b)
      return b
    } catch (err) {
      setBundle(null)
      setError(err instanceof Error ? err.message : String(err))
      return null
    } finally {
      setBusy(false)
    }
  }

  const download = async () => {
    const b = bundle ?? (await doBuild())
    if (!b) return
    const blob = new Blob([JSON.stringify(b, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = reportFilename()
    document.body.appendChild(a)
    a.click()
    a.remove()
    URL.revokeObjectURL(url)
    setNotice(`Saved ${a.download}. Send that file to whoever asked for it.`)
  }

  const copySummary = async () => {
    const b = bundle ?? (await doBuild())
    if (!b) return
    try {
      await navigator.clipboard.writeText(summaryText(b))
      setNotice('Summary copied.')
    } catch (err) {
      setError(`Could not copy: ${err instanceof Error ? err.message : String(err)}`)
    }
  }

  const noTurns = turns.length === 0

  return (
    <>
      <Button variant="outline" color="gray" size="xs" onClick={onOpen}>
        Report a problem
      </Button>
      <Modal opened={opened} onClose={close} title="Report a problem" size="lg" centered>
        <Stack gap="md">
          <Text size="sm" c="dimmed">
            This builds a file on your computer. Nothing is sent anywhere unless you send the file
            yourself.
          </Text>
          {noTurns ? (
            <Text size="sm">No turns to report yet. Send a message first.</Text>
          ) : (
            <Radio.Group
              label="Which turn?"
              value={selected}
              onChange={(v) => {
                setSelected(v)
                setBundle(null)
              }}
            >
              <Stack gap={4} mt={4}>
                {turns.map((t) => (
                  <Radio
                    key={t.index}
                    value={String(t.index)}
                    label={`${t.ts} · ${t.mode ?? 'unknown'} · ${t.preview}`}
                  />
                ))}
              </Stack>
            </Radio.Group>
          )}
          <Textarea
            label="What went wrong?"
            placeholder="Optional"
            maxLength={MAX_DESCRIPTION}
            autosize
            minRows={2}
            value={description}
            onChange={(e) => {
              setDescription(e.currentTarget.value)
              setBundle(null)
            }}
          />
          <Checkbox
            label="Include the full prompt and reply for this turn (contains your conversation)"
            description="Off: the file has only timings, labels and counts, no text of your messages."
            checked={includeFull}
            onChange={(e) => {
              setIncludeFull(e.currentTarget.checked)
              setBundle(null)
            }}
          />
          {error && (
            <Text size="sm" c="red">
              {error}
            </Text>
          )}
          {notice && <Text size="sm">{notice}</Text>}
          {bundle && (
            <ScrollArea h={260} type="auto">
              <Code block data-testid="report-preview">
                {JSON.stringify(bundle, null, 2)}
              </Code>
            </ScrollArea>
          )}
          <Group justify="flex-end">
            <Button variant="default" onClick={close}>
              Close
            </Button>
            <Button variant="default" onClick={doBuild} loading={busy} disabled={noTurns}>
              Preview
            </Button>
            <Button variant="default" onClick={copySummary} disabled={noTurns || busy}>
              Copy summary
            </Button>
            <Button onClick={download} disabled={noTurns || busy}>
              Download
            </Button>
          </Group>
        </Stack>
      </Modal>
    </>
  )
}
