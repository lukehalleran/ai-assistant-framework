import { useEffect, useState } from 'react'
import {
  Button,
  Divider,
  Group,
  NumberInput,
  Paper,
  ScrollArea,
  Select,
  Slider,
  Stack,
  Switch,
  Text,
} from '@mantine/core'
import { notifications } from '@mantine/notifications'
import { api } from '../../api/client'
import type { SettingsSnapshot } from '../../api/types'

// Settings (Gradio tab → SPA, 2026-07-14): same six sections, same semantics —
// each Apply mutates the running orchestrator AND persists to config.yaml via
// the shared gui/settings_core.py functions the Gradio tab calls.

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <Paper withBorder p="md" radius="md">
      <Text fw={700} size="sm" mb="sm">
        {title}
      </Text>
      <Stack gap="sm">{children}</Stack>
    </Paper>
  )
}

export default function SettingsPage() {
  const [snap, setSnap] = useState<SettingsSnapshot | null>(null)
  const [busy, setBusy] = useState<string | null>(null)

  // Streaming
  const [disableBestOf, setDisableBestOf] = useState(false)
  const [disableRewrite, setDisableRewrite] = useState(false)
  const [disableSummaries, setDisableSummaries] = useState(false)
  const [bestOfBudget, setBestOfBudget] = useState(0)
  // Web search
  const [wsEnabled, setWsEnabled] = useState(true)
  const [wsLimit, setWsLimit] = useState(100)
  // Duel
  const [duelEnabled, setDuelEnabled] = useState(false)
  const [duelM1, setDuelM1] = useState<string | null>(null)
  const [duelM2, setDuelM2] = useState<string | null>(null)
  // Tokens
  const [genMax, setGenMax] = useState(128)
  const [judgeMax, setJudgeMax] = useState(64)
  const [streamMax, setStreamMax] = useState(2048)
  // Temperature + cadence
  const [temperature, setTemperature] = useState(0.7)
  const [summaryN, setSummaryN] = useState(10)
  // Shutdown LLM steps: synthesis dreaming + code proposals
  const [synEnabled, setSynEnabled] = useState(false)
  const [synCount, setSynCount] = useState(8)
  const [propEnabled, setPropEnabled] = useState(true)
  const [propCount, setPropCount] = useState(5)

  useEffect(() => {
    api
      .getSettings()
      .then((s) => {
        setSnap(s)
        setDisableBestOf(s.streaming.disable_best_of)
        setDisableRewrite(s.streaming.disable_query_rewrite)
        setDisableSummaries(s.streaming.disable_llm_summaries)
        setBestOfBudget(s.streaming.best_of_latency_budget_s)
        setWsEnabled(s.web_search.enabled)
        setWsLimit(s.web_search.daily_credit_limit)
        setDuelEnabled(s.duel.enabled)
        setDuelM1(s.duel.model_1)
        setDuelM2(s.duel.model_2)
        setGenMax(s.tokens.best_of_max_tokens)
        setJudgeMax(s.tokens.judge_max_tokens)
        setStreamMax(s.tokens.streaming_max_tokens)
        setTemperature(s.temperature)
        setSummaryN(s.summary_every_n)
        setSynEnabled(s.synthesis.enabled)
        setSynCount(s.synthesis.candidates_per_session)
        setPropEnabled(s.proposals.enabled)
        setPropCount(s.proposals.max_per_session)
      })
      .catch((err) =>
        notifications.show({
          color: 'red',
          title: 'Settings load failed',
          message: err instanceof Error ? err.message : String(err),
        }),
      )
  }, [])

  const apply = async (section: Parameters<typeof api.putSettings>[0], body: Parameters<typeof api.putSettings>[1]) => {
    setBusy(section)
    try {
      const result = await api.putSettings(section, body)
      notifications.show({
        color: result.persisted ? 'green' : 'yellow',
        message: result.message,
      })
    } catch (err) {
      notifications.show({
        color: 'red',
        title: 'Apply failed',
        message: err instanceof Error ? err.message : String(err),
      })
    } finally {
      setBusy(null)
    }
  }

  if (!snap) {
    return (
      <Text size="sm" c="dimmed" p="md">
        Loading settings…
      </Text>
    )
  }

  return (
    <ScrollArea h="calc(100dvh - 56px)" offsetScrollbars style={{ flex: 1, minWidth: 0 }}>
      <Stack gap="md" p="md" maw={720} mx="auto">
        <Text fw={700}>⚙️ Runtime Settings</Text>

        <Section title="⚡ Faster streaming (reduce first-token latency)">
          <Switch
            label="Disable Best-of (no multi-sample reranking)"
            checked={disableBestOf}
            onChange={(e) => setDisableBestOf(e.currentTarget.checked)}
          />
          <Switch
            label="Disable query rewrite (skip pre-call)"
            checked={disableRewrite}
            onChange={(e) => setDisableRewrite(e.currentTarget.checked)}
          />
          <Switch
            label="Disable LLM summaries (skip pre-call)"
            checked={disableSummaries}
            onChange={(e) => setDisableSummaries(e.currentTarget.checked)}
          />
          <NumberInput
            label="Best-of latency budget (seconds)"
            min={0}
            max={120}
            step={0.5}
            value={bestOfBudget}
            onChange={(v) => setBestOfBudget(Number(v) || 0)}
          />
          <Group>
            <Button
              size="xs"
              loading={busy === 'streaming'}
              onClick={() =>
                apply('streaming', {
                  disable_best_of: disableBestOf,
                  disable_query_rewrite: disableRewrite,
                  disable_llm_summaries: disableSummaries,
                  best_of_latency_budget_s: bestOfBudget,
                })
              }
            >
              Apply streaming settings
            </Button>
          </Group>
        </Section>

        <Section title="🔍 Web search (Tavily API)">
          <Switch
            label="Enable web search"
            checked={wsEnabled}
            onChange={(e) => setWsEnabled(e.currentTarget.checked)}
          />
          <NumberInput
            label="Daily credit limit (Tavily free tier ≈ 33/day)"
            min={10}
            max={500}
            step={10}
            value={wsLimit}
            onChange={(v) => setWsLimit(Number(v) || 100)}
          />
          <Group>
            <Button
              size="xs"
              loading={busy === 'web-search'}
              onClick={() => apply('web-search', { enabled: wsEnabled, daily_credit_limit: wsLimit })}
            >
              Apply web search settings
            </Button>
          </Group>
        </Section>

        <Section title="🥊 Best-of / duel mode">
          <Switch
            label="Enable duel mode (two models + judge)"
            checked={duelEnabled}
            onChange={(e) => setDuelEnabled(e.currentTarget.checked)}
          />
          <Group grow>
            <Select
              label="Model 1"
              data={snap.model_choices}
              value={duelM1}
              onChange={setDuelM1}
              searchable
            />
            <Select
              label="Model 2"
              data={snap.model_choices}
              value={duelM2}
              onChange={setDuelM2}
              searchable
            />
          </Group>
          <Group>
            <Button
              size="xs"
              loading={busy === 'duel'}
              onClick={() => apply('duel', { enabled: duelEnabled, model_1: duelM1, model_2: duelM2 })}
            >
              Apply duel settings
            </Button>
          </Group>
        </Section>

        <Section title="✂️ Max tokens (length / speed)">
          <NumberInput
            label="Duel/Best-of max tokens (per answer)"
            min={16}
            max={10000}
            step={16}
            value={genMax}
            onChange={(v) => setGenMax(Number(v) || 128)}
          />
          <NumberInput
            label="Judge max tokens"
            min={16}
            max={2000}
            step={8}
            value={judgeMax}
            onChange={(v) => setJudgeMax(Number(v) || 64)}
          />
          <NumberInput
            label="Streaming max tokens (final answer)"
            min={256}
            max={10000}
            step={64}
            value={streamMax}
            onChange={(v) => setStreamMax(Number(v) || 2048)}
          />
          <Group>
            <Button
              size="xs"
              loading={busy === 'tokens'}
              onClick={() =>
                apply('tokens', {
                  best_of_max_tokens: genMax,
                  judge_max_tokens: judgeMax,
                  streaming_max_tokens: streamMax,
                })
              }
            >
              Apply token settings
            </Button>
          </Group>
        </Section>

        <Section title="🌡️ Model temperature">
          <Slider
            min={0}
            max={1.5}
            step={0.05}
            value={temperature}
            onChange={setTemperature}
            label={(v) => v.toFixed(2)}
            marks={[
              { value: 0, label: '0' },
              { value: 0.7, label: '0.7' },
              { value: 1.5, label: '1.5' },
            ]}
          />
          <Divider my={4} style={{ visibility: 'hidden' }} />
          <Group>
            <Button
              size="xs"
              loading={busy === 'temperature'}
              onClick={() => apply('temperature', { temperature })}
            >
              Apply temperature ({temperature.toFixed(2)})
            </Button>
          </Group>
        </Section>

        <Section title="📝 Summary cadence">
          <NumberInput
            label="Summary every N exchanges (applies at shutdown)"
            min={1}
            max={200}
            value={summaryN}
            onChange={(v) => setSummaryN(Number(v) || 10)}
          />
          <Group>
            <Button
              size="xs"
              loading={busy === 'summary-cadence'}
              onClick={() => apply('summary-cadence', { every_n: summaryN })}
            >
              Apply
            </Button>
          </Group>
        </Section>

        <Section title="💤 Synthesis dreaming (shutdown LLM step — API cost)">
          <Switch
            label="Enable synthesis dreaming"
            description="Generates cross-domain synthesis candidates at shutdown (LLM calls incl. the Opus coherence judge)"
            checked={synEnabled}
            onChange={(e) => setSynEnabled(e.currentTarget.checked)}
          />
          <Text size="xs" c="dimmed">
            Candidates per shutdown: {synCount}
          </Text>
          <Slider
            min={1}
            max={20}
            step={1}
            value={synCount}
            onChange={setSynCount}
            marks={[
              { value: 1, label: '1' },
              { value: 8, label: '8' },
              { value: 20, label: '20' },
            ]}
          />
          <Divider my={4} style={{ visibility: 'hidden' }} />
          <Group>
            <Button
              size="xs"
              loading={busy === 'synthesis'}
              onClick={() =>
                apply('synthesis', { enabled: synEnabled, candidates_per_session: synCount })
              }
            >
              Apply synthesis settings
            </Button>
          </Group>
        </Section>

        <Section title="🛠️ Code proposals (shutdown LLM step — API cost)">
          <Switch
            label="Enable code proposals"
            description="Generates goal-directed code proposals at shutdown (LLM calls)"
            checked={propEnabled}
            onChange={(e) => setPropEnabled(e.currentTarget.checked)}
          />
          <Text size="xs" c="dimmed">
            Max proposals per shutdown: {propCount}
          </Text>
          <Slider
            min={1}
            max={10}
            step={1}
            value={propCount}
            onChange={setPropCount}
            marks={[
              { value: 1, label: '1' },
              { value: 5, label: '5' },
              { value: 10, label: '10' },
            ]}
          />
          <Divider my={4} style={{ visibility: 'hidden' }} />
          <Group>
            <Button
              size="xs"
              loading={busy === 'proposals'}
              onClick={() =>
                apply('proposals', { enabled: propEnabled, max_per_session: propCount })
              }
            >
              Apply proposal settings
            </Button>
          </Group>
        </Section>
      </Stack>
    </ScrollArea>
  )
}
