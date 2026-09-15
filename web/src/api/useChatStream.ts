import { useCallback, useEffect, useReducer, useRef } from 'react'
import { EventStreamContentType, fetchEventSource } from '@microsoft/fetch-event-source'
import { api } from './client'
import { resetDebugBaseline } from './debugSession'
import { assertApiPath, authorizedFetch, authorizedHeaders, nonOkRequestMessage } from './launchAuth'
import type { ChatMessage, ChatRequest, CompletePayload, DebugRecord, DuelThinking } from './types'

// Server sends CUMULATIVE content on `message` events (replace-render): the
// reducer replaces the last assistant bubble instead of appending deltas.

interface StreamState {
  messages: ChatMessage[]
  streaming: boolean
  progressText: string
  progressLog: string[]
  thinkingText: string
  duelThinking: DuelThinking | null
  pendingActionId: string | null
  debugRecords: DebugRecord[]
  startedAt: number | null
  error: string | null
  // F13c-2a: owner decision 4 (PARENT_STATE.md) — a failed memory save is a
  // status-bar notice only, never chat text. See `send`'s post-stream check.
  storageNotice: boolean
}

const initialState: StreamState = {
  messages: [],
  streaming: false,
  progressText: '',
  progressLog: [],
  thinkingText: '',
  duelThinking: null,
  pendingActionId: null,
  debugRecords: [],
  startedAt: null,
  error: null,
  storageNotice: false,
}

// Generic keepalive heartbeats ("🔄 Processing... (16s)", "💭 Working... (8s)")
// update the status line but don't belong in the per-turn activity log.
const KEEPALIVE_RE = /^(🔄 Processing|💭 Working)/

// F13c-2a timing constants (see `send`): the notice shows for this long once
// triggered, and the one follow-up read of GET /api/debug fires this long
// after stream close, to catch a background save that settles just after.
const STORAGE_NOTICE_CLEAR_MS = 4000
const STORAGE_NOTICE_FOLLOWUP_MS = 2500

type Action =
  | { type: 'restore'; messages: ChatMessage[]; pendingActionId: string | null }
  | { type: 'stream_started'; userText: string }
  | { type: 'message_replaced'; content: string }
  | { type: 'progress'; text: string }
  | { type: 'thinking'; text: string }
  | { type: 'duel_thinking'; payload: DuelThinking }
  | { type: 'complete'; payload: CompletePayload }
  | { type: 'error'; message: string }
  | { type: 'stream_ended' }
  | { type: 'append_assistant'; content: string }
  | { type: 'clear_pending_action' }
  | { type: 'set_pending_action'; id: string | null }
  | { type: 'clear_failed'; message: string }
  | { type: 'cleared' }
  | { type: 'storage_notice'; show: boolean }

function replaceLastAssistant(messages: ChatMessage[], content: string): ChatMessage[] {
  const out = [...messages]
  const last = out[out.length - 1]
  if (last && last.role === 'assistant') {
    out[out.length - 1] = { role: 'assistant', content }
  } else {
    out.push({ role: 'assistant', content })
  }
  return out
}

function reducer(state: StreamState, action: Action): StreamState {
  switch (action.type) {
    case 'restore':
      return {
        ...state,
        messages: action.messages,
        pendingActionId: action.pendingActionId,
      }
    case 'stream_started':
      return {
        ...state,
        streaming: true,
        error: null,
        progressText: '',
        progressLog: [],
        thinkingText: '',
        duelThinking: null,
        startedAt: Date.now(),
        storageNotice: false,
        messages: [
          ...state.messages,
          { role: 'user', content: action.userText },
          { role: 'assistant', content: '' },
        ],
      }
    case 'message_replaced':
      return {
        ...state,
        thinkingText: '',
        messages: replaceLastAssistant(state.messages, action.content),
      }
    case 'progress': {
      const isKeepalive = KEEPALIVE_RE.test(action.text)
      const log =
        !isKeepalive && action.text && state.progressLog[state.progressLog.length - 1] !== action.text
          ? [...state.progressLog, action.text]
          : state.progressLog
      return { ...state, progressText: action.text, progressLog: log }
    }
    case 'thinking':
      return { ...state, thinkingText: action.text || 'Thinking…' }
    case 'duel_thinking':
      return { ...state, duelThinking: action.payload }
    case 'complete': {
      // Attach the turn's activity log to its debug record so the memory
      // panel can show what the agent did after the fact.
      const debug = action.payload.debug
        ? { ...action.payload.debug, activity_log: state.progressLog }
        : null
      return {
        ...state,
        messages: replaceLastAssistant(state.messages, action.payload.content),
        pendingActionId: action.payload.pending_action_id,
        debugRecords: debug ? [...state.debugRecords, debug] : state.debugRecords,
        progressText: '',
        thinkingText: '',
      }
    }
    case 'error':
      return {
        ...state,
        error: action.message,
        messages: replaceLastAssistant(
          state.messages,
          state.messages[state.messages.length - 1]?.content || `⚠️ ${action.message}`,
        ),
      }
    case 'stream_ended':
      return { ...state, streaming: false, progressText: '', thinkingText: '' }
    case 'append_assistant':
      return {
        ...state,
        messages: [...state.messages, { role: 'assistant', content: action.content }],
      }
    case 'clear_pending_action':
      return { ...state, pendingActionId: null }
    case 'set_pending_action':
      // Approval chaining (2026-09-09, F07): a decided action can hand off
      // to the next still-pending proposal from the same turn instead of
      // always clearing (id === null still clears, on the final item).
      return { ...state, pendingActionId: action.id }
    case 'clear_failed':
      return { ...state, error: action.message }
    case 'cleared':
      return { ...initialState }
    case 'storage_notice':
      return { ...state, storageNotice: action.show }
    default:
      return state
  }
}

export function useChatStream() {
  const [state, dispatch] = useReducer(reducer, initialState)
  const abortRef = useRef<AbortController | null>(null)
  const inFlightRef = useRef(false)
  // F13c-2a: identifies "this" send so a stale timer from an earlier send
  // (superseded by a new send or clearAll) can no longer show the notice.
  const sendIdRef = useRef(0)
  const storageTimersRef = useRef<{
    followUp: ReturnType<typeof setTimeout> | null
    clear: ReturnType<typeof setTimeout> | null
  }>({ followUp: null, clear: null })

  const clearStorageTimers = useCallback(() => {
    const timers = storageTimersRef.current
    if (timers.followUp) clearTimeout(timers.followUp)
    if (timers.clear) clearTimeout(timers.clear)
    timers.followUp = null
    timers.clear = null
  }, [])

  // Unmount-only cleanup (no timer is ever started from an effect body —
  // see R_common_rules on StrictMode double-invoke).
  useEffect(() => {
    return () => clearStorageTimers()
  }, [clearStorageTimers])

  // Restore session on mount via the authorized chokepoint; a
  // missing/invalid token surfaces as a visible error, not a silent fresh-session look-alike.
  useEffect(() => {
    authorizedFetch('/api/session')
      .then((r) => {
        if (r.status === 401) {
          dispatch({ type: 'error', message: nonOkRequestMessage(401, 'Session restore') })
          return null
        }
        return r.ok ? r.json() : null
      })
      .then((s) => {
        if (s) dispatch({ type: 'restore', messages: s.history, pendingActionId: s.pending_action_id })
      })
      .catch(() => {})
  }, [])

  const send = useCallback(
    async (req: ChatRequest) => {
      if (!req.text.trim() || inFlightRef.current) return
      // State updates render asynchronously, so `streaming` alone cannot stop
      // two clicks/key events fired in the same render from submitting twice.
      inFlightRef.current = true
      const ctrl = new AbortController()
      abortRef.current = ctrl
      clearStorageTimers()
      const mySendId = ++sendIdRef.current
      dispatch({ type: 'stream_started', userText: req.text })

      // Captured from the `complete` event, checked once the stream ends.
      let completeDebug: DebugRecord | null | undefined
      try {
        // R3: fetchEventSource bypasses authorizedFetch's chokepoint; scope it the same way first.
        assertApiPath('/api/chat')
        await fetchEventSource('/api/chat', {
          method: 'POST',
          headers: authorizedHeaders({ 'Content-Type': 'application/json' }),
          body: JSON.stringify(req),
          signal: ctrl.signal,
          openWhenHidden: true,
          // R2: replaces the library's defaultOnOpen; a non-ok response (e.g.
          // stale token after a restart) gets a clear, token-free message first.
          async onopen(response) {
            if (!response.ok) {
              throw new Error(nonOkRequestMessage(response.status, 'Chat request'))
            }
            const contentType = response.headers.get('content-type')
            if (!contentType?.startsWith(EventStreamContentType)) {
              throw new Error(`Expected content-type to be ${EventStreamContentType}, Actual: ${contentType}`)
            }
          },
          onmessage(ev) {
            if (!ev.data) return
            const data = JSON.parse(ev.data)
            switch (ev.event) {
              case 'message':
                dispatch({ type: 'message_replaced', content: data.content })
                break
              case 'progress':
                if (data.text) dispatch({ type: 'progress', text: data.text })
                break
              case 'thinking':
                dispatch({ type: 'thinking', text: data.text })
                break
              case 'duel_thinking':
                dispatch({ type: 'duel_thinking', payload: data })
                break
              case 'complete':
                completeDebug = data.debug
                dispatch({ type: 'complete', payload: data })
                break
              case 'error':
                dispatch({ type: 'error', message: data.message })
                break
            }
          },
          onerror(err) {
            // Rethrow to stop fetch-event-source's default infinite retry —
            // a chat POST must not be silently re-submitted.
            throw err
          },
        })
      } catch (err) {
        if (!ctrl.signal.aborted) {
          dispatch({ type: 'error', message: err instanceof Error ? err.message : String(err) })
        }
      } finally {
        if (abortRef.current === ctrl) abortRef.current = null
        inFlightRef.current = false
        dispatch({ type: 'stream_ended' })

        // Only for the still-current, non-aborted send: a background save
        // (F13c-1) can settle just after the stream closes, so an absent or
        // keyless `complete.debug` gets one follow-up read of GET /api/debug
        // before giving up; any fetch/parse error stays silent (no `error`
        // state) per the contract.
        if (!ctrl.signal.aborted && sendIdRef.current === mySendId) {
          const showNotice = () => {
            dispatch({ type: 'storage_notice', show: true })
            storageTimersRef.current.clear = setTimeout(() => {
              if (sendIdRef.current === mySendId) dispatch({ type: 'storage_notice', show: false })
            }, STORAGE_NOTICE_CLEAR_MS)
          }
          if (typeof completeDebug?.storage_failed === 'string' && completeDebug.storage_failed) {
            showNotice()
          } else if (completeDebug) {
            storageTimersRef.current.followUp = setTimeout(async () => {
              try {
                const { records } = await api.getDebugRecords()
                const last = records[records.length - 1]
                if (
                  sendIdRef.current === mySendId &&
                  typeof last?.storage_failed === 'string' &&
                  last.storage_failed
                ) {
                  showNotice()
                }
              } catch {
                // Silent per contract: no error state, no extra console noise.
              }
            }, STORAGE_NOTICE_FOLLOWUP_MS)
          }
        }
      }
    },
    [clearStorageTimers],
  )

  const abort = useCallback(() => {
    abortRef.current?.abort()
  }, [])

  const appendAssistant = useCallback((content: string) => {
    dispatch({ type: 'append_assistant', content })
  }, [])

  const clearPendingAction = useCallback(() => {
    dispatch({ type: 'clear_pending_action' })
  }, [])

  const setPendingAction = useCallback((id: string | null) => {
    dispatch({ type: 'set_pending_action', id })
  }, [])

  const clearAll = useCallback(async () => {
    // Supersede any send's pending follow-up/clear timer (F13c-2a).
    clearStorageTimers()
    sendIdRef.current += 1
    try {
      const response = await authorizedFetch('/api/session', { method: 'DELETE' })
      if (!response.ok) {
        let message = `Could not clear chat (${response.status}).`
        try {
          const body = await response.json()
          if (body?.detail) message = body.detail
        } catch {
          // Keep the status-based fallback for non-JSON error responses.
        }
        dispatch({ type: 'clear_failed', message })
        return
      }
      resetDebugBaseline() // server records are gone; new turns start at index 0
      dispatch({ type: 'cleared' })
    } catch (err) {
      dispatch({
        type: 'clear_failed',
        message: err instanceof Error ? err.message : String(err),
      })
    }
  }, [clearStorageTimers])

  return { ...state, send, abort, appendAssistant, clearPendingAction, setPendingAction, clearAll }
}
