import { useCallback, useEffect, useReducer, useRef } from 'react'
import { fetchEventSource } from '@microsoft/fetch-event-source'
import { resetDebugBaseline } from './debugSession'
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
}

// Generic keepalive heartbeats ("🔄 Processing... (16s)", "💭 Working... (8s)")
// update the status line but don't belong in the per-turn activity log.
const KEEPALIVE_RE = /^(🔄 Processing|💭 Working)/

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
  | { type: 'clear_failed'; message: string }
  | { type: 'cleared' }

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
    case 'clear_failed':
      return { ...state, error: action.message }
    case 'cleared':
      return { ...initialState }
    default:
      return state
  }
}

export function useChatStream() {
  const [state, dispatch] = useReducer(reducer, initialState)
  const abortRef = useRef<AbortController | null>(null)
  const inFlightRef = useRef(false)

  // Restore session on mount (refresh-safe UI)
  useEffect(() => {
    fetch('/api/session')
      .then((r) => (r.ok ? r.json() : null))
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
      dispatch({ type: 'stream_started', userText: req.text })

      try {
        await fetchEventSource('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(req),
          signal: ctrl.signal,
          openWhenHidden: true,
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
      }
    },
    [],
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

  const clearAll = useCallback(async () => {
    try {
      const response = await fetch('/api/session', { method: 'DELETE' })
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
  }, [])

  return { ...state, send, abort, appendAssistant, clearPendingAction, clearAll }
}
