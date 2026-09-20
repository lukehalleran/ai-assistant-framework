import { HttpStatusError } from '../../api/client'

// BC-80 (docs/BUG_CLASSES.md): the Curation Center's apply/dismiss/undo/scan
// actions run in a server worker that outlives the HTTP request
// (api/routes/curation.py `_run_operation` — "the worker owns the lock;
// cancelling its HTTP wait cannot release it"). The engine's queue + journal
// are the retained outcome. A dropped connection (phone loses signal
// mid-request, tab backgrounded, etc.) therefore does NOT mean the operation
// failed — it means the response was lost while the work may have completed
// on the server. This is the same false-failure shape fixed for notes sync
// on 2026-09-16 (see api/client.ts startNotesSyncAndPoll / HttpStatusError).

export interface CurationFailure {
  /** True when the response was lost (not an HTTP status) — the operation's
   * real outcome is unknown and must be re-read from the server, never
   * reported as "<verb> failed". */
  lost: boolean
  title: string
  message: string
  color: 'red' | 'yellow'
}

/** Classifies a rejection from a Curation Center API call. `verb` is the
 * action being reported ("Apply", "Dismiss", "Undo", "Scan", …). */
export function describeCurationFailure(err: unknown, verb: string): CurationFailure {
  if (err instanceof HttpStatusError) {
    return {
      lost: false,
      title: `${verb} failed`,
      message: err.detail || err.message,
      color: 'red',
    }
  }
  return {
    lost: true,
    title: 'Connection lost',
    message: `${verb} may have completed on the server — refreshing the queue.`,
    color: 'yellow',
  }
}
