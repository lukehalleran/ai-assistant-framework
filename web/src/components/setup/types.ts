import type { api, SetupStatus } from '../../api/client'

export type SetupClient = Pick<
  typeof api,
  | 'getSetupStatus'
  | 'postSetupOpenrouter'
  | 'postSetupKey'
  | 'postSetupProfile'
  | 'postSetupObsidian'
  | 'postSetupComplete'
>

export interface StepProps {
  client: SetupClient
  status: SetupStatus
  /** Re-read the server status after a save. */
  refresh: () => Promise<void>
  next: () => void
}

export function errText(err: unknown): string {
  return err instanceof Error ? err.message : String(err)
}
