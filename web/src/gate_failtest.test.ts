import { expect, it } from 'vitest'

// DO NOT MERGE. Deliberate failure for the gate test: the frontend job must
// fail, bug-class-gate must fail with it, and branch protection must block
// the merge. The throwaway PR is closed and this branch deleted afterwards.
it('fails on purpose so bug-class-gate can be seen to fail', () => {
  expect(1).toBe(2)
})
