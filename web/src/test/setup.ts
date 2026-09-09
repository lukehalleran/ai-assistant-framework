import { afterEach } from 'vitest'
import { cleanup } from '@testing-library/react'
import '@testing-library/jest-dom/vitest'

// `test.globals` is off (see vitest.config.ts) so React Testing Library's
// automatic afterEach cleanup never registers itself — do it explicitly.
afterEach(() => {
  cleanup()
})

// jsdom does not implement matchMedia; Mantine's color-scheme hooks call it
// on mount for every component under test.
if (typeof window !== 'undefined' && !window.matchMedia) {
  window.matchMedia = (query: string) => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: () => {},
    removeListener: () => {},
    addEventListener: () => {},
    removeEventListener: () => {},
    dispatchEvent: () => false,
  }) as unknown as MediaQueryList
}

// jsdom does not implement ResizeObserver; Mantine's ScrollArea/AppShell use
// it for layout tracking.
if (typeof window !== 'undefined' && !('ResizeObserver' in window)) {
  class ResizeObserverStub {
    observe() {}
    unobserve() {}
    disconnect() {}
  }
  // @ts-expect-error -- test shim, not a spec-complete ResizeObserver
  window.ResizeObserver = ResizeObserverStub
}
