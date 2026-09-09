import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'

// Minimal component/state behaviour lane (2026-09-09, audit F07/T09). Kept
// separate from vite.config.ts so the production build config never has to
// carry test-only settings.
export default defineConfig({
  plugins: [react()],
  test: {
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
    globals: false,
    css: false,
  },
})
