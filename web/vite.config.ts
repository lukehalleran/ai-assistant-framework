import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Dev server proxies API + health to the FastAPI backend (no CORS needed).
export default defineConfig({
  plugins: [react()],
  // 2026-09-03: the single main chunk had grown to ~1.07 MB. Vendor libraries
  // get their own long-cached chunks; the non-chat views are lazy-loaded in
  // App.tsx so the chat shell is what a cold load pays for.
  build: {
    chunkSizeWarningLimit: 700,
    rollupOptions: {
      output: {
        manualChunks(id: string) {
          if (!id.includes('node_modules')) return undefined
          if (id.includes('@mantine')) return 'mantine'
          if (/node_modules\/(react|react-dom|scheduler)\//.test(id)) return 'react'
          if (id.includes('highlight.js')) return 'highlight'
          if (id.includes('katex')) return 'katex'
          return 'vendor'
        },
      },
    },
  },
  server: {
    port: 5173,
    proxy: {
      '/api': { target: 'http://127.0.0.1:8000', changeOrigin: false },
      '/health': { target: 'http://127.0.0.1:8000', changeOrigin: false },
    },
  },
})
