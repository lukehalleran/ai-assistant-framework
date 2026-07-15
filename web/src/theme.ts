import { createTheme, type MantineColorsTuple } from '@mantine/core'

// Palette parity with gui/theme.py: bg #111827, surfaces #1f2937,
// borders/inputs #374151, primary #3b82f6, text #f3f4f6.

const daemonBlue: MantineColorsTuple = [
  '#eff6ff',
  '#dbeafe',
  '#bfdbfe',
  '#93c5fd',
  '#60a5fa',
  '#3b82f6',
  '#2563eb',
  '#1d4ed8',
  '#1e40af',
  '#1e3a8a',
]

const daemonDark: MantineColorsTuple = [
  '#f3f4f6', // 0: text
  '#d1d5db',
  '#9ca3af',
  '#6b7280',
  '#4b5563',
  '#374151', // 5: borders / inputs
  '#2b3544',
  '#1f2937', // 7: surfaces
  '#161e2b',
  '#111827', // 9: page background
]

export const theme = createTheme({
  fontFamily: "'JetBrains Mono', ui-monospace, Consolas, monospace",
  fontFamilyMonospace: "'JetBrains Mono', ui-monospace, Consolas, monospace",
  headings: { fontFamily: "'JetBrains Mono', ui-monospace, Consolas, monospace" },
  primaryColor: 'daemonBlue',
  primaryShade: 5,
  colors: {
    daemonBlue,
    dark: daemonDark,
  },
  defaultRadius: 'md',
})

export const USER_BUBBLE = '#1e40af'
export const BOT_BUBBLE = '#374151'
