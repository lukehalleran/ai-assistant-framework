// Browser-side defence in depth for clipboard exports. The API already
// redacts debug records, but keeping this boundary safe protects copies made
// from records supplied by older servers or tests/fixtures.

const replacements: Array<[RegExp, string]> = [
  [
    /\b(api[_ -]?key|access[_ -]?token|bearer|password|passwd)(\s*[:=]\s*)([^\s,;]+)/gi,
    '$1$2[REDACTED CREDENTIAL]',
  ],
  [/\b(?:sk|pk)-[A-Za-z0-9_-]{16,}\b/g, '[REDACTED CREDENTIAL]'],
  [/\b(?:home|mailing|street)?\s*address\s*[:=]\s*[^\r\n]+/gi, 'address: [REDACTED ADDRESS]'],
  [/\b(?:date\s+of\s+birth|dob)\s*[:=]\s*[^\r\n,;]+/gi, 'date of birth: [REDACTED DOB]'],
  [/\b(gtid|student\s*id)(\s*[:#=-]?\s*)\d{5,12}\b/gi, '$1$2[REDACTED ID]'],
  [/[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}/gi, '[REDACTED EMAIL]'],
  [/\b\d{3}-\d{2}-\d{4}\b/g, '[REDACTED SSN]'],
  [
    /(?<![\w\d])(?:\+?1[\s.()-]*)?\(?\d{3}\)?[\s.-]*\d{3}[\s.-]*\d{4}(?![\w\d])/g,
    '[REDACTED PHONE]',
  ],
  [/(?<![\w\d])\+\d{1,3}(?:[\s.-]?\(?\d{1,4}\)?){2,5}(?![\w\d])/g, '[REDACTED PHONE]'],
  [/\b\d{9}\b/g, '[REDACTED ID]'],
]

export function redactForExport(value: string): string {
  let text = value
  for (const [pattern, replacement] of replacements) {
    text = text.replace(pattern, replacement)
  }
  return text
}
