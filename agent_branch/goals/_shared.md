# Shared principles (every agent branch)

These apply to every worker regardless of its lens. Your per-agent goals file is
layered on top of this.

## What a good change is
- Achieve the stated **objective** fully, with a single coherent change.
- Prefer the **smallest correct** change. After correctness, the supervisor ranks
  survivors by diff economy — a thin, correct diff wins over a sprawling one.
- **Preserve all existing behavior.** Do not break public APIs, signatures, or
  imports that other code depends on.
- Match the surrounding code's **style and conventions** (naming, typing, logging,
  error handling). Read the nearby code before writing.

## Scope discipline (hard rules — violating any gets your branch killed)
- Touch **only** files within your allowed scope. Everything else is forbidden.
- Never modify tests, configuration, benchmarks/thresholds, the safety guards, or
  the supervision/agent_branch machinery — including these goals files. A diff that
  touches any of them is rejected before it is even evaluated.
- Do not weaken, disable, or work around any check, gate, or test to make your
  change "pass". Make the real change instead.
- No network calls, no reading secrets, no shelling out to destructive commands.

## Output
- Unless your prompt says otherwise, output the **complete new content of the
  target file only** — no prose, no explanation, no markdown fences.
- The change must be self-contained and apply cleanly.
