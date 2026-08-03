# System Prompt Trim Proposal (2026-08-02)

**Status update (same day):**
- **CORRECTION**: the live system prompt is `config/prompts/default_personality.txt`
  (~1.7K tok) + `config/prompts/operating_principles.txt` (~5.0K tok) ≈ 6.7K —
  `core/system_prompt.txt` (analyzed below) is only the FALLBACK path. The
  section-by-section table below is directionally right (the same blocks exist
  in operating_principles.txt) but sizes refer to the fallback file.
- **Lever 1 SHIPPED**: the three per-section guidance blocks
  ([USER'S PERSONAL NOTES] 377, [DAEMON DOCUMENTATION] 414, [TEMPORAL
  GROUNDING] 352 ≈ **1,177 tokens**) moved out of `operating_principles.txt`
  into `core/prompt/section_instructions.py` and now inject into the system
  prompt tail (post-cache-breakpoint) ONLY on turns whose context contains
  that section (keyed on the formatter's own ctx keys: `personal_notes`,
  `reference_docs`, `narrative_state`). Behavior-preserving on turns with the
  sections; most turns save the full 1.2K. The stale scoring example
  (0.35/0.25/0.20 weights) was corrected to the live vector in transit.
  Tests: `tests/unit/test_section_instructions.py`.
- **Lever 2 (Examples diet) still owner review**: `default_personality.txt`
  Examples block is ~697 tokens — that's your voice, cut it yourself or ask.

---

Original analysis below (sizes from the fallback `core/system_prompt.txt`).

## Why

The base system prompt is ~7,650 tokens, sent on **every turn** (the stable
prefix is prompt-cached on Anthropic/GPT models, but the active models —
kimi-3, deepseek — use implicit caching where the whole prompt still counts
toward context and cost on cache misses). Together with the per-turn tail
(topic/threads/tone/plan/intent style) the system side runs ~13K tokens/call.
The 2026-07-15 budget experiment showed *smaller* prompts were judged BETTER
(+1.26/25), so trimming here is likely free quality, not a sacrifice.

## Per-section size (chars/4 ≈ tokens)

| Section | ~tokens | Assessment |
|---|---|---|
| Critical: Response Scope | 878 | Largest block. Likely compressible 50% — repeated injunctions. |
| Examples | 835 | **Top trim candidate.** Few-shot examples are the classic budget-experiment loser; 2-3 tight examples usually match 8 verbose ones. |
| Interaction Rules | 701 | Overlaps Core Style + Response Approach; merge candidates. |
| Agentic Capabilities | 505 | Only relevant on agentic turns (~10% of turns). Candidate for **conditional injection** — the agentic controller already builds its own instructions. |
| Knowledge & Information Queries | 460 | Overlaps Source Quality / Recency Rule / Precision blocks. |
| Core Style | 450 | Keep — this is the voice. |
| [DAEMON DOCUMENTATION] Self-Knowledge | 414 | Section-usage instructions; only needed when the section is present → conditional injection candidate. |
| [USER'S PERSONAL NOTES] Obsidian | 377 | Same — conditional injection candidate. |
| Memory & Context Integration | 376 | Keep, maybe tighten. |
| Self-Notes | 372 | Conditional (only when daemon_self_notes present). |
| [TEMPORAL GROUNDING] | 352 | Conditional. |
| Document Generation | 206 | Conditional (doc-gen turns only). |
| Comparison Output Template (for countries) | 76 | Hyper-specific one-off template in every prompt. Move to intent-conditional or delete. |
| Wolfram / Calculations / Source Attribution / Precision x2 / Recency | ~450 combined | Mostly conditional-injection candidates (computation turns). |

## Two mechanical levers (no voice changes)

1. **Conditional section-instruction injection**: the formatter knows which
   context sections are present each turn. Moving the per-section usage
   instructions (Obsidian, Self-Knowledge, Self-Notes, Temporal Grounding,
   Document Generation, Wolfram, country-comparison template — **~1,900
   tokens**) out of the static prompt and injecting each only when its
   section exists would cut most turns by ~1.5-1.9K tokens with zero
   behavior change on the turns that need them. Caveat: these would land in
   the per-turn (uncached) region for explicit-cache models — for kimi/
   deepseek implicit caching this is still a clear win.
2. **Examples diet**: cut the Examples block from ~835 to ~300 tokens
   (keep the 2-3 most load-bearing examples).

Combined estimate: **-2.5 to -3K tokens/turn** without touching Core Style,
Interaction Rules, or safety/grounding rules.

## Suggested process

Same as the token-budget change: run `scripts/budget_experiment.py`-style
pairwise judging (trimmed vs current, seed corpus) before adopting. The
conditional-injection lever can ship without an experiment — it only removes
instructions on turns where their target section is absent.
