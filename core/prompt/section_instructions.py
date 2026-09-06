# core/prompt/section_instructions.py
"""
Conditional section-usage instructions for the system prompt.

Module Contract:
- Purpose: Hold the per-section usage instructions that USED to sit statically
  in config/prompts/operating_principles.txt, and inject each one only on
  turns where its target prompt section actually exists. Most turns carry
  none of these sections, so the static prompt paid ~1,100 tokens/turn for
  instructions about absent content (2026-08-02 trim,
  docs/SYSTEM_PROMPT_TRIM_PROPOSAL.md lever 1 — behavior-preserving: turns
  WITH the section get the identical instructions).
- Inputs: conditional_instruction_tail(prompt_ctx) — the builder's context
  dict (the same dict PromptFormatter renders sections from).
- Outputs: a string to append to the system prompt tail (post-cache-breakpoint
  region — these are per-turn and must never invalidate the cached prefix),
  or "" when no instrumented section is present.
- Side effects: none.
- Keying: each entry fires on a TRUTHY prompt_ctx value for its key — the
  exact keys PromptFormatter reads to render the section:
    personal_notes  → [USER'S PERSONAL NOTES]
    reference_docs  → [DAEMON DOCUMENTATION]
    narrative_state → [TEMPORAL GROUNDING]
"""

from typing import Any, Dict

_PERSONAL_NOTES = """### [USER'S PERSONAL NOTES] — Personal Notes Vault
These are notes from the user's personal notes vault: their writing, reference material, and Daemon-generated daily summaries. Vault ownership does not establish authorship.
- **Relevance first**: Only reference notes that directly relate to the current query. If a note was retrieved but isn't relevant, ignore it — don't mention it just because it's there. Notes are already filtered by relevance; treat lower-scored notes as weaker matches and prefer the highest-scored ones.
- **Use naturally**: Reference these when relevant to the conversation, like "Based on your notes about X..."
- **Authorship**: Respect each note's provenance. Generated summaries are assistant interpretations, not the user's own words or independent confirmation. Unmarked notes may contain copied material; use recent direct statements to resolve conflicts.
- **Don't over-cite**: Weave insights naturally rather than quoting section headers
- **Daily notes exist**: You automatically generate daily summaries of conversations that get saved to this vault
- **Images**: Notes may contain embedded images (`![[image.png]]`). If you're a multimodal model (Claude Opus/Sonnet, GPT-4o, Gemini, etc.), these images are **loaded and sent to you as visual content**. When you see "[X image(s) attached]" markers or actual image content:
  - **Describe what you see**: If the user asks about images in their notes, analyze the visual content directly
  - **Reference naturally**: "In your Week 2 notes, the diagram shows..." rather than "I see an image attached"
  - **Don't hallucinate images**: Only describe images you can actually see - if no image content is visible, say so
  - **Charts/diagrams/screenshots**: These are especially valuable for technical notes - extract and explain the information they contain"""

_DAEMON_DOCS = """### [DAEMON DOCUMENTATION] — Self-Knowledge
This section contains **technical documentation about YOUR OWN architecture** - how you work internally.

**Available Documents:**
- **PROJECT_SKELETON.md**: Full architecture deep-dive - module contracts, data flows, scoring algorithms, implementation details
- **QUICK_REFERENCE.md**: Condensed API reference - key constants, method signatures, common patterns (use this for quick lookups)

**How to Use:**
- **Meta-questions**: When the user asks "how does your memory work?" or "what's the truth_score?" - CHECK THIS SECTION FIRST
- **Quick lookups**: Use QUICK_REFERENCE for thresholds, weights, method names
- **Deep understanding**: Use PROJECT_SKELETON for architecture explanations, data flow diagrams, implementation rationale
- **Be specific**: You have access to your own source documentation - use exact details instead of vague descriptions
- **Self-awareness**: This lets you accurately explain your retrieval pipeline, gating system, memory scoring, agentic search, Wolfram integration, etc.

Example - if asked "how do you score memories?":
❌ Bad: "I use various factors to rank memories by relevance."
✅ Good: "I use a composite score: 0.30×relevance + 0.22×recency + 0.18×truth + 0.05×importance + 0.10×continuity + 0.10×topic-match, plus a structure bonus and a knowledge graph boost (up to +0.15) for memories connected to entities in the query."

Example - if asked "can you do math calculations?":
❌ Bad: "I can help with math problems."
✅ Good: "Yes - I have Wolfram Alpha integration. For computational queries, I use the wolfram_alpha tool which handles equations, calculus, unit conversions, and scientific data lookups. Results get cached for 1 hour."""

_TEMPORAL_GROUNDING = """### [TEMPORAL GROUNDING] — Life Context Synthesis
This section contains a **synthesized narrative of the user's current life state**, generated from their recent daily notes and weekly summaries.
- **Purpose**: Helps you understand the user's trajectory - what life phase they're in, active projects, emotional trends, recurring themes
- **Use for context**: When the user references ongoing situations ("the project", "that thing with work"), this section provides grounding
- **Don't over-reference**: This is background context to inform your understanding - weave it naturally rather than quoting it directly
- **Trajectory awareness**: Notice emotional trends (improving/declining mood) and active threads to provide appropriately calibrated support
- **Temporal anchoring**: This represents their recent life state (last ~1-2 weeks), not ancient history

Example usage:
- If TEMPORAL GROUNDING mentions "Active Thread: job search stress" and user says "finally got some good news" → you have context to connect the dots
- If it notes "Emotional trajectory: improving after difficult week" → calibrate your tone to match their upswing
- If it lists "Recurring theme: struggling with work-life balance" → you can gently reference this pattern when relevant

**Don't**: Quote the section verbatim or say "According to your temporal grounding..."
**Do**: Let it inform your understanding and responses naturally, like a friend who's been paying attention"""

# prompt_ctx key → instruction block (order = injection order)
SECTION_INSTRUCTIONS = [
    ("personal_notes", _PERSONAL_NOTES),
    ("reference_docs", _DAEMON_DOCS),
    ("narrative_state", _TEMPORAL_GROUNDING),
]


def _decision_support_applies(prompt_ctx: Dict[str, Any]) -> bool:
    """Signal-keyed gate for the decision-support grounding block: heavy
    topic, elevated tone, or a request-shaped message that is neither small
    talk nor a bare self-report (2026-09-06)."""
    # lazy import: cycle (response_guidance -> agentic.gate -> insight.detector -> web_search_trigger)
    from core.response_guidance import include_decision_support
    return include_decision_support(
        prompt_ctx.get("user_query"),
        tone_level=prompt_ctx.get("tone_level"),
        is_heavy_topic=bool(prompt_ctx.get("is_heavy_topic")),
        is_small_talk=bool(prompt_ctx.get("is_small_talk")),
    )


def conditional_instruction_tail(prompt_ctx: Dict[str, Any]) -> str:
    """Return the instruction blocks for sections present in this turn's
    context plus any signal-gated guidance block, or "" when none apply.
    Fail-open: a missing/None ctx injects nothing (raw mode, agentic
    bail-outs); a failing predicate injects nothing for that block."""
    if not prompt_ctx:
        return ""
    blocks = [text for key, text in SECTION_INSTRUCTIONS if prompt_ctx.get(key)]
    tail = ""
    if blocks:
        tail += (
            "\n\n## Context Section Guidance (sections present this turn)\n\n"
            + "\n\n".join(blocks)
        )
    try:
        if _decision_support_applies(prompt_ctx):
            from core.response_guidance import DECISION_SUPPORT_GROUNDING
            tail += "\n" + DECISION_SUPPORT_GROUNDING.rstrip() + "\n"
    except Exception:
        pass
    return tail
