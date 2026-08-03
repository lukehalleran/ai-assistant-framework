# Adaptive Exemplar Learning

*Created: 2026-08-03*

The subsystem that replaces hand-maintained keyword/exemplar lists with
per-user learned exemplars. Before 2026-08-02, every semantic classifier in
the pipeline (tone, need, web-search trigger, intent) ran against hardcoded
seed phrase lists, and each production miss was fixed by manually adding
phrases — a maintenance model that does not scale and never personalizes.
This subsystem gives each classifier a learning loop: seeds ship with the
code, confirmed classifications teach new exemplars, and a fresh instance
personalizes to its owner's phrasing over time.

For the tone-domain specifics see `TONE_DETECTION_SUMMARY.md`. For the intent
semantic tier see `ARCHITECTURE_GUIDE.md` §4 and `QUICK_REFERENCE.md`.

---

## Core Design

### The Store

`utils/adaptive_exemplars.py` — `AdaptiveExemplarStore`, a domain-generic
JSON-backed store at `data/adaptive_exemplars.json`:

- `record(domain, label, text, source, embedder=None, seed_texts=None)` —
  add a learned exemplar. Quality gates: ≥ 12 chars, clipped to 300; exact
  duplicates (case-folded) rejected; near-duplicates rejected via cosine
  ≥ 0.92 against seeds + already-learned (when an embedder is provided);
  per-label cap 40 with oldest-evicted.
- `get_learned(domain, label)` — learned texts for prototype building.
- `version` — monotonic counter bumped on every accepted record; consumers
  key their prototype/embedding caches on it so a new exemplar invalidates
  exactly one cache rebuild.
- Persistence: atomic write; **lenient** load (corrupt file = start empty).
  This is derived, self-healing state — the documented exception to the
  strict-load rule for user-data stores (losing it costs personalization,
  not data).
- `get_store()` — process-wide singleton.

### The One Rule: Independent-Channel Teachers

Learning must never be able to poison the classifier that consumes it.
Every adopter follows the same contract:

1. **Only channels INDEPENDENT of the semantic prototypes teach.** A
   deterministic keyword hit, an LLM arbiter verdict, an outcome signal
   (citation in the final response), a gate veto — never the semantic
   similarity score itself. The semantic tier consuming the exemplars can
   never teach itself (no self-reinforcement loop).
2. **Negative/fallback labels are never learned.** `conversational` (tone),
   `neutral` (need), `general` (intent) are absences of signal, not
   phrasings — teaching them would teach the miss.
3. **Heuristic backstops never teach.** The tone backstop floors borderline
   cases to CONCERN by margin arithmetic; that is a guess, not a
   confirmation.
4. **Elevated-tone turns never teach outcome domains** (a distress turn
   that happened to cite web results is not a search pattern to reinforce).

### Compounding Loop

An arbiter (or keyword stage) catches a borderline phrasing once → the
phrasing is recorded → the prototype shifts → the next paraphrase scores
above-borderline *semantically*, with no arbiter round-trip. Expensive/slow
channels bootstrap the cheap channel.

---

## Adopters (4 domains)

| Domain | Consumer | Prototypes | Teachers (independent channels) | Never teaches |
|--------|----------|------------|--------------------------------|---------------|
| `tone` | `utils/tone_detector.py` semantic stage | `CRISIS_EXEMPLARS` seeds + learned, per crisis level | Deterministic keyword stage; non-conversational LLM-arbiter verdict | Backstop; conversational verdicts; `conversational` label itself |
| `need` | `utils/need_detector.py` semantic stage | `NEED_EXEMPLARS` seeds + learned | High-confidence (≥ 0.8) keyword fast-path | `neutral` (absence of signal) |
| `web_search` | `utils/web_search_trigger.py` `_get_search_anchors()` — learned `search_worthy` merge into positive anchors, `no_search` into negative | seed anchors + learned | OUTCOME-based: response actually citing `[WEB_` markers → `search_worthy` (hook in `handlers._write_turn_telemetry(response_text=…)`, all 3 call sites); tone-corroborated agentic-gate veto → `no_search` (`gate.apply_intent_veto`) | Elevated-tone turns; the trigger's own semantic decision |
| `intent` | `core/intent_classifier.py` semantic tier (fires when regex conf < 0.50; cos ≥ 0.45 + 0.05 margin → conf 0.60, `source="semantic"`) | `INTENT_EXEMPLARS` seeds + learned (8 intents; GENERAL has no prototype) | Confident regex hits (≥ 0.85); STM refinements (`refine_with_stm(query=…)`) | The semantic tier itself; GENERAL |

Deliberate non-adopters: checks that key on FORM rather than vocabulary stay
structural — `gate._is_info_seeking` (question shape), casual-ack detection,
`fact_extractor._is_junk_object` (preposition/negation shape), the memory-mode
gate keywords (entity + recall regex; no clean semantic channel).

Related but separate: `scripts/auto_label_intents.py` removes the other manual
loop (hand-labeling telemetry for the intent confusion matrix) — an LLM labels
`logs/turn_records.jsonl`, `--verify <model>` keeps only dual-model agreements
as trusted; `scripts/label_intents.py` is the optional audit path.

---

## Config

- `TONE_EXEMPLAR_LEARNING`, `NEED_EXEMPLAR_LEARNING`,
  `INTENT_EXEMPLAR_LEARNING` — per-domain env toggles, default on.
- `INTENT_SEMANTIC_TIER` (default on), `INTENT_SEMANTIC_MIN_SIM=0.45`,
  `INTENT_SEMANTIC_MIN_MARGIN=0.05` — intent semantic tier tuning.
- Store path is fixed (`data/adaptive_exemplars.json`); tests sandbox it via
  an autouse fixture in `tests/conftest.py` (keyword-matching test messages
  would otherwise pollute the owner's real store).

## Warmup

`gui/launch._run_model_warmup` step 6 pre-computes tone + need exemplar
embeddings at startup (they were previously computed inside turn 1).

## Tests

- `tests/unit/test_adaptive_exemplars.py` (14) — store gates, cap, version,
  lenient load, singleton.
- `tests/unit/test_adaptive_adopters.py` (10) — need + web-search adopters,
  outcome teachers, elevated-tone guard.
- `tests/unit/test_intent_semantic_tier.py` (11) — tier firing/thresholds,
  teacher channels, GENERAL-never-learned, prototype merge.
- Tone learning covered inside `tests/unit/test_tone_arbiter_hardening.py`.

## Failure Modes to Watch

- **Prototype drift**: many learned exemplars for one label shift its
  centroid; the per-label cap (40) and near-dup gate bound this, but a
  systematically wrong teacher would still drift it — which is why only
  independent channels teach.
- **Store deletion** is safe: seeds always load; the system degrades to
  fresh-install behavior, then re-personalizes.
