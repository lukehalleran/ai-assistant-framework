# Literature Oracle — Claim-Level Documented-ness Instrument

*Methodology + results log. Companion to `docs/SYNTHESIS_VALIDATION.md` (same role: the
living evidence record). Status: BUILT, calibration pending. 2026-07-01.*

---

## 1. Purpose & non-goals

The stage-3 world-novelty oracle in `knowledge/synthesis_filter.py` is **blind**: its only
"known" signals are embedding-proximity (`cos(A,B)>0.45` — the field misleadingly named
`cooccurrence_similarity` — plus verbatim-claim FAISS `>0.88`). A documented cross-domain
identity that is *paraphrased* and *embedding-distant* evades both. The 2026-06-30 blindspot
run (`scripts/validate_novelty_blindspot.py`, MERGED grading CSV, top-6 + bottom-5 by
composite) found **11/11 sampled candidates documented in the literature** — 100%
false-novelty among the sampled accepts.

This instrument answers, per candidate, **"is this specific claim documented?"** with an
**evidence-backed, human-auditable verdict**. It exists to run ONE pre-registered
measurement: the pooled generator's genuine-novel yield (§6), which decides the fork
(rebuild stage-3 novelty around a claim-level oracle vs rethink the generator's concept
source first).

**Non-goals (hard boundaries):**
- NOT wired into production. `synthesis_filter.py` never imports it; no config.yaml /
  schema.py / app_config.py surface (constants are module-local in
  `knowledge/literature_oracle.py`, env-var overrides only).
- NOT a replacement for `knowledge/doc_cooccurrence.py` (pair-level wiki oracle —
  different question). May be logged as an aux column; never gated on.
- NOT proof of absolute novelty. `NOT_FOUND` means "not found by this fixed search
  procedure" — the instrument supports relative comparison under a fixed procedure and
  bounded claims only.

## 2. Verdicts & the evidence rule

```
DOCUMENTED_EXACT     this specific mapping/identity is published
DOCUMENTED_ADJACENT  the concepts are connected in the literature, but NOT this claim
NOT_FOUND            the search procedure surfaced no documentation
UNCERTAIN            evidence insufficient — including code-enforced downgrades
PARSE_FAILURE        adjudicator output unparseable after exactly 1 retry (LOUD, counted;
                     never defaulted into another verdict — the dead-coherence-judge lesson)
```

**Evidence rule (code-enforced in `LiteratureOracle.enforce_evidence_rule`, never
prompt-only):** a `DOCUMENTED_*` verdict is valid only with ≥1 citation whose `url` matches
a retrieved evidence item and whose `supporting_passage` (≤40 words) appears **verbatim**
(whitespace/case/unicode-normalized) in that item's title+text. Otherwise the verdict is
downgraded to `UNCERTAIN` with `downgraded_from_prior=true`. The adjudicator's own prior
(`llm_prior_verdict`) is logged in a separate field and **never drives the verdict** — an
LLM adjudicating from its training prior is unauditable and hallucination-prone.

## 3. Per-candidate pipeline

```
[Q] query generation (1 LLM call, temp 0)
      >=1 CANONICAL-names query (paraphrase-evasion countermeasure)
      >=1 relationship-type query ("X duality Y theorem"); <=400 chars each
      parse failure here -> deterministic fallback templates (logged, not terminal)
[R] retrieval
      arXiv API: all queries, free, >=3.1s spacing + retries (politeness lock)
      Tavily (WebSearchManager): up to 3 queries at QUICK depth (1 credit each),
        72h isolated cache (data/oracle/cache_chroma — no production ChromaDB writes)
      merge: canonical-URL dedupe (extracts beat snippets), arXiv capped at 4 slots
        (ORACLE_MAX_ARXIV_SLOTS — so web evidence is never crowded out; spare slots
        backfilled with remaining arXiv), cap 8 items
[A] adjudication (1 LLM call, temp 0.1, strict JSON, verdict-key order is an A/B knob)
      judged STRICTLY from the rendered evidence [E1..E8]
      rendering caps (2026-07-07): snippets 800 chars (ORACLE_EVIDENCE_SNIPPET_CHARS);
        tavily_extract full pages 6000 chars (ORACLE_EVIDENCE_EXTRACT_CHARS), window
        selected around claim-term density — previously extracts were cut to 800 chars
        too, so escalation paid 2 credits for text the adjudicator never saw
      parse failure -> 1 retry -> PARSE_FAILURE
[V] validation: evidence rule (above), then
[E] escalation (at most once): when the model says documented (verdict OR prior) but no
      qualifying passage exists in snippets -> re-search the canonical query at STANDARD
      depth (2 credits, extracts top-2 pages) -> one re-adjudication -> re-validate.
      Rationale: textbook identities live in books/reviews/Wikipedia, not arXiv abstracts;
      without escalation, true positives bleed into UNCERTAIN on snippet surface area.
```

Cost: worst case 5 Tavily credits/candidate (3 QUICK + 1 STANDARD) + 2–3 LLM calls.
Budget guard: runs stop cleanly before a candidate when remaining daily credits < 5 and
print the `--resume` command; results.jsonl append + `load_done_ids` make resumption exact.

## 4. Calibration protocol (BLOCKS §6 until PASS)

Set: `data/oracle/calibration_set.jsonl` (regenerate: `scripts/oracle_calibrate.py --init-set`).

| kind | n | what |
|---|---|---|
| positive_arxiv | 11 | the blindspot-run candidates (MERGED top-6 + bottom-5 reconstruction) |
| positive_textbook | 15 | hand-listed identities, claim_text in pooled-generator PARAPHRASE style |
| negative_mutated | 15 | documented identity with ONE side swapped for a plausible-but-wrong neighbor |

**Owner gates:**
- Every `negative_mutated` row ships `owner_verified:false`; the script **refuses** to score
  negatives until each is verified (a mutation we invent can accidentally be real — Kelly's
  detailed-balance↔queueing reversibility IS real and therefore sits in the positives).
- Owner confirms the 11-row reconstruction (evidence: top-1 `information theory↔control
  theory` = the "data-rate theorem" of the finding, present only in the MERGED CSV top-6;
  the blindspot run used `--top 6`).

**Pre-registered thresholds:**
- positives: ≥90% verdict ∈ {DOCUMENTED_EXACT, DOCUMENTED_ADJACENT}, AND ≥75%
  DOCUMENTED_EXACT specifically (the exact/adjacent split is the instrument's point).
- mutated negatives: ≤10% DOCUMENTED_EXACT (ADJACENT on a mutated identity is acceptable).
- parse failures: 0 in the final calibration run.
- On failure: iterate prompts / verdict-key order / escalation depth, re-run, and log every
  iteration in §5. Do NOT proceed to §6 on a failed calibration.

### Iteration log

| run_id | knobs | pos DOC_* | pos EXACT | neg EXACT | parse fails | PASS | changes made |
|---|---|---|---|---|---|---|---|
| *(pending)* | | | | | | | |

## 5. Calibration results

*(appended per run — see `data/oracle/runs/<run_id>/report.md`)*

## 6. Measurement pre-registration (the payoff run)

**Corpus** (`scripts/oracle_measure_generator.py --build-corpus`; NOT the "27/28-graded"
set — those are old-pipeline personal→wiki candidates, out of scope):

- `pooled_csv` — `discovery_grading_MERGED_20260630_140521.csv`: 34 ACCEPT + 11 QUEUE
- `anchored_json_{anchored,random}` — `anchored_gen_validation_20260630_121819.json`:
  5 + 8 accepts; rejects feed the contrast strata
- Strata: **primary N=40** (accepts, not pre-seen) · pre_seen=7 (the blindspot 11 minus its
  5 QUEUEs, plus calibration collisions — scored, reported, EXCLUDED from N) · queued=11 ·
  reject_coherence=10 + reject_composite=10 (seed 20260701)
- Dedup: claim-hash exact-drop; pair-level duplicates kept and logged.
- **Blinding:** manifest rows carry `stratum`/`pre_seen` bookkeeping fields, and `source` is
  passed to `score_candidate` — but none of these reach the LLM prompts (unit-tested with
  banned-token checks); status/composite/coherence live in `corpus_hidden.jsonl`, joined at
  report time only.
- **Contingency** (pre-registered, not run by default): if the decision rule lands in the
  ambiguous band, generate a fresh pooled batch with seeds {101,102,103,104}
  (`scripts/smoke_pooled_generator.py` pattern, temp chroma, no production writes) to grow N.

**Decision rule (COUNTS, not percentages — N≈40 makes % bands meaningless).**
X = NOT_FOUND among primary usable verdicts; U = UNCERTAIN among primary:
- validity gate: U > 0.25·N or >2 PARSE_FAILUREs ⇒ instrument insufficient — iterate the
  oracle, NOT the thresholds; no generator conclusion.
- **X ≤ 2** ⇒ ~0 genuine-novel yield confirmed quantitatively ⇒ proceed to fork (b)
  (generator concept-source rethink) with this oracle as the scorer for any new generator.
- **3 ≤ X ≤ 7** ⇒ owner reviews every NOT_FOUND individually (candidate genuine novels);
  optionally run the fresh-batch contingency.
- **X ≥ 8** ⇒ blindspot repair changes yield accounting but the generator retains discovery
  value; re-examine before generator changes.
- UNCERTAIN never counts toward X. Pre-seen / queued / reject strata are descriptive only
  (expected contrast: rejects skew NOT_FOUND/UNCERTAIN — they are incoherent, not documented).

**Prediction on record (owner signs before the run):** X ≤ 2, and DOCUMENTED_EXACT dominates
the primary stratum (the "top accepts are textbook identities" diagnosis).

**Sign-off (mechanically enforced — the runner refuses to start without a line matching the
CURRENT manifest SHA):** after `--build-corpus`, change the keyword below from PENDING to
PREREGISTERED on its own line:

```
PENDING-PREREGISTRATION: 45215198b3b10792e4d4239fdb0713bc2f368cd83a6df2d8ff68c4ff510a248f 2026-07-01
```

## 7. Measurement results

*(pending — per-stratum verdict counts, X/U/N, decision-rule outcome, NOT_FOUND survivors)*

## 8. Known limitations

- **Escalation is one-directional (biases toward the signed prediction):** the re-search
  fires only when the model believes the claim is documented but cannot cite it — there is
  no symmetric second look for shaky NOT_FOUNDs. That asymmetry pushes X (NOT_FOUND count)
  DOWNWARD, i.e. toward the pre-registered X ≤ 2. Defensible (NOT_FOUND is the conservative
  default for a novelty claim), but any X ≤ 2 outcome must be read with this in mind; if the
  band decision is close, run a NOT_FOUND-side escalation A/B as a logged iteration.
- **Positive-label provenance:** the `positive_arxiv` calibration rows were labeled
  DOCUMENTED by one temp-0, documented-leaning LLM call over ≤2200 chars of arXiv abstracts
  (`validate_novelty_blindspot.py`) — they are reconstructions, not verified documentation.
  Owner verification treats a calibration miss on these rows as "label suspect" before
  "oracle broken"; unverifiable positives are dropped, not tuned toward.
- **Snippet bias, partially mitigated:** escalation extracts only the top-2 pages of one
  query; a passage in a book chapter or paywalled review can still evade citation → UNCERTAIN
  inflation. The validity gate (U > 0.25·N) catches this failing loudly. Logged fix path if
  calibration recall fails on textbook positives with `downgraded_from_prior=true`: raise
  extraction depth or add a Wikipedia-domain query — as logged calibration iterations, never
  silent changes.
- **arXiv coverage gap:** classic textbook material predates or bypasses arXiv; Tavily-side
  evidence carries those. Calibration measures the combined channels, not arXiv alone.
- **Prior anchoring risk:** eliciting `llm_prior_verdict` in the same adjudication call may
  anchor the evidence verdict. If calibration shows suspicious prior/verdict correlation,
  A/B a separate no-evidence prior call (one more LLM call per candidate).
- **Tavily cache staleness:** 72h TTL; multi-day budget-split runs re-pay credits for
  re-queried items only (done candidates are never re-scored).
- **n is modest:** N≈40 primary. The count-based rule is sized to that; the fresh-batch
  contingency exists to grow it.
- Field names state what is computed (the `cooccurrence_similarity` lesson); any numeric
  threshold in the module carries its scale in a comment and a boundary unit test.
