# Plan — 2026-09-12: spurious approval warnings, and current-events evidence

Drafted from evidence, not from theory. Every claim below was verified against
`logs/turn_records.jsonl`, `data/corpus_v4.json`, the repo-root
`daemon_debug_20260911_*.log` files, or by calling the deployed predicate.
Nothing in W1/W2 is implemented yet — this is the plan.

```
STATE   Working tree carries the 2026-09-12 keyword-boundary + web-budget batch
        (uncommitted; commit_message_sep12.txt ready). The running daemon
        predates all of it. Three commits will be ahead of origin once
        committed. W1 and W2 below are NOT started.
CLASS   W1: BC-04 (classifier missing an anchor) + BC-76 (closure by
        phrase-append — this family has been widened phrase-by-phrase in four
        dated batches, which is the catalog's own escalation signal).
        W2: BC-61 (owner-domain-scoped vocabulary) + BC-04 + BC-47/BC-78
        (honesty when a capability could not run) + BC-59 (owner-identity
        leakage into a query).
```

---

## W1 — the "no card to approve" and "not on your calendar" warnings

### Evidence

16 warning appends are stored across 09-10 → 09-12 (12 `NO_CARD_NOTICE`, 4
calendar-state). Roughly ten are spurious. Two of them are from today:
`"Lol. Just waking up now"` and `"Pretty sure I'm a bit hung over…"`.

Sentence-level attribution, by calling the deployed
`split_claim_sentences()` + `_APPROVAL_PROMPT_RE` on each stored reply:

| turn | matched span | why it matched |
|---|---|---|
| 09-10 17:16 | `approve it` | in "…whether Congress would need to **approve it**" — a news summary |
| 09-11 17:31 | `it's there` | "…but it's there if the `car`/VIF question resurfaces" |
| 09-11 19:01 | `it's refusing to track reality when reality is right there` | THING=it, STATE=bare `there` |
| 09-11 19:13 | `it's the difference between a country that's given up` | STATE=`up` from "given **up**" |
| 09-11 20:21 | `That's the cleaned-up` | STATE=`up` from "cleaned-**up**" |
| 09-12 11:30 | `that tweet's a good one to wake up` | STATE=`up` from "wake **up**" |
| 09-12 11:49 | `…is about all there is to it` | STATE=bare `there` |

Mechanism: `_CLAIM_CARD_TEMPLATE` (`core/action_claim_guard.py:529-576`)
composes THING `(?:card|proposal|event|it|that|this)` + ≤45 non-terminal chars
+ MODAL `(?:is|'s|are|already|queued|…)` + ≤20 chars + STATE
`(?:up|there|ready|waiting|showing)`. With pronouns in THING and particles in
STATE, that grammar matches ordinary English. The second arm,
`approve\s+(?:it|that|this)` (`:591-592`), has no proximity requirement to
card vocabulary at all. `claims_pending_card` (`:712-731`) scopes per sentence
but only excludes questions and offer clauses — unlike its siblings
`detect_proposals`/`detect_completion_claims`, it never consults
`_detect_kind()`, so there is **no topical anchor whatsoever**.

Two things this is NOT: the adaptive/learned channel is uninvolved
(`data/adaptive_exemplars.json` has no `action_claim` domain; the seed
exemplars matched nothing across ~58 sentences), and the four note/calendar
turns where the notice fired were **correct** — including 09-10 20:57, where a
`propose_action` call really did fire and the controller's own
datetime-grounding guard rejected it (`core/agentic/controller.py:1296-1300`),
so "nothing was actually queued" was literally true.

### Steps

1. **A1 (the structural fix).** Add `_sentence_has_action_anchor(sentence)` to
   `core/action_claim_guard.py`: True when `_detect_kind(sentence)` resolves a
   kind (the calendar/note/email/github vocabulary the siblings already use)
   OR the sentence carries an explicit approval-surface noun
   (`card|proposal|approval|approve|queue[d]|pending|propose`). Require it in
   BOTH `claims_pending_card` and `claims_calendar_state` before the template
   or literal arms may fire. This is the CATEGORIZED-GENERIC + ANCHORS remedy
   (`docs/GENERALIZATION_AUDIT_20260901.md` §Remedy patterns), reusing an
   existing vocabulary rather than adding one.
2. **A2.** With A1 in place, the accumulated negative lookbehinds
   (`(?<!on\s)there`, the "confirm that"/"the queue is" carve-outs) become
   redundant defence. Keep them, and note in the docstring that the anchor is
   now the load-bearing guard — the next miss must not be closed by a ninth
   phrase (BC-76).
3. **A3. Regression fixtures, from the live data**: the 7 false-positive
   sentences above must return False; the 5 legitimate catches (09-10 16:12,
   09-10 19:20, 09-10 20:57, 09-11 10:10, 09-11 12:32/12:33 calendar) must
   stay True. Both directions in one parametrized test — the red control is
   the pre-fix behaviour, which must be shown to fail.
4. **A4 (separate, one instance, needs its own evidence).** 09-10 16:12 was a
   *correct* notice with an upstream cause: after the single-event card was
   rejected the model offered a recurring version, and the next turn's "yes"
   was not recognised as an affirmation of that REPLACED offer
   (`gate_reason: "no trigger"`, plain streaming, no `propose_action`). Trace
   `registry.offer_action_type` / `is_offer_affirmation` against a replaced or
   rejected prior offer before touching it — BC-74's precedence family.

### Acceptance

Replay the 16 stored replies through the deployed predicates: exactly the 5
legitimate ones still fire. No new phrase in any list (`dm29_phrase_append`
gains no row for this family). `tests/unit/test_sep10_probe_dump_actions.py`
and `test_sep07_calendar_offer_continuation.py` stay green.

---

## W2 — current events: the article and the searches

### Evidence, and the corrected premise

The owner clarified what he meant: articles **pasted into chat in the past**,
and an expectation that **user uploads hold some**. Both checked against the
live store (read-only sqlite over `data/chroma_db_v4`):

**Uploads contain no articles at all — ever, not just on 09-11.**
`reference_docs` holds 271 `user_upload` chunks across **97 distinct titles**:
syllabi, homework PDFs/DOCXs, MGT 6203 lecture transcripts, CSVs, photos,
Reddit screenshots, one old `DAEMONv0.7.py`. The ~30 `tmp*.txt/pdf` titles
(real names destroyed by the pre-2026-09-04 temp-name bug) were each opened —
all lecture transcripts and homework PDFs. So the expectation is reasonable
and the answer is zero.

**Pasted articles exist and are retrievable only by their opening topic.**
Two survive in the live corpus (2026-08-30: a parenting op-ed 5,355 chars; a
Truth-Social/autos piece 4,882 chars). Every conversation turn IS embedded —
7,175 documents, 7,175 vectors — but `conversations` is **never chunked**
(`add_conversation_memory` writes one document per turn,
`memory/storage/multi_collection_chroma_store.py:520`), while
`obsidian_notes`/`reference_docs` chunk at ~2,000 chars. With
`bge-small-en-v1.5` truncating silently at 512 tokens, a long turn's stored
vector IS its opening: measured over 60 documents >4,000 chars,
`cos(stored vector, own first 2,000 chars)` = **0.995 mean / 0.983 min**.
Median document is 751 chars, p99 20,284, max 473,826; 759 (10.6%) exceed the
window, so 5.68M of 12.2M characters cannot influence any vector. That does
NOT make those turns unsearchable — later passages still score 0.795 mean
against the head-derived vector because long turns are topically coherent —
but content topically distinct from the opening is unreachable (5 of the 60
have a mid-document passage below a 0.65 bar), no paragraph can be retrieved
on its own, and the text is attributed to the user.

`obsidian_notes` and `reference_docs` DO chunk at ~2,000 chars
(`knowledge/reference_docs_manager.py:143`, `knowledge/obsidian_manager.py:444`).
The reasonable assumption that "everything is embedded, so everything is
findable" holds for those two and fails for `conversations`.

**The two news links he shared on 09-11 both worked.** 13:22:27 (nj.com) and
19:14:30 (thedailybeast) were fetched and cited — `[WEB_7]`, `[WEB_58]` — and
the 13:22 reply correctly conceded the point against its own earlier answer.

So the felt failure has three parts, not two:

1. **14 turns got zero web evidence from a spent budget.** Tavily's daily cap
   was hit at 19:11 (`credits_today: 104` vs `daily_credit_limit: 100`), and
   from 19:13:28 onward every triggered turn recorded
   `web_error: "Daily credit limit reached. Remaining: 0.0"` (19:13, 19:52,
   19:55, 20:14, 20:22, 20:23, 20:26, 20:34, 20:37, 20:39, 20:40, 20:42,
   20:43, 22:10). The R/homework debugging had already spent ~44/100 before
   the news conversation began. Today's batch stops the futile loop; it does
   NOT yet make the reply honest — only one of those replies (20:42:44) told
   the user "the searches came back empty, so I'm working from general
   knowledge". The rest answered from priors silently.
2. **The pasted articles are matchable only by their headline topic** — ask
   about a detail in the body and nothing retrieves them. That is the defect
   matching "articles I gave you in the past were not pulled". Addressed by
   W2-B0 below and, structurally, by `docs/SOURCE_DOCUMENT_TIER_DESIGN.md`.
3. **If an article HAD been uploaded it would not have surfaced either.** The
   freshness leg and the roster both gate on `_DOCUMENT_CONTEXT_RE`
   (`core/prompt/gatherer_knowledge.py:102-107`) — attachments / pdf / docx /
   csv / dataset / homework / assignment / syllabus / lecture / transcript /
   "question N" / "part N". There is no *article / news / story / headline /
   link / piece* category, so a shared article is reachable only by naming its
   filename or by clearing `USER_UPLOADS_MIN_RELEVANCE`=0.62 (`:96`) — a high
   bar for a reactive follow-up ("no that's not right"). The roster never
   fired once during the political conversation.

**Search terms.** Every bad term set came from the trigger LLM
(`_classify_with_llm_unified` → `_build_llm_trigger_prompt`), which the
agentic loop then reuses verbatim as `initial_search_terms`
(`core/agentic/controller.py:539-559`; log line `[WebSearch] Using 3
pre-computed sub-queries`). Four patterns, with live examples:

| pattern | live terms | outcome |
|---|---|---|
| rhetorical framing searched literally | `35% oppose mass death survey 2026`; `1 trillion bribe electorate election fraud news` | 9 results, **0** `[WEB_` cites |
| philosophy/hypothetical searched at all | `medical ethics historical practices` (an ethics thought experiment) | 12 results, **0** cites |
| future speculation searched as current news | `Alberta conflict news September 2026` for "there will be a war in at least 2 years" | 9 results, 2 cites |
| owner identity leaked into an unrelated query | `current voting issues Illinois`, **`Georgia Tech voting information`** for "I am referring to voting" (a weighted-voting thought experiment) | credit-exhausted |

The location backstop DID run and logged
`Stripped unjustified location 'Springfield, Illinois' … -> 'voting regulations'`
— but `_location_patterns()` (`utils/location_resolver.py:290-312`) builds
city-anchored regexes only, so the bare state name survived. And
`Georgia Tech` was in the LLM's output *before* any strip:
`_build_llm_trigger_prompt` injects `User's school: …`
(`utils/web_search_trigger.py:1347`) on **every** trigger call, with a scope
instruction the model ignored here. `apply_institution` runs after the strip
and only ADDS names, so nothing removes an institution the LLM invented.

Year padding (`… 2026` on nearly every term) is by explicit prompt
instruction (`:1437`) — fine for real news, compounding for the three patterns
above.

### Steps

0. **B0 — passage-level retrieval inside long turns.** Measure before changing
   anything: a read-only probe over `conversations` that buckets documents
   above the embedder window by *kind* (third-party source text vs the user's
   own code/console pastes vs long ordinary prose). The right fix differs per
   bucket, and chunking a 473 KB console dump into the semantic index would be
   its own retrieval-crowding incident. Candidates, to be chosen with the
   probe's numbers in hand:
   - (i) route third-party source text to the proposed source-document tier,
     chunked, with provenance — the design direction
     (`docs/SOURCE_DOCUMENT_TIER_DESIGN.md` §6, staged);
   - (ii) as a cheap interim, chunk conversation documents above the window at
     write time — invasive (touches ids, dedup, scoring, the memory-id map),
     so not to be done casually;
   - (iii) leave code/console pastes unchunked and rely on the corpus keyword
     anchor (2026-08-26), which is what finds them today.
   Acceptance for the probe alone: a table of counts by bucket, and a stated
   recommendation. No behaviour change in this step.
1. **B1 — one more vocabulary CATEGORY, not one more phrase.** Extend
   `_DOCUMENT_CONTEXT_RE` with a shared-reading category: article, piece,
   story, headline, news item, writeup, link, screenshot, paste. Acceptance:
   "what does the article say about the oath" surfaces a fresh upload and the
   roster line; "how are the cats" still surfaces neither.
2. **B2 — symmetric strips for identity leakage.** (a) Derive a region/state
   pattern from the SAME resolved location the city pattern comes from (the
   resolver already holds "Springfield, Illinois"; split it) so a bare state
   name is stripped by the existing deterministic backstop. (b) Add
   `strip_unjustified_institution(terms, query)` to
   `utils/institution_resolver.py`, mirroring `strip_unjustified_location`:
   when the query carries no institution cue, remove a resolved institution
   name the LLM added. Acceptance: the live 20:22 term set loses both
   "Illinois" and "Georgia Tech"; "when is the drop deadline" keeps them.
3. **B3 — make the school line conditional** in `_build_llm_trigger_prompt`
   (inject only when the query carries an institution/logistics cue).
   Belt-and-suspenders behind B2, since prompt-only guards do not hold.
4. **B4 — a hypothetical/speculative shape predicate.**
   `query_checker.is_hypothetical_or_speculative()`: future modality ("there
   will be", "would be", "say a doctor…", "if we"), conditional framing, no
   attributed current-event claim. Consulted by
   `requires_fresh_public_evidence` and the gate's news arm, in the same style
   as `is_self_report` / `is_request_shaped`. Acceptance: 19:01 (doctor
   ethics) and 19:12 ("war in at least 2 years") do not route to search;
   "did X actually say Y this week" and the 18:40 trade-claim turn still do.
5. **B5 — honesty when the budget is spent.** The budget veto now prevents the
   loop; add the deterministic reply-side note (same shape as
   `[ATTACHMENT NOTE]`/`[DEADLINE NOTE]` in `utils/attachment_audit.py`) so a
   turn that WANTED fresh evidence and could not get it says so. Acceptance:
   a budget-exhausted news turn's reply states it could not check, instead of
   answering from priors silently.
6. **B6 — owner decision, not code.** The homework session spent ~44% of the
   day's Tavily budget before the news conversation started. Either raise
   `web_search.daily_credit_limit` / the Tavily plan, or add a per-session
   reserve. Worth a number from the owner before anyone builds a reserve.

---

## W3 — doc and guard hygiene (mostly done today)

Done in this session: `DM-30` (keyword-boundary corpus diff, with
`scripts/probe_keyword_boundary.py` behind it), `DM-31`
(`dm31_live_state_default`, **gated**) and `BC-78`; incidents appended to
BC-01, BC-04, BC-11, BC-12, BC-32, BC-46, BC-72 with their Closure and Status
lines brought up to date.

An independent doc audit of this batch found **ten** inconsistencies in my own
write-up — four wrong counts (`'down'` 140→154, `'war'` 85→105, `'dead'`
176→177, "31 corpus rows"→17), a three-second timestamp error repeated in
three files, a class tag that differed between `CLAUDE.md` and the changelog,
two catalog Closure lines not updated for the fix in the same batch, a
"four/six candidates" count that contradicted the baseline, and two test/suite
counts stale within the same day. All corrected, all re-verified by running
the deployed functions.

**Process proposal (cheap, and it closes exactly what went wrong):** a
published count must name the command that reproduces it. DM-30 does; the
heavy-row figures did not, which is why they drifted. Candidate: a small
read-only `scripts/probe_heavy_rows.py` that prints the flagged/neutralized/
still-evidence counts, cited wherever those numbers appear.

Remaining: `docs/METRICS_SNAPSHOT.md` and the README counts are from 09-07 and
are stale by ~10 batches plus today's three new files — the owner runs
`python scripts/generate_doc_metrics.py --update-readme` (it writes).

---

## Sequencing and ownership

| # | Work | Who | Depends on |
|---|---|---|---|
| 1 | W1 A1-A3 (anchor + fixtures) | cheap sub, frontier referees | — |
| 2 | W2 B2 + B3 (identity leakage) | cheap sub | — |
| 3 | W2 B1 (document vocabulary) | cheap sub | B0's answer (it only matters once articles are in a store) |
| 3b | W2 B0 probe (length buckets by kind) | cheap sub | — |
| 4 | W2 B4 (speculative shape) | frontier (routing-sensitive) | — |
| 5 | W2 B5 (search-unavailable note) | cheap sub | today's budget veto |
| 6 | W1 A4 (replaced-offer affirmation) | frontier, needs an evidence pass first | — |
| 7 | W3 metrics regeneration, B6 budget decision | owner | — |

```
CONTINGENCY  W1 A1: if requiring the anchor turns any of the 5 legitimate
             catches False, do NOT widen the anchor vocabulary — the failing
             case means the notice was firing on a sentence with no action
             content, and the right answer is a different sentence in the same
             reply (or the upstream A4 gap). Stop and report which.
             W2 B4: if the predicate suppresses the 18:40 trade-claim turn
             (attributed, checkable, correctly searched), it is too broad —
             require the ABSENCE of an attributed claim, not the presence of
             modality alone.
             W2 B1: if extending the vocabulary admits stale homework uploads
             into ordinary chat, the freshness leg is doing the work, not the
             cue — re-check `_upload_is_fresh` before relaxing anything else.
DATA NOTE    The ~30 `tmp*`-named uploads are permanently missing their real
             filenames (the bug is fixed since 2026-09-04, the old titles are
             not recoverable). They are duplicate MGT 6203 lecture transcripts
             and homework PDFs, so the loss is cosmetic — but a title search
             for them will never work, and the roster line renders them as
             `tmp…` (documented behaviour, HANDOFF_20260907 A4).
OWNER        commit today's batch; restart (nothing above is live);
             `git push` (three commits ahead); the B6 credit decision;
             the metrics regeneration.
```
