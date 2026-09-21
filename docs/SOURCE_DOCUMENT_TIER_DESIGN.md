# Design note — a source-document tier (proposed 15th collection)

*2026-09-12. Proposed, not scheduled. Written after an audit found that the
articles the owner remembers giving Daemon are either absent or unretrievable,
and that the reason is structural rather than a tuning miss. The owner's own
framing question — "would it make sense to embed these in a 15th DB the agent
could use to model the world external to the user?" — plus his follow-up, that
such a store would need **an ingestion process that samples articles with no
respect to user input**, are the subject of §4.*

---

## 1. What the store actually contains today (measured, read-only)

| finding | number | how |
|---|---|---|
| `user_upload` chunks in `reference_docs` | 271 across **97 distinct titles** | read-only sqlite over `data/chroma_db_v4` |
| …of which news/opinion articles | **0** | every title inspected; the ~30 `tmp*.txt/pdf` titles (real names lost to the pre-2026-09-04 temp-name bug) are all MGT 6203 lecture transcripts and homework PDFs |
| `conversations` documents | 7,173 | " |
| …embedded | **all of them, one vector per turn** (7,175 docs → 7,175 embeddings) | " — this is what [RELEVANT MEMORIES] searches |
| …chunked | **0** | `add_conversation_memory` stores one document per turn: `documents=[f"User: {query}\nAssistant: {response}"]` (`memory/storage/multi_collection_chroma_store.py:520`) |
| stored vector vs the document's own first 2,000 chars | **cos 0.995 mean, 0.983 min** (60 docs >4,000 chars) | deployed embedder on CPU |
| a passage from chars 3k-5k vs the stored vector | cos 0.795 mean, **5 of 60 below a 0.65 bar** | " |
| document length | median 751 chars, p90 2,069, p99 20,284, **max 473,826** | " |
| documents longer than ~2,000 chars | 759 (10.6%) | " |
| text beyond the embedder's window | **5.68M of 12.2M chars (46%)** — cannot influence any vector, which is NOT the same as unsearchable (see below) | " |

The embedder is `BAAI/bge-small-en-v1.5`, `max_seq_length: 512` — roughly
2,000 characters of English. Sentence-transformers truncates silently.

Read this precisely, because the coarse version misleads. **Conversations are
embedded and semantically searchable — always, one vector per turn.** What
they are not is CHUNKED, unlike `obsidian_notes` and `reference_docs`
(`knowledge/reference_docs_manager.py:143`, `knowledge/obsidian_manager.py:444`,
both via `utils/text_chunking.chunk_by_headers`). Combined with silent
truncation at 512 tokens, a long turn's vector is effectively the vector of
its opening. A pasted article is therefore:

1. stored in full — nothing is lost on disk, and once the document matches, its
   FULL text reaches the prompt (subject to the budget's middle-out trim);
2. matched on its opening ~2,000 characters only. Because articles are
   topically coherent, a query about the headline topic still finds it (later
   passages score 0.795 mean against the head-derived vector);
3. **unreachable by anything topically distinct from that opening** — a second
   subject raised late, a figure or name in paragraph 8, a multi-topic paste.
   Five of sixty long documents have a mid-document passage below a 0.65
   retrieval bar;
4. never chunked, so no paragraph can be retrieved on its own — the thing a
   document store exists to do;
5. attributed to the user as if they had written it.

The honest one-line summary: **long pastes are findable by their opening
topic, not by their contents.**

Two pasted articles survive in the live corpus (2026-08-30: a parenting op-ed,
5,355 chars; a Truth-Social/autos piece, 4,882 chars). Both are in that state.

**This is the strongest argument for the tier** — stronger than world
modelling. Chunking and provenance are what a document collection does and
what a conversation row structurally cannot.

## 2. Proposal

A 15th collection — working name `source_documents` — holding third-party
source text the user brought into the system, chunked and dated.

**Admitted by intent, not by traffic:**
- pasted article/report text (detected at ingress, see §6),
- uploaded articles and reports (not coursework — `doc_type` already
  distinguishes uploads since 2026-09-07),
- a web-fetched page **only** when it was cited in an answer or explicitly
  saved. Not every fetch: the store is already 852 MB, and auto-embedding
  every fetched page inflates it, puts third-party content into every backup,
  and admits low-intent volume.

**Per-chunk schema (all mandatory, all first-class — not free-text):**
`source_url`, `publication_date`, `ingested_at`, `author`/`outlet`,
`ingest_channel` ∈ {pasted, uploaded, cited_fetch, sampled},
`stance="reported"`, `chunk_index`/`total_chunks`, `sample_frame_id`
(§4, null for user-brought documents).

**Retrieval contract:**
- its **own prompt section** and its **own `PRIORITY_ORDER` budget row —
  never pooled** with `reference_docs` (self-docs starved uploads there once;
  see the 2026-09-07 pool fix) and never pooled with personal memory;
- an admission gate in the shape of `_should_include_reference_docs`: the turn
  has to be about the topic;
- excluded from the memory floor and top-up paths (BC-24: floors have
  re-admitted ungated content three times);
- rendered with its publication date in the line itself — "as of
  2026-03-04" — so a six-month-old "X is expected next month" cannot read as
  current. The concept already exists for facts (`claim_kind` / `event_date` /
  `observed_at`, 2026-09-06).

**Never a fact source.** `memory/fact_source.py` refuses quoted and
third-party spans through a list of exclusions that has grown per incident
(lyrics → `lived_in=Atlanta`, a quoted email superseding `enrolled_in`). A
collection the extractors simply do not read makes that structural.

## 3. Non-goals

- **Not a world model.** A dated, attributed document store is evidence the
  agent can cite. A world model is a set of beliefs it reasons from. The
  second is where framing capture becomes dangerous (§4), and nothing in this
  design updates a belief.
- **Not a general news cache.** `web_search_cache` (1,979 embeddings in
  `data/chroma_multi`) already exists for repeat queries; this is not that.
- **Not a replacement for `wiki_knowledge`**, which is background reference,
  not dated current events.

## 4. Framing capture, and the independent sampling channel

The owner's concern, which is the right one: a store built only from what he
saved is self-selected and over-samples what upset him — the same denominator
problem `memory/pattern_engine.py` documents for the corpus ("the corpus
over-samples hard days"). Used as evidence it is fine; used as a base rate it
is a distortion machine. His proposal: **also ingest articles chosen without
reference to his input**, so there is a denominator.

Three ways to get one, cheapest first:

**(a) Counter-evidence on demand — no store, no background cost.** Insight
mode already forces a counter-evidence facet into every decomposition
(`core/insight/facets.py`, mandatory since 2026-08-23). Point it at the web
for the same claim whenever the user's saved set is used as evidence. This
buys the *asymmetry check* without any new infrastructure, and it should ship
before either option below.

**(b) Topic-stratified sampling — a real denominator, moderate cost.** Derive
the topic from the user's saved set ("US trade policy"), then sample the same
topic from an explicit, inspectable source list on a rotation, recording the
frame. The axis is "same topic, different outlets", which is exactly the
question framing capture raises. Enables honest statements like: *of the 40
pieces sampled on this topic this month, 8 carry the framing you have been
reading.*

**(c) Fixed-basket background pull — a true user-independent baseline, least
useful per byte.** A small fixed set of wire services plus a spread across the
spectrum, pulled on a schedule regardless of what the user discusses. Most
volume, least relevance; only worth it if (b) proves too topic-narrow.

**Honesty requirement, whichever is chosen:** a "neutral" source list is
itself a curated choice by whoever writes it. The sampler makes the frame
**explicit and inspectable**, it does not make it neutral. Every sampled
chunk carries `sample_frame_id` → a stored record of {source list, window,
query/topic, how many fetched, how many admitted, how many failed}, so a
count derived from it can be defended or discarded later. Without that record
the denominator is decoration.

**Constraints the sampler must respect (all real today):**
- **Budget.** Tavily is capped at 100 credits/day (`config/config.yaml:139`)
  and the homework session spent 44 of them before a news conversation even
  started on 2026-09-11. A sampler must NOT compete with interactive turns:
  either its own reserve, or a zero-credit path — plain `httpx` + the existing
  `utils/page_extract.extract_page_text` over RSS/Atom, which needs no new
  dependency — or off by default.
- **Retention.** ~20 articles/day at 5-50 KB chunked is ~0.5 MB/day, ~180
  MB/year on top of 852 MB. Needs a TTL by publication age, and deletion goes
  through the curation ladder (quarantine first, human empties it).
- **The sampler is the one component that could turn the store into a prior.**
  Sampled documents must surface ONLY inside an insight/assessment request
  about that topic — never injected into an ordinary turn, never in the base
  prompt. If they ever reach ordinary turns, the store has become a belief
  feed and this design has failed.

## 5. Risk register

| risk | mechanism | mitigation |
|---|---|---|
| retrieval crowding | an article is 5-50× a turn; chunked, it dominates its topic neighbourhood — the mechanism that flooded a distress turn with 15 self-doc chunks (2026-07-25) | own section + cap, admission gate, excluded from floors/top-ups |
| stale world-state read as current | a dated claim retrieved without its date | publication date in the rendered line; freshness bar for current-events questions |
| framing capture | self-selected corpus used as a base rate | §4: counter-evidence facet, then a recorded sample frame; denominator caveat extended to this store |
| belief laundering (BC-75, open) | the agent's own prose persisted, then read back as evidence | only third-party source text; `ingest_channel` + `author` required; model-authored text never admitted |
| provenance collapse | article text attributed to the user (today's behaviour) | `stance="reported"`, extractors do not read the collection |
| cost/privacy | third-party content in every backup; store growth | admit by intent; TTL; sampler off by default |

## 6. Staging

1. **Detect and tag pasted source text at ingress.** `core/content_type_detector.py`
   already classifies lyrics/poems/code/quotes and carries a partial
   `this article` hint (line 83). An `article` content type is the same shape
   of work, testable offline against the two live pastes. Value on its own:
   fact extraction stops treating article prose as the user's words.
2. **Chunk-and-route.** Write tagged source text to the new collection,
   chunked with the existing `chunk_by_headers`/`chunk_by_size`; keep the
   conversation row as the turn it was, with a pointer. Decide deliberately
   whether to ALSO chunk long non-article pastes (code/console dumps are 46%
   of the invisible text and a different question — see the plan's W2-B0).
3. **Section, gate, budget row** + the dated render.
4. **Counter-evidence facet pointed at the web** (§4a) — the cheap half of the
   framing-capture answer.
5. **Only then** a sampler, and only with a recorded frame (§4b).
6. Fetched pages last, admitted on citation.

## 7. Open questions for the owner

- Does a *summary* of a long pasted article belong in `conversations` where
  the article text used to sit, so the conversation history stays readable?
- Retention: keep sampled articles forever, or TTL them by publication age
  while keeping the frame record (which is what the denominator needs)?
- Is the sampler worth any Tavily budget, or RSS-only from the start?
- Coursework uploads stay in `reference_docs` — agreed? (They are the user's
  own materials, not third-party reporting, and the homework flow depends on
  the current behaviour.)
