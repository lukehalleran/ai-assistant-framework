"""
core/insight/sweep.py

Module Contract
- Purpose: Run the ungated evidence sweep for an insight FacetPlan across ALL
  stores: six Chroma collections (semantic), the corpus (exact keyword — the
  raw-text channel that defeats the fact extractor's triple-shape bias), the
  knowledge graph (1-hop from resolved entities), plus temporal expansion
  around the strongest conversation hits.
- Inputs: FacetPlan + live components (chroma_store, corpus_manager,
  graph_memory, entity_resolver, memory_expander) + caps dict (defaults from
  INSIGHT_* config).
- Outputs: deduped, date-sorted (newest first), snippet-clipped list of
  EvidenceItem, proportionally trimmed to the total cap.
- Key behaviors:
  * NO cosine gate anywhere — generous caps replace gating (the memory gate's
    per-doc threshold is structurally unable to pass a collective-signal
    evidence set; that is the whole reason this mode exists).
  * Graph traversal skips the 'user' star hub and any node with degree ≥
    GRAPH_EXPANSION_HUB_DEGREE, and honors read-time TTL
    (_edge_is_stale_transient) — same discipline as query expansion.
  * Expansion only around top conversation hits (EXPANDABLE_COLLECTIONS;
    threads are NEVER expanded).
  * Wall-clock bounded: on sweep_timeout_s the accumulated partial evidence
    is returned, never an exception.
  * Chroma/corpus calls run in threads (asyncio.to_thread) and facets run
    concurrently.
- Side effects: read-only against every store.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
from datetime import datetime, timedelta
from typing import Optional

from core.insight.types import EvidenceItem, FacetPlan, FacetQuery
from memory.utils import is_junk_conversation_doc, is_junk_summary, is_quarantined
from utils.logging_utils import get_logger
from utils.ordered_slice import week_bucket_key as _ordered_week_bucket_key
from utils.ordered_slice import window_fair_sample as _ordered_window_fair_sample

logger = get_logger("insight_sweep")

SWEEP_COLLECTIONS = (
    "conversations", "summaries", "reflections",
    "facts", "obsidian_notes", "threads",
)


def default_caps() -> dict:
    from config import app_config as cfg
    return {
        "per_facet_cap": cfg.INSIGHT_PER_FACET_CAP,
        "total_evidence_cap": cfg.INSIGHT_TOTAL_EVIDENCE_CAP,
        "evidence_snippet_chars": cfg.INSIGHT_EVIDENCE_SNIPPET_CHARS,
        "external_snippet_chars": cfg.INSIGHT_EXTERNAL_SNIPPET_CHARS,
        "keyword_scan_max": cfg.INSIGHT_KEYWORD_SCAN_MAX,
        "expand_top_k": cfg.INSIGHT_EXPAND_TOP_K,
        "expand_window": cfg.INSIGHT_EXPAND_WINDOW,
        "sweep_timeout_s": cfg.INSIGHT_SWEEP_TIMEOUT_S,
    }


def _meta_date(metadata: dict) -> Optional[str]:
    # Longitudinal notes carry the event date separately from indexing time.
    # Prefer it defensively, including for legacy Chroma rows whose timestamp
    # was written when the note was embedded rather than when it occurred.
    for key in ("note_date", "date", "timestamp", "created_at", "last_seen"):
        v = (metadata or {}).get(key)
        if v:
            return str(v)
    return None


def _clip(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "…"


def _dedupe_key(item: EvidenceItem) -> str:
    if item.doc_id:
        return f"id:{item.doc_id}"
    return "tx:" + hashlib.sha1(item.text[:160].lower().encode()).hexdigest()


# Prior-sweep-output feedback-loop guard (round 3, 2026-09-04). A live run
# cited [E1] — the ASSISTANT'S REPLY from a PREVIOUS insight-mode sweep — as
# fresh evidence and minted it into a new "correction #5"; a misattribution
# from two runs ago was propagating run to run. memory_storage.store_interaction
# records the answering mode as `response_mode` on BOTH the corpus entry
# (corpus_manager.add_entry(response_mode=...)) and the Chroma "conversations"
# doc metadata (memory_storage.py ~L862/L941) — an insight-assembly / doc-
# generation / self-note turn's own request+reply are the mode's OWN
# scaffolding, not lived history, and BOTH halves are dropped (the user half
# is a request, not something that happened to the user).
_SWEEP_OUTPUT_MODES = frozenset({"insight-assembly", "doc-generation", "self-note"})

# Backstop for legacy entries stored before response_mode was recorded: the
# synthesizer's own rendered output is textually distinctive — evidence
# citation markers ("[E12]") and doctrine phrases that don't occur in
# ordinary conversation.
_SWEEP_SCAFFOLDING_RE = re.compile(
    r"denominator caveat|coverage note|\[E\d+\]", re.IGNORECASE,
)


def _is_sweep_output_mode(mode) -> bool:
    return isinstance(mode, str) and mode in _SWEEP_OUTPUT_MODES


def _looks_like_sweep_scaffolding(text: str) -> bool:
    return bool(_SWEEP_SCAFFOLDING_RE.search(text or ""))


async def run_sweep(
    plan: FacetPlan,
    *,
    chroma_store,
    corpus_manager,
    graph_memory=None,
    entity_resolver=None,
    memory_expander=None,
    caps: Optional[dict] = None,
    request_text: str = "",
    date_window: Optional[tuple[str, str]] = None,
) -> list[EvidenceItem]:
    """Run the full sweep. Returns partial evidence on timeout, never raises.

    ``request_text`` (2026-09-04): the raw user request, scanned for
    double-quoted cue phrases ("no I mean", "that's wrong") that become an
    extra literal corpus scan — the facet decomposer only keeps single-word
    keywords, so the user's own quoted correction language never became a
    search target otherwise. ``date_window``: an explicit ``(start, end)``
    ISO-date filter applied to items with a parseable date at finalize time
    (undated items are always kept).
    """
    caps = {**default_caps(), **(caps or {})}
    collected: list[EvidenceItem] = []  # shared accumulator — survives timeout
    _prior_sweep_dropped = [0]  # mutable cell: nested async funcs share it

    async def _sweep_facet(facet: FacetQuery) -> None:
        # --- Chroma semantic, all six collections, NO gate ---
        async def _query(coll: str):
            try:
                return await asyncio.to_thread(
                    chroma_store.query_collection, coll, facet.query_text,
                    caps["per_facet_cap"],
                )
            except Exception as e:
                logger.debug(f"[Insight] query_collection({coll}) failed: {e}")
                return []

        chroma_results = await asyncio.gather(*[_query(c) for c in SWEEP_COLLECTIONS])
        conversation_hits: list[dict] = []
        for coll, rows in zip(SWEEP_COLLECTIONS, chroma_results):
            for row in rows or []:
                # --- Hygiene filters (no cosine gate, but junk/quarantine guard) ---
                metadata = row.get("metadata") or {}

                # Skip quarantined docs (curation engine)
                if is_quarantined(metadata):
                    continue

                content = (row.get("content") or "").strip()
                if not content:
                    continue

                # Collection-specific junk filters
                if coll == "conversations":
                    if is_junk_conversation_doc(content=content):
                        continue
                    if _user_side_is_greeting_or_ack(content):
                        continue
                    # Prior-sweep-output feedback-loop guard (round 3).
                    if _is_sweep_output_mode(metadata.get("response_mode")):
                        _prior_sweep_dropped[0] += 1
                        continue
                    if _looks_like_sweep_scaffolding(content):
                        _prior_sweep_dropped[0] += 1
                        continue
                elif coll in ("summaries", "reflections"):
                    if is_junk_summary(content):
                        continue
                    if _looks_like_sweep_scaffolding(content):
                        _prior_sweep_dropped[0] += 1
                        continue
                elif coll == "facts":
                    # Skip superseded facts (is_current=False or superseded_by set)
                    if metadata.get("is_current") is False or metadata.get("superseded_by"):
                        continue
                    # Read-time junk guard (2026-09-06 retest: "psychologist |
                    # is | tuesday" rendered as evidence) — the deployed
                    # extractor predicate, never a second list.
                    if fact_triple_is_junk(content):
                        continue

                collected.append(EvidenceItem(
                    doc_id=row.get("id"),
                    text=content,
                    date=_meta_date(row.get("metadata")),
                    collection=coll,
                    speaker="",
                    facet=facet.name,
                    metadata=metadata,
                ))
                if coll == "conversations":
                    conversation_hits.append(row)

        # --- Corpus exact-keyword (raw-text channel, speaker-attributed) ---
        if facet.keywords and corpus_manager is not None:
            try:
                hits = await asyncio.to_thread(
                    lambda: corpus_manager.search_keyword(
                        facet.keywords, max_results=caps["keyword_scan_max"],
                        context_chars=caps["evidence_snippet_chars"],
                        include_entry=True,
                        authored_only=True,
                    )
                )
            except Exception as e:
                logger.debug(f"[Insight] corpus search_keyword failed: {e}")
                hits = []
            for h in hits:
                # Prior-sweep-output feedback-loop guard (round 3): both
                # halves of an insight-assembly/doc-generation/self-note
                # entry are dropped — the user half is a request, not
                # lived history.
                if _is_sweep_output_mode((h.get("entry") or {}).get("response_mode")):
                    _prior_sweep_dropped[0] += 1
                    continue
                excerpt = h.get("excerpt", "")
                if h.get("speaker") == "assistant" and _looks_like_sweep_scaffolding(excerpt):
                    _prior_sweep_dropped[0] += 1
                    continue
                ts = h.get("timestamp")
                collected.append(EvidenceItem(
                    doc_id=None,
                    text=excerpt,
                    date=ts.isoformat() if isinstance(ts, datetime) else (str(ts) if ts else None),
                    collection="corpus",
                    speaker=h.get("speaker", ""),
                    facet=facet.name,
                ))

        # --- Graph 1-hop from resolved entities (hub-aware, TTL-aware) ---
        if facet.entities and graph_memory is not None:
            _sweep_graph(facet, collected)

        # --- Temporal expansion around top conversation hits ---
        if memory_expander is not None and caps["expand_top_k"] > 0:
            for row in conversation_hits[:caps["expand_top_k"]]:
                try:
                    exp = await asyncio.to_thread(
                        memory_expander.expand, row.get("id"),
                        caps["expand_window"], "conversations",
                    )
                except Exception as e:
                    logger.debug(f"[Insight] expansion failed: {e}")
                    continue
                for turn in (exp or {}).get("turns", []):
                    content = (turn.get("content") or "").strip()
                    if not content:
                        continue
                    # Hygiene: skip quarantined/junk expanded turns
                    turn_meta = turn.get("metadata") or {}
                    if is_quarantined(turn_meta):
                        continue
                    if is_junk_conversation_doc(content=content):
                        continue
                    if (_is_sweep_output_mode(turn_meta.get("response_mode"))
                            or _looks_like_sweep_scaffolding(content)):
                        _prior_sweep_dropped[0] += 1
                        continue
                    collected.append(EvidenceItem(
                        doc_id=turn.get("id"),
                        text=content,
                        date=_meta_date(turn.get("metadata")) or _meta_date(turn),
                        collection="conversations",
                        speaker="",
                        facet=facet.name,
                        metadata=turn_meta,
                    ))

    def _sweep_graph(facet: FacetQuery, out: list[EvidenceItem]) -> None:
        from config.app_config import GRAPH_EXPANSION_HUB_DEGREE
        from memory.stance_classifier import effective_stance

        now = datetime.now()
        for mention in facet.entities:
            try:
                entity_id = (
                    entity_resolver.resolve(mention) if entity_resolver else mention.lower()
                )
                if not entity_id or entity_id == "user":
                    continue  # never fan out from the user star hub
                node = graph_memory.get_entity(entity_id)
                if node is None:
                    continue
                try:
                    degree = graph_memory.graph.degree(entity_id)
                except Exception:
                    degree = 0
                if degree >= GRAPH_EXPANSION_HUB_DEGREE:
                    logger.debug(f"[Insight] Skipping hub entity {entity_id} (deg={degree})")
                    continue
                for edge in graph_memory.get_relations(entity_id, direction="both"):
                    if graph_memory._edge_is_stale_transient(edge, now):
                        continue
                    _supp = getattr(graph_memory, "edge_is_suppressed", None)
                    if callable(_supp) and _supp(edge) is True:  # bool only: test doubles return mocks
                        continue
                    src = graph_memory.get_entity(edge.source_id)
                    tgt = graph_memory.get_entity(edge.target_id)
                    sentence = edge.to_natural_language(
                        src.display_name if src else edge.source_id,
                        tgt.display_name if tgt else edge.target_id,
                    )
                    ts = edge.last_seen or edge.first_seen
                    out.append(EvidenceItem(
                        doc_id=f"edge:{edge.edge_key()}",
                        text=sentence,
                        date=ts.isoformat() if isinstance(ts, datetime) else None,
                        collection="graph",
                        speaker="",
                        is_appraisal=effective_stance(edge.metadata) == "appraisal",
                        facet=facet.name,
                    ))
            except Exception as e:
                logger.debug(f"[Insight] graph sweep for {mention!r} failed: {e}")

    async def _quoted_phrase_scan() -> None:
        # Quoted cue phrases in the REQUEST itself ("no I mean", "that's
        # wrong") are the user's own correction language — a literal corpus
        # scan for them catches turns the semantic facets miss (the facet
        # decomposer only keeps single-word keywords). User-side only: the
        # cue is the user's own phrase, not an echo of it in a reply.
        #
        # 2026-09-04 round 2: search_keyword itself is newest-first and its
        # OWN output was already window-fair in isolation — the loss was
        # DOWNSTREAM, in _finalize's per-collection quota (see that
        # function's docstring). Fixed here defensively too: scan generously
        # past the eventual per-scan cap, drop request/same-day-repeat noise
        # BEFORE capping, then week-fair-select down to keyword_scan_max (>=2
        # per ISO week when available) so this scan's OWN contribution to
        # `collected` can never be 100% recent even before _finalize runs.
        from core.insight.facets import extract_quoted_phrases

        if corpus_manager is None:
            return
        phrases = extract_quoted_phrases(request_text)
        if not phrases:
            return
        kw_kwargs: dict = {}
        if date_window:
            try:
                start_d, end_d = date_window
                kw_kwargs["start"] = datetime.strptime(start_d, "%Y-%m-%d")
                kw_kwargs["end"] = (
                    datetime.strptime(end_d, "%Y-%m-%d")
                    + timedelta(days=1) - timedelta(seconds=1)
                )
            except ValueError:
                pass
        scan_cap = max(caps["keyword_scan_max"] * 4, 200)
        try:
            hits = await asyncio.to_thread(
                lambda: corpus_manager.search_keyword(
                    phrases, max_results=scan_cap,
                    context_chars=caps["evidence_snippet_chars"],
                    include_entry=True,
                    authored_only=True,
                    **kw_kwargs,
                )
            )
        except Exception as e:
            logger.debug(f"[Insight] quoted-phrase corpus scan failed: {e}")
            return
        raw_items: list[EvidenceItem] = []
        for h in hits:
            if h.get("speaker") != "user":
                continue
            # Prior-sweep-output feedback-loop guard (round 3): a prior
            # sweep's OWN request can echo a cue phrase as an example.
            if _is_sweep_output_mode((h.get("entry") or {}).get("response_mode")):
                _prior_sweep_dropped[0] += 1
                continue
            ts = h.get("timestamp")
            raw_items.append(EvidenceItem(
                doc_id=None,
                text=h.get("excerpt", ""),
                date=ts.isoformat() if isinstance(ts, datetime) else (str(ts) if ts else None),
                collection="corpus",
                speaker=h.get("speaker", ""),
                facet="quoted-phrase",
            ))
        if request_text:
            raw_items = exclude_current_request_evidence(
                raw_items, request_text,
                current_turn_date=datetime.now().isoformat(),
            )
        collected.extend(
            interleave_evidence_for_coverage(raw_items)[: caps["keyword_scan_max"]]
        )

    async def _date_range_scan() -> None:
        # Explicit date window (2026-09-04): semantic chroma queries are
        # date-BLIND, so a week with no strong semantic match to any facet
        # contributes nothing even when it has dated content — the same
        # class of gap the deliberation path's windowed retrieval arm fixed
        # for notes/facts (gui.handlers._window_scan_collection). Mirrored
        # here for the two collections most likely to hold a quotable
        # correction/clarification turn: conversations + summaries.
        if not date_window or chroma_store is None:
            return
        for coll in ("conversations", "summaries"):
            try:
                rows = await asyncio.to_thread(
                    window_scan_collection, chroma_store, coll,
                    date_window, caps["per_facet_cap"] * 2,
                )
            except Exception as e:
                logger.debug(f"[Insight] date-range scan for {coll} failed: {e}")
                continue
            for row in rows or []:
                metadata = row.get("metadata") or {}
                if is_quarantined(metadata):
                    continue
                content = (row.get("content") or "").strip()
                if not content:
                    continue
                if coll == "conversations":
                    if is_junk_conversation_doc(content=content):
                        continue
                    if _user_side_is_greeting_or_ack(content):
                        continue
                    if _is_sweep_output_mode(metadata.get("response_mode")):
                        _prior_sweep_dropped[0] += 1
                        continue
                elif is_junk_summary(content):
                    continue
                if _looks_like_sweep_scaffolding(content):
                    _prior_sweep_dropped[0] += 1
                    continue
                collected.append(EvidenceItem(
                    doc_id=row.get("id"),
                    text=content,
                    date=_meta_date(metadata),
                    collection=coll,
                    speaker="",
                    facet="date-range",
                    metadata=metadata,
                ))

    async def _run_all() -> None:
        await asyncio.gather(
            *[_sweep_facet(f) for f in plan.facets],
            _quoted_phrase_scan(), _date_range_scan(),
        )

    try:
        await asyncio.wait_for(_run_all(), timeout=caps["sweep_timeout_s"])
    except asyncio.TimeoutError:
        logger.warning(
            f"[Insight] Sweep timed out at {caps['sweep_timeout_s']}s — "
            f"returning {len(collected)} partial evidence items"
        )
    except Exception as e:
        logger.warning(f"[Insight] Sweep error: {e} — returning partial evidence")

    if _prior_sweep_dropped[0]:
        logger.debug(
            f"[Insight] Dropped {_prior_sweep_dropped[0]} prior-sweep-output "
            f"item(s) (insight-assembly/doc-generation/self-note entries or "
            f"legacy scaffolding-marker matches)"
        )

    return _finalize(collected, caps, date_window=date_window)


def _finalize(
    items: list[EvidenceItem], caps: dict, *,
    date_window: Optional[tuple[str, str]] = None,
) -> list[EvidenceItem]:
    """Dedupe → date-window filter → snippet-clip → date-sort (newest first)
    → proportional trim."""
    seen: set[str] = set()
    deduped: list[EvidenceItem] = []
    # Attribution must run on full records: clipping a combined exchange
    # first made assistant prose look like the user's evidence.
    from core.insight.provenance import label_evidence
    for item in label_evidence(items):
        key = _dedupe_key(item)
        if key in seen:
            continue
        seen.add(key)
        item.text = _clip(item.text, caps["evidence_snippet_chars"])
        deduped.append(item)

    deduped = filter_evidence_by_date_window(deduped, date_window)

    deduped.sort(key=lambda i: i.date or "", reverse=True)

    total_cap = caps["total_evidence_cap"]
    if len(deduped) <= total_cap:
        return deduped

    # Proportional per-collection allocation (floor 1) so one chatty store
    # can't crowd out the graph/notes channels entirely. ROOT CAUSE (round 2,
    # 2026-09-04 live incident): `group[:quota]` sliced each collection's
    # NEWEST `quota` items — a single "corpus" collection label is shared by
    # every facet's ordinary keyword scan AND the quoted-phrase scan, so a
    # broad/common facet keyword (verified with a synthetic 6-facet plan:
    # a lone "you" facet alone contributed 43 of 80 final items, ALL from
    # the newest ~3 weeks) filled the WHOLE "corpus" quota with recent noise
    # and silently dropped the quoted-phrase scan's older, explicitly
    # in-window hits (07-20/07-28/08-02/08-18 all lost in the repro) even
    # though total items stayed at/under total_evidence_cap the whole time —
    # each per-scan cap looked healthy in isolation. When an explicit date
    # window is active, select each collection's slice window-fairly
    # (ISO-week round-robin) instead of blindly newest-first.
    by_coll: dict[str, list[EvidenceItem]] = {}
    for item in deduped:
        by_coll.setdefault(item.collection, []).append(item)
    total = len(deduped)
    kept: list[EvidenceItem] = []
    for coll, group in by_coll.items():
        quota = max(1, int(round(total_cap * len(group) / total)))
        source = interleave_evidence_for_coverage(group) if date_window else group
        kept.extend(source[:quota])
    kept.sort(key=lambda i: i.date or "", reverse=True)
    dropped = len(deduped) - len(kept[:total_cap])
    if dropped > 0:
        logger.info(f"[Insight] Evidence trimmed: kept {total_cap} of {len(deduped)} ({dropped} dropped)")
    return kept[:total_cap]


# ---------------------------------------------------------------------------
# Window-fair rendering + self-reference exclusion + date-window filtering
# (2026-09-04). Live incident: a "from 2026-07-15 through today" theme sweep
# assembled 77 items spanning seven weeks; newest-first sort + a 12000-char
# render cap meant only the newest ~3 days' worth (37 items) ever reached the
# model, and 7 of those were the request turn itself / the reply about it.
# ---------------------------------------------------------------------------

def week_bucket_key(date_str: Optional[str]) -> Optional[tuple[int, int]]:
    """Return a sortable ``(iso_year, iso_week)`` key, or None when the date
    is missing/unparseable (an ISO date/datetime string's first 10 chars are
    expected to be ``YYYY-MM-DD``).

    2026-09-04: delegates to utils.ordered_slice.week_bucket_key (single
    source of truth for the week-bucketing logic shared with
    window_fair_sample / interleave_evidence_for_coverage below).
    """
    return _ordered_week_bucket_key(date_str)


def interleave_evidence_for_coverage(items: list[EvidenceItem]) -> list[EvidenceItem]:
    """Reorder deduped, newest-first evidence for WINDOW-FAIR RENDERING.

    Items are bucketed by ISO week, buckets are visited newest-week-first,
    and one item is taken per bucket in round-robin (each bucket keeps its
    own newest-first order) — so a hard-character-cap render
    (``provenance.render_evidence_block``) spans the whole request window
    instead of collapsing to only the most recent few days. Undated items
    are appended last. Pure reordering: never drops or invents items, and
    the output is a permutation of the input.

    2026-09-04: delegates to utils.ordered_slice.window_fair_sample (single
    source of truth for the round-robin week-bucketing algorithm — the same
    fairness problem the gatherer_memory/summarizer/coordinator adoptions
    of this batch all close instances of).
    """
    return _ordered_window_fair_sample(items, lambda item: item.date)


_SHINGLE_N = 8
_REQUEST_OVERLAP_THRESHOLD = 0.60
_WORD_RE = re.compile(r"[a-z0-9']+")
_ASSISTANT_MARKER_RE = re.compile(r"\bAssistant:\s*", re.IGNORECASE)


def _word_tokens(text: str) -> list[str]:
    return _WORD_RE.findall((text or "").lower())


def _shingles(text: str, n: int = _SHINGLE_N) -> set[str]:
    words = _word_tokens(text)
    if not words:
        return set()
    if len(words) < n:
        return {" ".join(words)}
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def _shingle_overlap(item_shingles: set[str], request_shingles: set[str]) -> float:
    """Fraction of ``item_shingles`` also present in ``request_shingles`` —
    asymmetric on purpose: a short item wholly contained in a long request
    should count as near-total overlap even though the request has far more
    shingles overall."""
    if not item_shingles or not request_shingles:
        return 0.0
    return len(item_shingles & request_shingles) / len(item_shingles)


_CONTENT_WORD_MIN_LEN = 4
_REPLY_WORD_OVERLAP_THRESHOLD = 0.60
# Same-day tightening (round 2, 2026-09-04): the previous day's near-repeat
# of the SAME request ("10:59 and 11:5x" turns) shares real correction
# vocabulary with a fresh restatement but not always enough of it to cross
# the any-day 60% bar — same-day proximity is itself corroborating evidence
# that a chunk IS the request being re-typed, so same-day items use a lower
# bar. Any-day items keep the stricter 60% bar (a coincidental topical echo
# from a different day is real history, not self-reference).
_SAME_DAY_OVERLAP_THRESHOLD = 0.30
_SAME_DAY_REPLY_WORD_OVERLAP_THRESHOLD = 0.30


def _content_words(text: str) -> set[str]:
    """Word tokens with short function words dropped — a coarser, paraphrase
    -tolerant signal than 8-word shingles, used only to catch a conversation
    doc's assistant-authored reply chunk that is clearly ABOUT the request
    even when its wording diverges (a paraphrased "I'll gather every
    correction you made about X" shares almost none of the request's exact
    8-word runs but almost all of its content words)."""
    return {w for w in _word_tokens(text) if len(w) >= _CONTENT_WORD_MIN_LEN}


def _user_side_is_greeting_or_ack(content: str) -> bool:
    """A conversation doc whose USER side is a bare greeting/ack ("Hey",
    "ok", "thanks") is not evidence about anything (live 15:57: [E5] "Hey").
    Reads the ``User:`` span the store renders; shape only."""
    text = content or ""
    if "User:" in text:
        user_part = text.split("User:", 1)[1].split("Assistant:", 1)[0]
    else:
        user_part = text
    user_part = " ".join(user_part.split())
    if not user_part or len(user_part.split()) > 6:
        return False
    # lazy import: cycle (query_checker <- gate <- insight.detector)
    from utils.query_checker import is_casual_acknowledgment, is_greeting_opener
    return bool(is_greeting_opener(user_part) or is_casual_acknowledgment(user_part))


def fact_triple_is_junk(content: str) -> bool:
    """True when a facts-collection doc ("subject | relation | object") carries
    a junk OBJECT by THE deployed extractor predicate
    (``memory.fact_extractor._is_junk_object``). Non-triple text is never
    junk here (other filters own it)."""
    parts = [p.strip() for p in (content or "").split("|")]
    if len(parts) != 3 or not parts[2]:
        return False
    subject, relation, obj = parts
    # Junk SUBJECT: a function word or an empty/one-letter token can't be an
    # entity ("and | is | reported as evidence", "means | is | plausibly …",
    # live 15:57). Closed grammatical set + shape, no entity vocabulary.
    if not subject or subject.casefold() in _JUNK_SUBJECTS or len(subject) < 2:
        return True
    # Bare numeric duration/quantity as the whole object on a non-schedule
    # relation ("user | belief | 3 months") carries no claim.
    if _BARE_QUANTITY_RE.fullmatch(obj) and not _SCHEDULE_RELATION_RE.search(relation):
        return True
    # lazy import: cycle (fact_extractor -> memory package -> insight consumers)
    from memory.fact_extractor import _is_junk_object
    try:
        return bool(_is_junk_object(obj, relation))
    except Exception:
        return False


_JUNK_SUBJECTS = frozenset({
    "and", "or", "but", "so", "means", "it", "this", "that", "these", "those",
    "there", "here", "which", "what", "who", "the", "a", "an", "of", "in", "on",
    "is", "was", "be", "to", "for", "with", "as", "if", "then", "also",
})
_BARE_QUANTITY_RE = re.compile(
    r"(?:about\s+|roughly\s+|~)?\d+(?:\.\d+)?\s*"
    r"(?:seconds?|minutes?|mins?|hours?|hrs?|days?|weeks?|wks?|months?|years?|yrs?)?",
    re.IGNORECASE,
)
_SCHEDULE_RELATION_RE = re.compile(
    r"schedule|_time$|_at$|duration|frequency|streak|days_|_days|dose|amount|count",
    re.IGNORECASE,
)


_ASSISTANT_DIRECTED_MAX_WORDS = 80


def exclude_assistant_directed_items(items: list[EvidenceItem]) -> list[EvidenceItem]:
    """Drop user-authored evidence that is a REQUEST TO THE ASSISTANT rather
    than an observation about the user's life (2026-09-06 retest: "Give me a
    detailed analysis…", "Does my history actually support…", "…can you read
    the last one I received in outlook…", "search my claim and fact check"
    all rendered as [E#] history). Shape only — ``query_checker.is_request_shaped``
    (imperative/“can you” openers, info-seeking cues, an address to the
    assistant) on a short user turn. Long user turns are kept: a request can
    frame a substantive report. Assistant, notes, facts, research items are
    never touched here."""
    # lazy import: cycle (query_checker <- gate <- insight.detector)
    from utils.query_checker import is_request_shaped
    kept: list[EvidenceItem] = []
    dropped = 0
    for item in items:
        text = (item.text or "").strip()
        if (item.collection in ("corpus", "conversations")
                and item.speaker != "assistant"
                and text
                and len(text.split()) <= _ASSISTANT_DIRECTED_MAX_WORDS
                and is_request_shaped(text)):
            dropped += 1
            continue
        kept.append(item)
    if dropped:
        logger.debug("[Insight Sweep] Dropped %d assistant-directed request items", dropped)
    return kept


def exclude_current_request_evidence(
    items: list[EvidenceItem],
    request_text: str,
    *,
    current_turn_date: Optional[str] = None,
    threshold: float = _REQUEST_OVERLAP_THRESHOLD,
    same_day_threshold: float = _SAME_DAY_OVERLAP_THRESHOLD,
) -> list[EvidenceItem]:
    """Drop evidence items that ARE the live request rather than history.

    Three criteria: an exact current-turn timestamp match, a near-duplicate
    excerpt of the request text (>= ``threshold`` of the item's own 8-word
    shingles also appear in the request — a lower ``same_day_threshold``
    applies when the item shares the SAME calendar day as
    ``current_turn_date``, since same-day proximity is itself corroborating
    evidence a chunk is a repeat/near-repeat of the live request rather than
    a coincidental topical echo from another day), or a conversation-doc's
    embedded assistant-reply chunk that mostly restates the request (same
    same-day tightening). Unrelated any-day items are kept — this is a
    TEXT-overlap test, not a blanket date filter. Logs the drop count at
    DEBUG.
    """
    if not request_text or not request_text.strip():
        return list(items)
    request_shingles = _shingles(request_text)
    request_content_words = _content_words(request_text)
    if not request_shingles:
        return list(items)
    cur_date = (current_turn_date or "")[:19] or None
    today = (current_turn_date or "")[:10] or None
    kept: list[EvidenceItem] = []
    dropped = 0
    for item in items:
        if cur_date and (item.date or "")[:19] == cur_date:
            dropped += 1
            continue
        is_same_day = bool(today) and (item.date or "")[:10] == today
        chunk_threshold = same_day_threshold if is_same_day else threshold
        if _shingle_overlap(_shingles(item.text), request_shingles) >= chunk_threshold:
            dropped += 1
            continue
        if "Assistant:" in (item.text or ""):
            reply_part = _ASSISTANT_MARKER_RE.split(item.text, maxsplit=1)[-1]
            reply_words = _content_words(reply_part)
            reply_threshold = (
                _SAME_DAY_REPLY_WORD_OVERLAP_THRESHOLD if is_same_day
                else _REPLY_WORD_OVERLAP_THRESHOLD
            )
            if (len(reply_words) >= 5 and request_content_words and (
                len(reply_words & request_content_words) / len(reply_words)
                >= reply_threshold
            )):
                dropped += 1
                continue
        kept.append(item)
    if dropped:
        logger.debug(
            f"[Insight] Excluded {dropped} self-referential evidence item(s) "
            f"(the current request or a reply about it)"
        )
    return kept


_ISO_DATE_FULL_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def filter_evidence_by_date_window(
    items: list[EvidenceItem], date_window: Optional[tuple[str, str]],
) -> list[EvidenceItem]:
    """Keep items inside an explicit inclusive ``[start, end]`` ISO date
    window; items with no parseable date are ALWAYS kept — an explicit
    window narrows DATED evidence, it must never silently drop evidence
    (graph/fact rows) that carries no reliable date at all."""
    if not date_window:
        return items
    start, end = date_window
    if not start or not end:
        return items
    kept: list[EvidenceItem] = []
    dropped = 0
    for item in items:
        date = (item.date or "")[:10]
        if not _ISO_DATE_FULL_RE.match(date):
            kept.append(item)
            continue
        if start <= date <= end:
            kept.append(item)
        else:
            dropped += 1
    if dropped:
        logger.debug(f"[Insight] Date window {start}..{end} dropped {dropped} item(s)")
    return kept


def window_scan_collection(chroma_store, collection_name, window, cap):
    """Chunks whose CONTENT date falls inside ``[start, end]`` (ISO strings).

    The canonical date-range retrieval arm — an explicit calendar window is a
    metadata question, not a similarity question. Scans the collection's
    metadata (small collections only — notes/facts/conversations/summaries, a
    few thousand chunks) and prefers ``note_date`` (content date) over
    index-time timestamps. Read-only; failures degrade to an empty list.

    Originally written for the longitudinal-deliberation path
    (``gui.handlers``, still exposed there as ``_window_scan_collection`` for
    backward compatibility — that name now delegates here); reused by
    ``run_sweep``'s theme-sweep date-range arm (2026-09-04) so a week with no
    strong semantic hit can still contribute dated evidence.
    """
    try:
        coll = chroma_store._get_collection(collection_name)
        if coll is None:
            return []
        data = coll.get(include=["documents", "metadatas"])
        start, end = window
        rows = []
        for doc, meta, chunk_id in zip(
            data.get("documents") or [],
            data.get("metadatas") or [],
            data.get("ids") or [],
        ):
            meta = meta or {}
            date = str(
                meta.get("note_date") or meta.get("date")
                or meta.get("timestamp") or ""
            )[:10]
            if not (start <= date <= end):
                continue
            rows.append((date, {"id": chunk_id, "content": doc, "metadata": meta}))
        # Date-sorted, evenly-sampled cap: a first-N cap starved the LATER
        # weeks of a long window (coverage bias would masquerade as a trend).
        rows.sort(key=lambda item: item[0])
        if len(rows) > cap:
            step = len(rows) / cap
            rows = [rows[int(i * step)] for i in range(cap)]
        return [row for _, row in rows]
    except Exception as exc:
        logger.warning(
            f"[Insight] window scan failed for {collection_name}: {exc}"
        )
        return []
