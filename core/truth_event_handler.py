"""
# core/truth_event_handler.py

Module Contract
- Purpose: Handle truth score events (corrections, confirmations, entity resolution, content attribution).
- Inputs:
  - get_recent_profile_facts(user_profile, limit) -> list
  - apply_truth_event(event, user_profile, logger) -> None
  - cascade_entity_resolution(events, memory_system, logger) -> None
  - apply_content_attributions(attributions, memory_system, logger) -> None
- Side effects: Modifies user profile truth scores, ChromaDB metadata.
"""

from datetime import datetime, date
from typing import Optional

from utils.logging_utils import get_logger
from core.correction_detector import CorrectionEvent

logger = get_logger("truth_event_handler")

_CRISIS_KEYWORDS = frozenset({
    'die', 'died', 'death', 'dead', 'dying', 'icu', 'emergency',
    'hospital', 'dnr', 'critical', 'serious', 'make it', 'losing',
    'loss', 'crisis', 'panic', 'euthan', 'terminal', 'end of life',
})


def get_recent_profile_facts(user_profile, limit: int = 30) -> list:
    """Gather recent current facts from user profile for correction/confirmation detection."""
    facts = []
    try:
        from memory.user_profile_schema import ProfileCategory
        for cat in ProfileCategory:
            cat_facts = user_profile.get_category(cat, include_historical=False)
            for f in cat_facts:
                if isinstance(f, dict) and f.get("fact_id"):
                    facts.append(f)
        facts.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return facts[:limit]
    except Exception as e:
        logger.debug(f"Failed to get profile facts: {e}")
        return []


def apply_truth_event(event: CorrectionEvent, user_profile) -> None:
    """Apply a correction/confirmation event to the matching profile fact."""
    try:
        from memory.truth_scorer import TruthScorer
        from memory.user_profile_schema import ProfileCategory

        for cat in ProfileCategory:
            cat_facts = user_profile.get_category(cat, include_historical=True)
            for fact in cat_facts:
                if not isinstance(fact, dict):
                    continue
                if fact.get("fact_id") != event.fact_id:
                    continue

                old_truth = float(fact.get("truth_score", 0.7))
                if event.event_type == "correction":
                    fact["truth_score"] = TruthScorer.apply_correction(old_truth)
                    logger.info(
                        f"Truth correction: {event.relation}='{event.old_value}' "
                        f"truth {old_truth:.2f} -> {fact['truth_score']:.2f}"
                    )
                elif event.event_type == "confirmation":
                    fact["truth_score"] = TruthScorer.apply_confirmation(old_truth)
                    fact["last_confirmed_at"] = datetime.now().isoformat()
                    fact["confirmation_count"] = fact.get("confirmation_count", 0) + 1
                    logger.info(
                        f"Truth confirmation: {event.relation}='{event.old_value}' "
                        f"truth {old_truth:.2f} -> {fact['truth_score']:.2f}"
                    )

                user_profile.save()
                return  # Found and updated

    except Exception as e:
        logger.warning(f"Failed to apply truth event: {e}")


def cascade_entity_resolution(events: list, memory_system) -> None:
    """Annotate crisis-era summaries/reflections with resolution metadata.

    When the user says "Flapjack did not die", finds summaries mentioning
    Flapjack + crisis keywords and marks them with a resolution note +
    elevated staleness_ratio so the prompt prefix signals resolution.
    """
    from config.app_config import STALENESS_ENABLED
    if not STALENESS_ENABLED:
        return

    chroma_store = None
    if memory_system and hasattr(memory_system, 'chroma_store'):
        chroma_store = memory_system.chroma_store
    if not chroma_store:
        return

    entity_resolver = getattr(memory_system, 'entity_resolver', None)
    today = date.today().isoformat()

    for event in events:
        # Resolve display name via entity resolver if available
        display_name = event.entity_name
        if entity_resolver:
            try:
                resolved_id = entity_resolver.resolve(event.entity_name)
                if resolved_id:
                    graph_memory = getattr(memory_system, 'graph_memory', None)
                    if graph_memory:
                        node = graph_memory.get_node(resolved_id)
                        if node and hasattr(node, 'display_name') and node.display_name:
                            display_name = node.display_name
            except Exception:
                pass

        entity_lower = event.entity_name.lower()
        resolution_note = f"{display_name} is alive (confirmed {today})"
        total_affected = 0

        for collection in ('summaries', 'reflections'):
            try:
                results = chroma_store.query_collection(
                    collection, event.entity_name, n_results=20
                )

                for doc in results:
                    content = (doc.get("content") or "").lower()

                    if entity_lower not in content:
                        continue

                    has_crisis = any(kw in content for kw in _CRISIS_KEYWORDS)
                    if not has_crisis:
                        continue

                    doc_id = doc.get("id")
                    if not doc_id:
                        continue

                    metadata = doc.get("metadata", {}) or {}
                    current_staleness = float(metadata.get("staleness_ratio", 0) or 0)
                    new_staleness = max(current_staleness, 0.65)

                    chroma_store.update_metadata(collection, doc_id, {
                        "resolution_note": resolution_note,
                        "staleness_ratio": new_staleness,
                        "resolution_date": today,
                        "resolution_entity": display_name,
                    })
                    total_affected += 1

            except Exception as e:
                logger.debug(
                    f"[EntityResolution] Failed to query {collection} for "
                    f"'{event.entity_name}': {e}"
                )

        if total_affected:
            logger.info(
                f"[EntityResolution] Annotated {total_affected} document(s) "
                f"with resolution for '{display_name}'"
            )


def apply_content_attributions(attributions: list, memory_system) -> None:
    """Apply retroactive attribution to the most recent shared content.

    When user says "it's by X", find the most recent conversation with
    content_type metadata and update its content_attribution field.
    """
    chroma_store = None
    if memory_system and hasattr(memory_system, 'chroma_store'):
        chroma_store = memory_system.chroma_store
    if not chroma_store:
        return

    try:
        recent = chroma_store.get_recent("conversations", limit=10)
        for item in (recent or []):
            md = item.get("metadata", {}) or {}
            if md.get("content_type") and not md.get("content_attribution"):
                doc_id = item.get("id")
                if not doc_id:
                    continue
                updates = {}
                for attr_event in attributions:
                    if attr_event.attribution_type in ("artist", "author"):
                        updates["content_attribution"] = attr_event.value
                    elif attr_event.attribution_type == "title":
                        updates["content_title"] = attr_event.value
                if updates:
                    chroma_store.update_metadata("conversations", doc_id, updates)
                    logger.info(
                        f"[ContentAttribution] Updated {doc_id} with {updates}"
                    )
                return  # Only update the most recent unattributed content
    except Exception as e:
        logger.debug(f"[ContentAttribution] Failed (non-fatal): {e}")
