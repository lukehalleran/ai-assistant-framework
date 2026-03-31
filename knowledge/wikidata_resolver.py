# knowledge/wikidata_resolver.py
"""
Entity resolution between personal graph entities and Wikidata entities.

Three matching strategies (in priority order):
  1. Exact alias match — personal entity name/alias matches Wikidata label/alias
  2. Description-enriched embedding — embed "{label} — {description}" for both
     sides, compare with cosine similarity.  Context disambiguates name collisions.
  3. Category-aware filtering — only compare personal entities against Wikidata
     entities in plausible domains (e.g., "rock climbing" only matches hobbies).

Each match gets a bridge_confidence score:
  - Exact alias: 1.0
  - Embedding >= 0.80: 0.9
  - Embedding 0.60-0.80: 0.7
"""

import re
from typing import Optional

from utils.logging_utils import get_logger

logger = get_logger("wikidata_resolver")

# Map personal graph relation types to plausible Wikidata domains.
# Used for category-aware matching.
PERSONAL_DOMAIN_MAP = {
    # Relations → plausible Wikidata domains
    "hobby": {"hobbies_interests", "fitness_exercise", "cross_domain_science"},
    "plays": {"hobbies_interests"},
    "favorite_game": {"hobbies_interests"},
    "favorite_beer": {"hobbies_interests"},
    "medications": {"health_medical"},
    "medication": {"health_medical"},
    "works_at": {"hobbies_interests", "cross_domain_science"},
    "works_on": {"computer_science", "cross_domain_science"},
    "studies": {"education_statistics", "cross_domain_science"},
    "course_interest": {"education_statistics", "computer_science"},
    "location": {"geography_places"},
    "lives_in": {"geography_places"},
    "pet": {"cross_domain_science"},
    "values": {"philosophy_mind", "psychology_relationships"},
}

# Default domains when relation is unknown
DEFAULT_DOMAINS = {
    "health_medical", "hobbies_interests", "cross_domain_science",
    "psychology_relationships", "education_statistics",
}


def _slugify(text: str) -> str:
    s = text.lower().strip()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", "_", s)
    return s


class WikidataEntityMapper:
    """Maps personal graph entities to Wikidata entities."""

    def __init__(self, graph_memory, entity_resolver, wikidata_entities: dict):
        """
        Args:
            graph_memory: GraphMemory instance with personal entities
            entity_resolver: EntityResolver for alias lookups
            wikidata_entities: dict of qid -> entity dict from wikidata_cache.json
        """
        self.graph = graph_memory
        self.resolver = entity_resolver
        self.wikidata = wikidata_entities

        # Build reverse index: lowercase label/alias → qid
        self._wiki_label_index: dict[str, str] = {}
        for qid, ent in self.wikidata.items():
            self._wiki_label_index[ent["label"].lower().strip()] = qid
            for alias in ent.get("aliases", []):
                key = alias.lower().strip()
                if key not in self._wiki_label_index:
                    self._wiki_label_index[key] = qid

    def map_personal_to_wikidata(
        self,
        embedding_threshold: float = 0.60,
        use_embeddings: bool = True,
    ) -> list[dict]:
        """Find mappings between personal entities and Wikidata entities.

        Returns list of match dicts with keys:
            personal_id, personal_display, wikidata_qid, wikidata_label,
            match_type, confidence, domain
        """
        matches = []
        matched_personal = set()

        # Collect personal entities with their relations (for domain filtering)
        personal_entities = []
        for nid, data in self.graph.graph.nodes(data=True):
            source = data.get("metadata", {}).get("source", "personal")
            if source != "personal":
                continue
            display = data.get("display_name", nid)
            # Get outgoing relations for domain inference
            out_rels = set()
            for _, _, edata in self.graph.graph.out_edges(nid, data=True):
                out_rels.add(edata.get("relation", ""))
            # Also check if this is a target of user edges
            for src, _, edata in self.graph.graph.in_edges(nid, data=True):
                if src == "user":
                    out_rels.add(edata.get("relation", ""))
            personal_entities.append({
                "id": nid,
                "display": display,
                "aliases": data.get("aliases", []),
                "relations": out_rels,
                "metadata": data.get("metadata", {}),
            })

        # Phase 1: Exact alias matching
        for pe in personal_entities:
            match = self._exact_match(pe)
            if match:
                matches.append(match)
                matched_personal.add(pe["id"])

        logger.info(f"Exact alias matches: {len(matches)}")

        # Phase 2: Description-enriched embedding matching
        if use_embeddings:
            unmatched = [pe for pe in personal_entities if pe["id"] not in matched_personal]
            if unmatched:
                emb_matches = self._embedding_match_batch(
                    unmatched, threshold=embedding_threshold
                )
                matches.extend(emb_matches)
                logger.info(f"Embedding matches: {len(emb_matches)}")

        return matches

    def _exact_match(self, personal_entity: dict) -> Optional[dict]:
        """Check if personal entity name or aliases exactly match a Wikidata label/alias."""
        names_to_check = [personal_entity["display"].lower().strip()]
        names_to_check.extend(a.lower().strip() for a in personal_entity.get("aliases", []))
        # Also try the entity ID itself
        names_to_check.append(personal_entity["id"])

        for name in names_to_check:
            if name in self._wiki_label_index:
                qid = self._wiki_label_index[name]
                wd = self.wikidata[qid]
                return {
                    "personal_id": personal_entity["id"],
                    "personal_display": personal_entity["display"],
                    "wikidata_qid": qid,
                    "wikidata_label": wd["label"],
                    "match_type": "exact_alias",
                    "confidence": 1.0,
                    "domain": wd.get("domain_category", "unknown"),
                }
        return None

    def _embedding_match_batch(
        self, personal_entities: list[dict], threshold: float = 0.60
    ) -> list[dict]:
        """Description-enriched embedding matching with category-aware filtering."""
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
        except ImportError:
            logger.warning("sentence-transformers not available, skipping embedding matches")
            return []

        model = SentenceTransformer("all-MiniLM-L6-v2")
        matches = []

        # Build per-domain Wikidata embedding pools
        wiki_by_domain: dict[str, list[tuple[str, str]]] = {}
        for qid, ent in self.wikidata.items():
            domain = ent.get("domain_category", "cross_domain_science")
            # Description-enriched text
            desc = ent.get("description", "")
            enriched = f"{ent['label']} — {desc}" if desc else ent["label"]
            wiki_by_domain.setdefault(domain, []).append((qid, enriched))

        # Pre-encode all Wikidata entities by domain
        wiki_embeddings: dict[str, tuple[list[str], any]] = {}
        for domain, items in wiki_by_domain.items():
            qids = [q for q, _ in items]
            texts = [t for _, t in items]
            embs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            wiki_embeddings[domain] = (qids, embs)

        # Match each personal entity
        for pe in personal_entities:
            # Determine plausible domains for this entity
            plausible = self._get_plausible_domains(pe)

            # Build enriched personal text
            pe_text = self._enrich_personal_text(pe)
            pe_emb = model.encode([pe_text], normalize_embeddings=True)

            best_match = None
            best_sim = 0.0

            for domain in plausible:
                if domain not in wiki_embeddings:
                    continue
                qids, embs = wiki_embeddings[domain]
                sims = (pe_emb @ embs.T)[0]
                max_idx = int(np.argmax(sims))
                sim = float(sims[max_idx])

                if sim > best_sim:
                    best_sim = sim
                    best_match = {
                        "qid": qids[max_idx],
                        "domain": domain,
                    }

            if best_match and best_sim >= threshold:
                wd = self.wikidata[best_match["qid"]]
                confidence = 0.9 if best_sim >= 0.80 else 0.7
                matches.append({
                    "personal_id": pe["id"],
                    "personal_display": pe["display"],
                    "wikidata_qid": best_match["qid"],
                    "wikidata_label": wd["label"],
                    "match_type": "embedding",
                    "confidence": confidence,
                    "similarity": round(best_sim, 3),
                    "domain": best_match["domain"],
                })

        return matches

    def _get_plausible_domains(self, personal_entity: dict) -> set[str]:
        """Determine which Wikidata domains a personal entity could map to."""
        domains = set()
        for rel in personal_entity.get("relations", []):
            mapped = PERSONAL_DOMAIN_MAP.get(rel)
            if mapped:
                domains.update(mapped)
        return domains if domains else DEFAULT_DOMAINS

    def _enrich_personal_text(self, personal_entity: dict) -> str:
        """Build description-enriched text for a personal entity.

        Pulls context from node metadata and graph neighbors.
        """
        display = personal_entity["display"]
        parts = [display]

        # Add metadata context
        meta = personal_entity.get("metadata", {})
        for key in ["occupation", "description", "category"]:
            if key in meta:
                parts.append(meta[key])

        # Add relation context from user edges
        rels = personal_entity.get("relations", set())
        if rels:
            # Pick the most informative relation
            useful_rels = [r for r in rels if r not in {"is", "unknown", "corrected_fact"}]
            if useful_rels:
                parts.append(f"({', '.join(useful_rels[:3])})")

        return " — ".join(parts)
