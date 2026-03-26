"""
Memory Staleness — Claim Tracker

Module Contract:
    - Purpose: Track which claims (subject, relation) are embedded in which
      summaries/reflections so that when a fact is corrected or contradicted,
      all documents containing the stale claim can have their staleness
      metadata updated (cascade invalidation).
    - Inputs: Summary/reflection text, fact correction events
    - Outputs: Staleness metadata updates on affected documents
    - Dependencies: entity_resolver (for canonicalization), app_config (settings)
    - Persistence: JSON at data/claim_index.json (same pattern as entity_aliases.json)
"""

import hashlib
import json
import os
import re
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from utils.logging_utils import get_logger

logger = get_logger("claim_tracker")


# ------------------------------------------------------------------
# Models
# ------------------------------------------------------------------

class ClaimKey(BaseModel):
    """A canonical (subject, relation) claim identifier."""
    subject: str = Field(description="Canonicalized subject entity")
    relation: str = Field(description="Canonicalized relation/predicate")
    claim_hash: str = Field(default="", description="MD5 hash of subject|relation")

    def model_post_init(self, __context) -> None:
        if not self.claim_hash:
            self.claim_hash = _compute_hash(self.subject, self.relation)


class IndexEntry(BaseModel):
    """A document reference stored in the claim index."""
    doc_id: str
    collection: str


# ------------------------------------------------------------------
# Hash computation
# ------------------------------------------------------------------

def _compute_hash(subject: str, relation: str) -> str:
    """Compute a short MD5 hash for a (subject, relation) pair."""
    canonical = f"{subject.lower().strip()}|{relation.lower().strip()}"
    return hashlib.md5(canonical.encode()).hexdigest()[:12]


# ------------------------------------------------------------------
# Canonicalization
# ------------------------------------------------------------------

def canonicalize_claim(
    subject: str,
    relation: str,
    entity_resolver=None,
) -> ClaimKey:
    """Build a canonical ClaimKey from raw subject/relation strings.

    Uses EntityResolver.resolve() for subject normalization and
    normalize_relation() for predicate normalization when available.
    Falls back to simple lowercasing + underscore normalization.
    """
    # Subject normalization
    canon_subject = subject.lower().strip()
    if entity_resolver:
        resolved = entity_resolver.resolve(subject)
        if resolved:
            canon_subject = resolved

    # Relation normalization
    canon_relation = re.sub(r"\s+", "_", relation.lower().strip())
    try:
        from memory.entity_resolver import normalize_relation
        canon_relation = normalize_relation(relation)
    except ImportError:
        pass

    return ClaimKey(
        subject=canon_subject,
        relation=canon_relation,
    )


# ------------------------------------------------------------------
# Claim extraction from text
# ------------------------------------------------------------------

# Patterns for extracting declarative claims from summary text
_TRIPLE_SEP_RE = re.compile(r"(.+?)\s*[|—–-]\s*(.+?)\s*[|—–-]\s*(.+)")
_DECLARATIVE_RE = re.compile(
    r"(?:^|\. )([A-Z][a-z]+(?:\s+[A-Z]?[a-z]+)*)\s+"
    r"(is|has|was|likes|dislikes|prefers|uses|works|lives|studies|owns|speaks|"
    r"moved|plans|wants|enjoys|created|built|loves|hates|made|started|plays|"
    r"works at|works on|lives in|born in|interested in|skilled at|married to|"
    r"friend of|sibling of|parent of|child of|pet of|spouse of)\s+"
    r"(.+?)(?:\.|,|;|$)",
    re.IGNORECASE,
)
# "Luke lives in X" / "The user works at Y" etc.
_SUBJECT_VERB_RE = re.compile(
    r"(?:^|\. )(?:the user|user|they|he|she|(?:[A-Z][a-z]+))\s+"
    r"((?:is|has|was|lives?|works?|studies|owns|speaks|moved|plans|wants|"
    r"enjoys|created|built|loves|hates|made|started|plays)(?:\s+(?:in|at|on|to|for|with))?)\s+"
    r"(.+?)(?:\.|,|;|$)",
    re.IGNORECASE,
)


def extract_claims_from_text(
    text: str,
    entity_resolver=None,
) -> List[ClaimKey]:
    """Extract (subject, relation) claim keys from summary/reflection text.

    Tries multiple strategies:
    1. Triple-separator patterns ("|", "—", "-")
    2. Declarative sentence patterns ("Luke lives in Atlanta")
    3. Subject-verb patterns ("the user works at Google")

    Returns deduplicated list of ClaimKeys (object intentionally discarded —
    we hash on (subject, relation) to catch all value changes).
    """
    claims: Dict[str, ClaimKey] = {}  # hash -> ClaimKey for dedup

    # Strategy 1: separator-based triples
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        m = _TRIPLE_SEP_RE.match(line)
        if m:
            ck = canonicalize_claim(m.group(1).strip(), m.group(2).strip(), entity_resolver)
            claims[ck.claim_hash] = ck

    # Strategy 2: declarative patterns
    for m in _DECLARATIVE_RE.finditer(text):
        subject = m.group(1).strip()
        relation = m.group(2).strip()
        ck = canonicalize_claim(subject, relation, entity_resolver)
        claims[ck.claim_hash] = ck

    # Strategy 3: subject-verb patterns (user-centric)
    for m in _SUBJECT_VERB_RE.finditer(text):
        relation = m.group(1).strip()
        ck = canonicalize_claim("user", relation, entity_resolver)
        claims[ck.claim_hash] = ck

    return list(claims.values())


# ------------------------------------------------------------------
# ClaimIndex — reverse index: claim_hash -> [doc references]
# ------------------------------------------------------------------

class ClaimIndex:
    """Reverse index mapping claim hashes to document IDs.

    Persisted as JSON at data/claim_index.json. Same pattern as
    entity_aliases.json — loaded at startup, saved on shutdown.
    """

    def __init__(self, persist_path: str = ""):
        self._persist_path = persist_path
        # claim_hash -> list of {doc_id, collection}
        self._index: Dict[str, List[Dict[str, str]]] = {}
        # doc_id -> list of claim_hashes (for fast removal)
        self._doc_claims: Dict[str, List[str]] = {}
        self._dirty = False

        if persist_path:
            self.load()

    def add_claims(
        self,
        doc_id: str,
        collection: str,
        claims: List[ClaimKey],
    ) -> None:
        """Register claims for a document in the index."""
        hashes = []
        for ck in claims:
            h = ck.claim_hash
            hashes.append(h)
            entry = {"doc_id": doc_id, "collection": collection}
            if h not in self._index:
                self._index[h] = []
            # Avoid duplicate entries
            if not any(e["doc_id"] == doc_id for e in self._index[h]):
                self._index[h].append(entry)
                self._dirty = True

        # Update reverse mapping
        existing = self._doc_claims.get(doc_id, [])
        for h in hashes:
            if h not in existing:
                existing.append(h)
        self._doc_claims[doc_id] = existing

    def get_documents_for_claim(self, claim_hash: str) -> List[Dict[str, str]]:
        """Return all documents containing a given claim.

        Returns list of {"doc_id": ..., "collection": ...}.
        """
        return list(self._index.get(claim_hash, []))

    def get_claims_for_document(self, doc_id: str) -> List[str]:
        """Return all claim hashes registered for a document."""
        return list(self._doc_claims.get(doc_id, []))

    def remove_document(self, doc_id: str) -> None:
        """Remove all index entries for a document."""
        hashes = self._doc_claims.pop(doc_id, [])
        for h in hashes:
            if h in self._index:
                self._index[h] = [
                    e for e in self._index[h] if e["doc_id"] != doc_id
                ]
                if not self._index[h]:
                    del self._index[h]
                self._dirty = True

    @property
    def total_claims(self) -> int:
        """Total unique claim hashes in the index."""
        return len(self._index)

    @property
    def total_documents(self) -> int:
        """Total unique documents tracked."""
        return len(self._doc_claims)

    # ------------------------------------------------------------------
    # Cascade: propagate staleness from a corrected fact
    # ------------------------------------------------------------------

    def cascade_staleness(
        self,
        claim_key: ClaimKey,
        chroma_store=None,
    ) -> List[Dict[str, str]]:
        """Find all documents containing a claim and update their staleness.

        For each affected document:
        - Counts total claims and stale claims
        - Computes staleness_ratio = stale_claims / total_claims
        - Updates metadata via chroma_store.update_metadata()

        Returns list of affected {doc_id, collection, staleness_ratio}.
        """
        affected = self.get_documents_for_claim(claim_key.claim_hash)
        if not affected:
            return []

        results = []
        for entry in affected:
            doc_id = entry["doc_id"]
            collection = entry["collection"]

            # Count total claims for this document
            all_hashes = self.get_claims_for_document(doc_id)
            total = len(all_hashes) if all_hashes else 1

            # Mark this claim hash as stale in our tracking
            stale_key = f"_stale_{claim_key.claim_hash}"
            if doc_id not in self._doc_claims:
                continue

            # Count how many distinct claims are stale for this doc
            # We store stale markers as metadata on the doc
            stale_count = 0
            if chroma_store:
                try:
                    # Read current stale_claims list from metadata
                    existing = chroma_store.get_document_metadata(collection, doc_id)
                    if existing:
                        stale_claims_str = existing.get("stale_claims", "")
                        stale_set = set(stale_claims_str.split(",")) if stale_claims_str else set()
                        stale_set.add(claim_key.claim_hash)
                        stale_count = len(stale_set)
                        staleness_ratio = round(min(stale_count / total, 1.0), 3)

                        chroma_store.update_metadata(collection, doc_id, {
                            "stale_claims": ",".join(sorted(stale_set)),
                            "staleness_ratio": staleness_ratio,
                        })
                        results.append({
                            "doc_id": doc_id,
                            "collection": collection,
                            "staleness_ratio": staleness_ratio,
                        })
                    else:
                        # Document doesn't exist (deleted?), clean up index
                        logger.debug(f"[Staleness] Doc {doc_id} not found in {collection}, removing from index")
                        self.remove_document(doc_id)
                except Exception as e:
                    logger.warning(f"[Staleness] Failed to update {doc_id} in {collection}: {e}")
            else:
                # No chroma_store — just report what would be affected
                stale_count = 1  # at least this one
                staleness_ratio = round(min(stale_count / total, 1.0), 3)
                results.append({
                    "doc_id": doc_id,
                    "collection": collection,
                    "staleness_ratio": staleness_ratio,
                })

        if results:
            logger.info(
                f"[Staleness] {len(results)} documents marked stale due to "
                f"correction on ({claim_key.subject}, {claim_key.relation})"
            )

        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Persist index to JSON file."""
        if not self._persist_path:
            return
        if not self._dirty:
            logger.debug("[ClaimIndex] No changes to save")
            return

        os.makedirs(os.path.dirname(self._persist_path) or ".", exist_ok=True)
        data = {
            "index": self._index,
            "doc_claims": self._doc_claims,
        }
        try:
            with open(self._persist_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            self._dirty = False
            logger.info(
                f"[ClaimIndex] Saved {self.total_claims} claims across "
                f"{self.total_documents} documents to {self._persist_path}"
            )
        except OSError as e:
            logger.warning(f"[ClaimIndex] Failed to save: {e}")

    def load(self) -> None:
        """Load index from JSON file."""
        if not self._persist_path or not os.path.exists(self._persist_path):
            return
        try:
            with open(self._persist_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._index = data.get("index", {})
            self._doc_claims = data.get("doc_claims", {})
            self._dirty = False
            logger.info(
                f"[ClaimIndex] Loaded {self.total_claims} claims across "
                f"{self.total_documents} documents from {self._persist_path}"
            )
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"[ClaimIndex] Failed to load: {e}")
            self._index = {}
            self._doc_claims = {}
