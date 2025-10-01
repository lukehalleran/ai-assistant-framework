# memory/fact_extractor.py
import re
import uuid
from datetime import datetime
from typing import List, Dict, Tuple, Optional

from memory.memory_interface import MemoryNode, MemoryType
from utils.logging_utils import get_logger, log_and_time

logger = get_logger("fact_extractor")
logger.debug("fact_extractor.py is alive")

# --- Generic filters & utilities (domain-agnostic) ---
_STOP_SUBJECTS = {
    "i","you","we","they","he","she","it","this","that","these","those","what","something","anything","everything",
    "there","here"  # added
}
_STOP_OBJECTS = {
    "it","this","that","there","here","something","anything","everything","what",
}
_GENERIC_NOUNS = {"thing","stuff","item","press","game","place","area","part"}
_UNIT_TOKENS = {"lb","lbs","pound","pounds","kg","kgs","kilogram","kilograms"}

_ARTICLE_PREFIXES = ("a ", "an ", "the ")

def _looks_like_units(s: str) -> bool:
    t = s.strip().lower()
    return t in _UNIT_TOKENS

def _has_number_unit(s: str) -> bool:
    # very generic "number + token" detector (e.g., 285 lb / 130 kilograms / 100lbs)
    return bool(re.search(r"\b\d{1,5}\s*(?:[a-zA-Z]+)?\b", s))

def _prefer_named_span(text: str, nlp_doc, fallback: str) -> str:
    """Given the original text and spaCy doc, try to lift a PROPN/ENTITY from it."""
    if not nlp_doc:
        return fallback
    seg = text.strip()
    # Try exact span entity match
    for ent in nlp_doc.ents:
        if ent.text in seg:
            return ent.text
    # Prefer a PROPN token inside the span
    for token in nlp_doc:
        if token.pos_ == "PROPN" and token.text in seg:
            return token.text
    return fallback

def _is_adj_only(doc) -> bool:
    """True if all tokens are ADJ/ADV/PRON/punct-ish — i.e., not a nounish content span."""
    if not doc:
        return False
    nounish = any(t.pos_ in {"NOUN","PROPN"} for t in doc)
    return not nounish

def _relation_allows_adj(rel: str) -> bool:
    """Allow adjective-only objects for certain attribute relations (e.g., color)."""
    rl = (rel or "").lower()
    # If relation clearly denotes a color/favorite-color, adjectives like "blue" are valid objects
    if "color" in rl or rl in {"likes_color", "favourite_color", "favorite_color"}:
        return True
    return False

def _is_clause_like(text: str) -> bool:
    """Heuristic: object looks like a clause (infinitive/gerund/passive-pp + prep)."""
    t = (text or "").strip().lower()
    if not t:
        return False
    # infinitive: "to <verb> ..."
    if re.match(r"^to\s+\w+", t):
        return True
    # gerund at start: "<verb>ing ..."
    if re.match(r"^\w+ing\b", t):
        return True
    # passive participle followed by common preposition: "<verb>ed in/with/by/for/to/on ..."
    if re.match(r"^\w+ed\s+(?:in|with|by|for|to|on)\b", t):
        return True
    return False

def _refine_is_relation(rel: str, obj: str) -> str:
    rl = (rel or "").strip().lower()
    if rl == "is":
        ol = (obj or "").strip().lower()
        if ol.startswith(_ARTICLE_PREFIXES):
            return "is_a"
    return rel

def _clean_triple(subj: str, rel: str, obj: str, nlp=None) -> Optional[Tuple[str,str,str]]:
    """Return a cleaned/normalized (s,r,o) or None to drop the triple."""
    if not subj or not obj or not rel:
        return None

    # Preserve original casing for spaCy checks (proper nouns lose signal when lowercased)
    subj_orig = re.sub(r"\s+", " ", subj).strip(" .,:;\"'`")
    obj_orig = re.sub(r"\s+", " ", obj).strip(" .,:;\"'`")

    s = subj_orig.lower()
    r = re.sub(r"\s+", " ", rel).strip(" .,:;\"'`").lower()
    o = obj_orig.lower()

    # Drop trivial subjects/objects
    if s in _STOP_SUBJECTS or o in _STOP_OBJECTS:
        return None

    # Drop bare units as subjects or objects (unless a numeric value is present)
    if _looks_like_units(s) and not _has_number_unit(f"{s} {o}"):
        return None
    if _looks_like_units(o) and not _has_number_unit(f"{s} {o}"):
        return None

    # Basic content length sanity
    if len(s) < 2 or len(o) < 2:
        return None
    if len(s) > 120 or len(o) > 300:
        return None

    # Upgrade relation 'is' -> 'is_a' when object starts with article (generic)
    r = _refine_is_relation(r, o)

    # Drop low-signal clause-y complements for plain "is" (e.g., "key is to <verb> ...")
    if r == "is" and _is_clause_like(o):
        return None

    # If spaCy is available, reject adjective-only complements (e.g., "is impressive"),
    # except for relations where adjectives are valid values (e.g., *_color)
    # IMPORTANT: Use original casing for spaCy parse (proper nouns need capitalization)
    if nlp is not None:
        try:
            o_doc_orig = nlp(obj_orig)  # Parse with original casing
            if _is_adj_only(o_doc_orig) and not _relation_allows_adj(r):
                return None
            # Prefer proper-noun/entity forms when available, but keep lowercased key later
            s_doc = nlp(subj_orig)
            s_pref = _prefer_named_span(subj_orig, s_doc, s)
            o_pref = _prefer_named_span(obj_orig, o_doc_orig, o)
            s, o = s_pref.lower(), o_pref.lower()  # Lowercase after preference extraction
        except Exception:
            pass

    # Very generic low-signal filter
    if s in _GENERIC_NOUNS and o in _GENERIC_NOUNS:
        return None

    return (s, r, o)

def _score_triple(subj: str, rel: str, obj: str, nlp=None) -> float:
    """Light, generic confidence shaping; no domain knowledge."""
    base = 0.6
    bonus = 0.0
    txt = f"{subj} {rel} {obj}"
    if _has_number_unit(txt):
        bonus += 0.05
    if nlp is not None:
        try:
            doc = nlp(txt)
            if any(t.pos_ == "PROPN" for t in doc):
                bonus += 0.05
            if any(ent.label_ in {"PERSON","ORG","PRODUCT","WORK_OF_ART","GPE"} for ent in doc.ents):
                bonus += 0.05
        except Exception:
            pass
    return max(0.4, min(0.9, base + bonus))

# ---------------- Optional deps (lazy) ----------------
_NLP = None
_REBEL = None

try:
    import spacy
    _NLP = spacy.load("en_core_web_sm")
    logger.debug("[FactExtractor] spaCy loaded: en_core_web_sm")
except Exception as e:
    logger.debug(f"[FactExtractor] spaCy not available or failed to load: {e}")

try:
    from transformers import pipeline
    _REBEL = pipeline(
        "text2text-generation",
        model="Babelscape/rebel-large",
        tokenizer="Babelscape/rebel-large",
        max_length=512,
        truncation=True
    )
    logger.debug("[FactExtractor] REBEL pipeline loaded: Babelscape/rebel-large")
except Exception as e:
    logger.debug(f"[FactExtractor] REBEL not available or failed to load: {e}")


# ---------------- Preference helpers (new) ----------------
# Add quick patterns for canonicalizing "my favorite X is Y" etc.
PREF_PATTERNS = [
    (r"\bmy favorite (?P<slot>video game|game)\b", "favorite_video_game"),
    (r"\bmy favorite (?P<slot>beer)\b", "favorite_beer"),
    (r"\bmy favorite (?P<slot>coffee)\b", "favorite_coffee"),
    (r"\bmy favorite (?P<slot>color)\b", "favorite_color"),
]

def _normalize_subject_obj(subj: str, rel: str, obj: str, user_name: str = "Luke") -> Tuple[str, str, str]:
    """Promote generic subjects, keep obvious entity casing lightly."""
    s, r, o = (subj or "").strip(), (rel or "").strip(), (obj or "").strip()

    # Map first-person pronouns/possessives to canonical 'user'
    if s.lower() in {"i", "me", "my", "mine", "we", "our", "ours", "us"}:
        s = "user"

    # If subject is too generic and object looks like the real entity, promote object to subject
    generic_subjects = {"game", "brew", "coffee", "beer", "dunkel", "dunkels"}
    if s.lower() in generic_subjects and o:
        s = o

    # Light normalization for well-known entities (examples; harmless if not matched)
    if s.lower() == "skyrim":
        s = "Skyrim"
    if o.lower() == "skyrim":
        o = "Skyrim"

    return s, r, o

def _canonicalize_preferences(q_text: str, r_text: str, s: str, r: str, o: str, user_name: str = "Luke") -> Tuple[str, str, str]:
    """Map preference phrasing to canonical triples like (Luke, favorite_*, Y) or (Luke, likes, X)."""
    ql = (q_text or "").lower()
    rl = (r_text or "").lower()

    # "my favorite X is Y" → Luke | favorite_X | Y
    for pat, relname in PREF_PATTERNS:
        if re.search(pat, ql) or re.search(pat, rl):
            return (user_name, relname, o or s)

    # "I like/love X" → Luke | likes | X
    like_triggers = ("i like ", "i love ", "i really like ", "i really love ")
    if any(t in ql for t in like_triggers) or any(t in rl for t in like_triggers):
        return (user_name, "likes", o or s)

    return (s, r, o)

def _refine_is_relation(rel: str, obj: str) -> str:
    """If relation is 'is' and object looks like a noun phrase with an article, use 'is_a'."""
    if rel.strip().lower() == "is":
        ol = (obj or "").strip().lower()
        if ol.startswith(("a ", "an ", "the ")):
            return "is_a"
    return rel


class FactExtractor:
    """Multi-stage fact extractor: coref/sentences -> spaCy rules -> REBEL -> regex fallback."""

    def __init__(self, use_rebel: bool = True, use_regex: bool = True):
        self.use_rebel = use_rebel
        self.use_regex = use_regex

        # Light, high-precision patterns
        # Generic, high‑precision patterns (applied per line)
        # Terminators include end‑of‑line to avoid spanning into the assistant text.
        self.fact_patterns = [
            # Generic copular statements, include 'am' for first person
            r'(\w+)\s+(?:is|are|was|were|am)\s+(.+?)(?:[\.;,!?:]|$)',
            # Simple preference statements
            r'(?:I|you|we)\s+(?:like|love|hate|prefer)\s+(.+?)(?:[\.;,!?:]|$)',
            # Possessive attribute: my|your|our <attr> is <val>
            r'(?:my|your|our)\s+(\w+)\s+(?:is|are)\s+(.+?)(?:[\.;,!?:]|$)'
        ]

        # Domain‑helpful patterns: lifting metrics like "my squat is 365 lb"
        self.lift_patterns = [
            # my squat is 365, squat: 365 lb, deadlift=405kg, etc.
            r'\b(?:my\s+)?(squat|bench(?:\s*press)?|deadlift|ohp|overhead\s*press)\s*(?:is|=|:)\s*(\d{2,4})\s*(lbs?|pounds?|kgs?|kg)?\b',
            # I squatted 365 lb / I benched 225
            r'\bI\s*(?:just\s*)?(squatted|benched|deadlifted|pressed)\s*(\d{2,4})\s*(lbs?|pounds?|kgs?|kg)?\b',
        ]

        # Possessive-like attribute statements without apostrophes
        # e.g., "my cats name is Flapjack", "my car make is Honda"
        self.possessive_attr_patterns = [
            r"\bmy\s+([a-zA-Z]+?)(?:'s|s)?\s+(name|make|model|color|birthday|birth\s*date)\s*(?:is|=|:)\s+([^\.;,!?:]+)"
        ]

        # Two-word user attributes like "my middle name is Andrew" → user | middle_name | Andrew
        # Keep specific to name parts to avoid overfitting.
        self.two_word_attr_patterns = [
            r"\b(?:my|our)\s+((?:first|middle|last)\s+name)\s*(?:is|=|:)\s+([^\.;,!?:]+)"
        ]

        # Employment/occupation patterns (explicit)
        #  - I work at/for/in X -> user | works_at | X
        #  - I work as (a|an) Y -> user | occupation | Y
        self.employment_patterns = [
            ("works_at", r"\b(?:i|we)\s+(?:work|worked|have\s+worked|currently\s+work)\s+(?:at|for|in)\s+([^\.;,!?:\n]+?)(?:[\.;,!?:]|$)"),
            ("occupation", r"\b(?:i|we)\s+(?:work|worked|have\s+worked|currently\s+work)\s+as\s+(?:an?\s+)?([^\.;,!?:\n]+?)(?:[\.;,!?:]|$)"),
        ]

        logger.debug(
            f"[FactExtractor.__init__] use_rebel={self.use_rebel} "
            f"(pipeline={'yes' if _REBEL else 'no'}), use_regex={self.use_regex}, "
            f"spaCy={'yes' if _NLP else 'no'}"
        )

    @log_and_time("Extract Facts")
    async def extract_facts(
        self,
        query: str,
        response: str,
        conversation_context: Optional[List[Dict]] = None
    ) -> List[MemoryNode]:
        q_preview = (query or "")[:120].replace("\n", " ")
        logger.debug(
            f"[FactExtractor] Received query-only (q_len={len(query or '')}; response_ignored=True) "
            f"q_preview='{q_preview}' ctx_items={len(conversation_context or [])}"
        )

        # Facts should be extracted ONLY from the user query (ignore assistant response)
        text = self._prep_text(query, "", conversation_context)
        t_preview = text[:200].replace("\n", " ")
        logger.debug(f"[FactExtractor] Prepared text len={len(text)} preview='{t_preview}'")

        triples: List[Tuple[str, str, str, float, str]] = []

        # 0) Generalized spaCy dependency patterns (grammar-first)
        spacy_triples = self._extract_with_spacy_rules(text)
        if spacy_triples:
            logger.debug(f"[FactExtractor] spaCy rules extracted {len(spacy_triples)} triples")
            for i, t in enumerate(spacy_triples[:5]):
                logger.debug(f"[FactExtractor] spaCy triple[{i}]: subj='{t[0]}' rel='{t[1]}' obj='{t[2]}' conf={t[3]}")
            triples.extend(spacy_triples)

        # 1) REBEL (if available & enabled)
        if self.use_rebel and _REBEL is not None:
            rebel_triples = self._extract_with_rebel(text)
            logger.debug(f"[FactExtractor] REBEL extracted {len(rebel_triples)} triples")
            for i, t in enumerate(rebel_triples[:5]):
                logger.debug(f"[FactExtractor] REBEL triple[{i}]: subj='{t[0]}' rel='{t[1]}' obj='{t[2]}' conf={t[3]}")
            triples.extend(rebel_triples)
        else:
            logger.debug("[FactExtractor] Skipping REBEL (disabled or pipeline unavailable)")

        # 2) Regex fallback (top up if fewer than 5)
        if self.use_regex and len(triples) < 5:
            regex_triples = self._extract_with_regex(text)
            logger.debug(f"[FactExtractor] Regex extracted {len(regex_triples)} triples")
            for i, t in enumerate(regex_triples[:5]):
                logger.debug(f"[FactExtractor] Regex triple[{i}]: subj='{t[0]}' rel='{t[1]}' obj='{t[2]}' conf={t[3]}")
            triples.extend(regex_triples)
        elif not self.use_regex:
            logger.debug("[FactExtractor] Regex extraction disabled")

        # 3) Canonicalize + dedupe (augmented with preference logic + generic cleaner & scorer)
        facts: List[MemoryNode] = []
        seen = set()
        kept = 0
        nlp_ref = _NLP  # pass once

        for subj, rel, obj, conf, method in triples:
            # First, normalize subject/object with light heuristics
            s1, r1, o1 = _normalize_subject_obj(subj, rel, obj, user_name="user")

            # Generic cleaning (drop low-signal triples, prefer named spans)
            cleaned = _clean_triple(s1, r1, o1, nlp=nlp_ref)
            if not cleaned:
                continue
            s2, r2, o2 = cleaned

            # Canonicalize preferences (maps to Luke | favorite_* | X or Luke | likes | X when applicable)
            s3, r3, o3 = _canonicalize_preferences(query, response, s2, r2, o2, user_name="user")

            # Final canon for the key (lowercasing etc. as you had)
            subj_c, rel_c, obj_c = self._canon(s3), self._canon(r3), self._canon(o3)
            # Enforce that we only keep facts about the user
            if subj_c != "user":
                continue
            if not subj_c or not rel_c or not obj_c:
                logger.debug(
                    f"[FactExtractor] Dropped empty field triple: subj='{subj}' rel='{rel}' obj='{obj}'"
                )
                continue

            key = f"{subj_c}|{rel_c}|{obj_c}"
            if key in seen:
                logger.debug(f"[FactExtractor] Deduped triple: {key}")
                continue
            seen.add(key)

            # Confidence shaping (generic)
            conf2 = max(conf, _score_triple(subj_c, rel_c, obj_c, nlp=nlp_ref))

            # Preference tagging if the relation indicates a preference
            tags = ["extracted_fact", method]
            if rel_c.startswith("favorite_") or rel_c in {"likes", "dislikes", "prefers"}:
                tags.append("personal_preference")

            node = self._to_node(
                subject=subj_c, relation=rel_c, object=obj_c,
                confidence=conf2, method=method, source_text=text, extra_tags=tags
            )
            facts.append(node)
            kept += 1
            logger.debug(
                f"[FactExtractor] Kept fact[{kept}]: id={node.id} '{node.content}' "
                f"(conf={conf2:.3f}, method={method})"
            )

            if kept >= 8:  # cap per turn (unchanged)
                logger.debug("[FactExtractor] Reached per-turn cap (8 facts); stopping")
                break

        # Post-filter: if a specific name part exists, drop generic name duplicate
        specific_name_vals = {
            (n.metadata.get('subject',''), n.metadata.get('object',''))
            for n in facts
            if n.metadata.get('relation') in {"first_name","middle_name","last_name"}
        }
        if specific_name_vals:
            before = len(facts)
            facts = [
                n for n in facts
                if not (
                    n.metadata.get('relation') == 'name' and
                    (n.metadata.get('subject',''), n.metadata.get('object','')) in specific_name_vals
                )
            ]
            after = len(facts)
            if after != before:
                logger.debug(f"[FactExtractor] Dropped {before - after} generic 'name' facts shadowed by specific name parts")

        # Emit a clear INFO line for each extracted fact so users can see
        # what was captured without enabling DEBUG.
        if facts:
            for i, node in enumerate(facts, start=1):
                meta = node.metadata or {}
                conf = meta.get("confidence")
                try:
                    conf_str = f"{float(conf):.2f}"
                except Exception:
                    conf_str = str(conf)
                method = meta.get("method", "?")
                logger.info(
                    f"[FactExtractor] Fact[{i}] {node.content} "
                    f"(conf={conf_str}, method={method}, id={node.id})"
                )

        logger.info(
            f"[FactExtractor] Final facts: {len(facts)} (from {len(triples)} triples; deduped={len(seen)})"
        )
        return facts

    def _extract_with_spacy_rules(self, text: str) -> List[Tuple[str, str, str, float, str]]:
        if _NLP is None:
            return []
        triples = []
        doc = _NLP(text)

        def tok_text(t):
            return t.text.strip()

        def head_is_nounish(t):
            # Use the syntactic head of the complement span
            h = t.head if t.dep_ in {"attr","acomp","oprd"} else t
            return h.pos_ in {"NOUN","PROPN"}

        # Heuristic: lightweight list of evaluative adjectives to downweight/drop.
        eval_adjs = {"adorable","fun","exciting","fantastic","special","great","nice","cool","awesome"}

        for sent in doc.sents:
            # --- (A) Copular: X is Y  → (X, is_a, Y_head)
            for cop in [t for t in sent if t.dep_ == "cop"]:
                head = cop.head               # predicate head
                subj = next((c for c in head.children if c.dep_ in {"nsubj","nsubj:pass"}), None)
                if not subj:
                    continue
                comp = next((c for c in head.children if c.dep_ in {"attr","oprd","acomp"}), None)
                if not comp:
                    continue

                s = subj.text
                o = comp.text
                # Keep if complement's head is nounish (avoid “is exciting/nice”)
                if head_is_nounish(comp) and len(o) > 1:
                    rel = "is_a"
                else:
                    # Skip clearly evaluative/adj-only statements
                    if comp.pos_ == "ADJ" or o.lower() in eval_adjs:
                        continue
                    rel = "is"

                # Prefer proper-name subjects in span (e.g., “my kitten Flapjack is …”)
                s_candidate = None
                for t in subj.subtree:
                    if t.pos_ == "PROPN":
                        s_candidate = t.text
                        break
                if s_candidate:
                    s = s_candidate

                # Preserve proper-case for PROPN complements (e.g., Skyrim, Flapjack)
                if comp.ent_type_ in {"PERSON","ORG","WORK_OF_ART","PRODUCT"} or comp.pos_ == "PROPN":
                    o = comp.text

                triples.append((s, rel, o, 0.72, "spacy_rules"))

            # --- (B) Possessive NAME:  <X>'s name is Y  → (X, name, Y)
            for name_tok in [t for t in sent if t.lemma_.lower() == "name" and t.pos_ == "NOUN"]:
                poss = next((c for c in name_tok.children if c.dep_ == "poss"), None)
                cop  = next((c for c in name_tok.children if c.dep_ == "cop"), None)
                attr = next((c for c in name_tok.children if c.dep_ in {"attr","acomp","oprd"}), None)

                # Common case: "name" is nsubj, head is copula (ROOT or AUX)
                # e.g., "My cat's name is Flapjack" → name --nsubj--> is --attr--> Flapjack
                if not (cop and attr):
                    head = name_tok.head
                    if head and head != name_tok:
                        # If name is subject of copula, look for attr in head's children
                        if name_tok.dep_ == "nsubj" and head.pos_ in {"AUX", "VERB"}:
                            cop = head  # The head IS the copula
                            attr = next((c for c in head.children if c.dep_ in {"attr","acomp","oprd"}), None)
                        else:
                            # Fallback: original pattern
                            cop  = cop  or next((c for c in head.children if c.dep_ == "cop"), None)
                            attr = attr or next((c for c in head.children if c.dep_ in {"attr","acomp","oprd"}), None)

                if poss and (cop or name_tok.lemma_ == "name") and attr:
                    # Determine if the 'name' has modifiers like 'middle/first/last'
                    mod = None
                    for c in name_tok.children:
                        if c.dep_ in {"amod","compound"}:
                            lem = c.lemma_.lower()
                            if lem in {"first","middle","last"}:
                                mod = lem
                                break

                    rel_name = f"{mod}_name" if mod else "name"

                    # Subject: if possessor is first-person (my/our), use 'user'
                    subj_text = tok_text(poss)
                    subj_lower = subj_text.lower()
                    userish = {"my","our","me","us"}

                    # Special case: "my cat's name" → user | cat_name | …
                    # If possessor is a NOUN and itself is possessed by my/our, attach noun into relation
                    rel_override = None
                    if poss.pos_ == "NOUN":
                        inner_poss = next((c for c in poss.children if c.dep_ == "poss" and c.lemma_.lower() in userish), None)
                        if inner_poss is not None:
                            rel_override = f"{poss.lemma_.lower()}_{rel_name}"
                            subj_text = "user"
                    elif subj_lower in userish:
                        subj_text = "user"

                    obj_text  = tok_text(attr)
                    triples.append((subj_text, rel_override or rel_name, obj_text, 0.80, "spacy_rules"))

            # --- (C) Preference: “my favorite <NOUN> is Y” → (user, likes:<NOUN>, Y)
            for fav in [t for t in sent if t.lemma_.lower() == "favorite" and t.pos_ in {"ADJ"}]:
                head = fav.head
                if head.pos_ != "NOUN":
                    continue
                # Find complement of a copular construction
                cop  = next((c for c in head.children if c.dep_ == "cop"), None)
                attr = next((c for c in head.children if c.dep_ in {"attr","oprd","acomp"}), None)
                if not attr:
                    h2 = head.head
                    if h2:
                        cop  = cop  or next((c for c in h2.children if c.dep_ == "cop"), None)
                        attr = attr or next((c for c in h2.children if c.dep_ in {"attr","oprd","acomp"}), None)
                if attr:
                    cat = head.lemma_.lower()   # e.g., beer, color, game
                    obj = attr.text
                    rel = f"likes_{cat}"
                    triples.append(("user", rel, obj, 0.70, "spacy_rules"))

        return triples

    # ---------------- helpers ----------------

    @log_and_time("Prep Text")
    def _prep_text(self, q: str, r: str, ctx: Optional[List[Dict]]) -> str:
        """Merge query/response labels for parsing. If response is empty, omit its label."""
        base = f"User: {q}"
        if r:
            base += f"\nAssistant: {r}"
        if _NLP is None:
            logger.debug("[FactExtractor] spaCy unavailable; returning base text")
            return base

        # Light normalization for common contractions to improve regex hits
        try:
            import re as _re
            base = _re.sub(r"\bI['’]m\b", "I am", base)
        except Exception:
            pass
        try:
            doc = _NLP(base)
            # If you add a coref component later, apply here (spacy-experimental/coreferee)
            text = doc.text
            logger.debug(
                f"[FactExtractor] spaCy processed text (len={len(text)}) "
                f"(tokens={len(doc)})"
            )
            return text
        except Exception as e:
            logger.warning(f"[FactExtractor] spaCy processing failed: {e}; using base text")
            return base

    @log_and_time("REBEL Extract")
    def _extract_with_rebel(self, text: str) -> List[Tuple[str, str, str, float, str]]:
        """
        Use REBEL to get triples with grounding check against source text.
        Output decoding per REBEL format: <triplet> subject <subj> relation <rel> object <obj>
        """
        if _REBEL is None:
            return []
        try:
            # Hard cap input length to keep within model limits even if upstream changes
            try:
                import os as _os
                max_chars = int(_os.getenv("REBEL_MAX_INPUT_CHARS", "1200"))
            except Exception:
                max_chars = 1200
            safe_text = (text or "")[:max_chars]
            out = _REBEL(
                safe_text,
                return_text=True,
                clean_up_tokenization_spaces=True,
                truncation=True,
                max_length=256,
            )[0].get("generated_text", "")
            out_preview = out[:250].replace("\n", " ")
            logger.debug(f"[FactExtractor] REBEL raw output preview: '{out_preview}'")
        except Exception as e:
            logger.error(f"[FactExtractor] REBEL generation failed: {e}")
            return []

        # Normalize source text for grounding check
        text_lower = (safe_text or "").lower()

        triples: List[Tuple[str, str, str, float, str]] = []
        chunks = out.split("<triplet>")
        parsed = 0
        grounded = 0
        for ch in chunks:
            ch = ch.strip()
            if not ch:
                continue
            try:
                subj = self._between(ch, "subject", "relation")
                rel  = self._between(ch, "relation", "object")
                obj  = ch.split("object")[-1].strip(" <>:;,. \n\t")
                if subj and rel and obj:
                    parsed += 1

                    # Grounding check: both subject and object must appear in source text
                    # (allows for minor normalization differences)
                    subj_lower = subj.lower().strip()
                    obj_lower = obj.lower().strip()

                    subj_grounded = subj_lower in text_lower or any(
                        token in text_lower for token in subj_lower.split() if len(token) > 2
                    )
                    obj_grounded = obj_lower in text_lower or any(
                        token in text_lower for token in obj_lower.split() if len(token) > 2
                    )

                    if not (subj_grounded and obj_grounded):
                        logger.debug(
                            f"[FactExtractor] REBEL dropped ungrounded triple: "
                            f"subj='{subj}' (grounded={subj_grounded}) "
                            f"obj='{obj}' (grounded={obj_grounded})"
                        )
                        continue

                    grounded += 1
                    triples.append((subj, rel, obj, 0.75, "rebel"))
            except Exception:
                continue
        logger.debug(
            f"[FactExtractor] REBEL parsed {parsed} triples, "
            f"{grounded} passed grounding check ({len(chunks)} chunks)"
        )
        return triples

    def _between(self, s: str, a: str, b: str) -> str:
        if a not in s or b not in s:
            return ""
        seg = s.split(a, 1)[1]
        seg = seg.split(b, 1)[0]
        return seg.strip(" <>:;,. \n\t")

    @log_and_time("Regex Extract")
    def _extract_with_regex(self, text: str) -> List[Tuple[str, str, str, float, str]]:
        """Apply regexes line‑by‑line to avoid crossing speaker boundaries.
        Includes generic patterns and lifting‑specific patterns.
        """
        found: List[Tuple[str, str, str, float, str]] = []

        # Work line‑wise to ensure EOL acts as a terminator
        lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
        # Strip speaker prefixes where present
        def strip_prefix(ln: str) -> str:
            if ln.lower().startswith("user:"):
                return ln.split(":", 1)[1].strip()
            if ln.lower().startswith("assistant:"):
                return ln.split(":", 1)[1].strip()
            return ln

        norm_lines = [strip_prefix(ln) for ln in lines]

        # 1) Generic patterns
        for pi, pattern in enumerate(self.fact_patterns):
            total_matches = 0
            for ln in norm_lines:
                try:
                    matches = re.findall(pattern, ln, flags=re.IGNORECASE)
                except Exception as e:
                    logger.warning(f"[FactExtractor] Regex pattern error idx={pi}: {e}")
                    continue
                total_matches += len(matches)
                for mi, m in enumerate(matches[:4]):
                    if isinstance(m, tuple):
                        if len(m) == 2:
                            # Pattern index 2 is (my|your|our) <attr> is <val>
                            if pi == 2:
                                attr, obj = m
                                subj, rel = "user", (attr or "").strip()
                            else:
                                subj, obj = m
                                rel = "is"
                        else:
                            subj, rel, obj = m[0], "is", " ".join(m[1:])
                    else:
                        subj, rel, obj = "user", "likes", m

                    # Normalize
                    subj = (subj or "").strip()
                    obj = (obj if isinstance(obj, str) else str(obj)).strip()
                    # If rel came from attribute phrase, normalize to underscored
                    rel = re.sub(r"\s+", "_", (rel or "").strip().lower())

                    # Map 1st‑person to stable subject
                    if subj.lower() in {"i", "me", "my", "we", "us"}:
                        subj = "user"

                    # Trim temporal adverbs in simple copular claims
                    if rel in {"is", "is_a"}:
                        obj = re.sub(r"\b(?:currently|right\s+now|today|presently|now)\b\.?$", "", obj, flags=re.IGNORECASE).strip()

                    if 4 < len(rel) + len(obj) < 200:
                        cleaned = _clean_triple(subj, rel, obj, nlp=_NLP)
                        if not cleaned:
                            continue
                        subj2, rel2, obj2 = cleaned

                        # Filter clause-like objects (infinitives, gerunds) for non-attribute relations
                        if rel2 in {"is", "is_a", "likes", "loves", "prefers"} and _is_clause_like(obj2):
                            logger.debug(f"[FactExtractor] Regex dropped clause-like object: rel='{rel2}' obj='{obj2}'")
                            continue

                        found.append((subj2, rel2, obj2, 0.6, "regex"))
                        logger.debug(f"[FactExtractor] Regex kept m[{mi}] -> subj='{subj2}' rel='{rel2}' obj='{obj2}'")
                    else:
                        logger.debug(f"[FactExtractor] Regex dropped m[{mi}] (length/rel bounds)")
            logger.debug(f"[FactExtractor] Pattern[{pi}] matched {total_matches}")

        # 1b) Possessive attribute patterns (map to user | <noun>_<attr> | value)
        for li, pattern in enumerate(self.possessive_attr_patterns):
            total_matches = 0
            for ln in norm_lines:
                try:
                    matches = re.findall(pattern, ln, flags=re.IGNORECASE)
                except Exception as e:
                    logger.warning(f"[FactExtractor] PossAttr regex error idx={li}: {e}")
                    continue
                total_matches += len(matches)
                for m in matches[:4]:
                    if not isinstance(m, tuple) or len(m) < 3:
                        continue
                    noun, attr, val = m[0], m[1], m[2]
                    noun = (noun or '').strip().lower().rstrip('s')
                    # Convert multi-token attribute to underscored
                    attr = re.sub(r"\s+", "_", (attr or '').strip().lower())
                    # Build relation
                    rel = f"{noun}_{attr}" if noun else attr
                    # Stop value at common conjunctions/punctuation
                    val = (val or '').strip()
                    val = re.split(r"\s+(?:and|but|however)\b|[,;]", val, maxsplit=1)[0].strip()

                    # Build canonical triple: user | rel | val
                    cleaned = _clean_triple('user', rel, val, nlp=_NLP)
                    if not cleaned:
                        continue
                    sj, rl, ob = cleaned
                    found.append((sj, rl, ob, 0.78, "regex"))
            if total_matches:
                logger.debug(f"[FactExtractor] PossAttrPattern[{li}] matched {total_matches}")

        # 1c) Two-word user attributes (e.g., "my middle name is Andrew")
        for li, pattern in enumerate(self.two_word_attr_patterns):
            total_matches = 0
            for ln in norm_lines:
                try:
                    matches = re.findall(pattern, ln, flags=re.IGNORECASE)
                except Exception as e:
                    logger.warning(f"[FactExtractor] TwoWordAttr regex error idx={li}: {e}")
                    continue
                total_matches += len(matches)
                for m in matches[:4]:
                    if not isinstance(m, tuple) or len(m) < 2:
                        continue
                    attr, val = m[0], m[1]
                    rel = re.sub(r"\s+", "_", (attr or '').strip().lower())
                    val = (val or '').strip()
                    val = re.split(r"\s+(?:and|but|however)\b|[,;]", val, maxsplit=1)[0].strip()

                    cleaned = _clean_triple('user', rel, val, nlp=_NLP)
                    if not cleaned:
                        continue
                    sj, rl, ob = cleaned
                    found.append((sj, rl, ob, 0.78, "regex"))
            if total_matches:
                logger.debug(f"[FactExtractor] TwoWordAttrPattern[{li}] matched {total_matches}")

        # 1d) Employment/occupation statements
        for ei, (rel_key, pattern) in enumerate(self.employment_patterns):
            emp_total = 0
            for ln in norm_lines:
                try:
                    matches = re.findall(pattern, ln, flags=re.IGNORECASE)
                except Exception as e:
                    logger.warning(f"[FactExtractor] Employment regex error idx={ei}: {e}")
                    continue
                # re.findall may return list of tuples or strings depending on grouping
                if not isinstance(matches, list):
                    continue
                emp_total += len(matches)
                for m in matches[:4]:
                    val = m[0] if isinstance(m, tuple) else m
                    val = (val or '').strip()
                    # Trim common temporal/qualifier tails
                    val = re.split(r"\b(?:right\s+now|currently|today)\b", val, maxsplit=1, flags=re.IGNORECASE)[0].strip()
                    # Drop leading articles
                    val = re.sub(r"^(?:the|a|an)\s+", "", val, flags=re.IGNORECASE)
                    # Stop value at conjunctions/punctuation just in case
                    val = re.split(r"\s+(?:and|but|however)\b|[,;]", val, maxsplit=1)[0].strip()

                    cleaned = _clean_triple('user', rel_key, val, nlp=_NLP)
                    if not cleaned:
                        continue
                    sj, rl, ob = cleaned
                    found.append((sj, rl, ob, 0.78, "regex"))
            if emp_total:
                logger.debug(f"[FactExtractor] EmploymentPattern[{ei}:{rel_key}] matched {emp_total}")

        # 2) Lifting patterns → canonical relation names
        def lift_rel(verb_or_noun: str) -> str:
            v = (verb_or_noun or "").lower()
            if v.startswith("squat") or v == "squatted":
                return "squat_max"
            if v.startswith("bench") or v == "benched":
                return "bench_max"
            if v.startswith("deadlift") or v == "deadlifted":
                return "deadlift_max"
            if v.startswith("ohp") or v.startswith("overhead") or v == "pressed":
                return "ohp_max"
            return f"lift_{v}"

        for li, pattern in enumerate(self.lift_patterns):
            lm_total = 0
            for ln in norm_lines:
                try:
                    matches = re.findall(pattern, ln, flags=re.IGNORECASE)
                except Exception as e:
                    logger.warning(f"[FactExtractor] Lift regex error idx={li}: {e}")
                    continue
                lm_total += len(matches)
                for m in matches[:4]:
                    # m can be (type, value, unit) or (verb, value, unit)
                    kind, value, unit = (m[0], m[1], (m[2] or "").lower()) if isinstance(m, tuple) else ("", str(m), "")
                    rel = lift_rel(kind)
                    # Normalize unit
                    unit_norm = "kg" if unit.startswith("kg") else ("lb" if unit else "")
                    obj = f"{value} {unit_norm}".strip()
                    subj = "user"

                    cleaned = _clean_triple(subj, rel, obj, nlp=_NLP)
                    if not cleaned:
                        continue
                    subj2, rel2, obj2 = cleaned
                    found.append((subj2, rel2, obj2, 0.72, "regex"))
            if lm_total:
                logger.debug(f"[FactExtractor] LiftPattern[{li}] matched {lm_total}")

        return found

    def _canon(self, s: str) -> str:
        return re.sub(r"\s+", " ", (s or "").strip()).strip(" .,:;\"'`").lower()

    def _safe_type(self):
        """Return MemoryType.FACT if available; fallback to string 'fact'."""
        try:
            return MemoryType.FACT
        except Exception:
            return "fact"

    def _to_node(
        self,
        subject: str,
        relation: str,
        object: str,
        confidence: float,
        method: str,
        source_text: str,
        extra_tags: Optional[List[str]] = None,
    ) -> MemoryNode:
        # Merge base + extra tags
        tags = (extra_tags or [])[:]
        if "extracted_fact" not in tags:
            tags.insert(0, "extracted_fact")
        if method and method not in tags:
            tags.append(method)

        node = MemoryNode(
            id=str(uuid.uuid4()),
            content=f"{subject} | {relation} | {object}",
            type=self._safe_type(),  # safe enum/string
            timestamp=datetime.now(),
            importance_score=0.6,
            tags=tags,
            metadata={
                "subject": subject,
                "relation": relation,
                "object": object,
                "confidence": float(confidence),
                "method": method,
                "source_excerpt": source_text[:400],
            }
        )
        logger.debug(
            f"[FactExtractor] Built MemoryNode(id={node.id}, type=FACT, "
            f"conf={confidence}, method={method})"
        )
        return node
