"""
Query corpus for prompt ablation testing.

A structured set of queries spanning the 9 IntentType categories and 4 tone
levels, each with expected behavior tags for downstream judging.

Inputs:
    - IntentType from core/intent_classifier.py (9 types)
    - Tone levels: HIGH, MEDIUM, CONCERN, CONVERSATIONAL
    - Manual query entries
Outputs:
    - eval/corpus.json -- labeled query sets with coverage metadata
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from eval.section_registry import SECTION_REGISTRY

# Import the real IntentType enum from the classifier
from core.intent_classifier import IntentType

# Tone levels matching utils/tone_detector.py
TONE_LEVELS = ("HIGH", "MEDIUM", "CONCERN", "CONVERSATIONAL")


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class ExpectedBehavior:
    """What a good response to this query should look like."""

    should_reference_memory: bool = False
    should_reference_personal: bool = False
    should_reference_docs: bool = False
    should_be_concise: bool = False
    should_be_thorough: bool = False
    expected_sections_used: List[str] = field(default_factory=list)
    must_not_include: List[str] = field(default_factory=list)
    max_expected_tokens: Optional[int] = None
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "should_reference_memory": self.should_reference_memory,
            "should_reference_personal": self.should_reference_personal,
            "should_reference_docs": self.should_reference_docs,
            "should_be_concise": self.should_be_concise,
            "should_be_thorough": self.should_be_thorough,
            "expected_sections_used": self.expected_sections_used,
            "must_not_include": self.must_not_include,
            "max_expected_tokens": self.max_expected_tokens,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ExpectedBehavior:
        return cls(**d)


@dataclass
class CorpusQuery:
    """One query in the ablation test corpus."""

    query_id: str
    query_text: str
    intent: IntentType
    tone: str  # HIGH, MEDIUM, CONCERN, CONVERSATIONAL
    expected: ExpectedBehavior = field(default_factory=ExpectedBehavior)
    tags: List[str] = field(default_factory=list)
    snapshot_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_id": self.query_id,
            "query_text": self.query_text,
            "intent": self.intent.value,
            "tone": self.tone,
            "expected": self.expected.to_dict(),
            "tags": self.tags,
            "snapshot_id": self.snapshot_id,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> CorpusQuery:
        d = dict(d)
        d["intent"] = IntentType(d["intent"])
        d["expected"] = ExpectedBehavior.from_dict(d["expected"])
        return cls(**d)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

_VALID_SECTION_KEYS = set(SECTION_REGISTRY.keys())


def validate_section_keys(keys: List[str]) -> List[str]:
    """Return any keys not found in SECTION_REGISTRY."""
    return [k for k in keys if k not in _VALID_SECTION_KEYS]


# ---------------------------------------------------------------------------
# Seed corpus: 27 queries, 3 per intent type
# ---------------------------------------------------------------------------

SEED_CORPUS: List[Dict[str, Any]] = [
    # --- FACTUAL_RECALL (3) ---
    {
        "query_id": "fact_001",
        "query_text": "What is my brother's name?",
        "intent": "factual_recall",
        "tone": "CONVERSATIONAL",
        "expected": {
            "should_reference_memory": True,
            "should_reference_personal": True,
            "should_be_concise": True,
            "should_be_thorough": False,
            "should_reference_docs": False,
            "expected_sections_used": ["memories", "user_profile"],
            "must_not_include": [],
            "max_expected_tokens": 100,
            "notes": "Should retrieve stored personal facts",
        },
        "tags": ["personal", "recall", "simple"],
    },
    {
        "query_id": "fact_002",
        "query_text": "What do you know about my job?",
        "intent": "factual_recall",
        "tone": "CONVERSATIONAL",
        "expected": {
            "should_reference_memory": True,
            "should_reference_personal": True,
            "should_be_concise": False,
            "should_be_thorough": True,
            "should_reference_docs": False,
            "expected_sections_used": ["memories", "user_profile"],
            "must_not_include": [],
            "max_expected_tokens": None,
            "notes": "Should synthesize across multiple stored facts",
        },
        "tags": ["personal", "recall", "synthesis"],
    },
    {
        "query_id": "fact_003",
        "query_text": "What's my cat's name and what does he look like?",
        "intent": "factual_recall",
        "tone": "CONVERSATIONAL",
        "expected": {
            "should_reference_memory": True,
            "should_reference_personal": True,
            "should_be_concise": False,
            "should_be_thorough": False,
            "should_reference_docs": False,
            "expected_sections_used": ["memories", "user_profile"],
            "must_not_include": [],
            "max_expected_tokens": None,
            "notes": "Multi-part factual recall",
        },
        "tags": ["personal", "recall", "multi-part"],
    },

    # --- TEMPORAL_RECALL (3) ---
    {
        "query_id": "temp_001",
        "query_text": "What did I tell you about my job last week?",
        "intent": "temporal_recall",
        "tone": "CONVERSATIONAL",
        "expected": {
            "should_reference_memory": True,
            "should_reference_personal": False,
            "should_be_concise": False,
            "should_be_thorough": True,
            "should_reference_docs": False,
            "expected_sections_used": ["memories", "recent_summaries"],
            "must_not_include": [],
            "max_expected_tokens": None,
            "notes": "Requires temporal filtering of memories",
        },
        "tags": ["temporal", "recall", "recency"],
    },
    {
        "query_id": "temp_002",
        "query_text": "When did I first mention Flapjack?",
        "intent": "temporal_recall",
        "tone": "CONVERSATIONAL",
        "expected": {
            "should_reference_memory": True,
            "should_reference_personal": False,
            "should_be_concise": True,
            "should_be_thorough": False,
            "should_reference_docs": False,
            "expected_sections_used": ["memories", "semantic_summaries"],
            "must_not_include": [],
            "max_expected_tokens": 150,
            "notes": "Requires earliest-mention retrieval",
        },
        "tags": ["temporal", "recall", "origin"],
    },
    {
        "query_id": "temp_003",
        "query_text": "What were we talking about yesterday?",
        "intent": "temporal_recall",
        "tone": "CONVERSATIONAL",
        "expected": {
            "should_reference_memory": True,
            "should_reference_personal": False,
            "should_be_concise": False,
            "should_be_thorough": True,
            "should_reference_docs": False,
            "expected_sections_used": ["recent_conversation", "recent_summaries"],
            "must_not_include": [],
            "max_expected_tokens": None,
            "notes": "Recent session recall",
        },
        "tags": ["temporal", "recall", "session"],
    },

    # --- EMOTIONAL_SUPPORT (3) ---
    {
        "query_id": "emo_001",
        "query_text": "I feel really anxious about work tomorrow",
        "intent": "emotional_support",
        "tone": "CONCERN",
        "expected": {
            "should_reference_memory": True,
            "should_reference_personal": True,
            "should_be_concise": False,
            "should_be_thorough": False,
            "should_reference_docs": False,
            "expected_sections_used": ["user_profile", "memories"],
            "must_not_include": [],
            "max_expected_tokens": None,
            "notes": "Should acknowledge feeling without being saccharine",
        },
        "tags": ["anxiety", "work", "support"],
    },
    {
        "query_id": "emo_002",
        "query_text": "I don't think I can do this anymore, everything feels pointless",
        "intent": "emotional_support",
        "tone": "HIGH",
        "expected": {
            "should_reference_memory": True,
            "should_reference_personal": True,
            "should_be_concise": False,
            "should_be_thorough": False,
            "should_reference_docs": False,
            "expected_sections_used": ["user_profile", "memories"],
            "must_not_include": [],
            "max_expected_tokens": None,
            "notes": "Crisis-level tone. Should trigger escalation tracker.",
        },
        "tags": ["crisis", "support", "escalation"],
    },
    {
        "query_id": "emo_003",
        "query_text": "Everything feels like it's falling apart right now",
        "intent": "emotional_support",
        "tone": "HIGH",
        "expected": {
            "should_reference_memory": True,
            "should_reference_personal": True,
            "should_be_concise": False,
            "should_be_thorough": False,
            "should_reference_docs": False,
            "expected_sections_used": ["user_profile", "memories"],
            "must_not_include": [],
            "max_expected_tokens": None,
            "notes": "Second HIGH tone query for coverage",
        },
        "tags": ["crisis", "support", "escalation"],
    },

    # --- CASUAL_SOCIAL (3) ---
    {
        "query_id": "casual_001",
        "query_text": "Hey man",
        "intent": "casual_social",
        "tone": "CONVERSATIONAL",
        "expected": {
            "should_reference_memory": False,
            "should_reference_personal": False,
            "should_be_concise": True,
            "should_be_thorough": False,
            "should_reference_docs": False,
            "expected_sections_used": [],
            "must_not_include": [],
            "max_expected_tokens": 100,
            "notes": "Minimal greeting, should not trigger heavy retrieval",
        },
        "tags": ["greeting", "minimal"],
    },
    {
        "query_id": "casual_002",
        "query_text": "Just got back from the gym, absolutely wrecked",
        "intent": "casual_social",
        "tone": "CONVERSATIONAL",
        "expected": {
            "should_reference_memory": False,
            "should_reference_personal": False,
            "should_be_concise": True,
            "should_be_thorough": False,
            "should_reference_docs": False,
            "expected_sections_used": [],
            "must_not_include": [],
            "max_expected_tokens": 150,
            "notes": "Daily check-in, brief response",
        },
        "tags": ["daily", "health"],
    },
    {
        "query_id": "casual_003",
        "query_text": "Lol that's hilarious",
        "intent": "casual_social",
        "tone": "CONVERSATIONAL",
        "expected": {
            "should_reference_memory": False,
            "should_reference_personal": False,
            "should_be_concise": True,
            "should_be_thorough": False,
            "should_reference_docs": False,
            "expected_sections_used": ["recent_conversation"],
            "must_not_include": [],
            "max_expected_tokens": 100,
            "notes": "Reaction to previous message, needs recent conversation context",
        },
        "tags": ["reaction", "continuity"],
    },

    # --- TECHNICAL_HELP (3) ---
    {
        "query_id": "tech_001",
        "query_text": "How does the memory scoring system work?",
        "intent": "technical_help",
        "tone": "CONVERSATIONAL",
        "expected": {
            "should_reference_memory": False,
            "should_reference_personal": False,
            "should_be_concise": False,
            "should_be_thorough": True,
            "should_reference_docs": True,
            "expected_sections_used": ["reference_docs", "memories"],
            "must_not_include": [],
            "max_expected_tokens": None,
            "notes": "Architecture question about Daemon internals",
        },
        "tags": ["architecture", "memory", "scoring"],
    },
    {
        "query_id": "tech_002",
        "query_text": "I'm getting a dimension mismatch error in ChromaDB, any ideas?",
        "intent": "technical_help",
        "tone": "CONVERSATIONAL",
        "expected": {
            "should_reference_memory": False,
            "should_reference_personal": False,
            "should_be_concise": False,
            "should_be_thorough": True,
            "should_reference_docs": False,
            "expected_sections_used": [],
            "must_not_include": [],
            "max_expected_tokens": None,
            "notes": "Debugging help, should not invent fixes",
        },
        "tags": ["debugging", "chromadb", "embeddings"],
    },
    {
        "query_id": "tech_003",
        "query_text": "Walk me through how to add a new retrieval source to the pipeline",
        "intent": "technical_help",
        "tone": "CONVERSATIONAL",
        "expected": {
            "should_reference_memory": False,
            "should_reference_personal": False,
            "should_be_concise": False,
            "should_be_thorough": True,
            "should_reference_docs": True,
            "expected_sections_used": ["reference_docs"],
            "must_not_include": [],
            "max_expected_tokens": None,
            "notes": "Step-by-step instruction request",
        },
        "tags": ["architecture", "howto", "extending"],
    },

    # --- CREATIVE_EXPLORATION (3) ---
    {
        "query_id": "creative_001",
        "query_text": "What if I built a language learning app that used spaced repetition?",
        "intent": "creative_exploration",
        "tone": "CONVERSATIONAL",
        "expected": {
            "should_reference_memory": False,
            "should_reference_personal": False,
            "should_be_concise": False,
            "should_be_thorough": True,
            "should_reference_docs": False,
            "expected_sections_used": [],
            "must_not_include": [],
            "max_expected_tokens": None,
            "notes": "Ideation, should explore the concept",
        },
        "tags": ["ideation", "apps", "exploration"],
    },
    {
        "query_id": "creative_002",
        "query_text": "I've been thinking about writing a novel set in a world where AI is sentient",
        "intent": "creative_exploration",
        "tone": "CONVERSATIONAL",
        "expected": {
            "should_reference_memory": True,
            "should_reference_personal": True,
            "should_be_concise": False,
            "should_be_thorough": True,
            "should_reference_docs": False,
            "expected_sections_used": ["memories", "user_profile"],
            "must_not_include": [],
            "max_expected_tokens": None,
            "notes": "Personal creative project, should reference user's interests",
        },
        "tags": ["writing", "creative", "personal"],
    },
    {
        "query_id": "creative_003",
        "query_text": "I've been thinking about whether I'm spending my time right lately",
        "intent": "creative_exploration",
        "tone": "CONCERN",
        "expected": {
            "should_reference_memory": True,
            "should_reference_personal": True,
            "should_be_concise": False,
            "should_be_thorough": False,
            "should_reference_docs": False,
            "expected_sections_used": ["memories", "user_profile"],
            "must_not_include": [],
            "max_expected_tokens": None,
            "notes": "Reflective/philosophical, needs personal context",
        },
        "tags": ["life", "priorities", "reflection"],
    },

    # --- META_CONVERSATIONAL (3) ---
    {
        "query_id": "meta_001",
        "query_text": "What tools do you have available?",
        "intent": "meta_conversational",
        "tone": "CONVERSATIONAL",
        "expected": {
            "should_reference_memory": False,
            "should_reference_personal": False,
            "should_be_concise": False,
            "should_be_thorough": True,
            "should_reference_docs": True,
            "expected_sections_used": ["reference_docs"],
            "must_not_include": [],
            "max_expected_tokens": None,
            "notes": "Self-knowledge question about Daemon capabilities",
        },
        "tags": ["capabilities", "tools"],
    },
    {
        "query_id": "meta_002",
        "query_text": "How does your memory retrieval work exactly?",
        "intent": "meta_conversational",
        "tone": "CONVERSATIONAL",
        "expected": {
            "should_reference_memory": False,
            "should_reference_personal": False,
            "should_be_concise": False,
            "should_be_thorough": True,
            "should_reference_docs": True,
            "expected_sections_used": ["reference_docs"],
            "must_not_include": [],
            "max_expected_tokens": None,
            "notes": "Architecture introspection",
        },
        "tags": ["architecture", "memory", "explanation"],
    },
    {
        "query_id": "meta_003",
        "query_text": "Do you have a visual memory?",
        "intent": "meta_conversational",
        "tone": "CONVERSATIONAL",
        "expected": {
            "should_reference_memory": True,
            "should_reference_personal": False,
            "should_be_concise": False,
            "should_be_thorough": True,
            "should_reference_docs": True,
            "expected_sections_used": ["reference_docs", "memories"],
            "must_not_include": [],
            "max_expected_tokens": None,
            "notes": "Capability question that may trigger memory search for past discussion",
        },
        "tags": ["capabilities", "visual", "self-knowledge"],
    },

    # --- PROJECT_WORK (3) ---
    {
        "query_id": "proj_001",
        "query_text": "What should I build next after the eval system?",
        "intent": "project_work",
        "tone": "CONVERSATIONAL",
        "expected": {
            "should_reference_memory": True,
            "should_reference_personal": False,
            "should_be_concise": False,
            "should_be_thorough": True,
            "should_reference_docs": True,
            "expected_sections_used": ["proposed_features", "proactive_insights"],
            "must_not_include": [],
            "max_expected_tokens": None,
            "notes": "Roadmap planning, needs project context",
        },
        "tags": ["roadmap", "features", "planning"],
    },
    {
        "query_id": "proj_002",
        "query_text": "I want to add visual memory to Daemon. What would that take?",
        "intent": "project_work",
        "tone": "CONVERSATIONAL",
        "expected": {
            "should_reference_memory": True,
            "should_reference_personal": False,
            "should_be_concise": False,
            "should_be_thorough": True,
            "should_reference_docs": True,
            "expected_sections_used": ["reference_docs", "proposed_features"],
            "must_not_include": [],
            "max_expected_tokens": None,
            "notes": "Feature architecture design",
        },
        "tags": ["features", "architecture", "visual"],
    },
    {
        "query_id": "proj_003",
        "query_text": "Can you check the status of the synthesis pipeline?",
        "intent": "project_work",
        "tone": "CONVERSATIONAL",
        "expected": {
            "should_reference_memory": True,
            "should_reference_personal": False,
            "should_be_concise": False,
            "should_be_thorough": True,
            "should_reference_docs": True,
            "expected_sections_used": ["reference_docs", "memories"],
            "must_not_include": [],
            "max_expected_tokens": None,
            "notes": "Project status query, needs both docs and memory",
        },
        "tags": ["status", "synthesis", "pipeline"],
    },

    # --- GENERAL (3) ---
    {
        "query_id": "gen_001",
        "query_text": "What's happening with the US economy right now?",
        "intent": "general",
        "tone": "CONVERSATIONAL",
        "expected": {
            "should_reference_memory": False,
            "should_reference_personal": False,
            "should_be_concise": False,
            "should_be_thorough": True,
            "should_reference_docs": False,
            "expected_sections_used": ["web_search_results"],
            "must_not_include": [],
            "max_expected_tokens": None,
            "notes": "Current events, should trigger web search",
        },
        "tags": ["current_events", "economics"],
    },
    {
        "query_id": "gen_002",
        "query_text": "Explain how transformer attention works",
        "intent": "general",
        "tone": "CONVERSATIONAL",
        "expected": {
            "should_reference_memory": False,
            "should_reference_personal": False,
            "should_be_concise": False,
            "should_be_thorough": True,
            "should_reference_docs": False,
            "expected_sections_used": ["wiki", "semantic_chunks"],
            "must_not_include": [],
            "max_expected_tokens": None,
            "notes": "Knowledge question, may use wiki or web",
        },
        "tags": ["ml", "transformers", "explanation"],
    },
    {
        "query_id": "gen_003",
        "query_text": "Can you help me with something?",
        "intent": "general",
        "tone": "CONVERSATIONAL",
        "expected": {
            "should_reference_memory": False,
            "should_reference_personal": False,
            "should_be_concise": True,
            "should_be_thorough": False,
            "should_reference_docs": False,
            "expected_sections_used": [],
            "must_not_include": [],
            "max_expected_tokens": 100,
            "notes": "Vague opener, should ask for clarification",
        },
        "tags": ["vague", "opener"],
    },
]


# ---------------------------------------------------------------------------
# Corpus Manager
# ---------------------------------------------------------------------------

class CorpusManager:
    """Manage the query corpus -- load, expand, associate with snapshots."""

    def __init__(self, corpus_path: Path = Path("eval/corpus.json")) -> None:
        self.corpus_path = corpus_path
        self.queries: Dict[str, CorpusQuery] = {}

        if self.corpus_path.exists():
            self.load()
        else:
            self._seed()

    def _seed(self) -> None:
        """Initialize with seed corpus, validating section keys."""
        for qdata in SEED_CORPUS:
            expected_data = dict(qdata["expected"])
            bad_keys = validate_section_keys(
                expected_data.get("expected_sections_used", [])
            )
            if bad_keys:
                raise ValueError(
                    f"Query '{qdata['query_id']}' references unknown section keys: "
                    f"{bad_keys}. Valid keys: {sorted(_VALID_SECTION_KEYS)}"
                )

            expected = ExpectedBehavior.from_dict(expected_data)
            query = CorpusQuery(
                query_id=qdata["query_id"],
                query_text=qdata["query_text"],
                intent=IntentType(qdata["intent"]),
                tone=qdata.get("tone", "CONVERSATIONAL"),
                expected=expected,
                tags=qdata.get("tags", []),
            )
            self.queries[query.query_id] = query
        self.save()

    def add_query(
        self,
        query_id: str,
        query_text: str,
        intent: IntentType,
        tone: str = "CONVERSATIONAL",
        expected: Optional[ExpectedBehavior] = None,
        tags: Optional[List[str]] = None,
    ) -> CorpusQuery:
        """Add a query to the corpus."""
        if expected is not None:
            bad_keys = validate_section_keys(expected.expected_sections_used)
            if bad_keys:
                raise ValueError(
                    f"Unknown section keys: {bad_keys}. "
                    f"Valid keys: {sorted(_VALID_SECTION_KEYS)}"
                )

        query = CorpusQuery(
            query_id=query_id,
            query_text=query_text,
            intent=intent,
            tone=tone,
            expected=expected or ExpectedBehavior(),
            tags=tags or [],
        )
        self.queries[query_id] = query
        self.save()
        return query

    def link_snapshot(self, query_id: str, snapshot_id: str) -> None:
        """Associate a captured snapshot with a corpus query."""
        if query_id in self.queries:
            self.queries[query_id].snapshot_id = snapshot_id
            self.save()

    def get_by_intent(self, intent: IntentType) -> List[CorpusQuery]:
        """Return queries matching the given intent."""
        return [q for q in self.queries.values() if q.intent == intent]

    def get_by_tone(self, tone: str) -> List[CorpusQuery]:
        """Return queries matching the given tone level."""
        return [q for q in self.queries.values() if q.tone == tone]

    def get_intent_coverage(self) -> Dict[str, int]:
        """Intent coverage counts."""
        counts: Dict[str, int] = {}
        for q in self.queries.values():
            counts[q.intent.value] = counts.get(q.intent.value, 0) + 1
        return counts

    def get_tone_coverage(self) -> Dict[str, int]:
        """Tone coverage counts."""
        counts: Dict[str, int] = {}
        for q in self.queries.values():
            counts[q.tone] = counts.get(q.tone, 0) + 1
        return counts

    def get_intent_gaps(self, min_per_intent: int = 3) -> List[str]:
        """Intents with fewer than min_per_intent queries."""
        coverage = self.get_intent_coverage()
        return [
            intent.value
            for intent in IntentType
            if coverage.get(intent.value, 0) < min_per_intent
        ]

    def get_tone_gaps(self, min_per_tone: int = 2) -> List[str]:
        """Tone levels with fewer than min_per_tone queries."""
        coverage = self.get_tone_coverage()
        return [
            tone for tone in TONE_LEVELS
            if coverage.get(tone, 0) < min_per_tone
        ]

    def save(self) -> None:
        """Save corpus to JSON."""
        data = {
            "queries": [q.to_dict() for q in self.queries.values()],
            "intent_coverage": self.get_intent_coverage(),
            "tone_coverage": self.get_tone_coverage(),
            "intent_gaps": self.get_intent_gaps(),
            "tone_gaps": self.get_tone_gaps(),
        }
        self.corpus_path.parent.mkdir(parents=True, exist_ok=True)
        self.corpus_path.write_text(json.dumps(data, indent=2))

    def load(self) -> None:
        """Load corpus from JSON."""
        data = json.loads(self.corpus_path.read_text())
        self.queries.clear()
        for qdata in data["queries"]:
            query = CorpusQuery.from_dict(qdata)
            self.queries[query.query_id] = query
