"""
config/schema.py

Pydantic v2 schema validation for config.yaml.

- Validates the full config dict at startup (after YAML load + resolve_vars + ensure_defaults).
- Returns the dict unchanged on success; exits with actionable errors on failure.
- All section models use extra="ignore" to allow forward-compatible config growth.
- Cross-field validators warn (log) rather than fail where appropriate.
"""

import sys
import logging
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

logger = logging.getLogger("config.schema")


# =============================================================================
# Sub-models (nested dicts with typed fields)
# =============================================================================

class CollectionBoosts(BaseModel):
    model_config = ConfigDict(extra="ignore")
    facts: float = Field(default=0.15, ge=0.0, le=1.0)
    summaries: float = Field(default=0.10, ge=0.0, le=1.0)
    conversations: float = Field(default=0.0, ge=0.0, le=1.0)
    semantic: float = Field(default=0.05, ge=0.0, le=1.0)
    wiki: float = Field(default=0.05, ge=0.0, le=1.0)


class ScoreWeights(BaseModel):
    model_config = ConfigDict(extra="ignore")
    relevance: float = Field(default=0.35, ge=0.0, le=1.0)
    recency: float = Field(default=0.25, ge=0.0, le=1.0)
    truth: float = Field(default=0.20, ge=0.0, le=1.0)
    importance: float = Field(default=0.05, ge=0.0, le=1.0)
    continuity: float = Field(default=0.10, ge=0.0, le=1.0)
    structure: float = Field(default=0.05, ge=0.0, le=1.0)
    topic_match: float = Field(default=0.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def check_sum(self) -> "ScoreWeights":
        total = (self.relevance + self.recency + self.truth +
                 self.importance + self.continuity + self.structure + self.topic_match)
        if abs(total - 1.0) > 0.05:
            logger.warning(f"score_weights sum to {total:.3f} (expected ~1.0)")
        return self


class SynthesisWeights(BaseModel):
    model_config = ConfigDict(extra="ignore")
    coherence: float = Field(default=0.35, ge=0.0, le=1.0)
    novelty: float = Field(default=0.60, ge=0.0, le=1.0)
    distance: float = Field(default=0.05, ge=0.0, le=1.0)
    structural: float = Field(default=0.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def check_sum(self) -> "SynthesisWeights":
        total = self.coherence + self.novelty + self.distance + self.structural
        if abs(total - 1.0) > 0.05:
            logger.warning(f"synthesis.weights sum to {total:.3f} (expected ~1.0)")
        return self


class BestOfSelectorWeights(BaseModel):
    model_config = ConfigDict(extra="ignore")
    heuristic: float = Field(default=1.0, ge=0.0)
    llm: float = Field(default=0.0, ge=0.0)


class TruthSourceScores(BaseModel):
    model_config = ConfigDict(extra="ignore")
    user_stated: float = Field(default=0.8, ge=0.0, le=1.0)
    corrected: float = Field(default=0.85, ge=0.0, le=1.0)
    llm_extracted: float = Field(default=0.7, ge=0.0, le=1.0)
    inferred: float = Field(default=0.5, ge=0.0, le=1.0)


class SystemPromptFile(BaseModel):
    model_config = ConfigDict(extra="ignore")
    name: str = "daemon"
    system_prompt_file: str = ""


class DefaultCoreDirective(BaseModel):
    model_config = ConfigDict(extra="ignore")
    query: str = "[CORE DIRECTIVE]"
    response: str = ""
    timestamp: str = ""
    tags: List[str] = Field(default_factory=lambda: ["@seed", "core", "directive", "safety"])


# =============================================================================
# Section models
# =============================================================================

class DaemonSection(BaseModel):
    model_config = ConfigDict(extra="ignore")
    version: str = "v4"
    mode: str = "dev"
    data_dir: str = "./data"
    log_dir: str = "./conversation_logs"
    in_harm_test: bool = False
    debug_mode: bool = False
    log_level: str = "INFO"
    device: str = "cuda"

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v: str) -> str:
        if v not in ("dev", "user"):
            raise ValueError(f"mode must be 'dev' or 'user', got '{v}'")
        return v


class MemorySection(BaseModel):
    model_config = ConfigDict(extra="ignore")
    corpus_file: str = "./data/corpus_v4.json"
    chroma_path: str = "./data/chroma_db_v4"
    max_recent: int = Field(default=50, ge=1)
    max_memories: int = Field(default=30, ge=1)
    max_final_memories: int = Field(default=5, ge=1)
    max_working_memory: int = Field(default=10, ge=1)
    mem_no: int = Field(default=5, ge=1)
    mem_importance_score: float = Field(default=0.6, ge=0.0, le=1.0)
    num_memories: int = Field(default=30, ge=1)
    prompt_max_mems: int = Field(default=15, ge=1)
    child_mem_limit: int = Field(default=3, ge=0)
    recency_decay_rate: float = Field(default=0.05, gt=0.0, le=1.0)
    truth_score_update_rate: float = Field(default=0.02, gt=0.0, le=1.0)
    truth_score_max: float = Field(default=0.95, ge=0.0, le=1.0)
    collection_boosts: CollectionBoosts = Field(default_factory=CollectionBoosts)
    default_summary_prompt_header: str = "Summary of last 20 exchanges:\n"
    default_tagging_prompt: str = ""
    retrieve_conversations: bool = True
    max_conversations: int = Field(default=15, ge=0)
    conversation_collection: str = "conversations"
    retrieve_summaries: bool = True
    max_summaries: int = Field(default=10, ge=0)
    summary_collection: str = "summaries"
    summary_interval: int = Field(default=20, ge=1)
    max_facts: int = Field(default=15, ge=0)
    max_reflections: int = Field(default=6, ge=0)
    hybrid_summaries_enabled: bool = True
    hybrid_reflections_enabled: bool = True
    user_profile_facts_per_category: int = Field(default=5, ge=1)
    prompt_max_recent: int = Field(default=10, ge=1)
    semantic_retrieval_limit: int = Field(default=100, ge=1)
    prompt_max_semantic: int = Field(default=15, ge=0)
    force_min_memories: bool = True


class ModelsSection(BaseModel):
    model_config = ConfigDict(extra="ignore")
    default: str = "llama"
    dream_model: str = "gpt-neo"
    openrouter_base: str = "https://openrouter.ai/api/v1"
    load_local_model: bool = True
    local_model_context_limit: int = Field(default=4096, ge=128)
    api_model_context_limit: int = Field(default=128000, ge=1024)
    default_max_tokens: int = Field(default=2048, ge=1)
    heavy_topic_max_tokens: int = Field(default=8192, ge=1)
    default_top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    default_top_k: int = Field(default=5, ge=1)
    default_temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    openai_api_key: str = ""
    active: str = ""


class GatingSection(BaseModel):
    model_config = ConfigDict(extra="ignore")
    confidence_threshold: float = Field(default=1.5, ge=0.0)
    gate_rel_threshold: float = Field(default=0.18, ge=0.0, le=1.0)
    gate_rel_threshold_retrieval: float = Field(default=0.60, ge=0.0, le=1.0)
    gate_deictic_min_retrieval: float = Field(default=0.61, ge=0.0, le=1.0)
    cosine_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    cosine_similarity_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    deictic_threshold: float = Field(default=0.60, ge=0.0, le=1.0)
    normal_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    deictic_anchor_penalty: float = Field(default=0.1, ge=0.0, le=1.0)
    deictic_continuity_min: float = Field(default=0.12, ge=0.0, le=1.0)
    use_reranking: bool = True
    rerank_use_llm: bool = True
    cross_encoder_weight: float = Field(default=0.7, ge=0.0, le=1.0)
    topic_confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    summary_cosine_threshold: float = Field(default=0.30, ge=0.0, le=1.0)
    reflection_cosine_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    semantic_chunks_gate_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    score_weights: ScoreWeights = Field(default_factory=ScoreWeights)


class FeaturesSection(BaseModel):
    model_config = ConfigDict(extra="ignore")
    enable_wikipedia: bool = True
    enable_semantic_search: bool = True
    enable_facts: bool = True
    enable_dreams: bool = False
    semantic_only_mode: bool = False
    enable_best_of: bool = True
    enable_query_rewrite: bool = True
    disable_llm_summaries: bool = False
    best_of_latency_budget_s: float = Field(default=2.0, ge=0.0)
    rewrite_timeout_s: float = Field(default=1.2, ge=0.0)
    best_of_model: Optional[str] = None
    best_of_max_tokens: int = Field(default=256, ge=1)
    best_of_min_question: bool = True
    best_of_generator_models: List[str] = Field(default_factory=list)
    best_of_selector_models: List[str] = Field(default_factory=list)
    best_of_selector_weights: BestOfSelectorWeights = Field(default_factory=BestOfSelectorWeights)
    best_of_selector_top_k: int = Field(default=0, ge=0)
    best_of_selector_max_tokens: int = Field(default=64, ge=1)
    best_of_duel_mode: bool = False
    # STM settings
    use_stm_pass: bool = True
    stm_model_name: str = "gpt-4o-mini"
    stm_max_recent_messages: int = Field(default=30, ge=1)
    stm_recent_hours: int = Field(default=24, ge=1)
    stm_inject_daily_notes_days: int = Field(default=2, ge=0)
    stm_min_conversation_depth: int = Field(default=3, ge=0)
    stm_topic_similarity_threshold: float = Field(default=0.4, ge=0.0, le=1.0)
    # Citations
    enable_memory_citations: bool = True
    max_citations_display: int = Field(default=10, ge=0)
    citation_content_length: int = Field(default=200, ge=1)
    enable_graph_attribution: bool = True
    enable_insight_attribution: bool = True
    attribution_verbosity: str = "moderate"


class WebSearchSection(BaseModel):
    model_config = ConfigDict(extra="ignore")
    enabled: bool = True
    api_key: str = ""
    timeout_s: float = Field(default=30.0, gt=0.0)
    max_content_chars: int = Field(default=10000, ge=100)
    daily_credit_limit: int = Field(default=100, ge=1)
    llm_first_enabled: bool = True
    trigger_model: str = "gpt-4o-mini"
    trigger_timeout_s: float = Field(default=5.0, gt=0.0)
    trigger_max_tokens: int = Field(default=150, ge=1)
    llm_heuristic_weight: float = Field(default=0.7, ge=0.0, le=1.0)
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    per_query_limit: int = Field(default=5, ge=1)
    cache_ttl_hours: int = Field(default=72, ge=1)
    credits_path: str = "data/web_search_credits.json"
    link_selector_model: str = "gpt-4o-mini"


class LocationSection(BaseModel):
    model_config = ConfigDict(extra="ignore")
    enabled: bool = True
    ip_lookup_enabled: bool = True
    ip_cache_ttl_hours: float = Field(default=6.0, gt=0.0)
    ip_lookup_timeout_s: float = Field(default=3.0, gt=0.0)
    override: str = ""


class AgenticSearchSection(BaseModel):
    model_config = ConfigDict(extra="ignore")
    enabled: bool = True
    max_rounds: int = Field(default=5, ge=1)
    context_budget_tokens: int = Field(default=8000, ge=1000)
    compression_model: str = "sonnet-4.5"
    prefer_native_tools: bool = True
    memory_search_enabled: bool = True
    memory_search_limit: int = Field(default=7, ge=1)
    reuse_decision_answer: bool = True
    decision_max_tokens: int = Field(default=1600, ge=100)
    # Per-round decision-LLM timeout (backstop against a stalled connection /
    # a genuinely hung provider call). Generous so it never cuts a legitimate
    # full-length decision round; on timeout the loop answers with gathered ctx.
    round_timeout_s: float = Field(default=75.0, ge=5.0)
    # Wall-clock budget for the rounds-2-N loop. Once exceeded, no new round is
    # started and the loop falls through to final synthesis. Bounds a runaway
    # multi-round session (max_rounds × slow-model latency) hanging the turn.
    loop_timeout_s: float = Field(default=120.0, ge=10.0)
    fetch_fastpath: bool = True
    fetch_fastpath_min_chars: int = Field(default=400, ge=1)


class InsightModeSection(BaseModel):
    """Insight / evidence-assembly turn-owning mode (facet sweep + provenance
    + adversarial assessment). Sibling of agentic_search."""
    model_config = ConfigDict(extra="ignore")
    enabled: bool = True
    max_facets: int = Field(default=6, ge=1)
    per_facet_cap: int = Field(default=10, ge=1)
    total_evidence_cap: int = Field(default=80, ge=1)
    evidence_snippet_chars: int = Field(default=280, ge=50)
    external_snippet_chars: int = Field(default=560, ge=50)
    keyword_scan_max: int = Field(default=50, ge=1)
    expand_top_k: int = Field(default=3, ge=0)
    expand_window: int = Field(default=2, ge=1)
    decompose_max_tokens: int = Field(default=700, ge=100)
    synthesis_max_tokens: int = Field(default=2600, ge=200)
    sweep_timeout_s: float = Field(default=45.0, ge=5.0)
    offer_enabled: bool = True
    doc_on_agreement: bool = True
    planner_model: Optional[str] = None


class PatternAnalysisSection(BaseModel):
    """Deterministic pattern engine + insight-mode pattern_temporal facet
    (2026-08-29). On-demand only — never injected into prompts uninvited."""
    model_config = ConfigDict(extra="ignore")
    enabled: bool = True
    default_window_days: int = Field(default=90, ge=1)
    max_exemplars: int = Field(default=12, ge=0)
    exemplars_per_bucket: int = Field(default=2, ge=0)
    keyword_hit_cap: int = Field(default=5000, ge=100)


class FileAccessSection(BaseModel):
    model_config = ConfigDict(extra="ignore")
    enabled: bool = True
    approved_folders: List[str] = Field(default_factory=lambda: ["."])
    max_read_bytes: int = Field(default=100000, ge=1)
    max_grep_results: int = Field(default=25, ge=1)
    max_list_entries: int = Field(default=200, ge=1)
    allowed_extensions: List[str] = Field(default_factory=lambda: [".py", ".md", ".txt", ".json", ".yaml"])


class MemoryExpansionSection(BaseModel):
    model_config = ConfigDict(extra="ignore")
    enabled: bool = True
    max_per_session: int = Field(default=3, ge=1)
    max_window: int = Field(default=5, ge=1)
    default_window: int = Field(default=3, ge=1)
    max_total_tokens: int = Field(default=2000, ge=100)
    anchor_char_limit: int = Field(default=600, ge=50)
    context_char_limit: int = Field(default=300, ge=50)
    anchor_char_limit_long: int = Field(default=3000, ge=100)
    context_char_limit_long: int = Field(default=2000, ge=100)


class TokenBudgetSection(BaseModel):
    model_config = ConfigDict(extra="ignore")
    default: int = Field(default=15000, ge=1000)
    local_model: int = Field(default=12000, ge=1000)
    floor: int = Field(default=8000, ge=1000)
    ceiling: int = Field(default=16000, ge=1000)
    context_fraction: float = Field(default=0.12, ge=0.01, le=1.0)

    @model_validator(mode="after")
    def check_ordering(self) -> "TokenBudgetSection":
        if self.floor > self.default:
            raise ValueError(f"token_budget.floor ({self.floor}) must be <= default ({self.default})")
        if self.default > self.ceiling:
            raise ValueError(f"token_budget.default ({self.default}) must be <= ceiling ({self.ceiling})")
        return self


class KnowledgeGraphSection(BaseModel):
    model_config = ConfigDict(extra="ignore")
    enabled: bool = True
    persist_path: str = "data/knowledge_graph.json"
    retrieval_depth: int = Field(default=2, ge=1)
    max_sentences: int = Field(default=15, ge=1)
    auto_save_threshold: int = Field(default=50, ge=1)
    min_confidence: float = Field(default=0.50, ge=0.0, le=1.0)
    aliases_path: str = "data/entity_aliases.json"
    max_prompt_sentences: int = Field(default=12, ge=1)
    scoring_boost_enabled: bool = True
    scoring_boost_cap: float = Field(default=0.15, ge=0.0, le=1.0)
    query_expansion_enabled: bool = True
    query_expansion_max_terms: int = Field(default=8, ge=0)


class SynthesisSection(BaseModel):
    model_config = ConfigDict(extra="ignore")
    enabled: bool = True
    batch_size: int = Field(default=100, ge=1)
    log_rejections: bool = True
    min_token_length: int = Field(default=10, ge=1)
    max_repetition_ratio: float = Field(default=0.5, ge=0.0, le=1.0)
    min_domains: int = Field(default=2, ge=1)
    distance_min: float = Field(default=0.20, ge=0.0, le=1.0)
    distance_max: float = Field(default=0.90, ge=0.0, le=1.0)
    use_percentile_thresholds: bool = False
    novelty_known_threshold: float = Field(default=0.88, ge=0.0, le=1.0)
    novelty_adjacent_threshold: float = Field(default=0.70, ge=0.0, le=1.0)
    cooccurrence_known_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    # Stage-3 known gate: direct cos(A,B). Replaced the inverted bigram-FAISS signal
    # (see docs/SYNTHESIS_VALIDATION.md revert note). ~0.45 → 6/7 known, 0 false-positive.
    concept_cosine_known_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    memory_similarity_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    coherence_model: str = "sonnet-4.5"
    coherence_min_level: str = "MODERATE"
    weights: SynthesisWeights = Field(default_factory=SynthesisWeights)
    composite_min_score: float = Field(default=0.70, ge=0.0, le=1.0)
    convergence_strong_paths: int = Field(default=3, ge=1)
    convergence_strong_sources: int = Field(default=2, ge=1)
    # Novelty sub-weights (in app_config but under synthesis section)
    novelty_w_claim: float = Field(default=0.25, ge=0.0, le=1.0)
    novelty_w_cooccurrence: float = Field(default=0.30, ge=0.0, le=1.0)
    novelty_w_specificity: float = Field(default=0.25, ge=0.0, le=1.0)
    novelty_w_internal: float = Field(default=0.20, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def check_distance_order(self) -> "SynthesisSection":
        if self.distance_min >= self.distance_max:
            raise ValueError(
                f"synthesis.distance_min ({self.distance_min}) must be < distance_max ({self.distance_max})"
            )
        return self


class SynthesisGeneratorSection(BaseModel):
    model_config = ConfigDict(extra="ignore")
    enabled: bool = True
    candidates_per_session: int = Field(default=5, ge=1)
    llm_concurrency: int = Field(default=5, ge=1)
    min_graph_nodes: int = Field(default=20, ge=1)


class SynthesisPooledSection(BaseModel):
    model_config = ConfigDict(extra="ignore")
    enabled: bool = True
    candidates_per_session: int = Field(default=8, ge=1)
    llm_concurrency: int = Field(default=5, ge=1)
    min_cos: float = Field(default=0.20, ge=0.0, le=1.0)
    max_cos: float = Field(default=0.45, ge=0.0, le=1.0)


class SynthesisRetrievalSection(BaseModel):
    model_config = ConfigDict(extra="ignore")
    enabled: bool = True
    structural_query_max_tokens: int = Field(default=100, ge=1)
    retrieval_k: int = Field(default=5, ge=1)
    min_similarity: float = Field(default=0.25, ge=0.0, le=1.0)
    bridge_on_accept: bool = True
    bridge_relation: str = "structural_parallel"


class SynthesisAuditSection(BaseModel):
    model_config = ConfigDict(extra="ignore")
    enabled: bool = True
    fp_halt_threshold: float = Field(default=0.50, ge=0.0, le=1.0)
    min_graded: int = Field(default=10, ge=1)


class GraphWalkSection(BaseModel):
    model_config = ConfigDict(extra="ignore")
    enabled: bool = True
    max_walk_length: int = Field(default=8, ge=2)
    walks_per_seed: int = Field(default=20, ge=1)
    restart_probability: float = Field(default=0.15, ge=0.0, le=1.0)
    min_path_length: int = Field(default=3, ge=2)
    max_candidates_per_session: int = Field(default=10, ge=1)
    boundary_crossing_required: bool = True
    min_bridge_edges: int = Field(default=40, ge=0)
    personal_return_bias: float = Field(default=2.0, ge=0.0)
    hub_degree_threshold: int = Field(default=15, ge=1)
    min_domains: int = Field(default=2, ge=1)


class ObsidianSection(BaseModel):
    model_config = ConfigDict(extra="ignore")
    enabled: bool = True
    vault_path: str = "~/Documents/Notes"
    chunk_threshold: int = Field(default=1500, ge=100)
    max_notes_prompt: int = Field(default=5, ge=0)
    gate_threshold: float = Field(default=0.60, ge=0.0, le=1.0)  # blended gate score; 2026-09-03: 0.45→0.60
    include_images: bool = True
    max_images_per_note: int = Field(default=3, ge=0)
    max_image_size_mb: float = Field(default=10.0, gt=0.0)
    multimodal_models: List[str] = Field(default_factory=list)


class DailyNotesSection(BaseModel):
    model_config = ConfigDict(extra="ignore")
    enabled: bool = True
    folder: str = "Daily"
    model: str = "sonnet-4.5"
    max_tokens: int = Field(default=800, ge=1)
    weekly_enabled: bool = True
    weekly_model: str = "sonnet-4.5"
    weekly_max_tokens: int = Field(default=1200, ge=1)
    monthly_enabled: bool = True
    monthly_model: str = "sonnet-4.5"
    monthly_max_tokens: int = Field(default=2000, ge=1)


class ReferenceDocsSection(BaseModel):
    model_config = ConfigDict(extra="ignore")
    auto_seed: bool = True
    seed_paths: List[str] = Field(default_factory=lambda: ["docs"])
    chunk_threshold: int = Field(default=2000, ge=100)
    max_prompt: int = Field(default=15, ge=0)
    gate_threshold: float = Field(default=0.40, ge=0.0, le=1.0)


class VisualMemorySection(BaseModel):
    model_config = ConfigDict(extra="ignore")
    enabled: bool = False
    clip_model: str = "ViT-B-32"
    clip_pretrained: str = "openai"
    max_images_prompt: int = Field(default=3, ge=0)
    caption_model: str = "gpt-4o-mini"
    caption_timeout_s: float = Field(default=10.0, gt=0.0)
    similarity_threshold: float = Field(default=0.20, ge=0.0, le=1.0)
    ingest_on_upload: bool = True
    ingest_on_obsidian_sync: bool = True
    index_path: str = "data/clip_index.faiss"
    meta_path: str = "data/clip_metadata.json"


class WikiEnrichmentSection(BaseModel):
    model_config = ConfigDict(extra="ignore")
    enabled: bool = True
    max_articles_per_session: int = Field(default=15, ge=1)
    min_text_length: int = Field(default=200, ge=1)
    shutdown_step_timeout_s: float = Field(default=30.0, gt=0.0)
    edge_relation_type: str = "mentioned_alongside"
    edge_default_weight: float = Field(default=0.5, ge=0.0, le=1.0)


class WikidataImportSection(BaseModel):
    model_config = ConfigDict(extra="ignore")
    enabled: bool = True
    persist_path: str = "data/wikidata_cache.json"
    entities_per_domain: int = Field(default=5000, ge=1)
    max_total_entities: int = Field(default=50000, ge=1)
    sparql_batch_size: int = Field(default=500, ge=1)
    embedding_match_threshold: float = Field(default=0.60, ge=0.0, le=1.0)


class WikidataEnrichmentSection(BaseModel):
    """Anchored shutdown-time Wikidata typed-edge enrichment (1-hop, capped)."""
    model_config = ConfigDict(extra="ignore")
    enabled: bool = True
    relation_whitelist: List[str] = Field(default_factory=lambda: [
        "instance_of", "subclass_of", "part_of", "has_part",
        "uses", "has_use", "has_effect", "has_cause", "main_subject",
    ])
    max_edges_per_entity: int = Field(default=5, ge=1)
    max_new_nodes_per_run: int = Field(default=25, ge=0)
    max_edges_per_run: int = Field(default=50, ge=1)
    shutdown_step_timeout_s: float = Field(default=30.0, gt=0.0)


class ThreadSurfacingSection(BaseModel):
    model_config = ConfigDict(extra="ignore")
    enabled: bool = True
    max_open: int = Field(default=50, ge=1)
    stale_days: int = Field(default=14, ge=1)
    deadline_grace_hours: int = Field(default=48, ge=0)
    max_surfaced: int = Field(default=3, ge=0)
    model_alias: str = ""


class StalenessSection(BaseModel):
    model_config = ConfigDict(extra="ignore")
    enabled: bool = True
    max_penalty: float = Field(default=0.4, ge=0.0, le=1.0)
    weight: float = Field(default=0.15, ge=0.0, le=1.0)
    steep_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    steep_multiplier: float = Field(default=2.0, ge=1.0)
    historical_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    reflection_weight_factor: float = Field(default=0.6, ge=0.0, le=1.0)
    index_path: str = "data/claim_index.json"


class HealthFramingDecaySection(BaseModel):
    """Read-time decay for stale *free-text* health framing.

    Structured illness/recovery relations already age out via
    ``relation_classifier`` (graph edges, facts collection, profile). This
    applies the SAME health-transient horizon
    (``user_profile.health_transient_ttl_hours``) to narrative memory text so an
    old "post-viral" / "recovering from illness" line in a conversation, note,
    or reflection stops reading as present-tense. The episode horizon is shared;
    only the penalty shape + collection scope live here.
    """
    model_config = ConfigDict(extra="ignore")
    enabled: bool = True
    weight: float = Field(default=0.25, ge=0.0, le=1.0)
    max_penalty: float = Field(default=0.4, ge=0.0, le=1.0)
    collections: List[str] = Field(
        default_factory=lambda: [
            "conversations", "obsidian_notes", "reflections",
            "summaries", "daemon_self_notes",
        ]
    )


class EscalationSection(BaseModel):
    model_config = ConfigDict(extra="ignore")
    enabled: bool = True
    threshold: int = Field(default=3, ge=1)
    deescalation_window: int = Field(default=2, ge=1)
    max_history: int = Field(default=10, ge=1)
    # Consecutive CONCERN-or-higher turns before a mild-but-persistent spiral
    # upgrades to grounding presence (higher than `threshold`, which is for
    # ELEVATED/CRISIS). Catches slow spirals built from many terse CONCERN turns.
    distress_threshold: int = Field(default=5, ge=1)
    # Max consecutive turns the sustained-distress upgrade may hold GROUNDING
    # before stepping down to GENTLE_REENGAGEMENT (distress counter resets).
    distress_grounding_max: int = Field(default=3, ge=1)


class CanarySection(BaseModel):
    """Runtime safety canary — log-only monitor for tone-flatline miswires."""
    model_config = ConfigDict(extra="ignore")
    enabled: bool = True
    # Consecutive negative-affect-but-CONVERSATIONAL user turns before a WARNING.
    consecutive_threshold: int = Field(default=4, ge=1)


class ValenceRetrievalSection(BaseModel):
    """Valence-aware memory retrieval — caps mood-congruent recall during distress."""
    model_config = ConfigDict(extra="ignore")
    enabled: bool = True
    # Max fraction of retrieved memories allowed to be high-negative-affect while
    # the session is in distress. The rest are backfilled with lower-affect,
    # still-relevant memories to break the rumination-amplifier loop.
    max_negative_fraction: float = Field(default=0.5, ge=0.0, le=1.0)
    # Negative-affect score (0..1) at/above which a memory counts as "negative".
    negative_threshold: float = Field(default=0.30, ge=0.0, le=1.0)


class CrossDedupSection(BaseModel):
    model_config = ConfigDict(extra="ignore")
    enabled: bool = True
    duplicate_threshold: float = Field(default=0.92, ge=0.0, le=1.0)
    contradiction_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    max_docs_per_collection: int = Field(default=1000, ge=1)
    on_shutdown: bool = True
    collections: List[str] = Field(
        default_factory=lambda: ["facts", "summaries", "procedural_skills", "proposals", "reflections"]
    )


class TruthScorerSection(BaseModel):
    model_config = ConfigDict(extra="ignore")
    enabled: bool = True
    initial_score: float = Field(default=0.7, ge=0.0, le=1.0)
    confirmed_boost: float = Field(default=0.08, ge=0.0, le=1.0)
    correction_penalty: float = Field(default=0.25, ge=0.0, le=1.0)
    contradiction_penalty: float = Field(default=0.15, ge=0.0, le=1.0)
    decay_rate: float = Field(default=0.02, ge=0.0, le=1.0)
    decay_floor: float = Field(default=0.3, ge=0.0, le=1.0)
    correction_detection: bool = True
    confirmation_detection: bool = True
    source_scores: TruthSourceScores = Field(default_factory=TruthSourceScores)


class PersonalVocabularySection(BaseModel):
    """Per-user owner-specific terms, kept out of shipped source.

    Each field is merged into a general default at load time by its consumer
    (user_profile_schema, context_surfacer, memory_storage, fact_extractor).
    All optional — an empty block yields a fully generic install.
    """
    model_config = ConfigDict(extra="ignore")
    category_tokens: Dict[str, List[str]] = Field(default_factory=dict)
    relation_categories: Dict[str, str] = Field(default_factory=dict)
    project_areas: Dict[str, List[str]] = Field(default_factory=dict)
    entity_casing: Dict[str, str] = Field(default_factory=dict)
    generic_subjects: List[str] = Field(default_factory=list)
    preference_slots: List[str] = Field(default_factory=list)


class UserProfileSection(BaseModel):
    model_config = ConfigDict(extra="ignore")
    ephemeral_relations: List[str] = Field(default_factory=list)
    ephemeral_max_history: int = Field(default=20, ge=1)
    ephemeral_ttl_hours: int = Field(default=24, ge=1)
    health_transient_ttl_hours: int = Field(default=96, ge=1)
    category_soft_cap: int = Field(default=200, ge=1)
    personal_vocabulary: PersonalVocabularySection = Field(default_factory=PersonalVocabularySection)


class IntentClassifierSection(BaseModel):
    model_config = ConfigDict(extra="ignore")
    enabled: bool = True
    stm_refinement_threshold: float = Field(default=0.50, ge=0.0, le=1.0)
    # Confidence assigned to an STM-refined intent. 0.60 lets a refined intent
    # reach the 0.60 routing floors (heavy-topic skip) without reaching the
    # 0.75 agentic-veto floor.
    stm_refined_confidence: float = Field(default=0.60, ge=0.0, le=1.0)
    section_gating_enabled: bool = True
    # Inject a short per-intent response-style block into the system prompt
    # tail (after the cache breakpoint); crisis tone levels suppress it.
    style_instructions_enabled: bool = True


class TurnTelemetrySection(BaseModel):
    model_config = ConfigDict(extra="ignore")
    enabled: bool = True
    path: str = "logs/turn_records.jsonl"


class LightPromptSection(BaseModel):
    """Lightweight-context path for terse casual acknowledgments."""
    model_config = ConfigDict(extra="ignore")
    enabled: bool = True
    max_words: int = Field(default=8, ge=1, le=20)


class BackupSection(BaseModel):
    """Automated local backups of the memory stores (utils/backup_manager.py)."""
    model_config = ConfigDict(extra="ignore")
    enabled: bool = True
    dir: str = "data/backups"
    retention: int = Field(default=5, ge=1)
    min_interval_hours: float = Field(default=12, ge=0)
    include_chroma: bool = True


class LogMaintenanceSection(BaseModel):
    """Startup log-growth bounding (utils/log_rotation.py)."""
    model_config = ConfigDict(extra="ignore")
    enabled: bool = True
    turn_records_max_mb: float = Field(default=50, gt=0)
    daily_notes_max_mb: float = Field(default=20, gt=0)
    audit_max_mb: float = Field(default=20, gt=0)
    debug_compress_age_days: float = Field(default=7, ge=0)
    debug_keep_days: float = Field(default=90, ge=1)


class CurationSection(BaseModel):
    """Autonomous curation engine (docs/AUTONOMOUS_CURATION_DESIGN.md)."""
    model_config = ConfigDict(extra="ignore")
    enabled: bool = True
    max_mode: Literal["off", "shadow", "queue", "auto"] = "queue"
    curator_modes: Dict[str, Literal["off", "shadow", "queue", "auto"]] = Field(
        default_factory=dict)
    scan_timeout_s: float = Field(default=45, gt=0)
    auto_rate_cap: int = Field(default=25, ge=0)
    anomaly_fraction: float = Field(default=0.05, gt=0, le=1)
    max_queue_items_per_curator: int = Field(default=50, ge=1)
    staleness_grace_hours: int = Field(default=48, ge=0)


class ApiSection(BaseModel):
    """FastAPI server settings (React frontend at /, Gradio admin UI at /admin)."""
    model_config = ConfigDict(extra="ignore")
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    cors_origins: List[str] = Field(default_factory=lambda: ["http://localhost:5173"])
    serve_frontend: bool = True
    frontend_dist_dir: str = "web/dist"


class EntityFactsSection(BaseModel):
    model_config = ConfigDict(extra="ignore")
    enabled: bool = True
    per_turn_cap: int = Field(default=4, ge=0)
    user_per_turn_cap: int = Field(default=6, ge=0)
    min_confidence: float = Field(default=0.55, ge=0.0, le=1.0)


class ScheduleExtractionSection(BaseModel):
    model_config = ConfigDict(extra="ignore")
    enabled: bool = True
    prompt_max_events: int = Field(default=10, ge=0)
    lookahead_days: int = Field(default=7, ge=1)
    bare_time_min_confidence: float = Field(default=0.50, ge=0.0, le=1.0)


class FactVerificationSection(BaseModel):
    model_config = ConfigDict(extra="ignore")
    enabled: bool = True
    llm_enabled: bool = True
    model: str = "gpt-4o-mini"
    user_trust_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    max_candidates: int = Field(default=10, ge=1)


class ProactiveSurfacingSection(BaseModel):
    model_config = ConfigDict(extra="ignore")
    enabled: bool = True
    min_graph_nodes: int = Field(default=20, ge=0)
    min_graph_edges: int = Field(default=15, ge=0)
    max_insights: int = Field(default=2, ge=0)
    cooldown_hours: int = Field(default=72, ge=1)
    model: str = ""
    history_path: str = "data/surfacing_history.json"
    max_prompt_insights: int = Field(default=2, ge=0)


class ResponsePlanningSection(BaseModel):
    model_config = ConfigDict(extra="ignore")
    enabled: bool = True
    model: Optional[str] = None
    max_tokens: int = Field(default=200, ge=1)
    timeout: float = Field(default=5.0, gt=0.0)
    review_enabled: bool = True
    review_model: Optional[str] = None
    review_max_tokens: int = Field(default=200, ge=1)
    review_confidence_threshold: float = Field(default=0.90, ge=0.0, le=1.0)
    review_timeout: float = Field(default=5.0, gt=0.0)


class GroundingCheckSection(BaseModel):
    """Factual-grounding floor (2026-08-28): post-generation false-claim
    check — deterministic pre-filter → LLM verifier → visible correction."""
    model_config = ConfigDict(extra="ignore")
    enabled: bool = True
    # 2026-09-04: LOG-ONLY by default (same class as the review-gate LOG-ONLY
    # fix, 2026-08-28) — telemetry over the window showed 42 verifier fires ->
    # 27 flags -> 25 shipped corrections, >=9 documented false, 0 documented
    # true. "correct" restores the pre-09-04 integrate/suffix behavior.
    mode: Literal["log_only", "correct"] = "log_only"
    model: Optional[str] = None  # None → falls back to response_planning.review_model
    confidence_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    timeout_s: float = Field(default=5.0, gt=0.0)
    max_tokens: int = Field(default=250, ge=1)
    min_response_chars: int = Field(default=40, ge=0)
    # 2026-08-29: integrate=True weaves the correction INTO the response via
    # a bounded rewrite (final-yield replacement, display==storage); the
    # appended suffix is the fallback.
    integrate: bool = True
    integrate_timeout_s: float = Field(default=6.0, gt=0.0)
    integrate_max_response_chars: int = Field(default=4000, ge=200)
    integrate_min_ratio: float = Field(default=0.75, gt=0.0)  # min (rewritten / original) length ratio
    integrate_max_ratio: float = Field(default=1.30, gt=0.0)  # max (rewritten / original) length ratio


class EmailIntegrationSection(BaseModel):
    """Email integration (2026-09-01): metadata-only read access to Gmail and Outlook
    via adapters conforming to EmailProvider protocol. Results cached in-memory (TTL).
    Consumers: agentic email_search tool, passive [RELEVANT EMAILS] context gatherer,
    pattern-engine email dimension."""
    model_config = ConfigDict(extra="ignore")
    enabled: bool = True
    gmail_enabled: bool = True
    outlook_enabled: bool = False
    outlook_client_id: str = ""  # Azure app registration Application ID; personal values in config.local.yaml
    outlook_tenant: str = "common"  # Azure tenant ID (common for multi-tenant)
    max_results: int = Field(default=20, ge=1, le=100)
    cache_ttl_seconds: float = Field(default=300.0, gt=0.0)
    passive_context_enabled: bool = True  # inject [RELEVANT EMAILS] into prompts
    passive_max_emails: int = Field(default=3, ge=0, le=10)
    passive_min_relevance: float = Field(default=0.35, ge=0.0, le=1.0)
    default_window_days: int = Field(default=7, ge=1, le=365)


class UncertaintyFallbackSection(BaseModel):
    model_config = ConfigDict(extra="ignore")
    enabled: bool = True
    semantic_threshold: float = Field(default=0.70, ge=0.0, le=1.0)
    max_length: int = Field(default=400, ge=1)


class ProvenanceSection(BaseModel):
    model_config = ConfigDict(extra="ignore")
    enabled: bool = True
    thinking_max_chars: int = Field(default=4000, ge=100)


class LlmCompressionSection(BaseModel):
    model_config = ConfigDict(extra="ignore")
    enabled: bool = False
    model: str = "gpt-4o-mini"
    timeout_s: float = Field(default=3.0, gt=0.0)
    ratio_threshold: float = Field(default=3.0, ge=1.0)
    max_batch: int = Field(default=8, ge=1)


class SessionDiffSection(BaseModel):
    model_config = ConfigDict(extra="ignore")
    enabled: bool = True
    max_committed: int = Field(default=20, ge=0)
    max_uncommitted: int = Field(default=20, ge=0)
    extensions: List[str] = Field(default_factory=lambda: [".py", ".yaml", ".yml"])


class GitStatsSection(BaseModel):
    model_config = ConfigDict(extra="ignore")
    enabled: bool = True
    timeout_s: int = Field(default=10, ge=1)
    max_output_lines: int = Field(default=50, ge=1)


class GitHubAPISection(BaseModel):
    model_config = ConfigDict(extra="ignore")
    enabled: bool = True
    timeout_s: int = Field(default=15, ge=1)
    max_output_lines: int = Field(default=80, ge=1)
    repo: str = Field(default="", description="Optional owner/repo override (auto-detected if empty)")


class GitMemorySection(BaseModel):
    model_config = ConfigDict(extra="ignore")
    enabled: bool = True
    include_diffs: bool = False
    default_limit: int = Field(default=200, ge=1)


class ProceduralSkillsSection(BaseModel):
    model_config = ConfigDict(extra="ignore")
    enabled: bool = True
    prompt_max_skills: int = Field(default=5, ge=0)
    dedup_threshold: float = Field(default=0.85, ge=0.0, le=1.0)


class SkillActivationSection(BaseModel):
    """Post-retrieval skill filtering and cooldown."""
    model_config = ConfigDict(extra="ignore")
    enabled: bool = True
    max_skills: int = Field(default=3, ge=0, description="Max skills to inject into prompt")
    min_score: float = Field(default=0.25, ge=0.0, le=1.0, description="Minimum relevance score")
    cooldown_hours: float = Field(default=48.0, ge=0.0, description="Hours before re-surfacing a skill")
    fetch_multiplier: int = Field(default=3, ge=1, description="Fetch N*max_skills candidates for filtering")
    stm_bonus: float = Field(default=0.10, ge=0.0, le=1.0, description="Score bonus for STM topic match")
    use_stm: bool = Field(default=True, description="Use STM topics for reranking")


class CodeProposalsSection(BaseModel):
    model_config = ConfigDict(extra="ignore")
    enabled: bool = True
    collection: str = "proposals"
    dedup_threshold: float = Field(default=0.70, ge=0.0, le=1.0)
    max_per_session: int = Field(default=5, ge=1)
    require_tests: bool = True
    prompt_enabled: bool = True
    prompt_max: int = Field(default=3, ge=0)
    keyword_dedup_tag_threshold: float = Field(default=0.60, ge=0.0, le=1.0)
    semantic_dedup_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    llm_ranking: bool = False
    llm_ranking_model: str = "gpt-4o-mini"
    weight_priority: float = Field(default=0.30, ge=0.0, le=1.0)
    weight_breadth: float = Field(default=0.20, ge=0.0, le=1.0)
    weight_recency: float = Field(default=0.10, ge=0.0, le=1.0)
    weight_goal_alignment: float = Field(default=0.40, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def check_weight_sum(self) -> "CodeProposalsSection":
        total = self.weight_priority + self.weight_breadth + self.weight_recency + self.weight_goal_alignment
        if abs(total - 1.0) > 0.05:
            logger.warning(f"code_proposals weights sum to {total:.3f} (expected ~1.0)")
        return self


class ImplTrackingSection(BaseModel):
    model_config = ConfigDict(extra="ignore")
    enabled: bool = True
    cooldown_seconds: int = Field(default=86400, ge=0)
    confidence_confirmed: float = Field(default=0.85, ge=0.0, le=1.0)
    confidence_likely: float = Field(default=0.60, ge=0.0, le=1.0)
    git_depth: int = Field(default=50, ge=1)
    at_shutdown: bool = True
    auto_complete: bool = False


class NarrativeContextSection(BaseModel):
    model_config = ConfigDict(extra="ignore")
    enabled: bool = True
    path: str = "./data/narrative_context.txt"
    max_tokens: int = Field(default=500, ge=1)
    weeklies_count: int = Field(default=3, ge=0)
    monthlies_count: int = Field(default=1, ge=0)
    dailies_count: int = Field(default=6, ge=0)
    synthesis_model: str = "sonnet-4.5"


class TagGenerationSection(BaseModel):
    model_config = ConfigDict(extra="ignore")
    enabled: bool = True
    model: str = "sonnet-4.5"
    max_tags: int = Field(default=10, ge=1)
    min_tags: int = Field(default=3, ge=1)

    @model_validator(mode="after")
    def check_min_max(self) -> "TagGenerationSection":
        if self.min_tags > self.max_tags:
            raise ValueError(
                f"tag_generation.min_tags ({self.min_tags}) must be <= max_tags ({self.max_tags})"
            )
        return self


class WolframSection(BaseModel):
    model_config = ConfigDict(extra="ignore")
    enabled: bool = True
    app_id: str = ""
    api_url: str = "https://www.wolframalpha.com/api/v1/llm-api"
    timeout_s: float = Field(default=30.0, gt=0.0)
    max_output_chars: int = Field(default=10000, ge=100)
    cache_ttl_seconds: int = Field(default=3600, ge=0)
    rate_limit_per_minute: int = Field(default=60, ge=1)


class SandboxSection(BaseModel):
    model_config = ConfigDict(extra="ignore")
    enabled: bool = True
    api_key: str = ""
    timeout_seconds: int = Field(default=60, ge=1)
    session_timeout_minutes: int = Field(default=30, ge=1)
    max_output_chars: int = Field(default=4000, ge=100)
    cache_ttl_seconds: int = Field(default=3600, ge=0)
    rate_limit_per_minute: int = Field(default=30, ge=1)


class SecuritySection(BaseModel):
    model_config = ConfigDict(extra="ignore")
    max_file_size: int = Field(default=10485760, ge=1)
    max_total_size: int = Field(default=52428800, ge=1)
    allowed_extensions: List[str] = Field(
        default_factory=lambda: [".txt", ".docx", ".csv", ".py", ".pdf", ".png", ".jpg"]
    )
    csv_formula_prefixes: List[str] = Field(
        default_factory=lambda: ["=", "+", "-", "@", "\t", "\r", "\n"]
    )


class BehavioralPatternsSection(BaseModel):
    model_config = ConfigDict(extra="ignore")
    enabled: bool = True
    min_turns: int = Field(default=3, ge=1)
    max_patterns: int = Field(default=3, ge=1)


class WikiSection(BaseModel):
    model_config = ConfigDict(extra="ignore")
    fetch_full: bool = True
    max_chars: int = Field(default=15000, ge=100)
    max_sentences: int = Field(default=0, ge=0)
    timeout_s: float = Field(default=1.2, gt=0.0)


class PathsSection(BaseModel):
    model_config = ConfigDict(extra="ignore")
    system_prompt: str = "core/system_prompt.txt"
    directives: str = "structured_directives.txt"
    system_prompt_file: Optional[Dict[str, Any]] = None


class PromptsSection(BaseModel):
    model_config = ConfigDict(extra="ignore")
    default_system_prompt: str = ""
    system_prompt: str = ""
    default_core_directive: Optional[Dict[str, Any]] = None


# =============================================================================
# Root model
# =============================================================================

class DocumentGenerationSection(BaseModel):
    model_config = ConfigDict(extra="ignore")
    enabled: bool = True
    output_dir: str = "documents"
    max_sources: int = Field(default=10, ge=1, le=50)
    report_max_sections: int = Field(default=5, ge=1, le=20)
    summary_max_sections: int = Field(default=3, ge=1, le=10)
    report_token_budget: int = Field(default=6000, ge=1000, le=16000)
    summary_token_budget: int = Field(default=2000, ge=500, le=8000)


class DaemonNotesSection(BaseModel):
    model_config = ConfigDict(extra="ignore")
    enabled: bool = True
    output_dir: str = "daemon_notes"
    max_per_prompt: int = Field(default=2, ge=0, le=10)
    collection_boost: float = Field(default=-0.05, ge=-1.0, le=1.0)


class ActionGuardSection(BaseModel):
    """Pending-proposal capture + claimed-action verification (anti-confabulation)."""

    model_config = ConfigDict(extra="ignore")
    pending_proposal_enabled: bool = True
    pending_proposal_ttl_turns: int = Field(default=2, ge=1, le=10)
    claim_guard_enabled: bool = True
    claim_self_repair_enabled: bool = True


class InternetActionsSection(BaseModel):
    model_config = ConfigDict(extra="ignore")
    enabled: bool = False
    # Telegram
    telegram_bot_token: str = ""
    telegram_default_chat_id: str = ""
    # Discord
    discord_webhook_url: str = ""
    # Email (SMTP)
    smtp_host: str = ""
    smtp_port: int = Field(default=587, ge=1, le=65535)
    smtp_user: str = ""
    smtp_password: str = ""
    smtp_from: str = ""
    # GitHub write (reuses existing gh CLI auth)
    github_write_enabled: bool = False
    # Headless browser
    playwright_enabled: bool = False
    playwright_timeout_s: int = Field(default=30, ge=5, le=120)
    # Google OAuth2
    google_client_id: str = ""
    google_client_secret: str = ""
    google_token_path: str = "data/google_token.json"
    google_calendar_enabled: bool = False
    google_calendar_max_events: int = Field(default=10, ge=1, le=50)
    google_calendar_lookahead_days: int = Field(default=7, ge=1, le=30)
    # Google Contacts (People API)
    google_contacts_enabled: bool = True
    google_other_contacts_enabled: bool = True
    # Gmail header search (fallback for contact resolution)
    google_gmail_search_enabled: bool = True
    # Safety
    action_ttl_seconds: int = Field(default=300, ge=30, le=3600)
    max_pending_actions: int = Field(default=5, ge=1, le=20)
    pending_actions_path: str = "data/pending_actions.json"
    audit_log_path: str = "logs/actions_audit.jsonl"


class DaemonConfig(BaseModel):
    """Root config model — validates the entire config.yaml structure."""
    model_config = ConfigDict(extra="ignore")

    daemon: DaemonSection = Field(default_factory=DaemonSection)
    memory: MemorySection = Field(default_factory=MemorySection)
    models: ModelsSection = Field(default_factory=ModelsSection)
    gating: GatingSection = Field(default_factory=GatingSection)
    features: FeaturesSection = Field(default_factory=FeaturesSection)
    paths: PathsSection = Field(default_factory=PathsSection)
    prompts: PromptsSection = Field(default_factory=PromptsSection)
    web_search: WebSearchSection = Field(default_factory=WebSearchSection)
    location: LocationSection = Field(default_factory=LocationSection)
    agentic_search: AgenticSearchSection = Field(default_factory=AgenticSearchSection)
    insight_mode: InsightModeSection = Field(default_factory=InsightModeSection)
    pattern_analysis: PatternAnalysisSection = Field(default_factory=PatternAnalysisSection)
    uncertainty_fallback: UncertaintyFallbackSection = Field(default_factory=UncertaintyFallbackSection)
    response_planning: ResponsePlanningSection = Field(default_factory=ResponsePlanningSection)
    grounding_check: GroundingCheckSection = Field(default_factory=GroundingCheckSection)
    email_integration: EmailIntegrationSection = Field(default_factory=EmailIntegrationSection)
    turn_telemetry: TurnTelemetrySection = Field(default_factory=TurnTelemetrySection)
    light_prompt: LightPromptSection = Field(default_factory=LightPromptSection)
    backup: BackupSection = Field(default_factory=BackupSection)
    log_maintenance: LogMaintenanceSection = Field(default_factory=LogMaintenanceSection)
    curation: CurationSection = Field(default_factory=CurationSection)
    file_access: FileAccessSection = Field(default_factory=FileAccessSection)
    memory_expansion: MemoryExpansionSection = Field(default_factory=MemoryExpansionSection)
    obsidian: ObsidianSection = Field(default_factory=ObsidianSection)
    daily_notes: DailyNotesSection = Field(default_factory=DailyNotesSection)
    tag_generation: TagGenerationSection = Field(default_factory=TagGenerationSection)
    narrative_context: NarrativeContextSection = Field(default_factory=NarrativeContextSection)
    intent_classifier: IntentClassifierSection = Field(default_factory=IntentClassifierSection)
    entity_facts: EntityFactsSection = Field(default_factory=EntityFactsSection)
    schedule_extraction: ScheduleExtractionSection = Field(default_factory=ScheduleExtractionSection)
    reference_docs: ReferenceDocsSection = Field(default_factory=ReferenceDocsSection)
    fact_verification: FactVerificationSection = Field(default_factory=FactVerificationSection)
    proactive_surfacing: ProactiveSurfacingSection = Field(default_factory=ProactiveSurfacingSection)
    knowledge_graph: KnowledgeGraphSection = Field(default_factory=KnowledgeGraphSection)
    session_diff: SessionDiffSection = Field(default_factory=SessionDiffSection)
    token_budget: TokenBudgetSection = Field(default_factory=TokenBudgetSection)
    llm_compression: LlmCompressionSection = Field(default_factory=LlmCompressionSection)
    thread_surfacing: ThreadSurfacingSection = Field(default_factory=ThreadSurfacingSection)
    staleness: StalenessSection = Field(default_factory=StalenessSection)
    health_framing_decay: HealthFramingDecaySection = Field(default_factory=HealthFramingDecaySection)
    implementation_tracking: ImplTrackingSection = Field(default_factory=ImplTrackingSection)
    provenance: ProvenanceSection = Field(default_factory=ProvenanceSection)
    synthesis: SynthesisSection = Field(default_factory=SynthesisSection)
    synthesis_generator: SynthesisGeneratorSection = Field(default_factory=SynthesisGeneratorSection)
    synthesis_pooled: SynthesisPooledSection = Field(default_factory=SynthesisPooledSection)
    synthesis_retrieval: SynthesisRetrievalSection = Field(default_factory=SynthesisRetrievalSection)
    synthesis_audit: SynthesisAuditSection = Field(default_factory=SynthesisAuditSection)
    wikidata_import: WikidataImportSection = Field(default_factory=WikidataImportSection)
    wikidata_enrichment: WikidataEnrichmentSection = Field(default_factory=WikidataEnrichmentSection)
    graph_walk: GraphWalkSection = Field(default_factory=GraphWalkSection)
    visual_memory: VisualMemorySection = Field(default_factory=VisualMemorySection)
    wiki_enrichment: WikiEnrichmentSection = Field(default_factory=WikiEnrichmentSection)
    procedural_skills: ProceduralSkillsSection = Field(default_factory=ProceduralSkillsSection)
    skill_activation: SkillActivationSection = Field(default_factory=SkillActivationSection)
    code_proposals: CodeProposalsSection = Field(default_factory=CodeProposalsSection)
    behavioral_patterns: BehavioralPatternsSection = Field(default_factory=BehavioralPatternsSection)
    escalation_tracker: EscalationSection = Field(default_factory=EscalationSection)
    valence_retrieval: ValenceRetrievalSection = Field(default_factory=ValenceRetrievalSection)
    canary: CanarySection = Field(default_factory=CanarySection)
    cross_dedup: CrossDedupSection = Field(default_factory=CrossDedupSection)
    user_profile: UserProfileSection = Field(default_factory=UserProfileSection)
    git_stats: GitStatsSection = Field(default_factory=GitStatsSection)
    github_api: GitHubAPISection = Field(default_factory=GitHubAPISection)
    git_memory: GitMemorySection = Field(default_factory=GitMemorySection)
    wolfram: WolframSection = Field(default_factory=WolframSection)
    sandbox: SandboxSection = Field(default_factory=SandboxSection)
    security: SecuritySection = Field(default_factory=SecuritySection)
    wiki: WikiSection = Field(default_factory=WikiSection)
    truth_scorer: TruthScorerSection = Field(default_factory=TruthScorerSection)
    document_generation: DocumentGenerationSection = Field(default_factory=DocumentGenerationSection)
    daemon_notes: DaemonNotesSection = Field(default_factory=DaemonNotesSection)
    action_guard: ActionGuardSection = Field(default_factory=ActionGuardSection)
    internet_actions: InternetActionsSection = Field(default_factory=InternetActionsSection)
    api: ApiSection = Field(default_factory=ApiSection)


# =============================================================================
# Public API
# =============================================================================

# Known section names (for unknown-key detection)
_KNOWN_SECTIONS = set(DaemonConfig.model_fields.keys())


def validate_config(config: dict) -> dict:
    """
    Validate the full config dict against the Pydantic schema.

    Args:
        config: The loaded + resolved + defaulted config dict.

    Returns:
        The config dict unchanged (on success).

    Raises:
        SystemExit: On validation failure (with formatted error messages).
    """
    from pydantic import ValidationError

    # Warn about unknown top-level sections (typo detection)
    unknown_sections = set(config.keys()) - _KNOWN_SECTIONS
    if unknown_sections:
        logger.warning(
            f"[CONFIG] Unknown sections in config.yaml (possible typos): {sorted(unknown_sections)}"
        )

    try:
        DaemonConfig(**config)
    except ValidationError as e:
        logger.error("=" * 60)
        logger.error("CONFIG VALIDATION FAILED - fix config.yaml and restart")
        logger.error("=" * 60)
        for err in e.errors():
            loc = " -> ".join(str(x) for x in err["loc"])
            input_val = err.get("input", "?")
            # Truncate long input values
            input_str = str(input_val)
            if len(input_str) > 80:
                input_str = input_str[:77] + "..."
            logger.error(f"  [{loc}] {err['msg']} (got: {input_str})")
        logger.error("=" * 60)
        sys.exit(1)

    return config
