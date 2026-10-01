"""
2026-09-27 lane B batch B4 (class: BC-89, BC-71).

FOLLOWUPS.md: "BC-89 remaining sites (ratchet ceiling 4): memory/fact_verification.py:398
(A/B/C adjudication, max_tokens=4), memory/user_profile_schema.py:850 (category, 10),
core/prompt/proposal_filter.py:576 (pairwise rank, 4), gui/wizard.py:429 (key smoke test, 10)
— each returns empty on a reasoning model and silently takes its fallback; add
disable_reasoning=True + headroom, lower the ceiling per fix."

Mechanism (BC-89, docs/BUG_CLASSES.md): a structured LLM call expecting a single
word/letter dials a tiny max_tokens without disabling the model's reasoning
channel. A reasoning-capable model spends the whole budget reasoning, returns
an empty string, and the caller's fail-safe silently takes the WRONG branch —
no exception, no retry. These four tests call each DEPLOYED site through its
public entry point with a mocked model_manager and assert the actual
`generate_once(...)` call carries `disable_reasoning=True` plus headroom
(max_tokens raised past the old single-digit budget), not a re-derivation of
the fix. Each test would fail against the pre-fix code (tiny max_tokens, no
disable_reasoning kwarg) — verified in laneB_0927_base.

Also covers the BC-71 doc-drift fix: gui/wizard.py's wiki-index prompt no
longer hardcodes a stale/wrong GitHub owner/repo path.
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock


# ---------------------------------------------------------------------------
# 1. memory/fact_verification.py — FactVerifier._adjudicate_conflict
# ---------------------------------------------------------------------------

class TestFactVerificationAdjudicationBudget:
    @pytest.mark.asyncio
    async def test_adjudication_call_disables_reasoning_with_headroom(self):
        from memory.fact_verification import FactVerifier

        chroma = MagicMock()
        coll = MagicMock()
        existing = [{
            "id": "f1", "content": "user | works_at | Google",
            "metadata": {"subject": "user", "relation": "works_at",
                         "object": "Google", "confidence": 0.7},
        }]
        coll.count.return_value = len(existing)
        chroma.collections = {"facts": coll}
        chroma.query_collection.return_value = existing

        mm = MagicMock()
        mm.generate_once = AsyncMock(return_value="A")

        v = FactVerifier(chroma_store=chroma, model_manager=mm)
        v._ephemeral = frozenset()

        result = await v.verify(
            subject="user", predicate="works_at", object_val="Meta",
            fact_text="user | works_at | Meta",
        )

        mm.generate_once.assert_awaited_once()
        _, kwargs = mm.generate_once.call_args
        assert kwargs.get("disable_reasoning") is True
        assert kwargs.get("max_tokens", 0) > 4
        # Behavior is unchanged for a real answer.
        assert result.reason == "llm_update"

    @pytest.mark.asyncio
    async def test_adjudication_empty_answer_still_falls_back_safely(self):
        """A reasoning model that STILL returns empty (e.g. mocked directly)
        must not raise — the existing unparseable-fallback path is untouched
        by this fix, only the odds of hitting it are lowered."""
        from memory.fact_verification import FactVerifier, FactVerdict

        chroma = MagicMock()
        coll = MagicMock()
        existing = [{
            "id": "f1", "content": "user | works_at | Google",
            "metadata": {"subject": "user", "relation": "works_at",
                         "object": "Google", "confidence": 0.7},
        }]
        coll.count.return_value = len(existing)
        chroma.collections = {"facts": coll}
        chroma.query_collection.return_value = existing

        mm = MagicMock()
        mm.generate_once = AsyncMock(return_value="")

        v = FactVerifier(chroma_store=chroma, model_manager=mm)
        v._ephemeral = frozenset()

        result = await v.verify(
            subject="user", predicate="works_at", object_val="Meta",
            fact_text="user | works_at | Meta",
        )
        assert result.verdict == FactVerdict.STORE_AND_FLAG
        assert result.reason == "llm_unparseable_fallback"


# ---------------------------------------------------------------------------
# 2. memory/user_profile_schema.py — categorize_relation_deep (Layer 5)
# ---------------------------------------------------------------------------

class TestProfileSchemaCategorizeDeepBudget:
    @pytest.mark.asyncio
    async def test_layer5_llm_call_disables_reasoning_with_headroom(self, monkeypatch):
        import memory.user_profile_schema as ups

        relation = "zzz_sep27b_b4_unclassified_relation"
        # Force layers 1-4 to miss so the deep path reaches the Layer 5 LLM
        # call, and make sure no stale cache entry from a previous run
        # short-circuits it.
        monkeypatch.setattr(ups, "categorize_relation", lambda r: ups.ProfileCategory.PREFERENCES)
        ups._category_cache._cache.pop(relation.lower(), None)

        mm = MagicMock()
        mm.generate_once = AsyncMock(return_value="career")

        result = await ups.categorize_relation_deep(relation, model_manager=mm)

        mm.generate_once.assert_awaited_once()
        _, kwargs = mm.generate_once.call_args
        assert kwargs.get("disable_reasoning") is True
        assert kwargs.get("max_tokens", 0) > 10
        assert result == ups.ProfileCategory.CAREER

        # Clean up so this test doesn't leave state for a later run in the
        # same process (the cache is a module-level singleton).
        ups._category_cache._cache.pop(relation.lower(), None)


# ---------------------------------------------------------------------------
# 3. core/prompt/proposal_filter.py — ProposalFilter._llm_pairwise_rank
# ---------------------------------------------------------------------------

class TestProposalFilterPairwiseRankBudget:
    @pytest.mark.asyncio
    async def test_pairwise_rank_call_disables_reasoning_with_headroom(self):
        from core.prompt.proposal_filter import ProposalFilter

        mock_model = AsyncMock()
        mock_model.generate_once = AsyncMock(return_value="A")

        pf = ProposalFilter(model_manager=mock_model)
        proposals = [
            {"content": "A proposal", "metadata": {"title": "A", "proposal_type": "feature",
                                                    "priority": 8, "reasoning": "good"},
             "relevance_score": 0.9},
            {"content": "B proposal", "metadata": {"title": "B", "proposal_type": "feature",
                                                    "priority": 5, "reasoning": "ok"},
             "relevance_score": 0.7},
        ]

        result = await pf._llm_pairwise_rank(proposals, limit=1)

        mock_model.generate_once.assert_awaited_once()
        _, kwargs = mock_model.generate_once.call_args
        assert kwargs.get("disable_reasoning") is True
        assert kwargs.get("max_tokens", 0) > 4
        assert result[0]["metadata"]["title"] == "A"

    @pytest.mark.asyncio
    async def test_pairwise_rank_empty_answer_does_not_silently_pick_b(self):
        """Regression for the exact BC-89 symptom in this file: `"A" in choice`
        on an empty string is False, so an empty answer used to make `b` win
        EVERY comparison with no error at all. With headroom+disable_reasoning
        this should be rare in production; here we still confirm the (a or "")
        guard added alongside the fix doesn't itself raise on empty."""
        from core.prompt.proposal_filter import ProposalFilter

        mock_model = AsyncMock()
        mock_model.generate_once = AsyncMock(return_value="")

        pf = ProposalFilter(model_manager=mock_model)
        proposals = [
            {"content": "A proposal", "metadata": {"title": "A", "proposal_type": "feature",
                                                    "priority": 8, "reasoning": "good"},
             "relevance_score": 0.9},
            {"content": "B proposal", "metadata": {"title": "B", "proposal_type": "feature",
                                                    "priority": 5, "reasoning": "ok"},
             "relevance_score": 0.7},
        ]

        # Must not raise (AttributeError on None, etc.) even on an empty answer.
        result = await pf._llm_pairwise_rank(proposals, limit=1)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# 4. gui/wizard.py — _handle_api_key smoke test + wiki-index doc drift
# ---------------------------------------------------------------------------

class TestWizardApiKeySmokeTestBudget:
    @pytest.mark.asyncio
    async def test_api_key_smoke_call_disables_reasoning_with_headroom(self, monkeypatch):
        import gui.wizard as wizard

        # Avoid touching the real .env file.
        monkeypatch.setattr(wizard, "write_api_key_to_env", lambda key: True)

        orchestrator = MagicMock()
        orchestrator.model_manager.reinitialize_clients.return_value = True
        orchestrator.model_manager.generate_once = AsyncMock(return_value="OK")

        state = wizard.WizardState(step=wizard.WizardStep.API_KEY)
        key = "sk-or-" + ("x" * 30)

        response, new_state, done = await wizard._handle_api_key(key, state, orchestrator)

        orchestrator.model_manager.generate_once.assert_awaited_once()
        _, kwargs = orchestrator.model_manager.generate_once.call_args
        assert kwargs.get("disable_reasoning") is True
        assert kwargs.get("max_tokens", 0) > 10
        assert done is False
        assert new_state.collected_data.get("api_key_saved") is True

    @pytest.mark.asyncio
    async def test_api_key_smoke_test_empty_response_fails_key(self, monkeypatch):
        """Unchanged behavior: a genuinely empty response (bad key / dead
        route) still reports failure instead of proceeding."""
        import gui.wizard as wizard

        monkeypatch.setattr(wizard, "write_api_key_to_env", lambda key: True)

        orchestrator = MagicMock()
        orchestrator.model_manager.reinitialize_clients.return_value = True
        orchestrator.model_manager.generate_once = AsyncMock(return_value="")

        state = wizard.WizardState(step=wizard.WizardStep.API_KEY)
        key = "sk-or-" + ("x" * 30)

        response, new_state, done = await wizard._handle_api_key(key, state, orchestrator)
        assert "didn't work" in response
        assert done is False


class TestWizardWikiIndexDocDrift:
    def test_wiki_index_prompt_has_no_stale_owner_repo_literal(self):
        """BC-71: the wizard used to hardcode github.com/lukeh/daemon/releases,
        which is not this project's real repository path. It should point at
        the Releases page generically instead of a literal that can drift."""
        import gui.wizard as wizard

        text = wizard._get_wiki_index_prompt()
        assert "github.com/lukeh/daemon" not in text
        assert "GitHub Releases page" in text
        assert "daemon-wiki-index-v1.zip" in text
