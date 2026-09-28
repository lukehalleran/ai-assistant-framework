"""tests/unit/test_sep27b_B1.py

Lane B, batch B1 (2026-09-27): "unavailable" is never "empty" — Google
sibling readers, passive email, semantic disabled.

class: BC-47, BC-58, BC-70, BC-72

Three closures under test:

1. Sibling `get_credentials() is None -> []` readers (google_calendar.py,
   gmail_search.py, google_contacts.py) now expose `unavailable_reason()`
   (the E1 shape, core/email/gmail_provider.py) instead of falling silent;
   `utils/preflight.py` WARNs (disk-only, no network) when a Google token
   file is expired with no refresh token.
2. Passive [RELEVANT EMAILS] (core/prompt/gatherer_knowledge.py +
   core/prompt/formatter.py) renders a named failure instead of nothing
   when `core.email.registry.provider_coverage()['failed']` is non-empty,
   and [ACTIVE FEATURES]' catch-all line names it too — using ONLY the
   curated, human-safe `provider_coverage()` vocabulary, never a raw
   `_section_outcomes` reason (privacy: test_reason_labels_never_appear_in_
   output in test_feature_inventory_outcomes.py must keep passing
   unmodified).
3. `knowledge/semantic_search.py` exposes a new cached `index_available()`
   (no full load attempt) for callers that want to tell a process-wide
   DISABLED index (missing/unmounted external FAISS index) apart from a
   loadable one. `_build_feature_inventory` renders
   `semantic=OFF(index not found)` instead of repeating "Could not check
   this turn: semantic" forever, keyed off the SAME "index_not_loaded"
   reason `SemanticSearchIndex.search()` already reports every call when
   the index is absent — no change to `_get_semantic_chunks_timed`'s
   in-flight/timeout/exception guards, which
   test_gatherer_outcomes_background_knowledge.py and
   test_sep09_latency_metrics.py pin unconditionally reaching the FAISS
   call (see this batch's handoff for why a scheduling short-circuit was
   NOT wired in).

No LLM/network/store access — everything mocked or file-path-controlled.
"""

import asyncio

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from utils.retrieval_outcome import OutcomeList, outcome_status


# ---------------------------------------------------------------------------
# Fixtures: reset the new module-level failure/cache state around every test
# so nothing leaks into (or out of) other test files in the same process.
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_sibling_module_state():
    import core.actions.google_calendar as calendar_mod
    import core.actions.gmail_search as gmail_search_mod
    import core.actions.google_contacts as contacts_mod
    import knowledge.semantic_search as sem_mod

    calendar_mod.clear_cache()
    gmail_search_mod.clear_cache()
    contacts_mod.clear_cache()
    sem_mod._availability_cache = {"value": None, "ts": 0.0}
    yield
    calendar_mod.clear_cache()
    gmail_search_mod.clear_cache()
    contacts_mod.clear_cache()
    sem_mod._availability_cache = {"value": None, "ts": 0.0}


# ---------------------------------------------------------------------------
# 1a. core/actions/google_calendar.py
# ---------------------------------------------------------------------------

class TestGoogleCalendarUnavailableReason:
    @pytest.mark.asyncio
    async def test_refresh_failure_records_reason(self):
        from core.actions.google_calendar import fetch_upcoming_events, unavailable_reason

        mock_auth = MagicMock()
        mock_auth.is_authenticated = True
        mock_auth.get_credentials.return_value = None
        mock_auth.auth_failure = "Gmail/Google authorization expired or was revoked"

        with patch("config.app_config.GOOGLE_CALENDAR_ENABLED", True), \
             patch("core.actions.google_auth.get_google_auth", return_value=mock_auth):
            result = await fetch_upcoming_events()

        assert result == []  # unchanged contract
        assert unavailable_reason() == "Gmail/Google authorization expired or was revoked"

    @pytest.mark.asyncio
    async def test_success_clears_recorded_failure(self):
        from core.actions.google_calendar import fetch_upcoming_events, unavailable_reason

        mock_auth = MagicMock()
        mock_auth.is_authenticated = True
        mock_auth.get_credentials.return_value = None
        mock_auth.auth_failure = "revoked"
        with patch("config.app_config.GOOGLE_CALENDAR_ENABLED", True), \
             patch("core.actions.google_auth.get_google_auth", return_value=mock_auth):
            await fetch_upcoming_events()
        assert unavailable_reason() == "revoked"

        # A later successful fetch clears it (module-level _LAST_FAILURE;
        # `auth.auth_failure` also reads healthy on the mock now).
        mock_auth2 = MagicMock()
        mock_auth2.is_authenticated = True
        mock_creds = MagicMock()
        mock_creds.token = "tok"
        mock_auth2.get_credentials.return_value = mock_creds
        mock_auth2.auth_failure = None

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"items": []}
        mock_client = AsyncMock()
        mock_client.get.return_value = mock_resp
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("config.app_config.GOOGLE_CALENDAR_ENABLED", True), \
             patch("core.actions.google_auth.get_google_auth", return_value=mock_auth2), \
             patch("httpx.AsyncClient", return_value=mock_client):
            await fetch_upcoming_events()

        assert unavailable_reason() is None

    def test_clear_cache_resets_failure(self):
        import core.actions.google_calendar as calendar_mod

        calendar_mod._record_failure("stale")
        assert calendar_mod.unavailable_reason() == "stale"
        calendar_mod.clear_cache()
        assert calendar_mod.unavailable_reason() is None

    def test_no_failure_by_default(self):
        from core.actions.google_calendar import unavailable_reason

        assert unavailable_reason() is None


# ---------------------------------------------------------------------------
# 1b. core/actions/gmail_search.py
# ---------------------------------------------------------------------------

class TestGmailSearchUnavailableReason:
    @pytest.mark.asyncio
    async def test_refresh_failure_records_reason(self):
        from core.actions.gmail_search import search_gmail_contacts, unavailable_reason

        mock_auth = MagicMock()
        mock_auth.is_authenticated = True
        mock_auth.has_scope.return_value = True
        mock_auth.get_credentials.return_value = None
        mock_auth.auth_failure = "Gmail token revoked"

        with patch("config.app_config.GOOGLE_GMAIL_SEARCH_ENABLED", True), \
             patch("core.actions.google_auth.get_google_auth", return_value=mock_auth):
            result = await search_gmail_contacts("Pat")

        assert result == []  # unchanged contract
        assert unavailable_reason() == "Gmail token revoked"

    def test_clear_cache_resets_failure(self):
        import core.actions.gmail_search as gmail_search_mod

        gmail_search_mod._record_failure("stale")
        assert gmail_search_mod.unavailable_reason() == "stale"
        gmail_search_mod.clear_cache()
        assert gmail_search_mod.unavailable_reason() is None


# ---------------------------------------------------------------------------
# 1c. core/actions/google_contacts.py
# ---------------------------------------------------------------------------

class TestGoogleContactsUnavailableReason:
    @pytest.mark.asyncio
    async def test_refresh_failure_records_reason(self):
        from core.actions.google_contacts import search_contacts, unavailable_reason

        mock_auth = MagicMock()
        mock_auth.is_authenticated = True
        mock_auth.has_scope.return_value = True
        mock_auth.get_credentials.return_value = None
        mock_auth.auth_failure = "Google Contacts token revoked"

        with patch("config.app_config.GOOGLE_CONTACTS_ENABLED", True), \
             patch("core.actions.google_auth.get_google_auth", return_value=mock_auth):
            result = await search_contacts("Maren")

        assert result == []  # unchanged contract
        assert unavailable_reason() == "Google Contacts token revoked"

    def test_generic_api_error_still_surfaces_via_unavailable_reason(self):
        """`unavailable_reason()` folds in the pre-existing `_last_error`
        (non-200 API errors) too, not only auth failures."""
        import core.actions.google_contacts as contacts_mod

        contacts_mod._last_error = "HTTP 403 (saved) — PERMISSION_DENIED: ..."
        with patch("core.actions.google_auth.get_google_auth", return_value=None):
            assert contacts_mod.unavailable_reason() == "HTTP 403 (saved) — PERMISSION_DENIED: ..."

    def test_clear_cache_resets_failure(self):
        import core.actions.google_contacts as contacts_mod

        contacts_mod._last_error = "stale"
        with patch("core.actions.google_auth.get_google_auth", return_value=None):
            assert contacts_mod.unavailable_reason() == "stale"
        contacts_mod.clear_cache()
        with patch("core.actions.google_auth.get_google_auth", return_value=None):
            assert contacts_mod.unavailable_reason() is None


# ---------------------------------------------------------------------------
# 1d. utils/preflight.py
# ---------------------------------------------------------------------------

class TestPreflightGoogleToken:
    def test_warns_when_expired_no_refresh(self):
        from utils.preflight import PreflightResult, _check_google_token

        mock_auth = MagicMock()
        mock_auth.token_expired_no_refresh = True
        result = PreflightResult()
        with patch("core.actions.google_auth.get_google_auth", return_value=mock_auth):
            _check_google_token(result)

        assert len(result.warnings) == 1
        assert "reauth_google.py" in result.warnings[0]
        assert not result.fatal  # never aborts

    def test_silent_when_healthy(self):
        from utils.preflight import PreflightResult, _check_google_token

        mock_auth = MagicMock()
        mock_auth.token_expired_no_refresh = False
        result = PreflightResult()
        with patch("core.actions.google_auth.get_google_auth", return_value=mock_auth):
            _check_google_token(result)

        assert not result.warnings

    def test_silent_when_unconfigured(self):
        from utils.preflight import PreflightResult, _check_google_token

        result = PreflightResult()
        with patch("core.actions.google_auth.get_google_auth", return_value=None):
            _check_google_token(result)

        assert not result.warnings

    def test_wired_into_run_preflight(self):
        """run_preflight() must not raise and must include the new check
        (never blocks startup even when it warns)."""
        from utils.preflight import run_preflight

        mock_auth = MagicMock()
        mock_auth.token_expired_no_refresh = True
        with patch("core.actions.google_auth.get_google_auth", return_value=mock_auth):
            result = run_preflight()

        assert any("reauth_google.py" in w for w in result.warnings)
        assert result.ok  # warnings never fatal


# ---------------------------------------------------------------------------
# 2a. core/prompt/gatherer_knowledge.py — get_google_calendar_events
# ---------------------------------------------------------------------------

class TestGetGoogleCalendarEventsOutcome:
    def _mixin(self):
        from core.prompt.gatherer_knowledge import KnowledgeRetrievalMixin

        mixin = KnowledgeRetrievalMixin.__new__(KnowledgeRetrievalMixin)
        mixin.memory_coordinator = None
        return mixin

    @pytest.mark.asyncio
    async def test_auth_failure_returns_unavailable_outcome(self):
        mixin = self._mixin()
        with patch("config.app_config.GOOGLE_CALENDAR_ENABLED", True), \
             patch("core.actions.google_calendar.fetch_upcoming_events",
                   new_callable=AsyncMock, return_value=[]), \
             patch("core.actions.google_calendar.unavailable_reason",
                   return_value="Gmail/Google authorization expired or was revoked"):
            result = await mixin.get_google_calendar_events()

        assert result == []  # still empty — a consumer that only checks truthiness is unaffected
        status, reason = outcome_status(result)
        assert status == "unavailable"
        assert reason == "Gmail/Google authorization expired or was revoked"

    @pytest.mark.asyncio
    async def test_genuine_empty_stays_plain_no_results(self):
        mixin = self._mixin()
        with patch("config.app_config.GOOGLE_CALENDAR_ENABLED", True), \
             patch("core.actions.google_calendar.fetch_upcoming_events",
                   new_callable=AsyncMock, return_value=[]), \
             patch("core.actions.google_calendar.unavailable_reason",
                   return_value=None):
            result = await mixin.get_google_calendar_events()

        assert result == []
        status, _ = outcome_status(result)
        assert status == "no_results"

    @pytest.mark.asyncio
    async def test_events_present_unaffected(self):
        mixin = self._mixin()
        events = [{"summary": "Standup", "start": "2026-05-27T10:00:00-05:00",
                   "end": "2026-05-27T11:00:00-05:00", "all_day": False, "location": ""}]
        with patch("config.app_config.GOOGLE_CALENDAR_ENABLED", True), \
             patch("core.actions.google_calendar.fetch_upcoming_events",
                   new_callable=AsyncMock, return_value=events):
            result = await mixin.get_google_calendar_events()

        assert len(result) == 1
        status, _ = outcome_status(result)
        assert status == "succeeded"


# ---------------------------------------------------------------------------
# 2b. core/prompt/gatherer_knowledge.py — get_relevant_emails
# ---------------------------------------------------------------------------

class TestGetRelevantEmailsOutcome:
    def _run(self, monkeypatch, messages, failed_providers):
        import core.prompt.gatherer_knowledge as gk
        import core.email.service as svc
        import core.email.registry as reg

        class _FakeService:
            async def search(self, *a, **k):
                return messages

        monkeypatch.setattr(svc, "get_email_service", lambda: _FakeService())
        monkeypatch.setattr(
            reg, "provider_coverage",
            lambda: ({"searched": [], "unconnected": {}, "failed": failed_providers}
                     if failed_providers else {"searched": [], "unconnected": {}}))

        gatherer = gk.KnowledgeRetrievalMixin.__new__(gk.KnowledgeRetrievalMixin)
        gatherer._distress_active = False
        gatherer.memory_coordinator = None
        return asyncio.run(
            gk.KnowledgeRetrievalMixin.get_relevant_emails(gatherer, "check my gmail inbox"))

    def test_failed_provider_returns_unavailable_outcome(self, monkeypatch):
        result = self._run(
            monkeypatch, messages=[],
            failed_providers={"gmail": "Gmail/Google authorization expired or was revoked"})

        assert result == []
        status, reason = outcome_status(result)
        assert status == "unavailable"
        assert "Gmail/Google authorization expired or was revoked" in reason

    def test_genuine_empty_stays_plain_no_results(self, monkeypatch):
        result = self._run(monkeypatch, messages=[], failed_providers={})

        assert result == []
        status, _ = outcome_status(result)
        assert status == "no_results"

    def test_no_cue_or_contact_never_calls_provider_coverage(self, monkeypatch):
        """A query with no email cue / contact still short-circuits BEFORE
        even calling the service (unchanged early-return contract)."""
        import core.email.registry as reg

        calls = []
        monkeypatch.setattr(reg, "provider_coverage", lambda: calls.append(1) or {})

        import core.prompt.gatherer_knowledge as gk

        gatherer = gk.KnowledgeRetrievalMixin.__new__(gk.KnowledgeRetrievalMixin)
        gatherer._distress_active = False
        gatherer.memory_coordinator = None
        out = asyncio.run(
            gk.KnowledgeRetrievalMixin.get_relevant_emails(gatherer, "what a nice sunny day"))

        assert out == []
        assert calls == []


# ---------------------------------------------------------------------------
# 3a. core/prompt/formatter.py — [RELEVANT EMAILS] failure rendering
# ---------------------------------------------------------------------------

def _formatter():
    from core.prompt.formatter import PromptFormatter
    return PromptFormatter(token_manager=MagicMock(), time_manager=None)


def _prompt_context(**overrides):
    ctx = {
        "recent_conversations": [], "memories": [], "user_profile": "", "summaries": [],
        "recent_summaries": [], "semantic_summaries": [], "reflections": [],
        "recent_reflections": [], "semantic_reflections": [], "dreams": [],
        "semantic_chunks": [], "wiki": [], "user_uploads": [], "proposed_features": [],
        "web_search_results": None, "codebase_changes": {},
    }
    ctx.update(overrides)
    return ctx


class TestRelevantEmailsFailureSection:
    def test_renders_failure_notice_instead_of_silence(self):
        context = _prompt_context(
            relevant_emails=[],
            _section_outcomes={"relevant_emails": {
                "status": "unavailable",
                "reason": "Gmail/Google authorization expired or was revoked",
            }},
        )
        prompt = _formatter()._assemble_prompt(context=context, user_input="anything from Pat?")

        assert "[RELEVANT EMAILS]" in prompt
        assert "FAILED" in prompt
        assert "Gmail/Google authorization expired or was revoked" in prompt
        assert "empty inbox" in prompt

    def test_genuine_empty_renders_no_section(self):
        context = _prompt_context(
            relevant_emails=[],
            _section_outcomes={"relevant_emails": {"status": "no_results", "reason": ""}},
        )
        prompt = _formatter()._assemble_prompt(context=context, user_input="hello")

        assert "[RELEVANT EMAILS]" not in prompt

    def test_messages_present_unaffected(self):
        context = _prompt_context(
            relevant_emails=[{
                "date": "2026-09-20", "sender": "Pat <pat@example.com>",
                "subject": "Re: plans", "snippet": "sounds good", "provider": "gmail",
            }],
            _section_outcomes={"relevant_emails": {"status": "succeeded", "reason": ""}},
        )
        prompt = _formatter()._assemble_prompt(context=context, user_input="anything from Pat?")

        assert "[RELEVANT EMAILS] n=1" in prompt
        assert "FAILED" not in prompt.split("[RELEVANT EMAILS]")[1].split("[", 1)[0]


# ---------------------------------------------------------------------------
# 3b. core/prompt/formatter.py — [ACTIVE FEATURES] catch-all reason (email)
# ---------------------------------------------------------------------------

class TestActiveFeaturesEmailReason:
    def test_catchall_appends_safe_reason_when_provider_failed(self, monkeypatch):
        import core.email.registry as reg

        monkeypatch.setattr(
            reg, "provider_coverage",
            lambda: {"searched": [], "unconnected": {},
                     "failed": {"gmail": "Gmail/Google authorization expired or was revoked"}})

        context = {"_section_outcomes": {
            "relevant_emails": {"status": "unavailable", "reason": "timeout"}}}
        result = _formatter()._build_feature_inventory(context)

        last_line = result.split("\n")[-1]
        assert last_line.startswith("Could not check this turn:")
        assert "relevant_emails (Gmail/Google authorization expired or was revoked)" in last_line

    def test_catchall_bare_name_when_no_provider_failure(self, monkeypatch):
        """Backward compatibility: when provider_coverage() has nothing in
        'failed' (e.g. simply not connected/disabled), the line stays
        EXACTLY the pre-existing bare-name form —
        test_feature_inventory_outcomes.py's exact-match assertions pin
        this shape and must keep passing unmodified."""
        import core.email.registry as reg

        monkeypatch.setattr(
            reg, "provider_coverage",
            lambda: {"searched": [], "unconnected": {"gmail": "disabled"}})

        context = {"_section_outcomes": {
            "upcoming_schedule": {"status": "failed", "reason": "ValueError"},
            "relevant_emails": {"status": "unavailable", "reason": "timeout"},
        }}
        result = _formatter()._build_feature_inventory(context)

        assert result.split("\n")[-1] == "Could not check this turn: relevant_emails, upcoming_schedule"

    def test_other_sections_never_carry_a_reason(self, monkeypatch):
        """Privacy floor: only 'relevant_emails' may show a reason; any
        other not-checked name stays bare even when it happens to fail
        alongside a failed email provider."""
        import core.email.registry as reg

        monkeypatch.setattr(
            reg, "provider_coverage",
            lambda: {"searched": [], "unconnected": {},
                     "failed": {"gmail": "Gmail/Google authorization expired or was revoked"}})

        marker = "SENSITIVE_EXCEPTION_TEXT_MUST_NOT_LEAK"
        context = {"_section_outcomes": {
            "upcoming_schedule": {"status": "failed", "reason": marker},
            "relevant_emails": {"status": "unavailable", "reason": "timeout"},
        }}
        result = _formatter()._build_feature_inventory(context)

        assert marker not in result
        assert "upcoming_schedule (" not in result


# ---------------------------------------------------------------------------
# 3c. core/prompt/formatter.py — semantic OFF(index not found)
# ---------------------------------------------------------------------------

class TestSemanticIndexDisabledRendering:
    def test_index_not_loaded_renders_off_label_and_excluded_from_catchall(self):
        context = {"_section_outcomes": {
            "semantic": {"status": "unavailable", "reason": "index_not_loaded"}}}
        result = _formatter()._build_feature_inventory(context)

        assert "semantic=OFF(index not found)" in result
        assert "Could not check this turn" not in result

    def test_genuine_timeout_still_uses_generic_catchall(self):
        """A real transient failure (not the disabled state) is NOT
        special-cased — it still falls into the ordinary catch-all,
        unaffected by the new dedicated label."""
        context = {"_section_outcomes": {
            "semantic": {"status": "unavailable", "reason": "timeout"}}}
        result = _formatter()._build_feature_inventory(context)

        assert "semantic=OFF" not in result
        assert result.split("\n")[-1] == "Could not check this turn: semantic"

    def test_no_semantic_outcome_unaffected(self):
        context = {"_section_outcomes": {}}
        result = _formatter()._build_feature_inventory(context)

        assert "semantic=" not in result
        assert "Could not check this turn" not in result


# ---------------------------------------------------------------------------
# 3d. knowledge/semantic_search.py — index_available()
# ---------------------------------------------------------------------------

class TestIndexAvailable:
    def test_false_when_files_missing(self, monkeypatch, tmp_path):
        import knowledge.semantic_search as sem

        monkeypatch.setattr(sem, "INDEX_PATH", str(tmp_path / "missing.faiss"))
        monkeypatch.setattr(sem, "META_PATH", str(tmp_path / "missing.parquet"))
        monkeypatch.setattr(sem, "get_index", lambda: MagicMock(loaded=False))

        assert sem.index_available() is False

    def test_true_when_already_loaded(self, monkeypatch):
        import knowledge.semantic_search as sem

        monkeypatch.setattr(sem, "get_index", lambda: MagicMock(loaded=True))
        assert sem.index_available() is True

    def test_true_when_files_present(self, monkeypatch, tmp_path):
        import knowledge.semantic_search as sem

        index_file = tmp_path / "vector_index_ivf.faiss"
        meta_file = tmp_path / "metadata.parquet"
        index_file.write_text("stub")
        meta_file.write_text("stub")
        monkeypatch.setattr(sem, "INDEX_PATH", str(index_file))
        monkeypatch.setattr(sem, "META_PATH", str(meta_file))
        monkeypatch.setattr(sem, "get_index", lambda: MagicMock(loaded=False))

        assert sem.index_available() is True

    def test_result_is_cached_within_ttl(self, monkeypatch, tmp_path):
        import knowledge.semantic_search as sem
        import os as os_mod

        monkeypatch.setattr(sem, "INDEX_PATH", str(tmp_path / "missing.faiss"))
        monkeypatch.setattr(sem, "META_PATH", str(tmp_path / "missing.parquet"))
        monkeypatch.setattr(sem, "get_index", lambda: MagicMock(loaded=False))

        calls = {"n": 0}
        real_exists = os_mod.path.exists

        def _counting_exists(path):
            calls["n"] += 1
            return real_exists(path)

        monkeypatch.setattr(sem.os.path, "exists", _counting_exists)

        first = sem.index_available()
        n_after_first = calls["n"]
        assert n_after_first > 0
        second = sem.index_available()

        assert first == second is False
        assert calls["n"] == n_after_first  # no new stat calls — cache hit
