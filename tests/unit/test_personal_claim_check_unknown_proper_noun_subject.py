"""An unknown TitleCase subject is not a person.

Incident 2026-10-20: course-module names ("Arena has finished the run",
"Process module completed the batch") kept a span personal because
``_is_personal_completion`` treated ANY untagged proper-noun subject as a
person. Now a PROPN subject is a person only when NER types it PERSON, the
name is a KNOWN person (graph person entity / cached contact via
``utils.known_person``), or it is the user's name. Class: BC-84 (same family
as the #93 CLM5 lane). Drives THE deployed ``audit_personal_claims``.
"""
import asyncio
import json

import pytest

import core.personal_claim_check as pcc
from core.personal_claim_check import audit_personal_claims, build_personal_evidence
import utils.known_person as known_person


class _Model:
    def __init__(self, raw):
        self.raw = raw

    async def generate_once(self, prompt, **kwargs):
        return json.dumps(self.raw)


def _wrap(text, wrapped):
    return text.replace(" ", "\n    ", 3) if wrapped else text


def _audit(span, kind="completion", wrapped=False):
    shown = _wrap(span, wrapped)
    payload = {"claims": [{"text": shown, "status": "insufficient", "kind": kind, "evidence": []}]}
    return asyncio.run(audit_personal_claims(
        "Okay. " + shown + " Moving on.", build_personal_evidence("what next?", {}), _Model(payload)))


class _Node:
    entity_type = "person"


class _Graph:
    def __init__(self, names):
        self.names = {n.lower() for n in names}

    def resolve_entity(self, name):
        return name.lower() if name.lower() in self.names else None

    def get_entity(self, eid):
        return _Node()


@pytest.fixture(autouse=True)
def _isolation(monkeypatch):
    monkeypatch.setattr(pcc, "get_user_display_name", lambda: "Tamsin", raising=False)
    monkeypatch.setattr(known_person, "_REGISTERED_GRAPH", None)
    # No contacts cache in tests.
    import core.actions.google_contacts as gc
    monkeypatch.setattr(gc, "is_cached_contact_name", lambda n: False)


MODULE_SPANS = [
    "Arena has finished the run.",
    "Process module completed the batch.",
    "Wrenfield has completed the queue.",
    "Dispose has released the entity.",
    "Create completed the batch.",
    "Decide has finished the run.",
]


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("span", MODULE_SPANS)
def test_module_name_subject_is_dropped_as_subject_matter(span, wrapped):
    result = _audit(span, wrapped=wrapped)
    assert result.claims == []
    assert result.subject_drops == 1


@pytest.mark.parametrize("wrapped", [False, True])
def test_unknown_name_dropped_known_person_kept(wrapped, monkeypatch):
    span = "Marlow has the forms."
    assert _audit(span, wrapped=wrapped).subject_drops == 1
    known_person.register_graph(_Graph(["Marlow"]))
    kept = _audit(span, wrapped=wrapped)
    assert kept.subject_drops == 0 and len(kept.claims) == 1


@pytest.mark.parametrize("wrapped", [False, True])
def test_user_name_and_ner_person_stay_personal(wrapped):
    assert _audit("Tamsin keeps the spare key.", wrapped=wrapped).subject_drops == 0
    assert _audit("Rowan has the forms.", wrapped=wrapped).subject_drops == 0


def test_known_person_lookup_is_shared_with_the_prompt_gatherer(monkeypatch):
    from core.prompt import gatherer_knowledge as gk
    g = _Graph(["Marlow"])
    assert gk._is_known_person(g, "Marlow") is True
    assert gk._is_known_person(g, "Arena") is False
    assert known_person.is_known_person(g, "Marlow") is True


@pytest.mark.parametrize("wrapped", [False, True])
def test_bare_intransitive_report_with_unknown_name_fails_closed(wrapped):
    # "HW4 submitted." is the elided-agent report shape: no object, no aux.
    result = _audit("HW4 submitted.", wrapped=wrapped)
    assert result.subject_drops == 0 and len(result.claims) == 1
