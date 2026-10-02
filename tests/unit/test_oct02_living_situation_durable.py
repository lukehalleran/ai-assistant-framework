"""2026-10-02: living-situation facts are standing circumstances.

apartment_condition matched relation_classifier's "_condition" suffix and aged out
in 24h; lives_in moved identity -> living_situation, so the location resolver must
read both categories."""
import json

from memory.relation_classifier import ephemeral_ttl_hours, is_ephemeral_relation
from memory.user_profile_schema import is_living_situation_relation


def test_living_situation_relations_never_expire():
    for rel in ("apartment_condition", "living_duration", "living_with",
                "housing_stress", "moving_out_timeline", "lease_end", "lives_in"):
        assert is_living_situation_relation(rel), rel
        assert ephemeral_ttl_hours(rel) is None, rel


def test_unrelated_condition_relations_still_transient():
    assert is_ephemeral_relation("health_condition")
    assert is_ephemeral_relation("current_condition")
    assert not is_living_situation_relation("health_condition")


def test_location_resolver_reads_living_situation_lives_in(tmp_path):
    from utils.location_resolver import LocationResolver
    profile = {"categories": {"identity": [], "living_situation": [
        {"relation": "lives_in", "value": "Springfield, IL", "is_current": True,
         "timestamp": "2026-10-02T12:00:00"}]}}
    p = tmp_path / "user_profile.json"
    p.write_text(json.dumps(profile), encoding="utf-8")
    r = LocationResolver(profile_path=str(p))
    assert "Springfield" in (r._location_from_profile() or "")
