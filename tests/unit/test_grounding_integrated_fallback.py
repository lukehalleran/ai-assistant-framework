"""A05a — integrated-fallback builder (F04 / G12 acceptance 3-6), INACTIVE.

`build_integrated_fallback` is a new PURE function beside the correction
section in core/grounding_check.py, with NO caller yet (A05b wires it into
gui/handlers.py in a later batch) — this file drives it directly.

Two kinds: "spliced" replaces the ONE sentence carrying the claim with a
visible correction sentence, keeping every other sentence/paragraph break
as-is; "standalone" (claim not located, or located ambiguously) drops the
flawed draft prose entirely for one short corrective reply. A trailing
action-proposal card is split off before matching and reattached verbatim
for both kinds. Fixtures appear in clean and wrapped/indented form per
BC-64 (a client-wrapped input can defeat a clean-only predicate).
"""
import pytest

from core.grounding_check import (
    GroundingVerdict,
    _MAX_CORRECTION_CHARS,
    _sentence_chunks,
    _truncate_correction,
)

try:
    from core.grounding_check import build_integrated_fallback, IntegratedFallback
except ImportError:  # expected before implementation — failing-before proof
    build_integrated_fallback = None
    IntegratedFallback = None


# A distinctive card, structurally identical to what handlers appends
# (`_PROPOSAL_CARD_RE = r"\n\n---\n\*\*[a-z][a-z_]*\*\*"`).
_CARD = (
    "\n\n---\n**calendar_create_event** — Tuesday, Sep 15 at 2:00 PM\n"
    "Approve or edit below."
)

_IDIOM = "> ⚠️ Correction:"


def _verdict(claim: str, correction: str = "The correct fact is X.",
             why_false: str = "It is well established to be false.",
             confidence: float = 0.9) -> GroundingVerdict:
    return GroundingVerdict(
        false_claim_present=True,
        claim=claim,
        why_false=why_false,
        confidence=confidence,
        correction=correction,
    )


def _require_impl():
    if build_integrated_fallback is None:
        pytest.fail(
            "build_integrated_fallback is not implemented yet in "
            "core/grounding_check.py (failing-before proof)."
        )


# ---------------------------------------------------------------------------
# Splice: claim verbatim in exactly one sentence
# ---------------------------------------------------------------------------

class TestSpliceVerbatimSingleSentence:
    CLAIM = "The refrigerator mother theory is closer to the truth"
    RESPONSE = (
        "I hear how hard this has been. "
        f"{CLAIM}. "
        "You are not alone in feeling this way. "
        "Let me know if you want to talk more."
    )

    def test_kind_is_spliced(self):
        _require_impl()
        result = build_integrated_fallback(self.RESPONSE, _verdict(self.CLAIM))
        assert result is not None
        assert result.kind == "spliced"

    def test_claim_sentence_absent(self):
        _require_impl()
        result = build_integrated_fallback(self.RESPONSE, _verdict(self.CLAIM))
        assert self.CLAIM not in result.text

    def test_correction_text_present(self):
        _require_impl()
        result = build_integrated_fallback(
            self.RESPONSE, _verdict(self.CLAIM, correction="Autism is neurodevelopmental.")
        )
        assert "Autism is neurodevelopmental." in result.text

    def test_other_sentences_preserved_in_order(self):
        _require_impl()
        result = build_integrated_fallback(self.RESPONSE, _verdict(self.CLAIM))
        opener = "I hear how hard this has been."
        closer = "Let me know if you want to talk more."
        middle = "You are not alone in feeling this way."
        assert opener in result.text
        assert middle in result.text
        assert closer in result.text
        assert result.text.index(opener) < result.text.index(middle)
        assert result.text.index(middle) < result.text.index(closer)

    def test_wrapped_claim_across_a_line_break_is_safe_standalone(self):
        """F1(a): a line break always ends a sentence, so a claim wrapped
        across a real newline (BC-64) spans two chunks and can't be located
        as one — still SAFE, since standalone never shows the claim either."""
        _require_impl()
        wrapped_response = (
            "I hear how hard this has been.\n"
            "The refrigerator mother theory is\n  closer to the truth.\n"
            "You are not alone in feeling this way.\n"
            "Let me know if you want to talk more."
        )
        result = build_integrated_fallback(wrapped_response, _verdict(self.CLAIM))
        assert result is not None
        assert result.kind == "standalone"
        assert "closer to the truth" not in result.text
        assert "refrigerator mother" not in result.text


# ---------------------------------------------------------------------------
# Token-overlap splice: reworded claim in one sentence vs. two-sentence control
# ---------------------------------------------------------------------------

class TestTokenOverlapSplice:
    CLAIM = "The refrigerator mother theory of autism is closer to the truth"
    # Reworded but high content-token overlap with CLAIM.
    REWORDED_SENTENCE = (
        "Honestly the refrigerator mother theory of autism is much closer "
        "to the truth than people admit"
    )

    def test_single_reworded_sentence_splices(self):
        _require_impl()
        response = (
            "That's a hard thing to sit with. "
            f"{self.REWORDED_SENTENCE}. "
            "I'm glad you brought it up."
        )
        result = build_integrated_fallback(response, _verdict(self.CLAIM))
        assert result is not None
        assert result.kind == "spliced"
        assert self.REWORDED_SENTENCE not in result.text
        assert "I'm glad you brought it up." in result.text

    def test_same_overlap_in_two_sentences_is_standalone(self):
        """Control: the SAME reworded content appears in two sentences, so
        the location is ambiguous and must NOT splice either one."""
        _require_impl()
        response = (
            f"{self.REWORDED_SENTENCE}. "
            "For what it's worth, I'd also say the refrigerator mother "
            "theory of autism lands closer to the truth than people admit. "
            "I'm glad you brought it up."
        )
        result = build_integrated_fallback(response, _verdict(self.CLAIM))
        assert result is not None
        assert result.kind == "standalone"
        assert self.REWORDED_SENTENCE not in result.text
        assert "I'm glad you brought it up." not in result.text


# ---------------------------------------------------------------------------
# Parent review round 1, F1/F2: segmentation (a line break always ends a
# sentence; terminal punctuation only ends one at end-of-line or before
# whitespace+non-lowercase) and location (word-bounded containment; digit
# tokens always count; overlap needs a content-token minimum).
# ---------------------------------------------------------------------------

class TestSegmentation:
    @pytest.mark.parametrize("response,claim", [
        ("It costs $2.50 per ride. Buses run hourly.", "costs $2"),
        ("Your class starts at 3 p.m. on Monday. Bring your laptop.",
         "starts at 3 p.m"),
    ])
    def test_decimal_and_abbreviation_never_fragment(self, response, claim):
        _require_impl()
        result = build_integrated_fallback(response, _verdict(claim))
        assert result is not None
        assert "50 per ride" not in result.text
        assert "m. on" not in result.text

    def test_heading_and_paragraph_break_survive_byte_identical(self):
        _require_impl()
        response = "Here's the plan:\n\nThe deadline is Friday.\nSee you then."
        result = build_integrated_fallback(response, _verdict("The deadline is Friday"))
        assert result.text.startswith("Here's the plan:\n\n")
        assert result.text.endswith("\nSee you then.")

    def test_only_the_matching_bullet_line_changes(self):
        _require_impl()
        response = ("Your schedule:\n- Math at 9\n- The deadline is Friday\n"
                    "- Gym at 5\nLet me know.")
        result = build_integrated_fallback(
            response, _verdict("The deadline is Friday", correction="The deadline is Saturday."))
        assert "\n- Math at 9\n" in result.text
        assert "\n- Gym at 5\n" in result.text
        assert result.text.endswith("Let me know.")
        # The "- " list marker is kept in front of the correction (F1 decision).
        assert "- Correction: The deadline is Saturday." in result.text

    @pytest.mark.parametrize("sample", [
        "a. b! c?", "x\n\ny.  \n", "no punct at all",
        "  lead ws. tail ws  \n\n", "1. first\n2. second",
    ])
    def test_chunk_concatenation_is_exact(self, sample):
        assert "".join(_sentence_chunks(sample)) == sample


class TestBoundedContainment:
    def test_may_does_not_match_inside_maybe(self):
        _require_impl()
        response = "Maybe we can meet later. Your exam is in June."
        result = build_integrated_fallback(
            response, _verdict("May", correction="The exam is in May."))
        assert result.kind == "standalone"
        assert "Your exam is in June" not in result.text

    def test_short_numeric_claim_does_not_match_inside_a_longer_number(self):
        _require_impl()
        response = "Lunch is at 11 AM. The appointment moved to 1:00 PM."
        result = build_integrated_fallback(
            response, _verdict("1 AM", correction="It is at 1 PM."))
        assert result.kind == "standalone"
        assert "appointment moved to 1:00 PM" not in result.text

    # Positive control: TestTokenOverlapSplice.test_single_reworded_sentence_splices.

    def test_overlap_claim_spanning_an_abbreviation_split_is_standalone(self):
        """Parent review round 2: "Sept. 15" still splits (whitespace then a
        digit). The half before the split reaches the 0.8 overlap while the
        claim's date token sits in the same-line neighbour — splicing only
        that half would leave " 15 at noon." (the flawed date) behind."""
        _require_impl()
        response = "The weekly team meeting is on Sept. 15 at noon. Bring notes."
        for text in (response, response.replace(" at noon", "\n  at noon")):
            result = build_integrated_fallback(text, _verdict(
                "the weekly team meeting is on Sept 15", correction="It is on Sept. 22."))
            assert result.kind == "standalone", text
            assert "15 at" not in result.text


# ---------------------------------------------------------------------------
# Standalone
# ---------------------------------------------------------------------------

class TestStandalone:
    DISTINCTIVE_DRAFT_TOKEN = "xylophone-quokka-transit"

    def test_claim_absent_is_standalone(self):
        _require_impl()
        claim = "Widgets are always blue"
        response = (
            f"Here is a totally unrelated draft mentioning {self.DISTINCTIVE_DRAFT_TOKEN} "
            "and nothing about widgets or their color at all."
        )
        result = build_integrated_fallback(response, _verdict(claim))
        assert result is not None
        assert result.kind == "standalone"
        assert self.DISTINCTIVE_DRAFT_TOKEN not in result.text

    def test_claim_in_two_sentences_is_standalone(self):
        _require_impl()
        claim = "The refrigerator mother theory is closer to the truth"
        response = (
            f"{claim}, in my honest opinion. "
            f"Again, I do think {claim.lower()} once you look at the history. "
            "Thanks for sharing that with me."
        )
        result = build_integrated_fallback(response, _verdict(claim))
        assert result is not None
        assert result.kind == "standalone"

    def test_correction_present_in_standalone(self):
        _require_impl()
        claim = "Widgets are always blue"
        response = f"Contains {self.DISTINCTIVE_DRAFT_TOKEN} only."
        result = build_integrated_fallback(
            response, _verdict(claim, correction="Widgets come in many colors.")
        )
        assert "Widgets come in many colors." in result.text

    def test_elevated_changes_lead_wording_only(self):
        _require_impl()
        claim = "Widgets are always blue"
        response = f"Contains {self.DISTINCTIVE_DRAFT_TOKEN} only."
        correction = "Widgets come in many colors."
        plain = build_integrated_fallback(response, _verdict(claim, correction=correction))
        elevated = build_integrated_fallback(
            response, _verdict(claim, correction=correction), elevated=True
        )
        assert plain.kind == "standalone"
        assert elevated.kind == "standalone"
        assert plain.text != elevated.text
        # The correction fact itself is unchanged by the elevated flag.
        assert correction in plain.text
        assert correction in elevated.text


# ---------------------------------------------------------------------------
# Proposal card: reattached verbatim, never counted toward sentence matching
# ---------------------------------------------------------------------------

class TestProposalCard:
    CLAIM = "The refrigerator mother theory is closer to the truth"

    def test_card_reattached_verbatim_when_spliced(self):
        _require_impl()
        response = (
            "I hear you. "
            f"{self.CLAIM}. "
            "Take care." + _CARD
        )
        result = build_integrated_fallback(response, _verdict(self.CLAIM))
        assert result.kind == "spliced"
        assert result.text.endswith(_CARD)

    def test_card_reattached_verbatim_when_standalone(self):
        _require_impl()
        claim = "Widgets are always blue"
        response = "Some unrelated draft text." + _CARD
        result = build_integrated_fallback(response, _verdict(claim))
        assert result.kind == "standalone"
        assert result.text.endswith(_CARD)

    def test_card_text_never_counts_toward_sentence_matching(self):
        """F4: claim exists ONLY inside the card text; a bug would splice
        into (and corrupt) the card — it must survive byte-identical."""
        _require_impl()
        claim = "Tuesday, Sep 15 at 2:00 PM"
        response = "An ordinary reply with no matching claim text." + _CARD
        result = build_integrated_fallback(response, _verdict(claim))
        assert result.kind == "standalone"
        assert result.text.endswith(_CARD)


# ---------------------------------------------------------------------------
# None: correction empty / non-substantive
# ---------------------------------------------------------------------------

class TestNoneSemantics:
    """Empty, whitespace-only, and non-substantive (pure advice-to-verify,
    per _substantive_correction_text/_is_advice_shaped) corrections all
    return None — A05b then ships the draft unmodified, same as today."""

    @pytest.mark.parametrize("correction", [
        "",
        "   \n\t  ",
        "Please verify the correct date.",
    ])
    def test_non_substantive_correction_returns_none(self, correction):
        _require_impl()
        result = build_integrated_fallback(
            "Some response text.", _verdict("a claim", correction=correction)
        )
        assert result is None


# ---------------------------------------------------------------------------
# Idiom guard: never the suffix idiom, never response + suffix
# ---------------------------------------------------------------------------

class TestIdiomGuard:
    CASES = [
        # (name, response, claim)
        (
            "spliced",
            "I hear you. The refrigerator mother theory is closer to the "
            "truth. Take care.",
            "The refrigerator mother theory is closer to the truth",
        ),
        (
            "standalone",
            "Contains no matching claim text at all.",
            "Widgets are always blue",
        ),
    ]

    @pytest.mark.parametrize("name,response,claim", CASES)
    def test_no_suffix_idiom_and_not_response_plus_suffix(self, name, response, claim):
        _require_impl()
        verdict = _verdict(claim, correction="The correct fact is X.")
        result = build_integrated_fallback(response, verdict)
        assert result is not None
        # Local literal shapes, not the retired suffix builder: the two forms
        # it used to append (plain / elevated).
        t = _truncate_correction(verdict.correction)
        plain = f"\n\n> ⚠️ Correction: {t}"
        elevated = (
            "\n\n> ⚠️ One thing I want to gently set straight, because it "
            f"matters: {t}"
        )
        for shape in (plain, elevated):
            assert result.text != response + shape
            assert not result.text.endswith(shape)
        assert _IDIOM not in result.text


# ---------------------------------------------------------------------------
# Truncation: correction over 300 chars truncated through the existing helper
# ---------------------------------------------------------------------------

class TestTruncation:
    LONG_CORRECTION = (
        "The refrigerator mother theory was thoroughly discredited by "
        "decades of neurodevelopmental research. It has no scientific "
        "support whatsoever and was abandoned by every major medical body. "
        "Autism is understood today as a neurodevelopmental condition with "
        "a strong genetic component, not something caused by cold or "
        "distant parenting as the discredited theory once claimed. "
        "Continuing to cite it does real harm to affected families "
        "everywhere and should be avoided by any careful writer."
    )

    # F3: the prior length-only bound passed even with `_truncate_correction`
    # monkeypatched to identity (batch packet). Assert against the real
    # helper directly: LONG_CORRECTION absent, its truncated form present.
    @pytest.mark.parametrize("claim,response", [
        ("Widgets are always blue", "Unrelated draft."),
        ("The refrigerator mother theory is closer to the truth",
         "Intro sentence here. The refrigerator mother theory is closer to "
         "the truth. Closing sentence here."),
    ])
    def test_correction_is_truncated_through_existing_helper(self, claim, response):
        _require_impl()
        assert len(self.LONG_CORRECTION) > _MAX_CORRECTION_CHARS
        result = build_integrated_fallback(
            response, _verdict(claim, correction=self.LONG_CORRECTION)
        )
        assert result is not None
        assert self.LONG_CORRECTION not in result.text
        assert _truncate_correction(self.LONG_CORRECTION) in result.text


# ---------------------------------------------------------------------------
# Inactivity proof (recorded in the evidence packet; not a pytest assertion
# of repo-wide grep behavior since that belongs in the batch packet, but a
# smoke check that the symbol is genuinely new/unused within this module's
# own public surface beyond its definition).
# ---------------------------------------------------------------------------

class TestNoActiveCaller:
    def test_not_in_public_all_export_yet(self):
        """A05a is INACTIVE by design (F04 / G12): the builder exists but is
        not exported or wired anywhere. It stays out of __all__ until A05b
        wires it in gui/handlers.py."""
        _require_impl()
        import core.grounding_check as gc
        assert "build_integrated_fallback" not in gc.__all__


# ---------------------------------------------------------------------------
# H01 (owner hardcoding review item 1): the claim-location thresholds
# externalize to config/config.yaml grounding_check: -> config/app_config.py
# GROUNDING_FALLBACK_* -> config/schema.py GroundingCheckSection, replacing
# the former module constants _CLAIM_OVERLAP_THRESHOLD (0.8) and
# _MIN_CLAIM_TOKENS_FOR_OVERLAP (3). Behaviour at the default config is
# byte-identical; every test above stays green unmodified.
# ---------------------------------------------------------------------------

class TestFallbackConfigDefaults:
    def test_app_config_overlap_threshold_default(self):
        from config import app_config
        assert app_config.GROUNDING_FALLBACK_CLAIM_OVERLAP_THRESHOLD == 0.8

    def test_app_config_min_claim_tokens_default(self):
        from config import app_config
        assert app_config.GROUNDING_FALLBACK_MIN_CLAIM_TOKENS == 3

    def test_schema_section_carries_both_defaults(self):
        from config.schema import GroundingCheckSection
        section = GroundingCheckSection()
        assert section.fallback_claim_overlap_threshold == 0.8
        assert section.fallback_min_claim_tokens == 3


class TestFallbackConfigBounds:
    def test_overlap_threshold_rejects_zero(self):
        from config.schema import GroundingCheckSection
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            GroundingCheckSection(fallback_claim_overlap_threshold=0.0)

    def test_overlap_threshold_rejects_above_one(self):
        from config.schema import GroundingCheckSection
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            GroundingCheckSection(fallback_claim_overlap_threshold=1.5)

    def test_min_claim_tokens_rejects_zero(self):
        from config.schema import GroundingCheckSection
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            GroundingCheckSection(fallback_min_claim_tokens=0)


class TestFallbackConfigYaml:
    def test_yaml_has_both_keys_with_defaults(self):
        from config.app_config import load_yaml_config
        raw = load_yaml_config("config.yaml")
        section = raw["grounding_check"]
        assert section["fallback_claim_overlap_threshold"] == 0.8
        assert section["fallback_min_claim_tokens"] == 3

    def test_yaml_values_validate_through_schema(self):
        from config.app_config import load_yaml_config
        from config.schema import GroundingCheckSection
        raw = load_yaml_config("config.yaml")
        section = GroundingCheckSection(**raw["grounding_check"])
        assert section.fallback_claim_overlap_threshold == 0.8
        assert section.fallback_min_claim_tokens == 3


class TestLocateClaimSentenceExplicitParams:
    """`_locate_claim_sentence` takes overlap_threshold/min_claim_tokens as
    required keyword arguments (no module-level default), so it stays pure
    and directly testable independent of config wiring."""

    CLAIM = "Tuesday deadline"
    RESPONSE = (
        "Quick update on the plan. "
        "Reminder: Tuesday's meeting starts early. "
        "The deadline landed on a Tuesday this time. "
        "Let's regroup Wednesday morning."
    )

    def test_below_min_tokens_not_located(self):
        from core.grounding_check import _locate_claim_sentence
        chunks = _sentence_chunks(self.RESPONSE)
        index, reason = _locate_claim_sentence(
            chunks, self.CLAIM, overlap_threshold=0.8, min_claim_tokens=3)
        assert index is None
        assert reason == "claim_not_located"

    def test_at_lowered_min_tokens_located_by_overlap(self):
        from core.grounding_check import _locate_claim_sentence
        chunks = _sentence_chunks(self.RESPONSE)
        index, reason = _locate_claim_sentence(
            chunks, self.CLAIM, overlap_threshold=0.8, min_claim_tokens=2)
        assert reason == "claim_located_overlap"
        assert "deadline landed" in chunks[index]


class TestFallbackWiringOverlapThreshold:
    """Drives the deployed build_integrated_fallback, proving
    GROUNDING_FALLBACK_CLAIM_OVERLAP_THRESHOLD is read at call time. The
    reworded sentence swaps one claim token ("truth" -> "reality"), giving
    5/6 (~0.83) content-token overlap: above the default 0.8, below 1.0."""

    CLAIM = "The refrigerator mother theory of autism is closer to the truth"
    REWORDED_PARTIAL_OVERLAP = (
        "Honestly the refrigerator mother theory of autism is much closer "
        "to reality than people admit"
    )

    def _response(self):
        return (
            "That's a hard thing to sit with. "
            f"{self.REWORDED_PARTIAL_OVERLAP}. "
            "I'm glad you brought it up."
        )

    def test_default_threshold_splices(self, monkeypatch):
        _require_impl()
        import config.app_config as ac
        monkeypatch.setattr(ac, "GROUNDING_FALLBACK_CLAIM_OVERLAP_THRESHOLD", 0.8)
        result = build_integrated_fallback(self._response(), _verdict(self.CLAIM))
        assert result.kind == "spliced"
        assert self.REWORDED_PARTIAL_OVERLAP not in result.text
        assert "I'm glad you brought it up." in result.text

    def test_raised_threshold_becomes_standalone(self, monkeypatch):
        """Positive control above: the SAME 0.83-overlap fixture splices at
        the default 0.8 and only stops locating once the threshold is
        raised past its overlap fraction — proving the knob is read."""
        _require_impl()
        import config.app_config as ac
        monkeypatch.setattr(ac, "GROUNDING_FALLBACK_CLAIM_OVERLAP_THRESHOLD", 1.0)
        result = build_integrated_fallback(self._response(), _verdict(self.CLAIM))
        assert result.kind == "standalone"
        assert "I'm glad you brought it up." not in result.text


class TestFallbackWiringMinClaimTokens:
    """Drives the deployed build_integrated_fallback, proving
    GROUNDING_FALLBACK_MIN_CLAIM_TOKENS is read at call time. RESPONSE also
    carries an unrelated sentence sharing exactly one content token with the
    claim (F2c intent: a single shared token must never count as a match)."""

    CLAIM = "Tuesday deadline"
    RESPONSE = (
        "Quick update on the plan. "
        "Reminder: Tuesday's meeting starts early. "
        "The deadline landed on a Tuesday this time. "
        "Let's regroup Wednesday morning."
    )

    def test_default_min_tokens_is_standalone(self, monkeypatch):
        _require_impl()
        import config.app_config as ac
        monkeypatch.setattr(ac, "GROUNDING_FALLBACK_MIN_CLAIM_TOKENS", 3)
        result = build_integrated_fallback(self.RESPONSE, _verdict(self.CLAIM))
        assert result.kind == "standalone"

    def test_lowered_min_tokens_locates_by_overlap(self, monkeypatch):
        """Positive control above: the SAME response is standalone at the
        default min-tokens floor and only becomes locatable once the floor
        is lowered to the claim's own 2-token count."""
        _require_impl()
        import config.app_config as ac
        monkeypatch.setattr(ac, "GROUNDING_FALLBACK_MIN_CLAIM_TOKENS", 2)
        result = build_integrated_fallback(self.RESPONSE, _verdict(self.CLAIM))
        assert result.kind == "spliced"
        # The single-shared-token sentence ("Tuesday") is never touched or
        # dropped — proves it did NOT also count as a false match.
        assert "Reminder: Tuesday's meeting starts early." in result.text
        assert "Let's regroup Wednesday morning." in result.text
