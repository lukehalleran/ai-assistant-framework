"""
# utils/location_resolver.py

Resolves the user's current location as a short place string ("Springfield,
IL") for localizing location-dependent web-search queries (weather, local news,
"near me").

Resolution chain (first hit wins; the session statement is checked right
after the override, the rest of the order is unchanged):
  1. Config override — `location.override` (set in the gitignored
     config.local.yaml for privacy) or DAEMON_USER_LOCATION env var
  2. IP geolocation — city-level, cached for LOCATION_IP_CACHE_TTL_HOURS and
     refreshed in a background daemon thread so the async hot path NEVER blocks
     on network I/O. Providers: ipinfo.io (HTTPS) then ip-api.com fallback.
     Until the first refresh lands, callers get the profile fallback.
  2b. (2026-10-27) THIS SESSION'S stated move — see "Session statement" below;
     sits directly under the config override, above everything else.
  3. User profile — most recent `lives_in` identity fact in
     data/user_profile.json whose value looks like a real "City, Region" place
     (filters junk like "joke state" or "a bad mood"; prefers is_current).

Key API:
  - get_user_location() -> Optional[str]   module-level, sync, non-blocking
  - strip_unjustified_location(terms, query, location, institution=None) ->
        List[str]
        backstop AFTER the trigger/decompose LLMs: removes the injected user
        location (city and, 2026-09-12, bare state) from generated search
        terms when the original query gave no reason to localize (2026-07-08
        incident: "college login" queries got "Springfield IL" appended,
        retrieval returned Springfield Community College, and the response
        asserted it was the user's school). `institution` protects a
        school-name span in a term from the bare-state removal.
  - LocationResolver                        cache + background-refresh manager

Side effects: outbound HTTPS to ipinfo.io / ip-api.com (disable via
location.ip_lookup_enabled); reads data/user_profile.json (mtime-cached).
Returns None when nothing resolves — callers must treat location as optional.
"""

import json
import os
import re
import threading
import time
from datetime import datetime
from typing import Optional

from utils.bootstrap import get_user_profile_path
from utils.logging_utils import get_logger
from utils.trigger_match import normalize_ws

logger = get_logger("location_resolver")

# A "session" is this process: stored user turns older than this belong to a
# previous run and are covered by the profile instead.
_PROCESS_START = time.time()

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None

try:
    from config.app_config import (
        LOCATION_ENABLED,
        LOCATION_IP_LOOKUP_ENABLED,
        LOCATION_IP_CACHE_TTL_HOURS,
        LOCATION_IP_LOOKUP_TIMEOUT_S,
        LOCATION_OVERRIDE,
    )
except ImportError:
    LOCATION_ENABLED = True
    LOCATION_IP_LOOKUP_ENABLED = True
    LOCATION_IP_CACHE_TTL_HOURS = 6.0
    LOCATION_IP_LOOKUP_TIMEOUT_S = 3.0
    LOCATION_OVERRIDE = os.getenv("DAEMON_USER_LOCATION", "")

# A stored lives_in value must look like "City, Region" to be trusted as a
# place — the profile accumulates sarcasm ("joke state") and mood junk
# ("a bad mood") under lives_in, and none of that should reach a search query.
_PLACE_RE = re.compile(r"^[A-Za-z][A-Za-z .'\-]*,\s*[A-Za-z][A-Za-z .'\-]{1,}$")

# Closed class: generic location NOUNS (home, work...) that spaCy will tag GPE
# in a "live in X" frame but that never name a city. Only consulted on the
# bare-place path below.
_GENERIC_PLACE_NOUNS = frozenset({
    "home", "house", "work", "office", "school", "campus", "apartment", "flat",
    "room", "dorm", "town", "city", "state", "country", "street", "building",
    "downtown", "uptown", "suburbs", "outside", "inside", "abroad",
})


def _is_bare_city(value: str) -> bool:
    """True for a 1-3 word city/region name as the per-turn extractor stores it
    ("chicago", "san diego", "salt lake city" - no region). The strict
    "City, Region" shape cannot admit it. Admitted only when ALL hold:
    (a) every token is a name-like word; (b) spaCy NER on "I live in {Title}."
    yields ONE GPE entity whose text equals the WHOLE title-cased value;
    (c) no token is a generic location noun or a spaCy stop word ("Joke State",
    "Empty House", "the city" fail even if tagged); (d) not a placeholder.
    (b)+(c) replace the old single-token rule: the span-equals-whole-value NER
    check rejects sarcasm phrases, and the closed-class token filters catch the
    two-word generic phrases NER can still tag. 4+ words stay rejected."""
    v = str(value or "").strip()
    tokens = v.split()
    if not 1 <= len(tokens) <= 3:
        return False
    if not all(re.fullmatch(r"[A-Za-z][A-Za-z.'\-]*", t) for t in tokens):
        return False
    if len(tokens) == 1 and len(v) < 2:
        return False
    if is_placeholder_location(v):
        return False
    if any(t.lower() in _GENERIC_PLACE_NOUNS for t in tokens):
        return False
    try:
        # lazy import: startup-cost (spaCy load on first bare-place check only)
        from memory.fact_extractor import _get_nlp
        nlp = _get_nlp()
        if nlp is None:
            return False
        if len(tokens) > 1:
            from spacy.lang.en.stop_words import STOP_WORDS  # lazy import: startup-cost (imports the spacy package)
            if any(t.lower() in STOP_WORDS for t in tokens):
                return False
        ents = nlp(f"I live in {v.title()}.").ents
    except Exception:  # degrades: bare-city check unavailable, only "City, Region" values are trusted
        return False
    gpes = [e for e in ents if e.label_ == "GPE"]
    return len(ents) == 1 and len(gpes) == 1 and gpes[0].text == v.title()


def _is_place_value(value: str) -> bool:
    v = str(value or "").strip()
    return bool(_PLACE_RE.match(v)) or _is_bare_city(v)


# Session-statement scan bounds: newest N NEW user turns per call, and a
# per-turn text cap (a long paste is not a statement about where the user lives).
_SESSION_MAX_NEW_TURNS = 3
_SESSION_MAX_TEXT_CHARS = 1500

_session_extractor = None
_session_turn_provider = None


def set_session_turn_provider(provider) -> None:
    """Register the callable returning this process's corpus entries (oldest
    first). Called once by the composition root (main.py) with
    `corpus_manager.visible_entries`."""
    global _session_turn_provider
    _session_turn_provider = provider


def _run_to_completion(coro):
    """Drive a coroutine that never actually suspends (FactExtractor.extract_facts
    has no awaits) without touching the running event loop; anything that
    would suspend is abandoned."""
    try:
        coro.send(None)
    except StopIteration as stop:
        return stop.value
    finally:
        coro.close()
    return None


def _without_verb_adverbs(text: str, nlp) -> str:
    """Remove adverbial modifiers of verbs ("I JUST moved to X" -> "I moved to X")
    by dependency label, so the extractor's `<subject> <verb> to <place>`
    patterns see the bare clause. Grammar-driven, not a word list."""
    doc = nlp(text)
    drop = {t.i for t in doc if t.dep_ == "advmod" and t.head.pos_ in ("VERB", "AUX") and t.pos_ == "ADV"}
    if not drop:
        return text
    return "".join(t.text_with_ws for t in doc if t.i not in drop)


def _with_region(text: str, place: str, nlp) -> str:
    """"Portland" + source "...Portland, Oregon." -> "Portland, Oregon" (adjacent
    GPE entities joined by a comma); otherwise the place unchanged."""
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "GPE" and ent.text.lower() == place.lower():
            nxt = doc[ent.end:ent.end + 2]
            if len(nxt) == 2 and nxt[0].text == "," and nxt[1].ent_type_ == "GPE":
                region = next((e for e in doc.ents if e.start == nxt[1].i), None)
                if region is not None:
                    return f"{ent.text}, {region.text}"
            return ent.text
    return place.title()   # bare city the NER missed; _is_bare_city validates the title form


def _stated_place_from_text(text: str) -> Optional[str]:
    """The place a user message says they live in/moved to, or None, via the
    deployed FactExtractor + the resolver's own place-shape gate."""
    global _session_extractor
    if len(text) > _SESSION_MAX_TEXT_CHARS:
        return None
    try:
        from memory.fact_extractor import FactExtractor, _get_nlp  # lazy import: layering (utils must not load memory/ at import)
        from memory.fact_source import _residence_resident_is_user  # lazy import: layering
        nlp = _get_nlp()
        if nlp is None:
            return None
        if _session_extractor is None:
            _session_extractor = FactExtractor(use_rebel=False)
        clean = _without_verb_adverbs(normalize_ws(text), nlp)   # line-wrapped client input
        for node in _run_to_completion(_session_extractor.extract_facts(clean, "")) or []:
            md = node.metadata or {}
            if md.get("relation") != "lives_in" or md.get("claim_kind") == "plan":
                continue
            obj = str(md.get("object") or "").strip()
            if not obj or not _residence_resident_is_user(clean, obj):
                continue
            place = _with_region(clean, obj, nlp)
            if _is_place_value(place) and not is_placeholder_location(place):
                return place
    except Exception as e:  # degrades: this turn yields no stated place
        logger.debug(f"[Location] session statement extraction failed: {type(e).__name__}")
    return None


# Retry failed IP lookups sooner than the success TTL, but not every turn.
_IP_FAILURE_RETRY_S = 600.0


# ----------------------------------------------------------------------
# Placeholder (deictic) locations — BC-04
# ----------------------------------------------------------------------
# A deictic phrase ("current location", "my area", "here", "near me") points AT
# the user's place; it is never a place. The closed-class words below are
# grammatical (determiners/deictics/generic location nouns), not a list of
# example phrases: a value built only from them can never name a place.
_PLACEHOLDER_TOKENS = frozenset({
    "current", "present", "my", "your", "the", "this", "that", "local",
    "nearby", "here", "there", "near", "me", "around", "in", "at", "where",
    "i", "am", "location", "area", "city", "town", "neighborhood",
    "neighbourhood", "place", "position", "whereabouts", "unknown", "none",
    "null", "n/a", "na", "unspecified", "undefined",
})
# Deictic placeholder PHRASES inside a longer query ("weather in my current
# location"); same closed-class vocabulary, phrase-shaped.
PLACEHOLDER_PHRASE_RE = re.compile(
    r"\b(?:(?:in|at|for|near|around|from)\s+)?"
    r"(?:(?:my|your|the)\s+)?(?:current|present)\s+"
    r"(?:location|city|town|area|place|position|neighbou?rhood)\b"
    r"|\b(?:(?:in|at|for|near|around)\s+)?my\s+"
    r"(?:location|area|city|town|neighbou?rhood)\b"
    r"|\bnear\s+me\b|\baround\s+here\b|\b(?:in|at)\s+here\b"
    r"|\bwhere\s+i\s+am\b",
    re.I,
)


def is_placeholder_location(value: Optional[str]) -> bool:
    """True when `value` is empty or built only from deictic/generic location
    words — never a real place (e.g. "current location", "my area", "here")."""
    if value is None:
        return True
    tokens = re.findall(r"[a-z/']+", str(value).lower())
    return not tokens or all(t in _PLACEHOLDER_TOKENS for t in tokens)


def strip_placeholder_location(text: str) -> str:
    """Remove deictic placeholder phrases from query text (whitespace-normalised)."""
    return re.sub(r"\s{2,}", " ", PLACEHOLDER_PHRASE_RE.sub(" ", text or "")).strip()


class LocationResolver:
    """Non-blocking user-location resolution with layered fallbacks."""

    def __init__(
        self,
        profile_path: Optional[str] = None,
        turn_provider=None,
        session_start: Optional[float] = None,
    ):
        self.profile_path = profile_path or get_user_profile_path()
        # Session-statement source (2026-10-27): a callable returning the
        # in-process corpus entries (chronological). None -> the module-level
        # provider the composition root registers (`set_session_turn_provider`).
        self.turn_provider = turn_provider
        self.session_start = _PROCESS_START if session_start is None else float(session_start)
        self._watermark: Optional[float] = None
        self._session_entries_seen: dict = {}
        self._session_location: Optional[str] = None
        self._ip_location: Optional[str] = None
        self._ip_fetched_at: float = 0.0
        self._ip_failed_at: float = 0.0
        self._refresh_lock = threading.Lock()
        self._refresh_in_flight = False
        self._profile_location: Optional[str] = None
        self._profile_mtime: Optional[float] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_location(self) -> Optional[str]:
        """Best currently-known location string, or None. Never blocks on network."""
        if not LOCATION_ENABLED:
            return None

        override = (LOCATION_OVERRIDE or "").strip()
        if override and not is_placeholder_location(override):
            return override

        # A move the user stated in THIS session beats stored/derived places:
        # the profile is only written at shutdown, so without this the old
        # place keeps localizing searches for the rest of the session.
        stated = self._location_from_session_statement()
        if stated:
            return stated

        if LOCATION_IP_LOOKUP_ENABLED and requests is not None:
            ttl_s = float(LOCATION_IP_CACHE_TTL_HOURS) * 3600.0
            age = time.time() - self._ip_fetched_at
            if self._ip_location and age < ttl_s:
                return self._ip_location
            self._start_background_refresh()
            # Stale-but-known IP location still beats the profile fallback
            # while the refresh is in flight.
            if self._ip_location:
                return self._ip_location

        return self._location_from_profile()

    # ------------------------------------------------------------------
    # IP geolocation (background)
    # ------------------------------------------------------------------

    def _start_background_refresh(self) -> None:
        if os.getenv("DAEMON_TEST_MODE"):
            # The test lane must not reach the network or log the machine's
            # city (2026-09-12: a pre-push run printed "[Location] IP
            # geolocation resolved: <city>" — outbound HTTPS plus a location
            # leak from pytest). Same doctrine as the DAEMON_TEST_MODE store
            # and backup guards; a test that wants this path monkeypatches
            # `_fetch_ip_location`.
            return
        if time.time() - self._ip_failed_at < _IP_FAILURE_RETRY_S:
            return
        with self._refresh_lock:
            if self._refresh_in_flight:
                return
            self._refresh_in_flight = True
        thread = threading.Thread(target=self._refresh_ip_location, daemon=True)
        thread.start()

    def _refresh_ip_location(self) -> None:
        try:
            loc = self._fetch_ip_location()
            if loc and not is_placeholder_location(loc):
                self._ip_location = loc
                self._ip_fetched_at = time.time()
                logger.info(f"[Location] IP geolocation resolved: {loc}")
            else:
                self._ip_failed_at = time.time()
                logger.debug("[Location] IP geolocation returned nothing")
        except Exception as e:
            self._ip_failed_at = time.time()
            logger.debug(f"[Location] IP geolocation failed: {e}")
        finally:
            with self._refresh_lock:
                self._refresh_in_flight = False

    def _fetch_ip_location(self) -> Optional[str]:
        """City-level location from the machine's public IP. Runs off-thread."""
        timeout = float(LOCATION_IP_LOOKUP_TIMEOUT_S)
        try:
            resp = requests.get("https://ipinfo.io/json", timeout=timeout)
            if resp.ok:
                data = resp.json()
                loc = self._format_place(data.get("city"), data.get("region"))
                if loc:
                    return loc
        except Exception as e:  # degrades: ipinfo.io provider unreachable, falls back to ip-api.com
            logger.warning(f"[Location] ipinfo.io lookup failed: {e}")
        try:
            resp = requests.get(
                "http://ip-api.com/json/?fields=status,city,regionName",
                timeout=timeout,
            )
            if resp.ok:
                data = resp.json()
                if data.get("status") == "success":
                    return self._format_place(data.get("city"), data.get("regionName"))
        except Exception as e:  # degrades: ip-api.com provider unreachable, IP geolocation unresolved this cycle
            logger.warning(f"[Location] ip-api.com lookup failed: {e}")
        return None

    @staticmethod
    def _format_place(city: Optional[str], region: Optional[str]) -> Optional[str]:
        city = (city or "").strip()
        region = (region or "").strip()
        if city and region:
            return f"{city}, {region}"
        return city or region or None

    # ------------------------------------------------------------------
    # Session statement (2026-10-27, class: BC-46)
    # ------------------------------------------------------------------

    def _location_from_session_statement(self) -> Optional[str]:
        """Newest place the user stated living in during THIS session, or None.

        Evidence = the user's own turns newer than the session start, taken from
        the IN-PROCESS corpus the Daemon already holds (a provider callable
        registered by the composition root -- no file is ever read here, so no
        loop-blocking JSON load; utils/ imports nothing from core/memory at
        module level). A statement is judged by the DEPLOYED per-turn fact
        extractor, so no residence regex lives here. A watermark makes each
        call O(new turns); each turn is extracted once.
        """
        provider = self.turn_provider or _session_turn_provider
        if provider is None:
            return self._session_location
        try:
            entries = provider()
        except Exception as e:  # degrades: session statement unavailable, the profile/IP chain answers
            logger.warning(f"[Location] Session turn provider failed: {type(e).__name__}")
            return self._session_location
        if not isinstance(entries, list) or not entries:
            return self._session_location
        try:
            self._scan_new_turns(entries)
        except Exception as e:  # degrades: this call keeps the previously found place
            logger.warning(f"[Location] Session statement scan failed: {type(e).__name__}")
        return self._session_location

    @staticmethod
    def _entry_time(entry) -> Optional[float]:
        ts = entry.get("timestamp")
        try:
            if isinstance(ts, datetime):
                return ts.timestamp()
            if isinstance(ts, str):
                return datetime.fromisoformat(ts).timestamp()
        except ValueError:
            return None
        return None

    def _scan_new_turns(self, entries) -> None:
        # lazy import: layering (memory/ must not load at utils import time)
        from memory.fact_source import is_test_turn  # lazy import: layering
        from memory.utils import is_quarantined  # lazy import: layering
        newest_t = None
        examined = 0
        for e in reversed(entries):
            if not isinstance(e, dict):
                continue
            t = self._entry_time(e)
            if t is None or t < self.session_start:
                break   # chronological: everything older is a previous session
            if newest_t is None:
                newest_t = t
            if self._watermark is not None and t < self._watermark:
                break   # already scanned on an earlier call
            if is_quarantined(e) or is_test_turn(e):
                continue
            text = e.get("user_text") if isinstance(e.get("user_text"), str) else e.get("query")
            if not isinstance(text, str) or not text.strip():
                continue
            examined += 1
            if examined > _SESSION_MAX_NEW_TURNS:
                break
            key = (t, hash(text))
            if key in self._session_entries_seen:
                continue   # extracted before (same-timestamp tail of the watermark)
            place = _stated_place_from_text(text)
            if len(self._session_entries_seen) > 256:
                self._session_entries_seen.clear()
            self._session_entries_seen[key] = place
            if place:
                self._session_location = place
                break
        if newest_t is not None:
            self._watermark = newest_t

    # ------------------------------------------------------------------
    # Profile fallback
    # ------------------------------------------------------------------

    def _location_from_profile(self) -> Optional[str]:
        try:
            mtime = os.path.getmtime(self.profile_path)
        except OSError:
            return None
        if self._profile_mtime == mtime:
            return self._profile_location

        location = None
        try:
            with open(self.profile_path, "r", encoding="utf-8") as f:
                profile = json.load(f)
            candidates = [
                # lives_in moved identity -> living_situation (2026-10-02); read both
                e for cat in ("identity", "living_situation")
                for e in profile.get("categories", {}).get(cat, [])
                if e.get("relation") == "lives_in"
                and _is_place_value(str(e.get("value", "")))
            ]
            if candidates:
                # Prefer current facts; among those, the most recent wins. If
                # every place-shaped fact was superseded (e.g. by junk that the
                # place filter rejected), fall back to the newest one anyway.
                current = [e for e in candidates if e.get("is_current")]
                pool = current or candidates
                best = max(pool, key=lambda e: str(e.get("timestamp", "")))
                location = str(best["value"]).strip()
        except Exception as e:  # degrades: profile location fact unreadable, lives_in fallback unavailable
            logger.warning(f"[Location] Profile location read failed: {e}")

        if location is not None and is_placeholder_location(location):
            location = None
        self._profile_location = location
        self._profile_mtime = mtime
        return location


# ----------------------------------------------------------------------
# Unjustified-localization backstop
# ----------------------------------------------------------------------

# Cues in the ORIGINAL user query that justify carrying the user's location
# into search terms. Deliberately mirrors WebSearchManager's localization
# shapes: physical-surroundings queries only.
_LOCAL_INTENT_RE = re.compile(
    r"\b(?:near\s+me|nearby|locally?|around\s+here|close\s+by|"
    r"my\s+(?:area|city|town|location|neighborhood))\b",
    re.I,
)
_WEATHER_SHAPE_RE = re.compile(
    r"\b(weather|forecast|heat\s+(?:advisory|warning|index|wave)|"
    r"air\s+quality|uv\s+index|wind\s+chill|excessive\s+heat)\b",
    re.I,
)
# Words that are weather-ish only alongside a current-conditions cue
# ("how hot is it outside" yes; "what temperature to bake salmon" no).
# Slightly broader than WebSearchManager's set on purpose: a false KEEP
# here just preserves pre-backstop behavior, a false STRIP breaks a
# legitimate local query.
_AMBIGUOUS_WEATHER_RE = re.compile(
    r"\b(temperature|humidity|hot|cold|warm|chilly|humid|rain(?:ing)?|"
    r"snow(?:ing)?|wind[gy]?)\b",
    re.I,
)
_CURRENT_CONDITIONS_RE = re.compile(
    r"\b(outside|outdoors|today|tonight|tomorrow|right\s+now|currently|"
    r"this\s+(?:week|weekend|morning|afternoon|evening))\b",
    re.I,
)

_US_STATES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
    "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho",
    "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas",
    "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi",
    "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
    "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma",
    "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah",
    "VT": "Vermont", "VA": "Virginia", "WA": "Washington", "WV": "West Virginia",
    "WI": "Wisconsin", "WY": "Wyoming", "DC": "District of Columbia",
}
_US_STATES_INV = {v.lower(): k for k, v in _US_STATES.items()}


def _city_variants(city: str) -> list:
    """'Springfield' and 'Springfield' are the same city to a geocoder and
    to the LLM — cover both spellings whichever one the resolver returned."""
    variants = [city]
    low = city.lower()
    if low.startswith("st. "):
        variants.append("Saint " + city[4:])
    elif low.startswith("st "):
        variants.append("Saint " + city[3:])
    elif low.startswith("saint "):
        variants.append("St. " + city[6:])
        variants.append("St " + city[6:])
    return variants


def _canonical_state_forms(state: str):
    """(abbrev, full_name) for `state` via the lookup table, regardless of
    how it was originally cased ("il" / "IL" / "Illinois" all resolve the
    same pair). Either element is None when `state` isn't a recognized US
    state — the bare-state patterns are then simply not added."""
    if not state:
        return None, None
    if state.upper() in _US_STATES:
        abbrev = state.upper()
        return abbrev, _US_STATES[abbrev]
    if state.lower() in _US_STATES_INV:
        return _US_STATES_INV[state.lower()], state
    return None, None


def _institution_protected_spans(text: str, institution: Optional[str]) -> list:
    """Character spans in `text` that name an institution — the resolved one
    (if given) plus any generally institution-shaped phrase ("Quellmoor
    University"). A location mention lying inside one of these spans is part
    of a SCHOOL'S name, not a reference to the place itself (2026-09-12:
    "Vermont Wrenfield drop deadline" must not read as the user naming the
    state of Vermont just because their school's name happens to contain it)."""
    spans = []
    if institution and institution.strip():
        inst = institution.strip()
        for m in re.finditer(re.escape(inst), text, re.I):
            spans.append((m.start(), m.end()))
    try:
        # lazy import: cycle (avoids a module-level cycle — institution_resolver imports this module's functions for scope_identity_terms.)
        from utils.institution_resolver import _NAMED_INSTITUTION_RE
    except Exception:  # degrades: institution-name regex unavailable, only the resolved name is protected
        _NAMED_INSTITUTION_RE = None
    if _NAMED_INSTITUTION_RE is not None:
        for m in _NAMED_INSTITUTION_RE.finditer(text):
            spans.append((m.start(), m.end()))
    return spans


def _contained_in_any(start: int, end: int, spans: list) -> bool:
    return any(s <= start and end <= e for s, e in spans)


def _overlaps_any(start: int, end: int, spans: list) -> bool:
    return any(start < e and s < end for s, e in spans)


def _sub_outside_spans(pattern, text: str, spans: list) -> str:
    """Like `pattern.sub("", text)`, but a match overlapping any of `spans`
    is left untouched instead of removed — the strip must never leave a
    fragment like "Wrenfield" behind by amputating half of "Vermont Wrenfield"."""
    if not spans:
        return pattern.sub("", text)
    out = []
    last = 0
    for m in pattern.finditer(text):
        if _overlaps_any(m.start(), m.end(), spans):
            continue
        out.append(text[last:m.start()])
        last = m.end()
    out.append(text[last:])
    return "".join(out)


# Proper-name spans (2026-09-12 follow-up review F4). A state or city name is
# also the leading word of newspapers, companies, characters and schools
# ("Washington Post", "Texas Instruments", "Indiana Jones", "New York Times").
# Removing the owner's state from such a name changes the search target:
# "Washington Post election coverage" became "Post election coverage".
# Capitalization is the only local signal, so it is read structurally and
# only in text that also has lowercase words (an all-TitleCase headline says
# nothing about proper nouns).
_NAME_TOKEN_RE = re.compile(r"[A-Za-z][\w&'’.\-]*")
_RUN_LEADING_ARTICLES = frozenset({"the", "a", "an"})
# "Washington State" names the place itself, not a separate entity.
_PLACE_DESIGNATORS = frozenset({"state"})


def _name_key(word: str) -> str:
    """Comparison key for one capitalized token: lowercase, possessive and
    abbreviation dot removed ("Washington's" -> "washington", "St." -> "st")."""
    key = word.lower()
    for suffix in ("'s", "’s"):
        if key.endswith(suffix):
            key = key[: -len(suffix)]
    return key.rstrip(".")


def _state_forms(state: str) -> list:
    """The state as given plus its abbreviation/full-name counterpart."""
    forms = []
    if state:
        forms.append(state)
        if state.upper() in _US_STATES:
            forms.append(_US_STATES[state.upper()])
        elif state.lower() in _US_STATES_INV:
            forms.append(_US_STATES_INV[state.lower()])
    return forms


def _location_words(location: str) -> frozenset:
    """Every word of the resolved location's city variants and state forms."""
    parts = [p.strip() for p in location.split(",")]
    city = parts[0] if parts else location.strip()
    state = parts[1] if len(parts) > 1 else ""
    words = set(_PLACE_DESIGNATORS)
    for form in _city_variants(city) + _state_forms(state):
        words.update(_name_key(w) for w in form.split())
    words.discard("")
    return frozenset(words)


def _city_state_patterns(location: str) -> list:
    """'City, ST' / 'City Statename' sequences — the complete resolved location."""
    parts = [p.strip() for p in location.split(",")]
    city = parts[0] if parts else location.strip()
    state = parts[1] if len(parts) > 1 else ""
    return [
        re.compile(rf"\b{re.escape(c)}\s*,?\s+{re.escape(s)}\b", re.I)
        for c in _city_variants(city)
        for s in _state_forms(state)
    ]


def _capitalized_runs(text: str) -> list:
    """Runs of two or more capitalized tokens separated only by whitespace,
    leading articles trimmed. Empty when `text` has no lowercase-initial
    word, because then capitalization carries no proper-noun signal."""
    tokens = list(_NAME_TOKEN_RE.finditer(text))
    if not any(tok.group()[0].islower() for tok in tokens):
        return []
    runs, current = [], []
    for tok in tokens:
        capitalized = tok.group()[0].isupper()
        if capitalized and current and text[current[-1].end():tok.start()].isspace():
            current.append(tok)
            continue
        if current:
            runs.append(current)
        current = [tok] if capitalized else []
    if current:
        runs.append(current)
    trimmed = []
    for run in runs:
        while run and _name_key(run[0].group()) in _RUN_LEADING_ARTICLES:
            run = run[1:]
        if len(run) >= 2:
            trimmed.append(run)
    return trimmed


def _proper_name_spans(
    text: str, location: str, query: Optional[str] = None, either_side: bool = False
) -> list:
    """Spans in `text` of capitalized runs that use a location word as part of
    a proper name rather than as a location.

    A run qualifies when it mixes location words with other words and either
    a non-location word FOLLOWS a location word (a place used as a modifier:
    "Washington Post", "New York Times"), or the user's own `query` contains
    the same run ("George Washington"). A run containing the complete
    "City State" sequence is the injected location itself and never
    qualifies; neither does a state appended after a name ("Chicago Tribune
    Illinois"). `either_side=True` drops the direction requirement — used on
    the user's query, where a state inside ANY name is not the user naming
    their state."""
    loc_words = _location_words(location)
    city_state = _city_state_patterns(location)
    query_lower = normalize_ws(query).lower() if query else ""
    spans = []
    for run in _capitalized_runs(text):
        keys = [_name_key(tok.group()) for tok in run]
        loc_idx = [i for i, key in enumerate(keys) if key in loc_words]
        other_idx = [i for i, key in enumerate(keys) if key not in loc_words]
        if not loc_idx or not other_idx:
            continue
        start, end = run[0].start(), run[-1].end()
        run_text = text[start:end]
        if any(p.search(run_text) for p in city_state):
            continue
        modifier = max(other_idx) > min(loc_idx)
        typed_by_user = bool(query_lower) and normalize_ws(run_text).lower() in query_lower
        if either_side or modifier or typed_by_user:
            spans.append((start, end))
    return spans


def _location_patterns(location: str) -> list:
    """Compiled patterns matching the location as the LLM tends to render it:
    'City, ST' / 'City ST' / 'City Statename' / bare 'City', plus (2026-09-12)
    the bare state alone with no city — "current voting issues Illinois" had
    no city to anchor on, so the state token survived the old city-only
    patterns. Longest/most-specific first so the state tail never survives a
    bare-city removal. The bare abbreviation matches ONLY as an uppercase
    whole token (case-sensitive) so lowercase "in"/"or"/"me" is never misread
    as a state code; the bare full name matches case-insensitively."""
    parts = [p.strip() for p in location.split(",")]
    city = parts[0] if parts else location.strip()
    state = parts[1] if len(parts) > 1 else ""

    state_forms = _state_forms(state)

    compiled = []
    for c in _city_variants(city):
        c_esc = re.escape(c)
        for s in state_forms:
            compiled.append(re.compile(rf"\b{c_esc}\s*,?\s+{re.escape(s)}\b\.?", re.I))
        compiled.append(re.compile(rf"\b{c_esc}\b", re.I))

    abbrev, full_name = _canonical_state_forms(state)
    if full_name:
        # "Washington State" first, so the designator never survives the
        # bare-name removal as a stray "State".
        compiled.append(re.compile(rf"\b{re.escape(full_name)}\s+State\b", re.I))
        compiled.append(re.compile(rf"\b{re.escape(full_name)}\b", re.I))
    if abbrev:
        compiled.append(re.compile(rf"\b{re.escape(abbrev)}\b"))  # case-sensitive
    return compiled


def query_justifies_location(
    query: str, location: str, institution: Optional[str] = None
) -> bool:
    """Does the user's own query give a reason to localize? True only for
    physical-surroundings shapes (weather/current conditions, near-me/local
    phrasing), when the user themselves named the place (city, or — 2026-09-12
    — its state by full name case-insensitively or by uppercase-only
    abbreviation), or Account, login, school, employer, product, etc. queries
    do NOT justify localization — the user's institutions are not determined
    by where they are sitting. A state mention that lies INSIDE an
    institution-name span ("Vermont Wrenfield drop deadline") does not count —
    that names the school, not the state; pass `institution` to protect it."""
    if not query:
        return False
    if _LOCAL_INTENT_RE.search(query):
        return True
    if _WEATHER_SHAPE_RE.search(query):
        return True
    if _AMBIGUOUS_WEATHER_RE.search(query) and _CURRENT_CONDITIONS_RE.search(query):
        return True
    # User typed the place themselves (any spelling variant of the city)
    q_low = query.lower()
    city = location.split(",")[0].strip()
    if any(v.lower() in q_low for v in _city_variants(city)):
        return True

    state = location.split(",")[1].strip() if "," in location else ""
    abbrev, full_name = _canonical_state_forms(state)
    if not (abbrev or full_name):
        return False
    # A state inside a school name or any other proper name ("the Washington
    # Post") names that thing, not the user's state.
    protected = _institution_protected_spans(query, institution) + _proper_name_spans(
        query, location, either_side=True
    )
    if full_name:
        for m in re.finditer(rf"\b{re.escape(full_name)}\b", query, re.I):
            if not _contained_in_any(m.start(), m.end(), protected):
                return True
    if abbrev:
        for m in re.finditer(rf"\b{re.escape(abbrev)}\b", query):  # case-sensitive
            if not _contained_in_any(m.start(), m.end(), protected):
                return True
    return False


def strip_unjustified_location(
    terms, query: str, location: Optional[str], institution: Optional[str] = None
):
    """Backstop behind the trigger/decompose LLM prompts: if the ORIGINAL
    query gives no reason to localize, remove the injected user location
    (city and, 2026-09-12, the bare state name/abbreviation) from every
    generated search term. An institution-name span inside a term (the
    resolved `institution`, or any institution-shaped phrase) is protected
    from the state removal — "Vermont Wrenfield drop deadline" must never
    become "Wrenfield drop deadline". Terms that were nothing but the location are
    dropped. Returns the (possibly unchanged) list; logs when it fires."""
    if not terms or not location:
        return terms
    if query_justifies_location(query or "", location, institution=institution):
        return terms

    patterns = _location_patterns(location)
    cleaned = []
    changed = False
    for term in terms:
        new = term
        for pat in patterns:
            protected = _institution_protected_spans(new, institution) + _proper_name_spans(
                new, location, query=query
            )
            new = _sub_outside_spans(pat, new, protected)
        if new != term:
            changed = True
            # tidy the amputation site: dangling connectors + doubled spaces
            new = re.sub(r"\s+(?:in|at|near|around|for|of)\s*$", "", new, flags=re.I)
            new = re.sub(r"\s{2,}", " ", new).strip(" ,;-")
        if new:
            cleaned.append(new)
    if changed:
        logger.info(
            f"[Location] Stripped unjustified location '{location}' from search "
            f"terms (query gave no local cue): {terms} -> {cleaned}"
        )
    return cleaned if changed else terms


_resolver: Optional[LocationResolver] = None
_resolver_lock = threading.Lock()


def get_user_location() -> Optional[str]:
    """Module-level accessor used by the web-search path. Non-blocking."""
    global _resolver
    if _resolver is None:
        with _resolver_lock:
            if _resolver is None:
                _resolver = LocationResolver()
    return _resolver.get_location()
