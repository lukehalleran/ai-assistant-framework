"""
# utils/location_resolver.py

Resolves the user's current location as a short place string ("Saint Charles,
IL") for localizing location-dependent web-search queries (weather, local news,
"near me").

Resolution chain (first hit wins):
  1. Config override — `location.override` (set in the gitignored
     config.local.yaml for privacy) or DAEMON_USER_LOCATION env var
  2. IP geolocation — city-level, cached for LOCATION_IP_CACHE_TTL_HOURS and
     refreshed in a background daemon thread so the async hot path NEVER blocks
     on network I/O. Providers: ipinfo.io (HTTPS) then ip-api.com fallback.
     Until the first refresh lands, callers get the profile fallback.
  3. User profile — most recent `lives_in` identity fact in
     data/user_profile.json whose value looks like a real "City, Region" place
     (filters junk like "joke state" or "a bad mood"; prefers is_current).

Key API:
  - get_user_location() -> Optional[str]   module-level, sync, non-blocking
  - strip_unjustified_location(terms, query, location) -> List[str]
        backstop AFTER the trigger/decompose LLMs: removes the injected user
        location from generated search terms when the original query gave no
        reason to localize (2026-07-08 incident: "college login" queries got
        "Springfield IL" appended, retrieval returned Springfield Community
        College, and the response asserted it was the user's school)
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
from typing import Optional

from utils.logging_utils import get_logger

logger = get_logger("location_resolver")

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

_DEFAULT_PROFILE_PATH = os.path.join("data", "user_profile.json")

# A stored lives_in value must look like "City, Region" to be trusted as a
# place — the profile accumulates sarcasm ("joke state") and mood junk
# ("a bad mood") under lives_in, and none of that should reach a search query.
_PLACE_RE = re.compile(r"^[A-Za-z][A-Za-z .'\-]*,\s*[A-Za-z][A-Za-z .'\-]{1,}$")

# Retry failed IP lookups sooner than the success TTL, but not every turn.
_IP_FAILURE_RETRY_S = 600.0


class LocationResolver:
    """Non-blocking user-location resolution with layered fallbacks."""

    def __init__(self, profile_path: Optional[str] = None):
        self.profile_path = profile_path or _DEFAULT_PROFILE_PATH
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
        if override:
            return override

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
            if loc:
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
        except Exception as e:
            logger.debug(f"[Location] ipinfo.io lookup failed: {e}")
        try:
            resp = requests.get(
                "http://ip-api.com/json/?fields=status,city,regionName",
                timeout=timeout,
            )
            if resp.ok:
                data = resp.json()
                if data.get("status") == "success":
                    return self._format_place(data.get("city"), data.get("regionName"))
        except Exception as e:
            logger.debug(f"[Location] ip-api.com lookup failed: {e}")
        return None

    @staticmethod
    def _format_place(city: Optional[str], region: Optional[str]) -> Optional[str]:
        city = (city or "").strip()
        region = (region or "").strip()
        if city and region:
            return f"{city}, {region}"
        return city or region or None

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
                e for e in profile.get("categories", {}).get("identity", [])
                if e.get("relation") == "lives_in"
                and _PLACE_RE.match(str(e.get("value", "")).strip())
            ]
            if candidates:
                # Prefer current facts; among those, the most recent wins. If
                # every place-shaped fact was superseded (e.g. by junk that the
                # place filter rejected), fall back to the newest one anyway.
                current = [e for e in candidates if e.get("is_current")]
                pool = current or candidates
                best = max(pool, key=lambda e: str(e.get("timestamp", "")))
                location = str(best["value"]).strip()
        except Exception as e:
            logger.debug(f"[Location] Profile location read failed: {e}")

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
    """'Springfield' and 'Saint Charles' are the same city to a geocoder and
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


def _location_patterns(location: str) -> list:
    """Compiled patterns matching the location as the LLM tends to render it:
    'City, ST' / 'City ST' / 'City Statename' / bare 'City'. Longest first so
    the state tail never survives a bare-city removal."""
    parts = [p.strip() for p in location.split(",")]
    city = parts[0] if parts else location.strip()
    state = parts[1] if len(parts) > 1 else ""

    state_forms = []
    if state:
        state_forms.append(state)
        if state.upper() in _US_STATES:
            state_forms.append(_US_STATES[state.upper()])
        elif state.lower() in _US_STATES_INV:
            state_forms.append(_US_STATES_INV[state.lower()])

    patterns = []
    for c in _city_variants(city):
        c_esc = re.escape(c)
        for s in state_forms:
            patterns.append(rf"\b{c_esc}\s*,?\s+{re.escape(s)}\b\.?")
        patterns.append(rf"\b{c_esc}\b")
    return [re.compile(p, re.I) for p in patterns]


def query_justifies_location(query: str, location: str) -> bool:
    """Does the user's own query give a reason to localize? True only for
    physical-surroundings shapes (weather/current conditions, near-me/local
    phrasing) or when the user themselves named the place. Account, login,
    school, employer, product, etc. queries do NOT justify localization —
    the user's institutions are not determined by where they are sitting."""
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
    return any(v.lower() in q_low for v in _city_variants(city))


def strip_unjustified_location(terms, query: str, location: Optional[str]):
    """Backstop behind the trigger/decompose LLM prompts: if the ORIGINAL
    query gives no reason to localize, remove the injected user location from
    every generated search term. Terms that were nothing but the location are
    dropped. Returns the (possibly unchanged) list; logs when it fires."""
    if not terms or not location:
        return terms
    if query_justifies_location(query or "", location):
        return terms

    patterns = _location_patterns(location)
    cleaned = []
    changed = False
    for term in terms:
        new = term
        for pat in patterns:
            new = pat.sub("", new)
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
