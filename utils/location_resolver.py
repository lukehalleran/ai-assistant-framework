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
