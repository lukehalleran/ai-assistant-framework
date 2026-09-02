"""
# core/email/outlook_auth.py

Module Contract
- Purpose: Outlook/Microsoft 365 device-code OAuth2 flow + token persistence.
  One-time setup: run scripts/auth_outlook.py in a terminal to open the
  consent URL and grant Mail.Read permission.
- Public interface:
  - get_access_token() -> Optional[str]: Load token from disk; refresh if expired.
  - run_device_code_flow() -> bool: Interactive device-code auth (used by script).
- Dependencies: httpx, stdlib only
- Side effects: Reads/writes token file at data/outlook_token.json (atomic, 0600 perms).
"""

import json
import os
import time
from pathlib import Path
from typing import Optional, Dict

from utils.logging_utils import get_logger

logger = get_logger("outlook_auth")


class OutlookAuthManager:
    """Manages Outlook/Microsoft 365 OAuth2 credentials for Mail.Read access."""

    def __init__(
        self,
        client_id: str,
        tenant: str = "common",
        token_path: str = "data/outlook_token.json",
    ):
        self._client_id = client_id
        self._tenant = tenant
        self._token_path = Path(token_path)
        self._token_cache: Optional[Dict] = None

    @property
    def is_configured(self) -> bool:
        """True if client_id is set."""
        return bool(self._client_id)

    @property
    def token_exists(self) -> bool:
        """True if token file exists."""
        return self._token_path.exists()

    @property
    def has_refresh_token(self) -> bool:
        """Disk-only check: refresh_token present (no network call)."""
        token = self._load_token()
        return bool(token and token.get("refresh_token"))

    def get_access_token(self) -> Optional[str]:
        """Load token from disk; refresh if expired. Returns access token or None."""
        token = self._load_token()
        if token is None:
            return None

        # Check expiry (60s slack)
        expires_at = token.get("expires_at", 0)
        if time.time() < (expires_at - 60):
            # Not expired
            return token.get("access_token")

        # Expired; try refresh
        refresh_token = token.get("refresh_token")
        if not refresh_token:
            logger.warning("[OutlookAuth] Token expired, no refresh_token")
            return None

        if self._refresh_access_token(refresh_token):
            token = self._load_token()
            return token.get("access_token") if token else None

        return None

    def _refresh_access_token(self, refresh_token: str) -> bool:
        """Refresh the access token using a refresh_token.

        Returns True on success (new token persisted), False otherwise.
        """
        try:
            import httpx

            # For a proper refresh, this should be async, but we'll do a sync call
            # via a helper. Actually, let's use httpx's sync Client instead.
            with httpx.Client() as client:
                resp = client.post(
                    f"https://login.microsoftonline.com/{self._tenant}/oauth2/v2.0/token",
                    data={
                        "grant_type": "refresh_token",
                        "refresh_token": refresh_token,
                        "client_id": self._client_id,
                        "scope": "Mail.Read offline_access User.Read",
                    },
                    timeout=10.0,
                )

            if resp.status_code != 200:
                logger.warning(f"[OutlookAuth] Refresh failed: HTTP {resp.status_code}")
                return False

            data = resp.json()
            new_token = {
                "access_token": data.get("access_token"),
                "refresh_token": data.get("refresh_token", refresh_token),
                "expires_at": time.time() + data.get("expires_in", 3600),
            }
            self._save_token(new_token)
            logger.debug("[OutlookAuth] Token refreshed and saved")
            self._token_cache = new_token
            return True

        except Exception as e:
            logger.warning(f"[OutlookAuth] Refresh exception: {e}")
            return False

    async def run_device_code_flow(self) -> bool:
        """Interactive device-code OAuth2 flow. Opens no browser; prints URL + code.

        Returns True on success (token persisted), False otherwise.
        Used by scripts/auth_outlook.py.
        """
        if not self._client_id:
            logger.error("[OutlookAuth] Cannot authenticate: client_id not configured")
            return False

        try:
            import httpx

            async with httpx.AsyncClient() as client:
                # Step 1: Get device code
                device_resp = await client.post(
                    f"https://login.microsoftonline.com/{self._tenant}/oauth2/v2.0/devicecode",
                    data={
                        "client_id": self._client_id,
                        "scope": "Mail.Read offline_access User.Read",
                    },
                    timeout=10.0,
                )

            if device_resp.status_code != 200:
                logger.error(
                    f"[OutlookAuth] devicecode failed: "
                    f"HTTP {device_resp.status_code}"
                )
                return False

            device_data = device_resp.json()
            verification_uri = device_data.get("verification_uri")
            user_code = device_data.get("user_code")
            device_code = device_data.get("device_code")
            poll_interval = device_data.get("interval", 5)
            expires_in = device_data.get("expires_in", 900)

            print(
                f"\n• Open this URL in any browser (same machine):\n"
                f"  {verification_uri}\n"
                f"• Enter this code: {user_code}\n"
                f"• Waiting for consent (timeout: {expires_in}s)...\n"
            )

            # Step 2: Poll for token
            start = time.time()
            while time.time() - start < expires_in:
                await __import__("asyncio").sleep(poll_interval)

                async with httpx.AsyncClient() as client:
                    grant_type = "urn:ietf:params:oauth:grant-type:device_code"
                    token_resp = await client.post(
                        (
                            f"https://login.microsoftonline.com/{self._tenant}/"
                            f"oauth2/v2.0/token"
                        ),
                        data={
                            "grant_type": grant_type,
                            "device_code": device_code,
                            "client_id": self._client_id,
                        },
                        timeout=10.0,
                    )

                if token_resp.status_code == 200:
                    token_data = token_resp.json()
                    token = {
                        "access_token": token_data.get("access_token"),
                        "refresh_token": token_data.get("refresh_token"),
                        "expires_at": time.time() + token_data.get("expires_in", 3600),
                    }
                    self._save_token(token)
                    logger.info("[OutlookAuth] Authentication successful")
                    print("• Consent granted! Token saved.\n")
                    self._token_cache = token
                    return True

                elif token_resp.status_code == 400:
                    # Still waiting or invalid request
                    err_code = token_resp.json().get("error", "")
                    if err_code == "authorization_pending":
                        continue
                    else:
                        logger.warning(f"[OutlookAuth] token poll error: {err_code}")
                        return False

                else:
                    code = token_resp.status_code
                    logger.warning(f"[OutlookAuth] token poll HTTP {code}")
                    return False

            logger.error("[OutlookAuth] Device code expired")
            return False

        except Exception as e:
            logger.error(f"[OutlookAuth] Authentication exception: {e}")
            return False

    def _save_token(self, token: Dict) -> None:
        """Persist token to disk atomically with 0600 perms."""
        self._token_path.parent.mkdir(parents=True, exist_ok=True)

        tmp_path = str(self._token_path) + ".tmp"
        fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(token, f, indent=2)
            os.replace(tmp_path, self._token_path)
        except Exception:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            raise
        logger.debug(f"[OutlookAuth] Token saved to {self._token_path}")

    def _load_token(self) -> Optional[Dict]:
        """Load token from disk. Returns dict or None."""
        if not self._token_path.exists():
            return None

        if self._token_cache is not None:
            return self._token_cache

        try:
            with open(self._token_path, "r", encoding="utf-8") as f:
                token = json.load(f)
            self._token_cache = token
            return token
        except Exception as e:
            logger.warning(f"[OutlookAuth] Failed to load token: {e}")
            return None


# Module-level singleton
_instance: Optional[OutlookAuthManager] = None


def get_outlook_auth() -> Optional[OutlookAuthManager]:
    """Return a shared OutlookAuthManager, or None if unconfigured."""
    global _instance
    if _instance is not None:
        return _instance

    try:
        from config.app_config import (
            EMAIL_OUTLOOK_CLIENT_ID,
            EMAIL_OUTLOOK_TENANT,
        )
    except ImportError:
        return None

    if not EMAIL_OUTLOOK_CLIENT_ID:
        return None

    _instance = OutlookAuthManager(
        client_id=EMAIL_OUTLOOK_CLIENT_ID,
        tenant=EMAIL_OUTLOOK_TENANT,
    )
    return _instance
