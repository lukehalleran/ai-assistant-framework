"""
# core/actions/google_auth.py

Module Contract
- Purpose: Google OAuth2 credential management for Gmail send + Calendar read/write.
- Public interface:
  - GoogleAuthManager: OAuth2 flow, token persistence, refresh, scope-upgrade detection, has_scope().
  - get_google_auth() -> GoogleAuthManager | None: Lazy singleton from config.
- Dependencies: google-auth-oauthlib, google.oauth2.credentials
- Side effects: Reads/writes token file at configured path. Opens browser for OAuth consent.
  Run as __main__ for one-time browser auth: python -m core.actions.google_auth
"""

import asyncio
import json
from pathlib import Path
from typing import Optional

from utils.logging_utils import get_logger

logger = get_logger("google_auth")

SCOPES = [
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/calendar.readonly",
    "https://www.googleapis.com/auth/calendar.events",
]

# Lazy singleton
_instance: "GoogleAuthManager | None" = None


def get_google_auth() -> "GoogleAuthManager | None":
    """Return a shared GoogleAuthManager, or None if unconfigured.

    Reads client_id / client_secret from app_config (which pulls from
    config.yaml with env-var overrides).  The same instance is reused
    for the lifetime of the process.
    """
    global _instance
    if _instance is not None:
        return _instance

    try:
        from config.app_config import (
            INTERNET_ACTIONS_GOOGLE_CLIENT_ID,
            INTERNET_ACTIONS_GOOGLE_CLIENT_SECRET,
            INTERNET_ACTIONS_GOOGLE_TOKEN_PATH,
        )
    except ImportError:
        return None

    if not INTERNET_ACTIONS_GOOGLE_CLIENT_ID or not INTERNET_ACTIONS_GOOGLE_CLIENT_SECRET:
        return None

    _instance = GoogleAuthManager(
        client_id=INTERNET_ACTIONS_GOOGLE_CLIENT_ID,
        client_secret=INTERNET_ACTIONS_GOOGLE_CLIENT_SECRET,
        token_path=INTERNET_ACTIONS_GOOGLE_TOKEN_PATH,
    )
    return _instance


class GoogleAuthManager:
    """Manages Google OAuth2 credentials for Gmail and Calendar APIs.

    Handles the installed-app OAuth2 flow, token persistence, automatic
    refresh, and scope-upgrade detection.
    """

    def __init__(
        self,
        client_id: str = "",
        client_secret: str = "",
        token_path: str = "data/google_token.json",
        scopes: Optional[list] = None,
    ):
        self._client_id = client_id
        self._client_secret = client_secret
        self._token_path = Path(token_path)
        self._scopes = scopes or SCOPES
        self._credentials = None

    @property
    def is_configured(self) -> bool:
        """True if client_id and client_secret are set."""
        return bool(self._client_id) and bool(self._client_secret)

    @property
    def is_authenticated(self) -> bool:
        """True if a valid (or refreshable) token exists."""
        creds = self._load_token()
        return creds is not None

    def get_credentials(self):
        """Load credentials, refreshing if expired.

        Returns google.oauth2.credentials.Credentials or None.
        """
        if self._credentials is not None and self._credentials.valid:
            return self._credentials

        creds = self._load_token()
        if creds is None:
            return None

        if creds.expired and creds.refresh_token:
            try:
                from google.auth.transport.requests import Request
                creds.refresh(Request())
                self._save_token(creds)
                logger.info("[GoogleAuth] Token refreshed successfully")
            except Exception as e:
                logger.warning(f"[GoogleAuth] Token refresh failed: {e}")
                return None

        self._credentials = creds
        return creds

    async def authenticate(self) -> bool:
        """Run the OAuth2 installed-app flow (opens browser).

        Returns True on success, False on failure.
        """
        if not self.is_configured:
            logger.error("[GoogleAuth] Cannot authenticate: client_id/secret not configured")
            return False

        try:
            from google_auth_oauthlib.flow import InstalledAppFlow

            client_config = {
                "installed": {
                    "client_id": self._client_id,
                    "client_secret": self._client_secret,
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                    "redirect_uris": ["http://localhost"],
                }
            }

            flow = InstalledAppFlow.from_client_config(client_config, self._scopes)

            # Run browser flow in executor to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            creds = await loop.run_in_executor(
                None, lambda: flow.run_local_server(port=0)
            )

            self._save_token(creds)
            self._credentials = creds
            logger.info("[GoogleAuth] Authentication successful")
            return True

        except Exception as e:
            logger.error(f"[GoogleAuth] Authentication failed: {e}")
            return False

    def has_scope(self, scope_url: str) -> bool:
        """Check if the stored token includes a specific scope."""
        creds = self._load_token()
        if creds is None or not creds.scopes:
            return False
        return scope_url in creds.scopes

    def needs_scope_upgrade(self) -> bool:
        """Check if stored token is missing any requested scope."""
        creds = self._load_token()
        if creds is None:
            return False

        if not creds.scopes:
            return True

        return not set(self._scopes).issubset(set(creds.scopes))

    def _save_token(self, credentials) -> None:
        """Persist credentials to token file."""
        self._token_path.parent.mkdir(parents=True, exist_ok=True)

        token_data = {
            "token": credentials.token,
            "refresh_token": credentials.refresh_token,
            "token_uri": credentials.token_uri,
            "client_id": credentials.client_id,
            "client_secret": credentials.client_secret,
            "scopes": list(credentials.scopes or []),
        }
        if credentials.expiry:
            token_data["expiry"] = credentials.expiry.isoformat()

        self._token_path.write_text(json.dumps(token_data, indent=2))
        logger.debug(f"[GoogleAuth] Token saved to {self._token_path}")

    def _load_token(self):
        """Load credentials from token file. Returns Credentials or None."""
        if not self._token_path.exists():
            return None

        try:
            from google.oauth2.credentials import Credentials

            token_data = json.loads(self._token_path.read_text())
            creds = Credentials(
                token=token_data.get("token"),
                refresh_token=token_data.get("refresh_token"),
                token_uri=token_data.get("token_uri", "https://oauth2.googleapis.com/token"),
                client_id=token_data.get("client_id", self._client_id),
                client_secret=token_data.get("client_secret", self._client_secret),
                scopes=token_data.get("scopes"),
            )

            # Restore expiry if present
            expiry_str = token_data.get("expiry")
            if expiry_str:
                from datetime import datetime, timezone
                try:
                    creds.expiry = datetime.fromisoformat(expiry_str).replace(tzinfo=None)
                except (ValueError, TypeError):
                    pass

            return creds

        except Exception as e:
            logger.warning(f"[GoogleAuth] Failed to load token: {e}")
            return None


if __name__ == "__main__":
    import sys

    auth = get_google_auth()
    if auth is None:
        print("ERROR: GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET env vars must be set.")
        sys.exit(1)

    if auth.is_authenticated and not auth.needs_scope_upgrade():
        print(f"Already authenticated. Token at: {auth._token_path}")
        sys.exit(0)

    print("Opening browser for Google OAuth consent...")
    ok = asyncio.run(auth.authenticate())
    if ok:
        print(f"Success! Token saved to {auth._token_path}")
    else:
        print("Authentication failed. Check the logs.")
        sys.exit(1)
