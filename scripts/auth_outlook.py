#!/usr/bin/env python3
"""Authenticate Outlook/Microsoft 365 OAuth (Mail.Read access).

One-time setup: run this script to grant Mail.Read permission to Daemon.
It uses the device-code flow, which works on any terminal (no browser
installed required).

What it does:
  1. Reads EMAIL_OUTLOOK_CLIENT_ID from config (or config.local.yaml).
     Get your client_id from:
       - Open https://portal.azure.com
       - App registrations → New registration
       - Name: "Daemon Email Integration" (or similar)
       - Supported account types: "Accounts in any organizational directory"
       - Platform: Add → Mobile and desktop applications
         → Redirect URI: http://localhost
       - Enable "Allow public client flows" (in Authentication)
       - Copy the Application (client) ID
       - API Permissions → Add permission → Delegated → Mail.Read
       - Copy the client_id into config/config.local.yaml:
           email_integration:
             outlook_client_id: "<your app id>"
  2. Prints a device verification URL and code.
  3. You open the URL on ANY device and enter the code.
  4. Daemon saves the token to data/outlook_token.json.

Usage (run in your own terminal):
    python scripts/auth_outlook.py

Token refresh is automatic. To re-authenticate, just run this script again.
"""
import asyncio
import sys
from pathlib import Path

# Ensure project root is importable when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


async def main() -> int:
    from core.email.outlook_auth import get_outlook_auth

    auth = get_outlook_auth()
    if auth is None or not auth.is_configured:
        print("❌ Outlook auth is not configured.")
        print("   Follow the setup steps in this script's docstring, then set:")
        print("   EMAIL_OUTLOOK_CLIENT_ID in config/config.local.yaml")
        print("   (or env DAEMON_OUTLOOK_CLIENT_ID)")
        return 1

    token_path = Path(auth._token_path)
    if token_path.exists():
        backup = token_path.with_suffix(token_path.suffix + ".bak")
        backup.write_bytes(token_path.read_bytes())
        backup.chmod(0o600)  # contains bearer/refresh tokens
        print(f"• Backed up existing token → {backup}")

    print("\n• Starting device-code OAuth flow…")
    print("  Scopes: Mail.Read, User.Read (offline_access for refresh)\n")

    ok = await auth.run_device_code_flow()
    if not ok:
        print("❌ Authentication failed — the token backup (if any) is intact.")
        return 1

    print("✅ Authentication successful. Token saved to", token_path)
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
