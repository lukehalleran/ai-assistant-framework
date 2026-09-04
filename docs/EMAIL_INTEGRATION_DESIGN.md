# EMAIL_INTEGRATION_DESIGN.md — Provider-Agnostic Email Layer

*2026-09-01. Owner requirements: (1) analysis queries over mail ("what's going
on this week for me in gmail"); (2) passive contextual retrieval — discussing
the therapist pulls relevant emails into the prompt when they clear a
relevance bar; (3) pattern analysis over email; (4) Outlook as a co-equal
provider, and adding a NEW provider must be easy — every user gets at least
Gmail + Outlook. Generality doctrine applies: provider-agnostic core,
per-provider adapters behind a registry (the ACTION_SPECS lesson: adding one
= one adapter + one registry row, consumers inherit).*

## Doctrine decisions (v1)

- **Live-fetch only, never persisted.** Email bodies/snippets are NEVER
  written to chroma/corpus/facts in v1 — 5-min TTL in-memory cache
  (google_calendar pattern). Emails are external ground truth, not memories;
  persisting them creates a privacy surface and a dedup problem we don't
  need. Pattern analysis runs over live-fetched HEADERS in-window.
- **Metadata-first.** Fetch headers + provider snippet/bodyPreview only.
  Full bodies are a later, explicitly-requested capability.
- **Read-only.** Send stays where it is (`core/actions/email.py`, the
  human-gated action path). This layer never writes/deletes/labels mail.
- **Passive section is cue-gated + distress-suppressed.** The [RELEVANT
  EMAILS] section only fires on an email cue or a contact-anchored query,
  applies a semantic relevance bar, caps at 3, and is suppressed on elevated
  tone (the refdocs-suppression doctrine — inbox content mid-distress is an
  amplifier).
- **Gmail auth already covers read** (`gmail.readonly` granted since the
  contact-resolution work). Outlook uses the OAuth2 device-code flow via raw
  httpx (no new dependency, no MSAL) — owner does a one-time Azure app
  registration + `scripts/auth_outlook.py` login; token persisted like
  `google_token.json` (atomic, 0600, refresh handling).

## Architecture

```
core/email/
├── provider.py         # EmailMessage dataclass + EmailProvider Protocol
├── registry.py         # PROVIDERS registry: name → (factory, enabled-flag)
│                       #   adding a provider = one adapter + one row
├── gmail_provider.py   # Gmail REST via httpx + existing google_auth
├── outlook_provider.py # Microsoft Graph via httpx + device-code token store
└── service.py          # EmailService singleton: fan-out across enabled
                        #   providers, merge newest-first, TTL cache, health
```

**Frozen interface contract** (both integration surfaces code against this):

```python
@dataclass
class EmailMessage:
    provider: str        # "gmail" | "outlook"
    message_id: str
    thread_id: str = ""
    sender: str = ""     # "Name <addr>"
    to: str = ""
    subject: str = ""
    snippet: str = ""    # plain text, provider snippet/bodyPreview
    date: str = ""       # ISO 8601
    unread: bool = False
    web_link: str = ""

class EmailProvider(Protocol):
    name: str
    def is_configured(self) -> bool: ...
    async def health(self) -> dict: ...          # {"available", "detail"}
    async def search(self, query, *, window_days=30, limit=20) -> list[EmailMessage]: ...
    # passive [RELEVANT EMAILS] context calls search() with window_days=EMAIL_DEFAULT_WINDOW_DAYS
    # (config email.default_window_days, 7) — 2026-09-03; it was a hardcoded 30 before.
    async def recent(self, *, window_days=7, limit=25) -> list[EmailMessage]: ...

get_email_service() -> EmailService   # .search()/.recent()/.health() fan-out
```

## Surfaces

1. **Agentic tool `email_search`** — explicit queries. SearchDecision flags
   `wants_email_search` + `email_query`/`email_window_days`, DISPATCH_TABLE
   row, native + XML tool definitions, tool-health from provider registry.
   Narrow Tier-1 gate arm: word-bounded email noun (email/emails/inbox/
   gmail/outlook/mail) + info-seeking shape + ≤30 words → tools (under-fires;
   Tier-4 tool advertisement covers the rest). "What's going on this week in
   gmail" → tool with window=7 → model synthesizes over returned headers.
2. **Passive [RELEVANT EMAILS] prompt section** — gatherer task fires only
   when the query has an email cue OR a rare-proper-noun/contact anchor
   (extract_rare_proper_nouns doctrine); provider search seeded with the
   anchor; results scored subject+snippet vs query in MiniLM space
   (both-sides-fresh, self-consistent); bar `EMAIL_PASSIVE_MIN_RELEVANCE`,
   cap `EMAIL_PASSIVE_MAX` = 3; formatter section + PRIORITY_ORDER row
   (the 08-14 metered-keys parity test enforces this); suppressed on
   elevated tone.
3. **Pattern dimension `email`** — deterministic counts over live-fetched
   in-window headers: volume per bucket, top sender domains, sent-vs-received
   where derivable. On-demand only (pattern doctrine), denominator caveats
   inherited.

## Config (`email_integration` YAML section → schema → EMAIL_* constants)

enabled, gmail_enabled, outlook_enabled, outlook_client_id (config.local),
outlook_tenant ("common"), max_results, cache_ttl_seconds, passive flags
(enabled / max=3 / min_relevance), window defaults. Committed config.yaml
ships generic (outlook_client_id blank) per the personal-vocabulary doctrine.

## Outlook specifics

Device-code flow, raw REST: POST `login.microsoftonline.com/{tenant}/oauth2/
v2.0/devicecode` (scope `Mail.Read offline_access User.Read`) → user enters
code at microsoft.com/devicelogin → poll `/token` → persist
`data/outlook_token.json` (atomic, 0600) with refresh_token; refresh on
expiry. Graph reads: `GET /v1.0/me/messages?$select=...&$top=N&$filter=
receivedDateTime ge ...` and `$search` (with `ConsistencyLevel: eventual`)
for query search. Owner setup (one-time): Azure portal app registration,
"Mobile and desktop applications" platform, public client flows enabled,
delegated Mail.Read — steps in `scripts/auth_outlook.py --help`.

## Deferred (explicitly)

Full-body fetch + summarize; email → thread/commitment extraction; proactive
email surfacing ("you got a reply from X"); write actions (archive/label);
IMAP generic provider (the registry makes it a one-adapter add later);
persistent email index. Each needs its own design pass.

## Status

2026-09-01: v1 built (two cheap-executor batches + frontier review) — core
package + both adapters + agentic tool + passive section + pattern dimension.
Outlook adapter is code-complete but DORMANT until the owner registers the
Azure app and runs `scripts/auth_outlook.py`.
