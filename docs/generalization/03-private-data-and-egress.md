# G03: Private Local Data and Egress

## Objective

Keep personal information under the user's control, make every off-device
transmission visible and policy-enforced, and reach a verifiable network-off
runtime. Local storage alone is not private inference.

## Threat model

### Protected assets

- Conversation text and attachments.
- User profile, facts, summaries, reflections, threads, and claims.
- Vector embeddings, graph structure, retrieval scores, and learned exemplars.
- Generated notes, exports, diagnostics, telemetry-like local logs, and backups.
- API tokens, OAuth refresh tokens, model/runtime configuration, and action
  receipts.
- Incident bundles and evaluation snapshots.

Embeddings and metadata are personal data. They must not be treated as harmless
because they are not plain text.

### In-scope threats

- Accidental upload to a hosted model, coding agent, search API, or telemetry
  endpoint.
- Logs, crash dumps, backups, clipboard content, and support bundles leaking
  private text.
- Another local browser origin calling the loopback API.
- A stolen powered-off device or copied backup exposing data.
- Dependency or model downloads executing untrusted code.
- A retrieved webpage or document causing prompt injection or memory poisoning.
- Uninstall or deletion leaving derived data behind.
- Configuration drift silently re-enabling hosted fallback.

### Explicit non-protections

- A compromised operating system, administrator, kernel, or malicious process
  while Daemon is unlocked can observe application memory.
- Explicit email, calendar, search, and web requests necessarily disclose some
  information to their destination.
- Full-disk protection depends on Windows security configuration unless an
  application-managed encrypted vault is later adopted.

These limits must appear in the privacy documentation.

## Privacy architecture

### Local data root

Use a non-roaming directory by default so Windows does not silently synchronize
personal data through a roaming profile:

```text
%LOCALAPPDATA%\Daemon\
  data\
  indexes\
  uploads\
  logs\
  backups\
  models\
  cache\
  exports\
  temp\
```

Application binaries may live separately. A user-selected data root must be
checked for cloud-sync indicators and receive an explicit warning.

### Central egress broker

All outbound application requests must be created through one policy service:

```text
EgressRequest
  destination_id
  capability
  privacy_class
  payload_summary
  contains_profile_data
  contains_conversation_text
  requires_confirmation
  retention_expectation
  timeout
```

The broker enforces privacy mode, destination allowlists, request minimization,
timeouts, and a payload-free local activity ledger. Libraries that perform
implicit downloads or telemetry must be configured before import or isolated in
the model/update manager.

## Step-by-step plan

### G03-T01: Create the complete data inventory

1. Enumerate every file, database, collection, index, cache, secret, in-memory
   snapshot, backup, and generated artifact.
2. Record owner module, schema version, sensitivity, path, encryption behavior,
   retention, export, deletion cascade, and backup inclusion.
3. Include Chroma segments, FAISS/CLIP indexes, graph JSON, model caches, debug
   prompts, screenshots, OAuth files, and archived logs.
4. Fail CI when a new persistent path is introduced without an inventory entry.
5. Generate the user-facing storage summary from the inventory.

Acceptance:

- `G03-A01`: A fresh lifecycle produces no unregistered durable file.
- `G03-A02`: Every inventory item has tested export and deletion behavior or an
  explicit non-exportable/non-deletable rationale.

### G03-T02: Consolidate the local data root

1. Route all mutable locations through `AppPaths` from G02.
2. Use restrictive current-user filesystem ACLs at creation.
3. Detect and warn about OneDrive, Dropbox, network shares, removable drives,
   and world-readable locations.
4. Keep application binaries, model artifacts, and personal data separable for
   upgrade and uninstall.
5. Build a dry-run migration from legacy project-relative and AppData paths.
6. Verify copy, checksums, schema load, rollback, and only then switch roots.

Acceptance:

- `G03-A03`: Migration failure leaves the original root intact and selected.
- `G03-A04`: A standard user account, not administrator, can install and operate
  within its own root.

### G03-T03: Protect secrets and powered-off data

1. Store API/OAuth secrets through Windows Credential Manager or DPAPI, never in
   plaintext `.env` for packaged mode.
2. Set owner-only permissions on tokens, private configuration, and backups.
3. Encrypt exported backup archives with authenticated encryption and a recovery
   workflow that is tested before relying on it.
4. Treat full-disk encryption as a documented requirement for stolen-device
   protection until an application vault is validated.
5. Evaluate application-level encryption for stores and indexes, including
   performance and search implications, before promising it.
6. Never invent custom cryptography.

Acceptance:

- `G03-A05`: Secret scans find no packaged-mode token in files, logs, dumps, or
  support bundles.
- `G03-A06`: A modified encrypted backup fails authentication without overwriting
  current data.

### G03-T04: Build the egress inventory

1. Enumerate direct clients for hosted models, Wikipedia, web search, Wolfram,
   arXiv, PubMed, Stack Exchange, Hacker News, location lookup, Google, Telegram,
   Discord, GitHub, model downloads, package updates, and URL fetch.
2. Include dependency telemetry and model-library downloads.
3. Assign each destination a stable ID and minimal required fields.
4. Mark whether personal context is ever required.
5. Create a removal or broker-migration issue for every direct client.

Acceptance:

- `G03-A07`: Static analysis rejects new `httpx`, `requests`, socket, or provider
  client creation outside approved network modules.
- `G03-A08`: Runtime integration tests observe no unknown destination.

### G03-T05: Implement `PrivacyMode` enforcement

1. Add `HOSTED_TRANSITION`, `LOCAL_CONNECTED`, and `OFFLINE` as runtime policy,
   not UI preferences alone.
2. Check mode at the egress broker and model gateway, below feature code.
3. Bind mode into the session manifest and diagnostics.
4. Make mode changes explicit and visible; never let a provider error switch it.
5. In `OFFLINE`, block update checks, geolocation, remote models, and every tool
   that needs the network.
6. Show capability degradation before the session begins.

Acceptance:

- `G03-A09`: A deny-all firewall run completes the offline scenario suite.
- `G03-A10`: Local model failure never causes a hosted request.
- `G03-A11`: Privacy mode cannot be weakened by imported config without visible
  confirmation.

### G03-T06: Minimize connected-tool disclosure

1. Pass only the tool query, not the whole conversation or profile.
2. Remove user identity, private entities, paths, and unrelated location before
   egress unless the operation explicitly requires them.
3. Present a request preview for high-sensitivity or consequential requests.
4. Allow destination-level policies: always allow, ask, or deny.
5. Record destination, timestamp, capability, result status, and payload class,
   but not payload text by default.
6. Provide a one-click network pause.

Acceptance:

- `G03-A12`: Golden tests prove unrelated profile facts never enter web queries.
- `G03-A13`: The activity screen explains every observed outbound connection.

### G03-T07: Eliminate implicit hosted inference

1. Assign every model call a role in G10.
2. Require a backend declaration and privacy class for each role.
3. Remove default provider aliases from stable packaged configuration.
4. Reject unknown model endpoints.
5. During transition, show which roles remain hosted and what context class they
   receive.
6. At completion, keep hosted development adapters outside the runtime package.

Acceptance:

- `G03-A14`: A call-site parity test proves every inference call has a registered
  role and backend policy.
- `G03-A15`: Stable runtime tests contain no hosted provider credentials or DNS
  requests.

### G03-T08: Isolate development agents from live data

1. Move live data outside the source checkout and agent workspace.
2. Add filesystem-deny rules for data roots, backups, exports, logs, and model
   prompt snapshots when Claude Fable or another hosted coding agent runs.
3. Generate synthetic reproductions for development.
4. Add a local pre-send scanner for names, paths, emails, phone numbers, tokens,
   and known private entities.
5. Require manual review of any sanitized incident bundle before hosted use.
6. Record only the sanitized bundle hash in the incident ledger.

Acceptance:

- `G03-A16`: A red-team coding-agent session cannot read the configured private
  data root.
- `G03-A17`: Known private canaries cause the pre-send check to fail.

### G03-T09: Harden logs, crashes, and diagnostics

1. Default production logging to metadata, IDs, hashes, and bounded redacted
   excerpts.
2. Prohibit full prompts, model responses, OAuth payloads, and file contents in
   standard logs.
3. Disable or encrypt crash dumps that may contain process memory.
4. Build a local diagnostic bundle generator with preview and field-level opt-in.
5. Store incident evidence in an encrypted local ledger with retention controls.
6. Test log rotation and deletion across archived files.

Acceptance:

- `G03-A18`: A seeded private-canary workflow leaves no canary in normal logs or
  default support bundles.
- `G03-A19`: Retention cleanup covers compressed and archived logs.

### G03-T10: Verify deletion and uninstall

1. Delegate semantic deletion behavior to G07.
2. On uninstall, distinguish binaries, model packs, user data, and encrypted
   backups.
3. Make data deletion an explicit, separately confirmed option.
4. Produce a deletion report listing every inventory item attempted.
5. Run a forensic filesystem scan within the declared threat model after test
   uninstall.

Acceptance:

- `G03-A20`: Delete-all removes every registered personal artifact outside
  explicitly retained backups.
- `G03-A21`: Preserving data across uninstall permits a verified reinstall and
  migration.

## Privacy release gates

- No "fully private" claim in `HOSTED_TRANSITION`.
- No "offline" claim until an OS-level deny-all test passes.
- No diagnostic upload without preview and explicit action.
- No hidden telemetry.
- No local-model or application update without artifact verification.
- No coding agent access to live personal stores.

## Exit gate

G03 is validated when all durable data is inventoried, all egress is policy
controlled, secrets and backups are protected, development agents cannot read
live data, `OFFLINE` passes under a deny-all boundary, and deletion covers every
registered derivative.

