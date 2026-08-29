# G01: Product Contract

## Objective

Define exactly who Daemon serves, what it does in each privacy mode, what it does
not claim, and which evidence is required before a statement appears in release
documentation.

This contract prevents "generalization" from becoming an unlimited feature list
or an untestable promise.

## Supported target for 1.0

### User

- One adult human per installed instance.
- Primarily communicates through typed American English.
- May use regional or social dialect, slang, fragments, typos, profanity,
  code-switching, assistive technology, or a personally configured style.
- Controls what Daemon stores and which network capabilities are enabled.
- Is not assumed to share the owner's identity, education, location, health,
  family structure, interests, technology skill, or communication cadence.

### Platform

- Windows 11 x64 is the first required operating system.
- A documented CPU, RAM, disk, and optional GPU matrix determines available
  local model packs.
- Daemon launches through a signed executable and installer.
- The React/FastAPI boundary may run over loopback only.

### Core capabilities

- Private local conversation and persistent memory.
- User-visible fact provenance, correction, staleness, and deletion.
- Local retrieval, reranking, classification, extraction, planning, and response
  generation.
- Optional connected tools with explicit capability permissions and visible
  egress behavior.
- Human-approved consequential actions.
- Backup, restore, export, reset, upgrade, rollback, and uninstall.

## Explicit non-goals for 1.0

- Multi-user or multi-tenant service.
- Hosted web application or remote account system.
- Support for minors.
- Clinical diagnosis, therapy, emergency response, legal representation, or
  fiduciary financial advice.
- Guaranteed correctness, universal dialect coverage, or guaranteed emotional
  interpretation.
- Autonomous external actions without approval.
- Silent cloud inference or telemetry.
- Identical behavior on hardware outside the supported matrix.

## Capability and privacy matrix

The UI and documentation must publish a generated matrix from runtime metadata,
not maintain a hand-written list that can drift.

| Capability | Hosted transition | Local connected | Offline |
|---|---|---|---|
| Conversation inference | Hosted or local by declared role | Local | Local |
| Memory and retrieval | Local | Local | Local |
| Embeddings and reranking | Local and pinned | Local and pinned | Local and pinned |
| Web and academic search | Optional | Optional | Unavailable |
| Email/calendar/contacts | Optional | Optional | Unavailable |
| Local files and notes | Local with permission | Local with permission | Local with permission |
| Software/model updates | Optional | Optional | Manual sideload only |
| Telemetry | Off | Off | Off |

## Step-by-step plan

### G01-T01: Create a versioned product contract

1. Add a machine-readable `product_contract.yaml` schema.
2. Record supported population, platform, modes, capabilities, and non-goals.
3. Give every statement an ID and version.
4. Make documentation and the About/Privacy UI read from this source where
   practical.
5. Require a contract version in incident and evaluation manifests.

Acceptance:

- `G01-A01`: Configuration validation rejects unknown privacy modes.
- `G01-A02`: UI and documentation tests agree with the contract capability matrix.
- `G01-A03`: Changing a supported claim requires an evaluation evidence link.

### G01-T02: Define user-visible limitation language

1. Write concise text for local inference, connected tools, and offline mode.
2. State that connected requests disclose request data to their destination.
3. State that emotional inference may be wrong and can be disabled or corrected.
4. State that high-stakes responses are informational and source-dependent.
5. Prohibit "works for everyone," "fully private" in connected modes, and
   "understands you" without qualification.

Acceptance:

- `G01-A04`: Static release-copy check rejects prohibited universal claims.
- `G01-A05`: A first-time user can identify whether a sample operation leaves
  the device in usability testing.

### G01-T03: Define hardware tiers

1. Build a hardware probe for architecture, RAM, free disk, GPU vendor, VRAM,
   and supported acceleration backend.
2. Define a minimum tier for utility models and a recommended tier for the main
   conversational model.
3. Benchmark first-token latency, tokens per second, peak RAM/VRAM, and sustained
   thermal behavior.
4. Disable or downgrade capabilities that do not meet their declared envelope.
5. Never download a model before showing size, license, disk need, and expected
   performance for the detected hardware.

Acceptance:

- `G01-A06`: Unsupported hardware receives a clear result before model download.
- `G01-A07`: Each supported tier completes a fixed end-to-end workload without
  out-of-memory termination.

### G01-T04: Define lifecycle support

1. Establish release channels: development, owner-canary, external-alpha, beta,
   and stable.
2. Define how long data migrations and the previous executable are supported.
3. Define model-pack compatibility independently from application version.
4. Define security update and rollback policy.
5. Define how unsupported configurations fail without damaging data.

Acceptance:

- `G01-A08`: A stable release can open data from the two previous supported
  schema versions through tested migrations.
- `G01-A09`: Rollback behavior is documented and tested before every stable build.

### G01-T05: Establish the claims review

1. Map every public capability claim to an automated, manual, or external test.
2. Label evidence as owner-only, synthetic, external, or independently reviewed.
3. Prevent owner-only evidence from supporting population claims.
4. Add dates, model/runtime versions, sample sizes, and limitations.
5. Retire a claim automatically when its model or critical subsystem changes
   until revalidation completes.

Acceptance:

- `G01-A10`: Every statement in the stable capability matrix has current evidence.
- `G01-A11`: An expired evidence record blocks stable release generation.

## Decisions required before implementation

- Confirm Windows 11 x64 as the only mandatory 1.0 OS.
- Set minimum and recommended hardware after the runtime spike, not by guess.
- Decide whether connected web search may run without per-query confirmation.
- Decide whether health-related memory storage is opt-in or enabled with clear
  category controls.
- Decide the exact boundary between companion behavior and prohibited clinical
  framing.

## Exit gate

G01 is validated when the product contract is machine-readable, the UI and docs
derive their behavior from it, every supported claim has an evidence type, and
unbounded claims are absent from stable release material.

