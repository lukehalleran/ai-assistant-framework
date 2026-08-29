# Daemon Generalization and Local-First Roadmap

Status: planning baseline

Target window: 18-24 months

Primary target: a single-user Windows desktop executable for adult users who
communicate primarily in typed American English.

This directory turns the existing owner-shape audit into an implementation and
validation program. It does not claim that universal generalization is possible.
The target is a bounded, testable product claim backed by external evidence.

## Fixed assumptions

1. One installed Daemon instance serves one human profile. Multi-tenancy and a
   hosted SaaS control plane are out of scope.
2. Daemon is distributed and launched as a Windows executable. A loopback HTTP
   server may remain an internal implementation detail, but it is never exposed
   as a hosted product.
3. Personal stores, logs, indexes, model state, and backups are local by default.
   There is no default telemetry.
4. Hosted inference is transitional. It is not part of the final runtime.
5. Within 18-24 months, every inference role must have a qualified local model.
6. Explicit network tools may remain available in connected mode. They are
   separate from local inference and must disclose and minimize egress.
7. Owner dogfooding is the primary early discovery loop. It is not population
   validation.
8. Claude Fable may assist development using source code and sanitized incident
   bundles. It must not receive raw private stores or live transcripts.
9. Human approval remains mandatory for consequential actions and code changes.
10. The first supported population is adults. Supporting minors is a separate
    safety and product program.

## Privacy vocabulary

The product must use these terms exactly. "Local-first" is too ambiguous for a
privacy indicator.

| Mode | Inference | Other network access | Honest privacy claim |
|---|---|---|---|
| `HOSTED_TRANSITION` | Some prompts leave the device | Explicit tools may connect | Local storage, not private inference |
| `LOCAL_CONNECTED` | All model inference is local | Only enabled tools and updates connect | Private local inference with disclosed tool egress |
| `OFFLINE` | All model inference is local | Denied | No application network egress during the session |

Complete privacy cannot be claimed while hosted inference receives conversation
context. Connected web, email, calendar, and search tools also disclose the
minimum data needed for their request. The UI must make this distinction visible.

## Product claim at completion

The intended claim is:

> Daemon is a private, single-user Windows desktop assistant for adults who use
> typed American English. Its memory and inference run locally. Optional network
> tools transmit only disclosed request data. Behavior is evaluated across
> documented communication styles and user groups, with explicit limitations.

The product must never claim to work for "any and all" users. Generalization is
an ongoing measured property, not a completed boolean state.

## Target architecture

```text
Daemon.exe
  |
  +-- DesktopHost
  |     process lifecycle, single-instance lock, updater, crash recovery
  |
  +-- Loopback UI/API
  |     bundled React assets, FastAPI, origin checks, per-launch token
  |
  +-- UserContext + AppPaths
  |     one profile, one local data root, no owner literals
  |
  +-- PrivacyPolicy + EgressBroker
  |     privacy mode, destination policy, request minimization, local audit
  |
  +-- ModelGateway
  |     role registry, local runtime, schemas, scheduling, capability checks
  |
  +-- MemoryKernel
  |     corpus, vector stores, graph, profile, claims, provenance, deletion
  |
  +-- ActionBroker
  |     capability permissions, proposals, approval, execution receipts
  |
  +-- Evaluation + Incident Ledger
        frozen cases, dogfood incidents, external cohorts, release gates
```

The local model runtime should be a separately managed sidecar process rather
than loading every generative model into the desktop process. A loopback,
OpenAI-compatible runtime such as llama.cpp is the initial reference option, not
a permanent hard dependency. The runtime and model artifacts must be pinned,
hash-verified, license-recorded, and replaceable behind `ModelGateway`.

## Workstreams

| ID | Plan | Outcome |
|---|---|---|
| G01 | [Product contract](01-product-contract.md) | Bounded users, capabilities, non-goals, and claims |
| G02 | [Owner-neutral runtime](02-owner-neutral-runtime.md) | No owner identity or calibration in global behavior |
| G03 | [Private local data and egress](03-private-data-and-egress.md) | Enforced data boundary and honest privacy modes |
| G04 | [Adaptive personalization](04-adaptive-personalization.md) | Safe per-user learning without global overfitting |
| G05 | [Population evaluation](05-population-evaluation.md) | External, stratified evidence rather than owner-only metrics |
| G06 | [Safety and security](06-safety-and-security.md) | Threat-modeled memory, tools, actions, and local API |
| G07 | [User memory control](07-user-memory-control.md) | Inspect, correct, retain, export, and delete every derivative |
| G08 | [Windows executable](08-windows-executable.md) | Reproducible install, update, recovery, and offline operation |
| G09 | [Accessibility and communication](09-accessibility-and-communication.md) | WCAG 2.2 AA plus cognitive and language usability |
| G10 | [Local model migration](10-local-model-migration.md) | Hosted inference removed role by role |
| G11 | [Dogfooding and incident process](11-dogfooding-and-incidents.md) | Fable-assisted fixes without owner-overfit or data disclosure |
| G12 | [Response integrity and atomic revision](12-response-integrity.md) | One coherent, evidence-bounded answer across display, storage, and indexing |

## Dependency order

```text
G01 Product contract
  +--> G02 UserContext/AppPaths
  |      +--> G03 private storage and egress
  |      +--> G07 memory control
  |      +--> G08 executable packaging
  |
  +--> G04 adaptation policy
  |      +--> G05 population evaluation
  |
  +--> G06 safety/security
  |      +--> G08 executable release
  |
  +--> G09 accessibility
  |
  +--> G10 role-based local inference
         +--> G05 local-model qualification
         +--> G08 model-pack distribution

G11 dogfooding and incidents runs across every phase.
G12 response integrity gates any post-generation factual review before it can
enter display, storage, or indexing.
External population claims wait for G02-G10 release gates.
```

## Delivery waves

### Wave 0: Freeze and baseline, weeks 0-4

1. Ratify G01 and assign stable requirement IDs.
2. Declare `HOSTED_TRANSITION` in the UI and documentation.
3. Freeze new feature families unless they close a roadmap requirement.
4. Snapshot the current owner corpus metrics and representative latency/cost.
5. Inventory every model call, network call, personal-data store, and executable
   artifact.
6. Start the incident ledger and use the G11 classification process for every
   dogfood failure.
7. Separate live private data from the development checkout and agent-readable
   paths.

Exit gate: scope is approved, all current egress and stores have owners, and no
new feature bypasses the roadmap review.

### Wave 1: Neutral and private foundations, months 1-3

1. Introduce typed `UserContext`, `AppPaths`, `PrivacyMode`, and data inventory.
2. Remove production owner literals and direct `data/...` path ownership.
3. Build the centralized egress broker and network activity ledger.
4. Move secrets to Windows Credential Manager or DPAPI-protected storage.
5. Add memory pause, session-only mode, and category-level storage policy.
6. Add static owner-literal, absolute-path, and direct-network CI checks.
7. Produce a clean synthetic profile and fresh-install fixture.

Exit gate: a random synthetic user can complete the primary workflow without
Luke-specific behavior, and all network egress is either brokered or explicitly
waived with an owner and removal issue.

### Wave 2: Executable alpha and local utility models, months 3-6

1. Restore a reproducible Windows build around the React/FastAPI product.
2. Add the managed local runtime and hardware probe.
3. Move embeddings, reranking, classification, extraction, and summarization to
   pinned offline-capable artifacts.
4. Add signed model manifests, resumable downloads, and offline sideloading.
5. Implement privacy-mode UI, network-off integration tests, and encrypted
   support bundles.
6. Complete keyboard, focus, streaming, zoom, and screen-reader foundations.

Exit gate: a clean Windows VM installs the executable, downloads or sideloads a
qualified utility model pack, completes core memory workflows, and uninstalls
without losing or silently retaining user data.

### Wave 3: Local conversational core, months 6-12

1. Move query rewrite, tone/intent fallback, fact verification, and agent tool
   decisions through `ModelGateway` role contracts.
2. Qualify a local main-response model on supported hardware tiers.
3. Reduce prompt and concurrency demands to local-runtime budgets.
4. Run local candidates in shadow comparison before promotion.
5. Remove silent hosted fallback. Local failure must fail honestly.
6. Complete action, prompt-injection, memory-poisoning, and loopback API red
   teams.
7. Begin trusted clean-install alpha with synthetic or volunteered data only.

Exit gate: ordinary chat, memory, corrections, and tool decisions run locally at
declared quality and latency thresholds. Hosted use is explicit opt-in and never
automatic.

### Wave 4: Local completeness and external beta, months 12-18

1. Migrate document generation, synthesis, evaluation judges, and optional vision.
2. Run every release suite under `LOCAL_CONNECTED` and `OFFLINE`.
3. Expand from trusted alpha to a consented, diverse external cohort.
4. Measure cold-start and longitudinal behavior by task and subgroup.
5. Complete WCAG 2.2 AA audit plus manual assistive-technology testing.
6. Ship correction, deletion cascade, export, and restore UI as release blockers.

Exit gate: no runtime inference role requires a hosted provider, and the product
has external evidence beyond the owner corpus.

### Wave 5: Network-off 1.0, months 18-24

1. Remove hosted provider credentials and aliases from the default distribution.
2. Keep any hosted development oracle in separate developer tooling only.
3. Perform an independent security and privacy review.
4. Run the full product behind a deny-all network boundary.
5. Verify signed install, update, rollback, model-pack, backup, and deletion paths.
6. Publish the support matrix, evaluation report, known limitations, threat model,
   model licenses, and privacy behavior.

Exit gate: the executable passes the master definition of done below.

## Master definition of done

All items are required for the intended 1.0 claim.

- [ ] No owner-specific literal, path, calibration datum, or private incident is
      required by production behavior.
- [ ] One authoritative `UserContext` and `AppPaths` reach every subsystem.
- [ ] Every durable datum appears in the data inventory with retention, deletion,
      backup, and encryption behavior.
- [ ] Every outbound connection goes through the egress policy or has a temporary,
      tested exception with an expiration milestone.
- [ ] `OFFLINE` passes with application egress blocked at the operating-system level.
- [ ] No runtime inference silently falls back to a hosted provider.
- [ ] Every model role has a pinned artifact, capability declaration, license,
      hardware envelope, and qualification report.
- [ ] Consequential tools require explicit permission and produce execution receipts.
- [ ] The user can inspect, correct, delete, export, and restore memory and all
      relevant derivatives.
- [ ] The Windows installer, upgrade, rollback, and uninstall paths pass on clean VMs.
- [ ] The UI passes WCAG 2.2 AA automated and manual gates.
- [ ] External evaluation reports aggregate and worst-group outcomes with confidence
      intervals and documented limitations.
- [ ] Dogfood incidents have generalized regression tests rather than exact owner
      phrase patches unless explicitly classified as personal preference.
- [ ] Security, privacy, and accessibility reviews include people other than the
      owner and the development model.
- [ ] Release documentation makes no universal-user, clinical, or zero-risk claim.

## Program controls

### Requirement IDs

Use `Gxx-Ryy` for requirements, `Gxx-Tyy` for implementation tasks, and
`Gxx-Ayy` for acceptance tests. Tests and commits should cite the relevant ID.

### Status values

Use only `not_started`, `in_progress`, `blocked`, `validated`, and `retired`.
"Implemented" is not a completion state until the acceptance evidence exists.

### Evidence package

Each validated requirement stores:

1. The requirement and threat or user need.
2. The implementation commit.
3. Automated tests and exact command.
4. Manual or external evaluation evidence when required.
5. Known limitations.
6. Rollback procedure.
7. Revalidation triggers, including model or runtime upgrades.

### Change allocation

Until Wave 3 is complete, reserve most development capacity for consolidation:

| Work | Target share |
|---|---:|
| Generalization, privacy, local inference, packaging | 70% |
| Dogfood defects and safety incidents | 20% |
| New capabilities | 10% maximum |

P0 privacy, data-loss, unauthorized-action, and acute-safety incidents override
the allocation.

## Required external participation

The early program can remain owner-operated, but these gates cannot be
self-certified:

- Population language and usability evaluation.
- Accessibility testing with assistive-technology users.
- Security and privacy review.
- Mental-health and other high-risk policy review where such behavior remains.
- Clean-install and longitudinal beta behavior.

Fable and synthetic personas may accelerate preparation. They do not replace
independent people because they can reproduce the same assumptions as the system
being evaluated.

## Reference standards and implementation inputs

- NIST AI RMF Generative AI Profile:
  https://www.nist.gov/publications/artificial-intelligence-risk-management-framework-generative-artificial-intelligence
- NIST Privacy Framework:
  https://www.nist.gov/privacy-framework
- OWASP Top 10 for Agentic Applications 2026:
  https://genai.owasp.org/resource/owasp-top-10-for-agentic-applications-for-2026/
- WCAG 2.2:
  https://www.w3.org/TR/WCAG22/
- W3C cognitive accessibility guidance:
  https://www.w3.org/WAI/cognitive/
- llama.cpp local runtime and server documentation:
  https://github.com/ggml-org/llama.cpp
  https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md
