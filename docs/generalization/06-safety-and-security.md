# G06: Safety and Security

## Objective

Protect a private, memory-bearing, tool-using desktop agent against unintended
actions, untrusted content, memory poisoning, local API abuse, model/runtime
supply-chain risk, and harmful high-risk behavior.

The design is single-user, not trust-free. A single user still encounters
malicious webpages, documents, repositories, model files, browser origins, and
dependencies.

## Required security properties

1. A model cannot grant itself a capability.
2. Untrusted content cannot redefine system goals or become user truth.
3. Consequential actions require explicit user authorization.
4. Action completion claims require an execution receipt.
5. Local-model failure cannot weaken privacy mode or invoke a hosted fallback.
6. Corrupt or newer data is quarantined, never silently overwritten.
7. The loopback API is inaccessible to unauthorized browser origins.
8. Model, runtime, installer, and update artifacts are verified before execution.
9. Safety routing is evaluated across communication styles and is correctable.
10. No automated self-modification reaches the active installation without human
    review, tests, and an ordinary release path.

## Threat-model domains

| Domain | Examples |
|---|---|
| Agent goals | Prompt injection, goal hijacking, instruction smuggling |
| Tools | Wrong arguments, excessive scope, unauthorized or repeated execution |
| Memory | Poisoning, false provenance, stale truth, external content as biography |
| Local API | Cross-origin requests, CSRF, websocket/session theft, exposed bind |
| Files/network | Path traversal, SSRF, unsafe URL schemes, oversized payloads |
| Runtime | Malicious model, `trust_remote_code`, tampered binary, dependency update |
| Sandbox | Host escape, secret access, persistence, resource exhaustion |
| Human safety | Tone misclassification, overconfidence, high-stakes misinformation |
| Development | Coding-agent data access, destructive commands, unreviewed generated code |

Use the OWASP Top 10 for Agentic Applications as one input, not as the complete
threat model:
https://genai.owasp.org/resource/owasp-top-10-for-agentic-applications-for-2026/

## Step-by-step plan

### G06-T01: Create a living threat model

1. Draw trust boundaries for UI, API, model runtime, memory stores, files,
   network tools, action executors, updater, installer, and development agents.
2. Inventory assets, entry points, privileges, and failure effects.
3. Map known incidents and tests to threats.
4. Assign severity, mitigation, detection, recovery, and residual risk.
5. Review whenever a tool, model backend, integration, or persistence format is
   added.

Acceptance:

- `G06-A01`: Every exposed capability appears in the threat model.
- `G06-A02`: Stable release has no unowned critical or high threat.

### G06-T02: Harden the loopback boundary

1. Bind only to `127.0.0.1` and, where supported, IPv6 loopback.
2. Generate a high-entropy per-launch token passed to the bundled UI through a
   protected startup channel.
3. Require token, strict Origin validation, and CSRF protection on state-changing
   routes.
4. Use restrictive CORS rather than wildcard development settings.
5. Separate read-only diagnostics from mutations.
6. Reject external Host headers and never expose the model sidecar publicly.
7. Test hostile local webpages and browser extensions within the declared threat
   limits.

Acceptance:

- `G06-A03`: Requests from an unapproved local origin cannot read or mutate state.
- `G06-A04`: Network scans observe no non-loopback listener.

### G06-T03: Centralize capability permissions

1. Give each tool a stable capability ID and risk level.
2. Separate capability enablement from per-action approval.
3. Define parameter schemas, data-access scope, destination policy, and timeout.
4. Default write, send, delete, execute, and code-modification capabilities off.
5. Show an approval card with exact action, target, important parameters, and
   expected effect.
6. Bind approval to an immutable proposal hash and short expiration.
7. Produce a signed/local execution receipt or an explicit failure result.

Acceptance:

- `G06-A05`: Changed parameters invalidate prior approval.
- `G06-A06`: A response cannot claim completion without the matching receipt.
- `G06-A07`: Retry and timeout cannot execute an action twice.

### G06-T04: Treat retrieved content as untrusted data

1. Attach source and trust labels to web pages, documents, emails, notes, tool
   results, and generated summaries.
2. Keep retrieved instructions out of system/developer authority channels.
3. Add deterministic detection for instruction-smuggling markers and suspicious
   tool requests.
4. Prohibit external content from granting permissions or changing privacy mode.
5. Require corroboration or user confirmation before external content becomes a
   personal fact.
6. Preserve quotations and provenance so the final response can distinguish
   source claims from system claims.

Acceptance:

- `G06-A08`: Prompt-injected documents cannot trigger tools, alter policy, or
  enter autobiographical memory as user-stated facts.
- `G06-A09`: Source labels survive summarization and synthesis.

### G06-T05: Harden memory ingestion

1. Define allowed writers and provenance for every collection.
2. Validate schemas and content length before persistence.
3. Separate user statements, imported documents, external sources, assistant
   inference, and confirmed facts.
4. Quarantine contradictions and suspicious bulk insertions.
5. Rate-limit learned relations and adaptive exemplars.
6. Verify staleness and supersession propagation transactionally where possible.
7. Add integrity checks and last-known-good recovery.

Acceptance:

- `G06-A10`: Memory-poisoning red teams cannot promote external or assistant text
  to user-confirmed truth.
- `G06-A11`: Interrupted writes recover without losing the prior valid state.

### G06-T06: Secure files, URLs, and network tools

1. Normalize and resolve paths before authorization.
2. Restrict file roots and reject traversal, devices, named pipes, and unsafe
   symbolic-link escapes.
3. Validate URL scheme, host, DNS resolution, redirect chain, response size, and
   content type.
4. Block loopback, link-local, private-network, metadata-service, and local-file
   fetches unless a separate explicit capability requires them.
5. Apply time, memory, output, and concurrency limits.
6. Send all external requests through G03 egress policy.

Acceptance:

- `G06-A12`: SSRF and path-traversal suites cover encoded and redirect variants.
- `G06-A13`: Oversized or slow responses terminate within declared bounds.

### G06-T07: Isolate code execution and self-improvement

1. Keep sandbox execution outside the desktop process with no private data mount
   by default.
2. Provide explicit read-only inputs and capture bounded outputs.
3. Disable network unless the specific experiment requires and declares it.
4. Apply CPU, memory, time, process, and filesystem limits.
5. Treat generated patches as untrusted proposals.
6. Require human diff review, focused tests, full release gates, and ordinary
   installation before activation.
7. Never auto-merge or auto-update the live owner installation.

Acceptance:

- `G06-A14`: Sandbox red team cannot read secrets or private data roots.
- `G06-A15`: Generated code cannot bypass the standard release channel.

### G06-T08: Secure the local model supply chain

1. Pin runtime binaries and model artifacts by hash.
2. Record source, license, architecture, quantization, tokenizer, prompt template,
   and known upstream revision.
3. Disable arbitrary `trust_remote_code` in the packaged path.
4. Download only through the update/model manager with signature or hash
   verification and resumable staging.
5. Scan archives before extraction and prevent path escape.
6. Install atomically, preserve the previous pack, and support rollback.
7. Re-run role qualification after any artifact change.

Acceptance:

- `G06-A16`: Tampered runtime/model/update artifacts never execute.
- `G06-A17`: Model-pack rollback restores the last qualified behavior.

### G06-T09: Define high-risk response policy

1. Inventory medical, mental-health, legal, financial, abuse, self-harm, violence,
   and other high-impact routes already present.
2. Define what Daemon may do, must not do, and when it should ask, source, defer,
   or encourage immediate human help.
3. Separate emotional support from diagnosis or therapy claims.
4. Avoid inferring location or presenting jurisdiction-specific resources without
   appropriate context.
5. Permit users to disable proactive emotional inference.
6. Use qualified external reviewers for policy and test cases.
7. Evaluate false escalation and missed escalation across language variation.

Acceptance:

- `G06-A18`: High-risk policies have reviewed test suites and documented residual
  risk.
- `G06-A19`: Communication style alone does not systematically change access to
  ordinary assistance or cause disproportionate escalation.

### G06-T10: Add security testing to release CI

1. Add static analysis, secret scanning, dependency review, SBOM generation, and
   artifact provenance.
2. Add API authorization, prompt injection, tool abuse, memory poisoning, SSRF,
   path traversal, archive extraction, and update tampering suites.
3. Fuzz parsers and action protocols.
4. Run the application in low-privilege clean Windows VMs.
5. Schedule an independent review before stable 1.0.
6. Track remediation and retest evidence in the threat model.

Acceptance:

- `G06-A20`: Stable release has no unresolved critical/high finding.
- `G06-A21`: Every mitigated high finding has a regression test or explicit
  monitoring control.

## Severity and response

| Severity | Examples | Release behavior |
|---|---|---|
| P0 | Privacy breach, destructive data loss, unauthorized external action | Stop use/release; preserve evidence; fix immediately |
| P1 | Dangerous safety failure, repeatable injection, secret exposure | Disable capability or roll back; fix before continuation |
| P2 | Material wrong behavior without immediate harm | Enter incident queue; focused fix and regression |
| P3 | Style, preference, low-impact friction | Personalize or batch; do not distort global policy |

## Exit gate

G06 is validated when the threat model covers every capability, the loopback API
and model runtime are contained, tools are capability- and approval-bound,
untrusted content cannot control or poison the system, artifacts are verified,
high-risk policy is externally reviewed, and independent security testing has no
unresolved critical/high result.

