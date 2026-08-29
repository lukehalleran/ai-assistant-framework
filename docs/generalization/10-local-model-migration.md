# G10: Local Model Migration

## Objective

Remove hosted inference from the Daemon runtime role by role while preserving
measured quality, privacy, structured output, tool safety, latency, and support on
declared Windows hardware tiers.

The final stable application must not contain a silent or automatic hosted
fallback. Claude Fable may remain a development reference against sanitized
material outside the packaged runtime.

## Current gap

`models/model_manager.py` currently combines a large hosted-provider registry
with a basic in-process Hugging Face local path. The local path does not provide
the complete lifecycle, capability, structured-output, scheduling, artifact,
hardware, or privacy contracts needed by the application. Roughly forty modules
invoke generation for distinct tasks.

This is not a one-line base-URL replacement.

## Target model architecture

### `ModelRole`

Every inference call declares a stable role, for example:

| Role family | Example roles |
|---|---|
| Conversation | final response, decision-answer recovery, clarification |
| Agent | tool selection, tool synthesis, action proposal |
| Retrieval | query rewrite, intent fallback, memory query expansion |
| Memory | fact extraction, fact verification, summaries, threads, skills |
| Safety/style | tone arbitration, response planning, grounding check |
| Research | document generation, synthesis candidate, coherence judge |
| Evaluation | pairwise judge, objective checks, grading assistance |
| Multimodal | image caption, visual interpretation |

Each role contract declares:

```text
role_id
privacy_class
required_capabilities
input_schema
output_schema
maximum_context
maximum_output
latency_budget
determinism_policy
failure_policy
allowed_backends_by_privacy_mode
qualification_suite
```

### `ModelGateway`

All application inference goes through one gateway that provides:

- Role lookup and backend selection.
- Capability and context validation.
- OpenAI-compatible request normalization where useful.
- Schema-constrained output and validation.
- Streaming and cancellation.
- Queueing and resource scheduling.
- Privacy-mode enforcement.
- Payload-safe telemetry and reproducibility manifests.
- Explicit failure without hidden provider fallback.

### Local runtime

The first reference runtime should be a managed llama.cpp sidecar because it has
Windows builds, GGUF support, CPU/GPU backends, an OpenAI-compatible loopback
server, structured JSON, reranking, and tool-use support. It remains replaceable
behind the gateway.

References:

- https://github.com/ggml-org/llama.cpp
- https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md

## Step-by-step plan

### G10-T01: Inventory every inference call

1. Find direct and indirect generation calls across core, memory, knowledge,
   utils, GUI, evaluation, and agent branches.
2. Record prompt inputs, maximum observed context, output form, model assumption,
   timeout, temperature, reasoning behavior, and side effects.
3. Assign one role ID to each call site.
4. Identify calls that can become deterministic code rather than model work.
5. Identify duplicate prompts and dead/retired paths.
6. Add a parity test that fails on unregistered model calls.

Acceptance:

- `G10-A01`: Every runtime inference call maps to exactly one role.
- `G10-A02`: The inventory records whether personal data can enter each role.

### G10-T02: Separate providers, roles, and model artifacts

1. Replace provider-specific model aliases in application logic with role IDs.
2. Define backend adapters for hosted transition, local OpenAI-compatible, and
   deterministic/no-model implementations.
3. Move capability declarations to artifact manifests rather than slug substring
   checks.
4. Keep model selection in configuration generated from qualified combinations.
5. Deprecate direct `ModelManager.generate*` use outside the gateway.
6. Retain a narrow compatibility shim with call-count telemetry until removed.

Acceptance:

- `G10-A03`: Application code selects roles, never provider brands.
- `G10-A04`: Removing hosted credentials does not prevent gateway construction.

### G10-T03: Build the managed local runtime

1. Package a pinned llama.cpp CPU build for the initial proof.
2. Add qualified acceleration variants only after hardware tests.
3. Launch on an ephemeral loopback port with per-launch authentication.
4. Implement health, model load/unload, graceful shutdown, timeout, and crash
   recovery.
5. Capture bounded metadata logs without prompts.
6. Disable built-in tools and UI capabilities not used by Daemon.
7. Deny non-loopback binding.

Acceptance:

- `G10-A05`: DesktopHost detects, starts, authenticates, and stops the sidecar.
- `G10-A06`: Sidecar crash produces a recoverable local error and no hosted call.

### G10-T04: Create signed model manifests

Each model artifact records:

1. Source repository and exact revision.
2. Model and tokenizer license.
3. Architecture and parameter class.
4. GGUF file and quantization.
5. Cryptographic hash and byte size.
6. Chat template and special-token policy.
7. Context limit and tested runtime flags.
8. Qualified roles and hardware tiers.
9. Evaluation report ID.
10. Known failure modes and prohibited roles.

Acceptance:

- `G10-A07`: Runtime refuses an unmanifested or hash-mismatched model.
- `G10-A08`: License review is complete before a pack enters beta/stable.

### G10-T05: Build hardware probing and qualification

1. Detect CPU architecture/features, RAM, disk, GPU vendor, VRAM, and supported
   runtime backends.
2. Benchmark a small representative workload before recommending a pack.
3. Define conservative context and concurrency based on usable memory.
4. Record first-token latency, throughput, peak RAM/VRAM, and model-load time.
5. Detect thermal throttling and sustained degradation where practical.
6. Offer CPU fallback only when its role latency remains usable.
7. Never assume model size alone predicts quality or fit.

Acceptance:

- `G10-A09`: Recommended configuration completes the full role smoke suite
  without out-of-memory failure.
- `G10-A10`: Unsupported hardware receives a clear capability result before a
  large download.

### G10-T06: Establish local utility-model pack

Migrate lower-risk, structured, high-volume roles first:

1. Confirm embeddings and reranking are pinned, local, and fully offline.
2. Migrate tag/topic generation and query checking.
3. Migrate structured intent fallback and query rewrite.
4. Migrate fact extraction and thread extraction.
5. Migrate summaries and memory consolidation.
6. Migrate non-critical scoring/judging roles.
7. Replace suitable model calls with deterministic logic when it performs as well.

Each promotion follows shadow comparison, frozen evaluation, latency/resource
qualification, owner canary, and rollback availability.

Acceptance:

- `G10-A11`: Utility pack meets role-specific G05 gates without hosted access.
- `G10-A12`: Structured-output parse/retry failure remains below its declared
  release threshold.

### G10-T07: Make structured output reliable

1. Define JSON Schema or grammar for every structured role.
2. Ask the runtime for constrained output where supported.
3. Validate strictly and reject extra privileged fields.
4. Permit bounded repair only for harmless syntax errors.
5. Never repair action target, permission, or security-sensitive semantics.
6. Track schema failures by model/template/runtime combination.

Acceptance:

- `G10-A13`: Fuzzed output cannot smuggle undeclared tool arguments.
- `G10-A14`: Structured role failure produces no partial persistence or action.

### G10-T08: Migrate agent decisions and tools

1. Qualify tool selection independently from final-response quality.
2. Test every tool schema, no-tool response, ambiguous request, unavailable tool,
   and malicious retrieved instruction.
3. Keep deterministic gate and capability enforcement outside the model.
4. Validate arguments before proposals.
5. Require user approval and execution receipts as in G06.
6. Stress multi-round loop termination, cancellation, and context inventory.

Acceptance:

- `G10-A15`: Local agent model meets correct-tool and unnecessary-tool gates.
- `G10-A16`: No model output bypasses action policy or argument validation.

### G10-T09: Migrate final conversation generation

1. Freeze a representative owner, synthetic, and approved external response suite.
2. Evaluate candidate local models blind against the current reference.
3. Measure instruction following, grounded use of memory, corrections,
   uncertainty, tone, citation fidelity, and refusal behavior.
4. Tune prompt/template and retrieval budget before considering fine-tuning.
5. Run local output in shadow mode on owner-authorized turns.
6. Promote to owner-canary with one-click rollback.
7. Make local the default only after longitudinal use meets gates.

Acceptance:

- `G10-A17`: Main local model passes high-risk invariants and predeclared quality
  floors on supported hardware.
- `G10-A18`: Owner canary completes the declared longitudinal window without a
  regression requiring hosted default restoration.

### G10-T10: Redesign prompts for local budgets

1. Measure actual tokens for every rendered section.
2. Remove duplicate instructions and context.
3. Make retrieval more selective rather than relying on enormous context.
4. Use compact structured state for identity, permissions, and session facts.
5. Cache stable prefixes locally where the runtime supports it.
6. Set role- and hardware-specific context/output budgets.
7. Test truncation for safety instructions, current query, provenance, and action
   state.

Acceptance:

- `G10-A19`: No required safety/action/current-query section can be truncated by
  lower-priority memory.
- `G10-A20`: Prompt size stays within the qualified context and memory envelope.

### G10-T11: Add resource scheduling

1. Centralize local generation concurrency.
2. Prioritize visible response and safety-critical work over background synthesis.
3. Bound queues and cancel obsolete work.
4. Coordinate model loading/unloading to avoid VRAM thrash.
5. Pause background tasks on battery, thermal pressure, memory pressure, or active
   conversation where configured.
6. Expose understandable busy/degraded status.

Acceptance:

- `G10-A21`: Worst-case concurrent workflow remains within memory bounds.
- `G10-A22`: Cancellation releases runtime work and does not persist partial data.

### G10-T12: Migrate high-cost and optional roles

1. Document generation.
2. Insight and synthesis generation.
3. Coherence and evaluation judges.
4. Visual captioning and multimodal response.
5. Development proposal generation, if kept in the product.

Optional roles may require a higher hardware tier or remain disabled. A feature
may not retain hosted inference merely to preserve a checkbox.

Acceptance:

- `G10-A23`: Capability UI accurately reflects unavailable roles by hardware/model
  pack.
- `G10-A24`: Disabled optional roles do not cause background hosted calls.

### G10-T13: Decommission hosted runtime inference

1. Set `LOCAL_CONNECTED` as packaged default.
2. Remove hosted provider onboarding from user mode.
3. Remove hosted credentials and aliases from stable runtime configuration.
4. Delete automatic provider fallbacks.
5. Move Fable/reference adapters to developer-only tooling with sanitized inputs.
6. Run static and runtime scans for provider endpoints.
7. Run the full product behind OS-level deny-all networking.

Acceptance:

- `G10-A25`: Stable executable contains no reachable hosted inference path.
- `G10-A26`: Full offline scenario suite passes with empty provider credentials
  and blocked DNS/network.

## Promotion protocol for one role

1. Register role and current baseline.
2. Select candidate artifacts compatible with license and hardware.
3. Run deterministic/schema and adversarial tests.
4. Run frozen quality suite.
5. Measure hardware/resource envelope.
6. Shadow against current behavior using authorized data.
7. Review failure taxonomy, not only mean score.
8. Promote to owner-canary behind a role flag.
9. Observe longitudinally.
10. Promote to default and retain rollback.
11. Remove hosted role only after release evidence is archived.

## Suggested migration order

| Window | Roles |
|---|---|
| Months 0-3 | Gateway, inventory, embeddings/reranker, deterministic replacements |
| Months 3-6 | Classification, rewrite, extraction, summaries, consolidation |
| Months 6-9 | Fact verification, planning, safety support, tool selection |
| Months 9-12 | Main conversational response |
| Months 12-18 | Documents, synthesis, judges, optional vision |
| Months 18-24 | Hosted removal, network-off qualification, stable model packs |

## Exit gate

G10 is validated when every runtime call is role-registered, every stable role has
a pinned and licensed local artifact qualified on declared hardware, structured
and tool behavior meet safety gates, resource scheduling is bounded, hosted
fallback is absent, and the full product passes with network inference blocked.

