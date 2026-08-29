# G02: Owner-Neutral Runtime

## Objective

Make one codebase behave correctly for an arbitrary fresh user without relying
on the owner's name, paths, institutions, hobbies, incidents, cadence, corpus
distribution, or manually curated identity relations.

Depersonalization removes owner assumptions. It does not by itself establish
population generalization.

## Current anchors

The existing `docs/GENERALIZATION_AUDIT.md` correctly separates:

- Mechanism-general behavior.
- Owner-tuned parameters and seeds.
- Owner-specific literals and paths.

Known production examples include hardcoded note prompts, `Luke Notes` paths,
owner-derived intent/tone examples, and profile promotion rules shaped by one
life. Tests containing the name Luke are not automatically defects; runtime
behavior containing it is.

## Target design

### `UserContext`

One immutable per-turn object should contain:

```text
profile_id
display_name
pronouns
timezone
locale
location_policy
communication_preferences
memory_policy
privacy_mode
enabled_capabilities
data_root
session_id
```

Subsystems may request only the fields they need. They may not read a global
profile file or assume `user_id="default"` independently.

### `AppPaths`

One object should resolve every mutable path:

```text
data_root
stores
indexes
uploads
logs
backups
models
cache
exports
temporary_files
```

No production module should own a literal `data/...`, home-directory path, or
owner vault path after migration.

## Step-by-step plan

### G02-T01: Build an owner-assumption inventory

1. Extend the current audit with machine-readable findings.
2. Scan production code, prompts, config defaults, migrations, installer files,
   and generated assets separately from tests and historical documentation.
3. Classify each hit as runtime defect, test fixture, historical record, example,
   or private data leak.
4. Assign an owner and removal milestone to every runtime defect.
5. Add a CI allowlist with reason and expiration for intentional examples.

Acceptance:

- `G02-A01`: CI fails on new owner names, absolute owner paths, or known private
  identifiers in production scopes.
- `G02-A02`: Every existing production hit has a disposition.

### G02-T02: Introduce `UserContext`

1. Define the typed schema and constructors for fresh install, persisted profile,
   test user, and migration fallback.
2. Construct it once during application startup and once per session where
   session-specific fields are needed.
3. Thread it through orchestration, prompt assembly, memory, notes, retrieval,
   actions, and background processing.
4. Remove direct reads of identity fields from global files inside those modules.
5. Add a temporary compatibility adapter with telemetry for every legacy read.
6. Delete each adapter once call sites reach zero.

Acceptance:

- `G02-A03`: A call-graph/static check finds no production identity read outside
  the profile/context boundary.
- `G02-A04`: Parallel tests with different synthetic users never share identity.

### G02-T03: Introduce `AppPaths`

1. Enumerate all mutable paths and assign them to `AppPaths`.
2. Move the default root to a non-roaming, per-user Windows application directory.
3. Update constructors to require the relevant path rather than inventing one.
4. Preserve command-line/test overrides through explicit factories.
5. Migrate existing owner data with a dry-run report, backup, and rollback.
6. Add a path-origin diagnostic screen.

Acceptance:

- `G02-A05`: Production static checks reject direct relative mutable paths.
- `G02-A06`: Running two test roots concurrently produces no shared files.
- `G02-A07`: Migration is idempotent and leaves the source untouched until commit.

### G02-T04: Neutralize prompts and generators

1. Replace literal names and pronouns with required template fields.
2. Replace owner vault conventions with configurable note destinations and
   naming templates.
3. Make missing name, pronouns, location, and timezone valid states.
4. Remove institution, gym, game, medication, and family examples from global
   instructions unless they are deliberately diverse examples in tests.
5. Validate templates before startup and fail clearly on unresolved placeholders.
6. Render prompts for a matrix of synthetic identities and inspect diffs.

Acceptance:

- `G02-A08`: Every production prompt renders with an empty optional profile.
- `G02-A09`: Randomized names, Unicode names, multiword names, and custom pronoun
  sets render without owner leakage or grammar failure.

### G02-T05: Generalize profile prominence

1. Replace hardcoded quick-profile relation lists with documented prominence
   signals such as confirmation, durability, user pinning, recurrence, and
   currentness.
2. Retain a minimal universal identity core only where necessary.
3. Permit user-defined relations and categories without code changes.
4. Keep safety-sensitive inference from auto-promoting protected or uncertain
   attributes.
5. Show why a fact became prominent and allow demotion.

Acceptance:

- `G02-A10`: Diverse synthetic profiles promote their central facts without a
  relation-specific code change.
- `G02-A11`: Unconfirmed inferred sensitive facts never enter the always-visible
  profile block.

### G02-T06: Separate global policy from personal adaptation

1. Label every threshold, exemplar set, relation seed, and keyword list as global,
   population-derived, or user-learned.
2. Store user-learned state under the user data root.
3. Prevent dogfood incidents from directly editing global defaults without the
   G11 generalization test.
4. Version seed data independently and provide reset-to-neutral behavior.
5. Record the source of every active adaptive item.

Acceptance:

- `G02-A12`: Deleting user-learned state returns behavior to a deterministic
  neutral baseline.
- `G02-A13`: Exporting application code/config contains no learned owner data.

### G02-T07: Create the neutral-profile matrix

Create at least these synthetic profiles, using fictional data:

1. Empty profile and first turn.
2. User who withholds name and pronouns.
3. Terse user with frequent fragments.
4. Verbose user with long context.
5. User with nontraditional family and household relations.
6. Night-shift user with nonstandard sleep schedule.
7. User with permanent disability and assistive-technology preferences.
8. User with multiple schools, jobs, residences, or changed identity facts.
9. User with many people sharing common names.
10. User with no notes, integrations, or internet capabilities.

Each profile must cover onboarding, ten scripted turns, correction, restart,
shutdown processing, export, and deletion.

Acceptance:

- `G02-A14`: The full matrix passes without owner-specific output or manual file
  changes.
- `G02-A15`: Stored facts and graph edges remain isolated to the active fixture.

## Migration rules

- Never rewrite live owner data in place without backup and dry-run output.
- Preserve source excerpts and provenance when replacing subject identity.
- Do not bulk-replace the string `Luke` across tests or history.
- Do not turn owner incidents into generic language rules without counterexamples.
- Do not make optional profile fields required merely because owner data has them.

## Exit gate

G02 is validated when a fresh synthetic user completes the lifecycle through one
`UserContext` and `AppPaths`, production scopes contain no owner-specific runtime
assumptions, owner-learned state is separable and resettable, and the neutral
profile matrix passes.

