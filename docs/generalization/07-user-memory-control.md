# G07: User Memory Control

## Objective

Give the user understandable and complete control over what Daemon remembers,
how it interprets that information, how long it persists, where it is used, and
how every derivative is removed.

The goal is not merely a delete button. It is a traceable memory lifecycle.

## Memory lifecycle

```text
observed
  -> classified by source and sensitivity
  -> allowed or denied by memory policy
  -> stored with provenance
  -> transformed into facts/summaries/graph/embeddings
  -> retrieved and shown with explanation
  -> confirmed, corrected, superseded, expired, hidden, or deleted
  -> cascaded through derivatives and backups according to policy
```

## User-selectable memory modes

| Mode | Behavior |
|---|---|
| `off` | No durable conversation or derived memory after the session |
| `session_only` | Temporary context is removed at session end |
| `ask_to_store` | Durable memory requires an explicit proposal |
| `automatic_bounded` | Approved categories store under retention rules |
| `custom` | Per-category and per-source policy |

Consequential actions and privacy mode remain separate controls.

## Step-by-step plan

### G07-T01: Define the memory inventory and source taxonomy

1. Reuse the G03 data inventory and enumerate semantic memory objects rather than
   files alone.
2. Define source classes: user-stated, user-confirmed, imported personal note,
   external source, assistant inference, graph-derived, and generated summary.
3. Define sensitivity classes and default policy for health, sexuality, religion,
   finance, legal matters, precise location, credentials, and third-party data.
4. Record which transformations and collections each memory may enter.
5. Reject unknown source or sensitivity values in stable mode.

Acceptance:

- `G07-A01`: Every stored object has source, provenance, sensitivity policy,
  created time, and owning profile.
- `G07-A02`: A transformation cannot increase trust beyond its strongest valid
  source without explicit confirmation.

### G07-T02: Implement policy before storage

1. Evaluate memory mode and category policy before writing conversation-derived
   or imported content.
2. Keep response delivery independent from permission to remember it.
3. Provide "do not remember this" and "remember this" turn-level controls.
4. Allow a session to pause memory without changing the default.
5. Ensure shutdown processing follows the same policy and cannot recreate denied
   memories.
6. Test every background writer against the policy boundary.

Acceptance:

- `G07-A03`: A denied turn leaves no corpus, vector, graph, summary, adaptive, log,
  or backup-derived personal record beyond minimal ephemeral process state.
- `G07-A04`: Session-only mode survives crash/restart cleanup according to its
  documented recovery policy.

### G07-T03: Build a unified memory ledger

1. Present memories by human concept, not storage collection alone.
2. Show content, source, creation, last confirmation, current/stale status,
   confidence, sensitivity, related derivatives, and recent use.
3. Link summaries and graph edges back to source events.
4. Distinguish exact user words from extracted or inferred statements.
5. Permit search, filter, sort, and source inspection.
6. Avoid exposing internal embeddings or model reasoning as if meaningful to the
   user.

Acceptance:

- `G07-A05`: A user can locate the memory responsible for a disputed response.
- `G07-A06`: Every displayed derivative resolves to its available source lineage.

### G07-T04: Implement correction as a first-class event

1. Preserve the original claim and record the correction event.
2. Mark supersession and currentness consistently across profile, claims, graph,
   vector metadata, summaries, and prompt rendering.
3. Propagate staleness to unsupported derivatives.
4. Rebuild or flag narrative summaries that materially rely on the old claim.
5. Show the user what changed and what remains uncertain.
6. Add repair tools for historical inconsistencies with dry-run and backup.

Acceptance:

- `G07-A07`: A correction no longer surfaces the obsolete claim as current in any
  primary retrieval or profile path.
- `G07-A08`: Correction propagation is idempotent and recoverable after
  interruption.

### G07-T05: Add pin, hide, forget, and delete semantics

1. Pin controls prominence without changing factual confidence.
2. Hide removes a memory from ordinary use but preserves it for user review.
3. Forget removes active retrieval and derived influence, subject to the visible
   backup policy.
4. Delete removes source and all derivatives where possible.
5. Explain the difference before confirmation.
6. Never turn delete into soft-hide without saying so.

Acceptance:

- `G07-A09`: Each operation has a distinct state transition and regression suite.
- `G07-A10`: Hidden/forgotten/deleted content is absent from prompt snapshots and
  model inputs according to its contract.

### G07-T06: Implement deletion cascade

1. Build a dependency graph from source records to embeddings, facts, graph
   edges, summaries, reflections, claims, adaptive items, generated notes, caches,
   indexes, logs, and backups.
2. Preview affected objects before deletion.
3. Write a local deletion transaction/ledger and process idempotently.
4. Recompute shared derivatives from remaining sources rather than deleting valid
   shared knowledge.
5. Verify absence after completion and report exceptions.
6. Define how encrypted immutable backups expire or are reissued without the data.

Acceptance:

- `G07-A11`: Seeded canary deletion removes every unique derivative in the data
  inventory.
- `G07-A12`: Deleting one source does not destroy a fact independently supported
  by another retained source.

### G07-T07: Add retention controls

1. Define defaults by memory/source/sensitivity class.
2. Let the user choose indefinite, bounded, session-only, or never-store policy
   where safe.
3. Show upcoming automatic expiration.
4. Keep durability of a fact separate from retention of raw conversation text.
5. Run retention locally and record a payload-free receipt.
6. Include archives, compressed logs, caches, and temporary files.

Acceptance:

- `G07-A13`: Time-controlled tests prove expiration across every registered store.
- `G07-A14`: Retention cannot silently delete pinned or legally/user-required data
  contrary to the displayed policy.

### G07-T08: Build export and restore

1. Offer human-readable and machine-restorable export formats.
2. Include schema versions, provenance, settings, and model-independent data.
3. Make large model packs and reconstructible indexes optional.
4. Encrypt restorable archives and verify checksums before restore.
5. Preview merge versus replace behavior.
6. Restore into a staging root, validate, and atomically activate.
7. Support offline transfer to a new installation.

Acceptance:

- `G07-A15`: Export -> fresh install -> restore preserves declared data and
  correction state.
- `G07-A16`: Wrong password, corrupt archive, or unsupported schema cannot damage
  the current root.

### G07-T09: Make backup behavior understandable

1. Show when backups occur, where they live, whether they are encrypted, and what
   they contain.
2. Distinguish application rollback from personal-data backup.
3. Include backup retention in delete/forget explanations.
4. Let the user create and verify a backup before destructive maintenance.
5. Test restore regularly, not only backup creation.

Acceptance:

- `G07-A17`: The UI can prove the newest backup is restorable using a non-live
  staging check.
- `G07-A18`: Delete-all offers explicit handling of retained backups.

### G07-T10: Test memory understanding with users

1. Ask external users to predict what each memory mode will store.
2. Ask them to locate and correct an intentionally wrong fact.
3. Ask them to prevent a sensitive turn from being remembered.
4. Ask them to delete it and explain whether backups retain it.
5. Revise labels and workflow where predictions do not match behavior.

Acceptance:

- `G07-A19`: Users can complete core control tasks without developer guidance.
- `G07-A20`: User predictions match observed storage behavior at the predeclared
  usability threshold.

## Non-negotiable invariants

- Storage permission is evaluated before every writer, including shutdown jobs.
- Assistant inference is never presented as a user statement.
- Correction is stronger than passive recurrence.
- Delete covers derivatives, not only the visible source row.
- Export never silently omits non-reconstructible personal data.
- Backup retention is disclosed during deletion.
- Memory controls operate without hosted inference.

## Exit gate

G07 is validated when users can predict and control storage, inspect provenance,
correct truth, control prominence, delete all derivatives, export and restore on
a fresh install, and verify the effect without trusting a hidden implementation
detail.

