# G08: Windows Executable and Release Engineering

## Objective

Deliver Daemon as a signed, reproducible, single-user Windows 11 x64 application
that installs, launches, upgrades, rolls back, operates offline, repairs, exports,
and uninstalls without repository access or a Python development environment.

## Current gap

The existing installer documentation depends on a PyInstaller spec that is no
longer present, describes the legacy Gradio path, and assumes hosted API keys.
The current React/FastAPI application and future local-model sidecar need a new
packaging baseline.

## Proposed process architecture

```text
Daemon.exe / launcher
  +-- validates installation and user data schema
  +-- acquires single-instance lock
  +-- starts local model runtime when required
  +-- starts FastAPI on ephemeral loopback port with launch token
  +-- serves bundled React production assets
  +-- opens or owns the UI surface
  +-- coordinates shutdown, backup, and crash recovery
```

Model packs remain separate, versioned artifacts. They must not make the core
installer tens of gigabytes or require rebuilding the application for every
model change.

## Step-by-step plan

### G08-T01: Write packaging decision records

1. Compare PyInstaller, Nuitka, and a thin native launcher around a packaged
   Python environment for startup, size, native dependencies, antivirus behavior,
   debugging, and updateability.
2. Compare default-browser UI with a desktop webview shell for accessibility,
   security, lifecycle, and maintenance.
3. Retain Inno Setup only if it meets signing, per-user install, upgrade, rollback,
   and uninstall needs.
4. Select one supported path and record rejected alternatives.
5. Build a minimal proof before porting all dependencies.

Acceptance:

- `G08-A01`: The selected proof runs on a clean Windows 11 VM without Python,
  Node, Git, or administrator privileges where feasible.
- `G08-A02`: Decision record includes local-model sidecar and update implications.

### G08-T02: Separate build-time and runtime assets

1. Build React production assets in CI and serve the immutable output locally.
2. Inventory Python packages, native DLLs, spaCy/model data, prompt files, config
   schema, migrations, and licenses.
3. Remove development tools, tests, private data, and provider credentials from
   the package.
4. Give each packaged asset an explicit manifest entry and checksum.
5. Fail the build on missing assets or unexpected private files.

Acceptance:

- `G08-A03`: Package manifest matches installed files exactly.
- `G08-A04`: Private canary and secret scans pass the final installer contents.

### G08-T03: Implement `DesktopHost`

1. Acquire a per-user single-instance lock.
2. Resolve `AppPaths` and run preflight without modifying data on failure.
3. Select an available ephemeral loopback port.
4. Generate and pass per-launch UI/API authentication.
5. Start backend and model sidecar as child processes with job-object/process-tree
   cleanup.
6. Display bounded startup progress and actionable failures.
7. Coordinate graceful shutdown, cancellation, backup, and last-state persistence.
8. Recover orphaned sidecars and stale locks safely.

Acceptance:

- `G08-A05`: Forced backend or sidecar termination does not leave an exposed or
  permanently wedged installation.
- `G08-A06`: Launching twice focuses the existing instance rather than corrupting
  stores.

### G08-T04: Package the local runtime separately

1. Choose verified CPU and acceleration builds for the supported hardware tiers.
2. Bind the runtime to loopback with a per-launch credential.
3. Store runtime version and hash in its manifest.
4. Install/upgrade it atomically and preserve the previous qualified version.
5. Capture health, load progress, memory use, and bounded logs.
6. Never permit arbitrary runtime flags from conversation content.

Acceptance:

- `G08-A07`: Runtime cannot bind externally or start an unverified model.
- `G08-A08`: Runtime upgrade rollback restores a passing role suite.

### G08-T05: Build model-pack management

1. Publish model packs independently from the application.
2. Show license, source, size, hardware fit, role qualification, and expected
   performance before download.
3. Support resumable download, checksum verification, staging, and atomic install.
4. Support offline sideload from a verified manifest.
5. Track disk use and protect active/rollback packs from cleanup.
6. Permit removal only when no required role depends on the pack.

Acceptance:

- `G08-A09`: Interrupted or tampered downloads never replace the active pack.
- `G08-A10`: A fully offline installation can sideload and qualify a model pack.

### G08-T06: Implement schema migration and rollback

1. Give every persistent store a schema version and compatibility policy.
2. Preflight all stores before changing any.
3. Create an encrypted verified backup before migration.
4. Migrate into staging where format permits.
5. Validate counts, referential integrity, and representative reads.
6. Activate atomically and preserve a rollback marker.
7. Never allow an older executable to overwrite newer unsupported data.

Acceptance:

- `G08-A11`: Power-loss/fault injection at each migration stage preserves a
  recoverable state.
- `G08-A12`: Rollback instructions are executable and release-tested.

### G08-T07: Build signed release and update flow

1. Produce releases from a clean, pinned build environment.
2. Generate SBOM, license report, checksums, provenance, and build manifest.
3. Sign executable, installer, update manifest, and runtime/model metadata.
4. Use staged owner-canary, alpha, beta, and stable channels.
5. Make update checking transparent and disabled in offline mode.
6. Download to staging, verify, offer release notes, and retain rollback.
7. Do not auto-update model packs across qualification boundaries.

Acceptance:

- `G08-A13`: A forged update manifest or installer is rejected.
- `G08-A14`: Update failure returns to the prior executable and data schema.

### G08-T08: Create clean-machine test automation

Test each release on snapshot-based Windows VMs:

1. Fresh per-user install.
2. First-run onboarding with no hosted credentials.
3. Model download and offline sideload.
4. Local chat, memory, correction, restart, and shutdown.
5. Connected tool enable/deny behavior.
6. Offline run under network denial.
7. Upgrade from every supported prior version.
8. Failed migration and rollback.
9. Backup, restore, export, reset, and delete-all.
10. Uninstall preserving data, reinstall, then uninstall deleting data.
11. Standard user, high DPI, high contrast, and screen-reader smoke tests.

Acceptance:

- `G08-A15`: Stable release is blocked unless the clean-machine matrix passes.
- `G08-A16`: No test depends on the developer checkout or environment variables.

### G08-T09: Define performance and resource budgets

1. Measure installer size separately from model-pack size.
2. Set startup, model-load, first-token, and shutdown targets per hardware tier.
3. Bound disk growth for logs, caches, indexes, and backups.
4. Bound CPU/GPU use during idle and background processing.
5. Support cancellation and prevent parallel model jobs from exhausting memory.
6. Provide a low-resource mode with honest capability changes.

Acceptance:

- `G08-A17`: Sustained workload remains within declared peak memory and disk
  budgets.
- `G08-A18`: Low-resource mode never silently routes to hosted inference.

### G08-T10: Build support and recovery tooling

1. Add local health checks for stores, runtime, model pack, disk, ports, and
   permissions.
2. Provide safe repair actions with preview and backup.
3. Generate a user-previewed, redacted, encrypted diagnostic bundle.
4. Keep raw personal text excluded by default.
5. Include application/config/model hashes and recent error IDs.
6. Document offline recovery and factory reset.

Acceptance:

- `G08-A19`: Support bundle passes G03 private-canary scans.
- `G08-A20`: Repair operations are idempotent and cannot delete data without a
  separate confirmation.

### G08-T11: Make uninstall semantics exact

1. Remove binaries and inactive runtime assets by default.
2. Ask separately about model packs, personal data, and external backups.
3. Explain irreversible deletion and backup exceptions.
4. Produce a local completion report.
5. Verify deletion against the G03 inventory.

Acceptance:

- `G08-A21`: Preserve-data uninstall supports clean reinstall.
- `G08-A22`: Delete-all uninstall leaves no registered personal artifact in
  application-owned locations.

## Release artifacts

- Signed installer and executable.
- Application manifest and checksums.
- SBOM and third-party licenses.
- Runtime manifest.
- Separate signed model-pack manifests.
- Migration and rollback matrix.
- Clean-VM test report.
- Privacy, capability, hardware, and known-limitations documents.

## Exit gate

G08 is validated when a clean standard-user Windows 11 VM can install, onboard,
run locally, work offline, update, roll back, restore, and uninstall without
developer tools; all artifacts are verified; private data is separate; and the
clean-machine matrix blocks stable release on failure.

