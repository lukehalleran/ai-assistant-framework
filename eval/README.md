# Eval System — Phase 1: Instrumentation & Snapshot Replay

## What Phase 1 Does

Phase 1 provides the measurement foundation for prompt section ablation testing:

1. **Section Registry** — canonical definition of all 26 prompt sections with ablation
   eligibility, structural requirements, and assembly order.

2. **Snapshot Capture** — captures the full assembled prompt (post-hygiene) with both
   structured content and formatted text per section, plus provenance metadata.

3. **Snapshot Replay** — reconstructs the prompt from saved formatted text, ordered by
   assembly_order. Verifies exact and normalized hash match. Does NOT call retrieval.

4. **No-Store Generation** — calls `ModelManager.generate_once()` directly. Bypasses
   the orchestrator entirely. Cannot enable any persistence side effect.

5. **Persistence Guard** — fingerprints ChromaDB collections and JSON data files
   before/after eval generation. Raises if anything changed.

## What Phase 1 Does NOT Do

- Generate ablation variants (Phase 2)
- Build a query corpus (Phase 3)
- Run batch eval generation (Phase 4)
- Judge response quality (Phase 5)
- Check for hallucination/accuracy (Phase 6)
- Aggregate results into scorecards (Phase 7)
- Modify production prompt behavior (Phase 8)

## How to Enable Snapshot Capture

```bash
# Enable capture (writes snapshots to eval/snapshots/)
DAEMON_EVAL_CAPTURE=1 python main.py

# Strict mode: raise on capture errors instead of warning
DAEMON_EVAL_CAPTURE=1 DAEMON_EVAL_CAPTURE_STRICT=1 python main.py

# Custom snapshot directory
DAEMON_EVAL_SNAPSHOT_DIR=/tmp/my_snapshots DAEMON_EVAL_CAPTURE=1 python main.py
```

When enabled, every prompt assembly writes a snapshot JSON to `eval/snapshots/`.
When disabled (default), zero overhead — the hook returns immediately.

## Where Snapshots Are Saved

```
eval/snapshots/
  2026-05-04_a1b2c3d4.json
  2026-05-04_e5f6g7h8.json
  ...
```

Filename format: `YYYY-MM-DD_<snapshot_id>.json`

## How to Replay a Snapshot

```python
from pathlib import Path
from eval.snapshots import load_snapshot, SnapshotReplay

snapshot = load_snapshot(Path("eval/snapshots/2026-05-04_a1b2c3d4.json"))
replay = SnapshotReplay()

# Full replay (all sections, ordered by assembly_order)
prompt_text = replay.replay_from_layer(snapshot, "post_hygiene")

# Verify hash match
assert replay.verify_replay_normalized(snapshot, "post_hygiene")

# Replay with subset of sections (for ablation)
ablated = replay.replay_from_layer(
    snapshot, "post_hygiene",
    sections_to_include=["recent_conversation", "memories", "current_query"],
)
```

## How to Run No-Store Generation

```python
from eval.no_store_generation import EvalGenerator, EvalGenerationConfig
from eval.persistence_guard import PersistenceGuard

# Safety: all persistence flags must be False (enforced by EvalGenerationConfig)
config = EvalGenerationConfig()

# Use your model_manager instance
gen = EvalGenerator(config=config, model_manager=model_manager)

# Guard against accidental writes
guard = PersistenceGuard(chromadb_client=chroma_client)
before = guard.capture()

result = await gen.generate(
    assembled_prompt=prompt_text,
    model="sonnet-4.5",
    temperature=0.3,
    system_message="You are Daemon, a helpful assistant.",
)

after = guard.capture()
before.assert_same_as(after)  # Raises if anything was written

print(result.response_text)
```

**WARNING:** Eval generation must NEVER be run through `process_user_query()`,
because that path stores interactions and updates memory.

## How to Run Tests

```bash
# All eval tests
pytest tests/test_eval/ -v

# Individual test files
pytest tests/test_eval/test_section_registry.py -v
pytest tests/test_eval/test_snapshots.py -v
pytest tests/test_eval/test_replay_hash.py -v
pytest tests/test_eval/test_no_store_generation.py -v
pytest tests/test_eval/test_persistence_guard.py -v
```

## How This Supports Later Phases

| Phase | Uses From Phase 1 |
|-------|-------------------|
| Phase 2 (Variants) | Section registry (ablatable sections), SnapshotReplay (subset replay) |
| Phase 3 (Corpus) | Snapshot capture (real queries), section_registry (coverage tracking) |
| Phase 4 (Harness) | EvalGenerator, PersistenceGuard, SnapshotReplay |
| Phase 5 (Judge) | EvalGenerationResult (response pairs for comparison) |
| Phase 6 (Checks) | PromptSnapshot (ground truth context for grounding checks) |
| Phase 7 (Reporting) | SectionSnapshot.token_count, section_registry categories |
| Phase 8 (Policy) | Section registry (inclusion rules), SnapshotReplay (shadow prompts) |

## File Structure

```
eval/
  __init__.py              # Package docstring
  PLAN.md                  # Full 8-phase plan (orientation doc)
  README.md                # This file (Phase 1 usage)
  schema.py                # Pure data models (no Daemon imports)
  section_registry.py      # Canonical section definitions (27 entries)
  snapshots.py             # Capture, save/load, replay, hash verification
  no_store_generation.py   # Side-effect-free LLM generation
  persistence_guard.py     # State fingerprinting and diffing
  snapshots/               # Saved snapshot JSON files (gitignored)

tests/test_eval/
  __init__.py
  test_section_registry.py # 29 tests
  test_snapshots.py        # 13 tests
  test_replay_hash.py      # 14 tests
  test_no_store_generation.py # 17 tests
  test_persistence_guard.py # 16 tests
```
