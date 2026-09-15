# Owner hardcoding review of this session's diff (2026-09-13): dispositions and schedule

Source: the owner's read-only hardcoding review against the generalization docs, sent mid-A05b-3. The owner asked for no interruption of in-flight work. T03 had already been integrated; A05b-3 was running and was not interrupted. The parent recorded this file after A05b-3 handed off, so the batch's manifest check was not disturbed.

## Batch IDs

New H-series IDs: H01–H03 ("hardcoding review").
- H-series IDs are unused in `docs/PLAN_20260913_generalization_execution.md`, whose batch IDs are A/B/C/F/Q.
- The plan's `Gnn-Tnn` tags are requirement tasks, not batches.
- The parent's tone batches T01–T03 (`batches/T01.md`…) are unrelated to those `Gnn-Tnn` tags.

## H01 — EXTERNALIZE the integrated-fallback location thresholds (owner item 1)

- **Finding.** A05a's builder adds bare constants in `core/grounding_check.py`: `_CLAIM_OVERLAP_THRESHOLD = 0.8` (≈939) and `_MIN_CLAIM_TOKENS_FOR_OVERLAP = 3` (≈947). The sibling grounding knobs come from `config/config.yaml` `grounding_check:` (≈208-223) through `config/app_config.py` (`GROUNDING_CHECK_CFG.get(...)`, ≈1465-1508) and a call-time import in `core/grounding_check.py`.
- **Precedent.** `docs/GENERALIZATION_AUDIT_20260901.md` P2 #12 (integrator ratio bounds, "CONFIG-EXPOSE") was fixed the same way on 2026-09-01.
- **Remedy.** Two config keys in the `grounding_check:` section plus typed `app_config` constants. The schema field goes in `config/schema.py` `GroundingCheckSection` (≈1130) if that section declares its fields. `build_integrated_fallback` reads them at call time with today's values as defaults, so behaviour is unchanged.
- **Ownership.** `config/config.yaml`, `config/app_config.py` and `config/schema.py` are shared seams: one owner at a time, and no other queued batch holds them concurrently. `config/bug_class_*.json` are class-guard files and are not touched.
- **Queue position.** Immediately after A05d. A05d is the other owner of `core/grounding_check.py`, so H01 completes the A05 chain's file sequence.

## H02 — categorized-generic stressor coverage across life domains (owner item 2)

- **Finding.**
  - T02's mild tier and Stage 1 read `CONCERN_KEYWORDS` / `MEDIUM_CRISIS_KEYWORDS`.
  - Audit item 9 (student/knowledge-worker stressor vocabulary) remains open for family and caregiving. The 2026-09-01 extension added only financial hardship and grief.
  - Parent count on the candidate:
    - CONCERN (130 entries): family 0, caregiving 0, housing 1, health 5, money 9, work 7, school 2, grief/loss 6.
    - MEDIUM (106 entries): family 1, everything else 0 or 2.
  - `CRISIS_EXEMPLARS["concern"]` (26 seeds) holds only generic worry/sadness statements with no domain stress shapes.
- **Baseline probe** (parent, 2026-09-13).
  - Script and output: `docs/execution/generalization/probes/h02_baseline_probe.py` and `…_output.txt`.
  - Ran the real `detect_crisis_level(model_manager=None)`, with the locally cached embedder offline, on the post-T03 candidate (`utils/tone_detector.py` `3d65d1a4…`).
  - Run with `docs/execution/generalization/probes/run_sandboxed_probe.py`: the adaptive store is sandboxed to an empty temp file and `TONE_EXEMPLAR_LEARNING=0`, i.e. fresh-user state. `data/` sizes and mtimes were unchanged.
  - An earlier unsandboxed run read, and wrote to, the clone's gitignored `data/adaptive_exemplars.json`. That is the parent process error recorded under "Parent process incident" below. It showed 9/21 only because two caregiving rows matched exemplars learned in the clone.
  - Correction, parent recount against the round-2 per-row table: earlier messages and drafts said "10/21" (unsandboxed) and "8/21" (sandboxed). Those were miscounts of the family rows; the correct figures are 9/21 and **7/21**.
  - **Domain stress messages: 7/21 reach ≥ CONCERN.**
    - family 3/3 (two only through unrelated MEDIUM phrases such as "haven't slept in days" and "falling apart");
    - caregiving **0/3** with a fresh store;
    - health 1/3; work 2/3; school 1/3;
    - **money 0/3** ("I don't know how we'll make rent this month", "cash flow is tight and payroll is due friday");
    - **housing 0/3** ("our landlord is trying to evict us").
  - **Neutral controls: 14/14 CONVERSATIONAL** ("my kids have soccer on saturday", "I paid my rent today", "the landlord fixed the sink"…). No over-escalation today; H02 must keep it that way.
- **Doctrine** (`GENERALIZATION_AUDIT_20260901.md` remedy patterns #2 and #5; `DEVELOPMENT_WORKFLOW.md` vocabulary-miss rule; BC-76):
  - categorized-generic coverage spanning life domains, where categories are extended rather than the mechanism cloned;
  - seeds+learned through the existing tone semantic channel;
  - no phrase appends.
- **Design first (parent), then implementation batch(es).**
  - A baseline probe with the real detector: stress versus neutral controls per domain.
  - A design doc covering:
    - a domain category × generic strain structure, not bare domain nouns (BC-28: "my kids", "rent" alone must not score);
    - seeds across domains;
    - the producer-side flattening that T02's mild tier and Stage 1 inherit (BC-58);
    - BC-28 over-escalation controls;
    - the carry-over impact.
  - A probe of the candidate against both sets before any worker starts.
- **Design:** `docs/execution/generalization/H02_design.md`. It covers three probe rounds and the H02a/H02b split.
- **Owner decisions D-H02-1..3, decided 2026-09-13:**
  - no new seeds;
  - "us" only as the object of composed threat_loss shapes, with controls;
  - impersonal business stress left to the semantic and learned channels.
- **Round-3 result:** 22/23 stress, 0/41 controls changed.
- **Queue position.** After A03b-2 and R06 (R06 answers class-guard request CGR-005). H02 needs its design completed first; the design runs as parent work in parallel. It touches `utils/tone_detector.py` only, and T01–T03 are already integrated.

## H03 — synthetic worked examples in resolver comments (owner item 3, low priority)

- **Finding** (names withheld here). Real institution names appear as worked examples in comments and docstrings only, with no code path:
  - `utils/institution_resolver.py`: 1, 54, 123, 402, 446;
  - `utils/location_resolver.py`: 309, 341, 516, 562.
  - Note (2026-09-14): several of these lines are the first lines of docstrings; the names sit further inside, and some wrap across lines. The parent's H03 brief carries the current per-line anchors as `file:line` only, with no names.
- **BC-58 siblings, same kind (comments/docstrings):**
  - `utils/web_search_trigger.py`: 1378, 1489, 1656, plus a two-line-wrapped example near 1541;
  - `memory/entity_resolver.py`: 297.
- **Siblings that need an owner decision, reported rather than fixed:**
  - `scripts/build_wiki_subset.py:107` has one name as a code string. The script builds a wiki subset, so the list is likely public institutions; H03 confirms.
  - The most frequent name appears as a test fixture about 140 times across 13 unit-test files. The largest are `test_sep12_search_identity_scope.py` (78), `test_institution_resolver.py` (31) and `test_private_sphere_search_guard.py` (11).
- **Remedy.** Replace the comment and docstring examples in the two named files, plus the same-kind siblings, with synthetic institutions, e.g. "Example State University" or the already-synthetic "Springfield Community College". No code or behaviour change. The fixture question stays with the owner.
- **Queue position.** End of queue (low priority), after the F-series. It can move up if the owner wants it earlier.

## Parent process incident (recorded for the owner)

- **What happened.** The parent's ad-hoc tone probes called `detect_crisis_level` / `ContextPipeline._detect_tone` from the clone's working directory, outside pytest.
  - Probes involved: R05 review, T01/T02 acceptance, T02/T03 follow-up, and the first H02 baseline.
  - `utils/adaptive_exemplars._STORE_PATH` is the relative path `data/adaptive_exemplars.json`, and tone exemplar learning defaults to on.
  - So Stage 1 keyword hits in those runs could record synthetic probe messages into the clone's store. The last write is timestamped 22:30:24 on 2026-09-13, the first H02 baseline run.
- **Scope.**
  - The clone's `data/` is a real directory with its own inode, not a link to `/home/lukeh/Daemon_v1/data`, and the store file is gitignored. The owner's live store was not touched.
  - pytest runs were not affected: the conftest sandbox covers both the workers' runs and the parent's reruns.
  - The parent's earlier "`ls -A data` unchanged" checks compared names only, so they missed the content change.
- **Effect on evidence.**
  - Deterministic results (Stage 0/1, T01–T03 triggers) do not depend on the store.
  - All tone probes were re-run sandboxed: empty temp store, learning off, `data/` sizes and mtimes unchanged.
  - The T03 follow-up probe was identical.
  - The H02 baseline changed from 9/21 (unsandboxed) to **7/21** (sandboxed): two caregiving rows had matched clone-learned exemplars. Correction (2026-09-14): this line first said "10/21 to 8/21". Both figures were parent miscounts; see the H02 section's correction.
  - Probe evidence from here on uses `docs/execution/generalization/probes/run_sandboxed_probe.py`, and each run records `data/` sizes and mtimes.
- **Not done by the parent (owner decision).** The parent did not read, purge or reset the clone's `data/adaptive_exemplars.json`, because the plan forbids traversing or changing `data/`. If the owner wants the clone store clean, the options are `scripts/purge_adaptive_exemplars.py` against the clone, or deleting that gitignored file.
- **Owner action, 2026-09-13 23:00 local.** The owner deleted the clone's gitignored `data/adaptive_exemplars.json`, which held 11 probe entries. The backup is at `~/daemon_checkpoints/backups/generalization_clone_adaptive_exemplars_polluted_20260913.json`. No other `data/` change.
  - Any `data/` before/after snapshot that spans 23:00 and shows that file gone reflects this owner action, not a batch.
  - From then on, the clone's live code path starts from an empty learned tone store, which is the same fresh-user state the sandboxed probes use.

## Resulting queue

1. A05b-3 (integrated; parent fix D1 applied)
2. A05b-4 (agentic-route `cancelled_before_review` receipt; A05b-3 parent review D2)
3. A05c
4. A05d
5. **H01**
6. A03b-2
7. R06
8. **H02** (design ready by then)
9. S01
10. S02
11. S03
12. F-series (CGR-007–010)
13. **H03**

### Queue as of 2026-09-14 (after A05d)

The list above is kept as recorded. A05b-3, A05b-4, A05b-5 (with parent fix D3), A05c and A05d are integrated. Remaining, in order:

1. **H01**
2. A03b-2
3. R06 (CGR-005)
4. **H02a**, categorize the keyword tables (no behaviour change)
5. **H02b**, the domain-strain rule
6. S01
7. S02
8. S03
9. F-series F1–F13 (CGR-007–010), per `failure_outcome_design.md` including its 2026-09-14 parent amendments
10. **H03**
