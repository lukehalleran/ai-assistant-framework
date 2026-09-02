# GENERALIZATION_AUDIT_20260901.md — Deep Sweep #2: "Me-Shaped" Code

*2026-09-01. Five-lens parallel agent sweep (owner identifiers / domain
vocabularies / LLM prompts / calibrated constants / learning-loop coverage),
every headline finding hand-verified against source. Supplements
`docs/GENERALIZATION_AUDIT.md` (2026-08-21, still-open worklist — its items
are NOT repeated here unless the status changed). Trigger: the private-sphere
search guard shipped today started life as a coursework-only guard and had to
be generalized on review — this sweep hunts everything else with that shape.*

**Target:** generalized software that can learn about and adapt to ANY user
typing American English — school, job, hobbies, family, health, any category.
**Not the goal:** expanding owner-specific tools case-by-case as new needs hit.

## Remedy patterns (the codebase's own, now five)

1. **EXTERNALIZE** — literal PII → config.local.yaml / profile-threaded.
2. **SEEDS+LEARNED** — neutral seeds, per-user exemplars (`adaptive_exemplars`).
3. **AUTO-PROMOTE / DERIVE** — vocabulary grows from observed data
   (`learned_relations`) or is computed from profile/telemetry at runtime.
4. **CALIBRATE-ON-DATA** — thresholds re-derived from the user's own corpus
   (probe scripts), not constants.
5. **CATEGORIZED-GENERIC + ANCHORS** *(new, 2026-09-01)* — one vocabulary
   spanning life domains with user-specific anchors passed as parameters
   (`terms_are_private_sphere_generic(terms, anchors)`); extend categories,
   never clone the mechanism.

---

## P0 — breaks or misbehaves for ANY other user immediately

| # | Item | Where | Problem | Remedy |
|---|------|-------|---------|--------|
| 1 | **Calendar timezone hardcoded America/Chicago** | `core/actions/google_calendar_create.py:165,187` · `google_calendar_modify.py:30` (`_DEFAULT_TZ`) · `core/agentic/types.py:1125` (tool schema teaches it) | every timed event for a non-Central user lands at the wrong wall time — the exact bug class we spent three live rounds fixing for the owner, permanently installed for everyone else | DERIVE: profile `timezone` fact → system tz (`tzlocal`) → only then a constant; update tool-schema text to "user's local timezone" |
| 2 | **`weekly_notes_generator.py` hardcodes Luke + vault path** | `utils/weekly_notes_generator.py:13,72,102` | the 08-21 audit caught daily+monthly; weekly has the identical problem and was missed | EXTERNALIZE: same fix as its siblings — thread `user_name`, config vault path |
| 3 | **"The Georgia Tech one?" few-shot in COMMITTED prompts** | `config/prompts/default_personality.txt:84` · `core/system_prompt.txt:113` | few-shots are steering: a fresh clone's assistant learns to presume a tech-school, project-demo interview and invented shared history ("The X one?") for every user | NEUTRALIZE-EXAMPLES: generic institution + balanced example set |
| 4 | **Vault fallback path in visual backfill script** | `scripts/backfill_visual_memory.py:72` (`~/Documents/Luke Notes`) | script silently targets the owner's vault when config key absent | EXTERNALIZE: require config or skip gracefully |

## P1 — wrong-for-others vocabularies and seeds (the private-sphere-guard class)

| # | Item | Where | Missing-domain story | Remedy |
|---|------|-------|----------------------|--------|
| 5 | **Insight self-model nouns = owner's clinical picture** | `core/insight/detector.py` `_SELF_MODEL_NOUNS` (anxiety, depression, medications, triggers, drinking, trauma, recovery…) | a user with ADHD ("my procrastination"), chronic pain ("my pain patterns"), or sobriety work ("my sobriety") never routes into insight mode — the list is what THIS owner discusses, not universal self-model vocabulary | CATEGORIZED-GENERIC (add executive-function / pain / addiction-recovery / grief / parenting / finance categories) + LEARNED: an explicit "gather everything about my X" teaches X |
| 6 | **Insight audience markers = therapy care team only** | `core/insight/detector.py` `_PERSONAL_MARKER_RE` context ("for my therapist, psychiatrist, psychologist, counselor, doctor") | "write up my pattern for my **sponsor** / **coach** / **mentor** / **pastor** / **lawyer**" misses document routing | CATEGORIZED-GENERIC: trusted-role list spanning clinical / peer-support / professional-advisor |
| 7 | **"therapist" baked into deliberation framing machinery** | `core/insight/deliberation.py:36` (`_FRAMING_TERMS`), `:232` (prompt rule), `:363` (entity filter) | the Zelphex-era fossils were removed but "therapist" survived as a hardcoded framing/filter token in three places — a user whose reported-speech frame is "my boss says" or "my sponsor says" gets no equivalent treatment | CATEGORIZED-GENERIC: role-generic reported-speech frame handling (therapist/doctor/boss/sponsor/coach/partner/parent) |
| 8 | **Gate tool keywords are developer-shaped** | `core/agentic/gate.py` TOOL/COMPUTATION lists (github, git stats, execute python, numpy, pandas, sandbox) | a data analyst ("run this SQL"), academic ("run the ANOVA"), or artist never hits Tier-1 tool routing for their discipline's vocabulary; only devs get fast-path tools | SEEDS+LEARNED or DERIVE from profile occupation; Tier-4 LLM already backstops, so this is precision-for-others, not breakage |
| 9 | **Tone CONCERN stressor list = student/knowledge-worker** | `utils/tone_detector.py` CONCERN keywords (work stress, school stress, deadline, performance) | caregiver ("kids haven't slept in days"), small-business ("cash flow is tight"), creative ("blocked for weeks") stress shapes score CONVERSATIONAL | CATEGORIZED-GENERIC seed extension; semantic tier + learned exemplars already absorb per-user idiom, so extend seeds only where a domain has ZERO coverage |
| 10 | **HEAVY_KEYWORDS conflate personal vs professional mental-health vocabulary** | `utils/query_checker.py` heavy-topic list (antidepressant, psychiatrist, bipolar, PTSD…) | a psychiatric nurse, therapist, or researcher discussing their WORK trips heavy-topic routing (intent gates, tone context) on professional vocabulary | needs care — this list feeds SAFETY paths and must keep under-triggering as the failure mode; if addressed, gate on first-person framing ("my antidepressant" vs "antidepressant-resistant patients"), never by shrinking the list |
| 11 | **ANCHORS GAP: institution = school only** | `utils/institution_resolver.py` (school relations only) · `web_search_trigger.py` guard call passes `[user_institution]` · trigger prompt injects school only | a working adult's primary institution is their EMPLOYER; their queries ("HR policy", "my standup") have no anchor, and employer/group names never reach the private-sphere guard, `apply_institution`, or the trigger prompt. `learned_relations` already auto-promotes `works_at`/`member_of`-class data and NOTHING consumes it | **flagship build**: generalize to a "user's named institutions" resolver (school + employer + orgs/groups from profile facts, same mtime-cache + shaped-value validation), feeding all three consumers |

## P2 — latent / constants / coverage bias

| # | Item | Where | Note |
|---|------|-------|------|
| 12 | Grounding integrator ratio bounds not config-exposed | `core/grounding_check.py` (`0.75/1.30` hardcoded) | other grounding knobs have env/YAML overrides; these don't — CONFIG-EXPOSE |
| 13 | `USER_UPLOADS_MIN_RELEVANCE=0.62` | `gatherer_knowledge.py` | calibrated twice on the owner's specific stale uploads (homework docx, two cat photos); another user's upload mix shifts the right bar — CALIBRATE-ON-DATA candidate once a probe exists |
| 14 | `INSIGHT_SYNTHESIS_MAX_TOKENS=4200` and sweep caps | `app_config.py` | sized for an owner-scale corpus; fresh installs waste tokens / sparse-evidence syntheses — acceptable defaults, document as such |
| 15 | EVENT-DISTRESS / political keyword coverage | `utils/tone_detector.py` | entries are AmE-general but accreted from the owner's 2026 news anxieties; coverage follows his feeds. Remedy is the P3 doctrine from the 08-21 audit (adaptive loops + visible failure), not longer lists |
| 16 | Stance evaluative lexicon has no learning channel | `memory/stance_classifier.py` | hand-curated thick-evaluative terms; feasible outcome teacher exists (see loop table) but poisoning risk is high — design carefully or leave |
| 17 | `visual_profile_context.txt` has no generation path | `data/` (gitignored), consumed by `scripts/backfill_visual_entities.py` only | hand-written owner/pet visual profile; fresh user has none and no way to make one — DERIVE from profile + graph pet/species metadata when visual memory is used |
| 18 | `PRONOUN_MAP` supports 3 sets, silent they/them fallback | `core/orchestrator.py:1041` | mechanism is CORRECT (pronouns ARE profile-threaded; determiner "her" is right); gap is only custom pronoun sets — accept fallback or pass the profile string through verbatim |
| 19 | Notes-generator example vocabulary | `core/agentic/types.py:541` ("name=Luke, age=33"), `gatherer_knowledge.py:239` ("Luke U_handle" signature example) | comments/examples in prompt-adjacent strings; low steering risk, tidy when touched |

## Rejected agent claims (verified false or reclassified — do not re-file)

- `PRONOUN_MAP` "she/her/her is a bug" — **no**: determiner form is correct for
  the prompt usage ("use she/her (she/her/her)").
- `data/narrative_context.txt` contains owner life state — it's **derived
  data**, regenerated at shutdown per user; a fresh install regenerates. Not a
  code finding.
- "refrigerator mother" in grounding `_SUBJECT_RE` — **KEEP**: a regression fix
  that generalizes (any competent grounding floor should know it).
- Multi-valued medication relations in `relation_classifier` — **KEEP**:
  correct cardinality for any multi-medicated user, not owner-tuning.
- "he/him hardcoded in system prompt" — the live pronoun line is
  profile-threaded through `PRONOUN_MAP`; only the 3-set limitation stands (#18).

## Learning-loop coverage map (which systems can adapt to a user at all)

| Verdict | Systems |
|---------|---------|
| **HAS-LOOP** (seeds + learned, teachers wired) | tone detector (exemplars), need detector, intent semantic tier, web-trigger anchors |
| **FEASIBLE — teacher signal already in telemetry** | ack-starter whitelist (light-path route followed by deep-retrieval need), proper-noun stoplist (extracted noun that anchors a successful retrieval), positive-recovery lexicon (CONVERSATIONAL verdict followed by escalation = false positive), recency keywords ("latest" + no `[WEB_` citation = mis-fire), gate tool keywords (Tier-4 routed a tool the Tier-1 lists missed → candidate keyword) |
| **FEASIBLE — needs new signal, high poisoning risk** | stance evaluative lexicon, private-sphere guard suppression correctness (needs explicit user override signal) |
| **STRUCTURAL — regex/shape logic, learning doesn't fit** | anaphora/fragment/continuation shapes, is_personal_doc_search, insight request-shape patterns, content-type format markers, **action registry intent patterns** (false positive fires a real-world action — must stay strict) |

Doctrine reminder for feasible loops: teachers must be channels independent of
the classifier being taught; backstops and suppressions never teach.

## Priority worklist

1. **P0 batch** (#1–4): small, mechanical, safe — timezone derivation,
   weekly-notes threading, few-shot neutralization, script path. One session.
2. **Anchors resolver** (#11): the highest-leverage general mechanism — one
   "named institutions" resolver feeding the private-sphere guard, the trigger
   prompt, and `apply_institution`; wire `learned_relations` promotions in as
   anchor candidates.
3. **Insight vocabulary generalization** (#5–7): categorized-generic seed
   extension + the "gather everything about my X teaches X" learned channel.
4. **Feasible learning loops** from the coverage map, one at a time, each with
   its poisoning-guard design reviewed.
5. Still parked from 08-21 (unchanged): identity-fact auto-promotion (the
   "never inject a fact again" mechanism), first-run/periodic self-calibration
   jobs, P0 items of that audit (notes generators daily/monthly, box-test
   strings, quick-key curation).

## Status

2026-09-01 (same night): worklist items 1 and 3 EXECUTED via two cheap-executor
batches + frontier review — P0 #1–4 fixed (`utils/timezone_resolver.py`,
`utils/user_identity.py` + all three notes generators incl. the ImportError-
branch vault fallbacks, few-shot neutralization, backfill script), P1 #5–7 and
#9 vocabulary extensions applied, #11 partially built (`get_user_anchors()`
with employer/org relations feeds the private-sphere guard; `apply_employer`
term backstop and learned_relations→anchors wiring still open), #12 and #18
fixed. Still open: #8, #10 (safety-path, needs design), #13–17, learned
channels, and the 08-21 parked items. Hand-verified: #1–4, #5, #7, #11, #17,
#18 checked against source; constants table spot-checked.
