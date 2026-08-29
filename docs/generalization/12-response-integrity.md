# G12: Response Integrity and Atomic Revision

This workstream is intentionally separate from tone, escalation, graph, and
agentic-routing changes. It addresses a specific production failure: a factual
reviewer appended a correction that contradicted an earlier sentence instead of
producing one coherent answer.

## Contract

Every generated answer has one user-visible and indexable final form:

```text
draft -> structured review -> PASS | REWRITE | FALLBACK -> final
```

The reviewer never writes user-facing prose. It returns structured data only.
If the review identifies a material factual or entailment problem, the system
rewrites the complete answer or uses a concise integrated fallback. It never
appends a correction suffix.

The following invariant is mandatory:

```text
display_text == stored_text == indexed_text == final_text
```

Drafts, reviewer verdicts, and failed revisions may be retained in a private
debug ledger, but they must not enter conversational memory or semantic search.

## Reviewer Requirements

The reviewer must identify the exact claim span before judging it. It must not
strengthen a claim during correction. At minimum, classify claims as:

- predictive or correlational
- causal
- deterministic
- normative or rhetorical
- assistant appraisal or endorsement

The reviewer must abstain when it lacks sufficient evidence for an empirical
claim. A generic caveat is not a substitute for checking entailment.

Boolean fields must be parsed strictly. JSON strings such as `"false"` must not
be treated as true merely because they are non-empty strings.

## Sanitized Incident Case

User intent: express anger that childhood geography is treated as irrelevant to
achievement, while making a probabilistic regression/distributional claim.

Bad draft pattern:

```text
Your accomplishment is solely determined by your ZIP code.
```

Required behavior:

```text
Childhood geography is a strong structural predictor of later outcomes and can
partly shape them causally, but it is not a complete or deterministic account of
any individual life. The point is that achievement is often narrated as pure
individual merit while unequal starting conditions are omitted.
```

The correction must be integrated into the answer. A suffix such as
`Correction: ZIP code does not determine everything` fails this contract because
the draft remains visible and retrievable.

## Acceptance Tests

1. A claim that says `predicts` is never rewritten as `determines` without an
   explicit source-supported reason.
2. An already-qualified answer receives `PASS`, not a redundant addendum.
3. A material error produces one rewritten answer with no contradictory draft
   sentence.
4. Rewrite failure produces an integrated fallback, not an appended warning.
5. `display_text`, `stored_text`, `indexed_text`, and `final_text` are identical.
6. The draft and reviewer JSON are absent from retrieval results.
7. A quoted or pasted webpage cannot trigger the reviewer to invent a claim from
   page chrome or treat a user rhetorical comparison as violent intent.

## Ownership Boundary

G12 should initially use new modules and tests. Integration into
`gui/handlers.py` happens only after the standalone contract passes review. It
must not modify the in-flight tone/escalation/graph batch.

## Fable Handoff

When the current batch is checkpointed, provide Fable this document plus a
synthetic fixture containing only the sanitized case above. Require a diff
summary, focused tests, and an explicit statement that no suffix-based
correction path remains on the production route.
