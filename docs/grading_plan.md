# Synthesis Audit & Grading Protocol

## 1. Overview

The Synthesis Audit Queue is the ground-truth mechanism for Daemon's knowledge graph. Because LLMs are inherently biased toward finding patterns (apophenia), the Coherence Judge cannot be trusted as the final arbiter for permanent graph edges.

This protocol dictates how human-in-the-loop review evaluates provisional bridge edges, establishes empirical False Positive (FP) and False Negative (FN) rates, and builds the labeled dataset required for future classifier training.

**Implementation:** GUI "Synthesis" tab, `memory/synthesis_memory.py` audit methods, auto-halt in `shutdown_processor.py`.

---

## 2. The 1-5 Structural Grading Rubric

Grading is strictly based on structural mechanics, not thematic similarity.

### 1: Pure Hallucination / Logical Failure

**Definition:** The connection is nonsensical, hallucinated, or relies on a fundamental misunderstanding of one of the concepts.

**Action:** Reject.

**Example:** `1993-02-09 <> Convergence space` — No functional relationship. The date is treated as a "point" in topological space purely because the word "convergence" exists.

### 2: The Surface Metaphor (LLM Trap)

**Definition:** The nodes share a broad thematic category or a linguistic pun, but the underlying rules, constraints, or mechanisms are completely different.

**Action:** Reject.

**Example:** `shower after workouts <> Babulang ritual` — Both involve water/cleansing, but the functional mechanism of removing sweat does not map to the cultural rules of the festival.

### 3: Structurally True, but Trivial

**Definition:** A valid functional mechanism is shared, but it is too broad or obvious to yield novel analytical value.

**Action:** Reject. (For classifier training, grade 3 = reject. The classifier should learn to reject trivial connections.)

**Example:** `bartending <> Fluid dynamics` — Accurate, but functionally useless for insight.

### 4: Structural Isomorphism (The Target)

**Definition:** The underlying system rules, causal loops, or mathematical properties of both domains map onto each other perfectly across different disciplines. Applying the Wikipedia concept changes how the personal fact is analyzed.

**Action:** Accept. Promote to permanent bridge edge.

**Example:** Connecting the token-budget decay rate in `MEMORY_SYSTEM` to pharmacokinetic half-life equations — the exponential decay model is structurally identical, and the pharmacokinetic framing yields concrete tuning insights (therapeutic window = useful context range).

### 5: Deep Convergence (Breakthrough)

**Definition:** A profound, perfectly mapped structural connection that synthesizes a completely novel analytical framework for a complex personal node.

**Action:** Accept. Promote to permanent bridge edge with high initial weight.

---

## 3. Evaluation Heuristics (For Unfamiliar Domains)

When the pipeline proposes a connection involving an obscure or highly technical Wikipedia concept, apply these analytical tests to verify structural integrity:

### The "De-Jargon" Test

Strip away all field-specific nouns. Evaluate only the verbs, constraints, and logic gates. If the connection sounds profound only because of the vocabulary, it is a **2**.

### The Variable Swap

If the two systems are truly isomorphic, changing a variable in the Wikipedia concept's ruleset should logically predict the outcome in the personal fact's ruleset. If it doesn't, it is a **2**.

### The Underwriter's Rule

Treat unverified edges as unpriced liabilities. If the underlying mechanics cannot be verified, default to **Reject**. A False Negative (missed insight) is always preferable to a False Positive (graph poisoning).

---

## 4. Mapping to Implementation

The GUI Synthesis tab uses a 1-5 slider. The system maps grades to FP/FN statistics:

| Grade | Label | Classification | Effect |
|-------|-------|---------------|--------|
| 1 | Hallucination | Invalid (FP if accepted) | Reject, counts toward FP rate |
| 2 | Surface metaphor | Invalid (FP if accepted) | Reject, counts toward FP rate |
| 3 | True but trivial | Invalid (FP if accepted) | Reject, counts toward FP rate |
| 4 | Structural isomorphism | Valid | Accept, promote bridge edge |
| 5 | Deep convergence | Valid | Accept, promote bridge edge with high weight |

**For classifier training:** Grades 1-3 = negative class, grades 4-5 = positive class. The 3/4 boundary is the decision threshold.

**Average grade** is tracked as a quality signal. Avg < 2.5 across recent grading indicates the pipeline is producing mostly junk.

---

## 5. Audit Operations Plan

### Daily Batch Review

Grade the pending queue of accepted insights and composite rejects blindly (without seeing generator tags). The GUI hides generator labels by default to prevent bias.

### Threshold Calibration

- If the **FN rate** (grade 4-5 insights found in composite rejects) rises, incrementally lower the composite threshold from 0.65.
- If the **FP rate** (grade 1-3 insights found in accepted) exceeds 50% over 10+ graded items, the system **auto-halts** synthesis dreaming (`SYNTHESIS_AUDIT_FP_HALT_THRESHOLD = 0.50`).

### Auto-Halt Mechanism

The shutdown processor checks audit stats before running synthesis dreaming:

```
if fp_rate > SYNTHESIS_AUDIT_FP_HALT_THRESHOLD
   AND total_graded >= SYNTHESIS_AUDIT_MIN_GRADED:
    → skip synthesis, log warning
```

This prevents the pipeline from flooding the graph with junk edges when the coherence judge is clearly failing.

### Classifier Bootstrap

Once the audit queue accumulates **300+ manually graded pairs** (balanced across valid/invalid), the dataset will be exported to train an SVM/KNN classifier to replace or augment the LLM Coherence Judge.

**Current volume:** 74 results from the 2026-04-01 diagnostic run (composite at 0.55). At the current session rate (~4 accepted per production run), reaching 300 requires either:
- Multiple diagnostic runs at relaxed thresholds (faster)
- ~75 production sessions (slower)

---

## 6. Diagnostic Run Protocol

To generate grading data efficiently:

1. Lower `composite_min_score` to 0.55 in `config/config.yaml`
2. Run `python scripts/test_end_to_end_synthesis.py --candidates 150 --runs 1`
3. Revert `composite_min_score` to 0.65
4. Grade the results through the Synthesis tab
5. Check audit stats — if FP rate indicates threshold adjustment, update composite accordingly

---

## 7. Configuration

```yaml
# config/config.yaml
synthesis_audit:
  enabled: true
  fp_halt_threshold: 0.50   # auto-halt if FP rate exceeds this
  min_graded: 10             # minimum graded results before auto-halt can trigger
```

```python
# config/app_config.py
SYNTHESIS_AUDIT_ENABLED = True
SYNTHESIS_AUDIT_FP_HALT_THRESHOLD = 0.50
SYNTHESIS_AUDIT_MIN_GRADED = 10
```
