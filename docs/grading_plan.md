# Synthesis Audit & Grading Protocol

## 1. Overview

The Synthesis Audit Queue is the ground-truth mechanism for Daemon's knowledge graph. Because LLMs are inherently biased toward finding patterns (apophenia), the Coherence Judge cannot be trusted as the final arbiter for permanent graph edges.

**Current status:** All three synthesis generators (retrieval, graph walk, cross-store) are disabled in `config.yaml`. No new synthesis candidates are being generated. Existing results in the audit queue are available for grading. Generators remain frozen until the grading approach is validated.

**Implementation:** GUI "Synthesis" tab, `memory/synthesis_memory.py` audit methods, auto-halt in `shutdown_processor.py`.

---

## 2. Two-Layer Grading

The original 1-5 rubric asked graders to evaluate structural validity — essentially peer-reviewing cross-domain claims. This is paralyzing because the difference between "surface metaphor" (2) and "structural isomorphism" (4) requires expertise in both domains, and a genuine breakthrough (5) is by definition something you can't recognize until it proves out.

The new approach separates grading into two layers: fast binary screening, then a loose gut-feel slider.

### Layer 1: Binary Screening (answer first, don't overthink)

Three yes/no questions that require no expertise:

| Question | Maps to | Training signal |
|----------|---------|-----------------|
| **"Does this make me think about something differently?"** | Structural validity | If no → likely grade 1-2 |
| **"Is the mechanism it describes real, as far as I know?"** (yes/no/unsure) | Structural validity | If no → grade 1. If unsure → could be 3-5 |
| **"Have I heard this connection before?"** | Novelty | Separates "valid but known" from "valid and novel" |

These three questions cleanly separate into the two problems the pipeline needs to solve:
- **Questions 1-2** → "Is this a real structural parallel?" (detect)
- **Question 3** → "Is this already known?" (subtract)

This maps directly to the detect-then-subtract architecture: train a structural classifier on Q1+Q2 labels, use FAISS for Q3.

### Layer 2: Gut-Feel Slider (1-5, first instinct)

After answering the binary questions, pick a number. Don't agonize — your first instinct is probably right and the noise will average out over 300 examples.

| Grade | Label | Rough meaning |
|-------|-------|---------------|
| 1 | Nonsense | This connection doesn't make sense |
| 2 | Surface metaphor | Sounds similar but mechanisms are different |
| 3 | True but obvious | Valid mechanism, but obvious or already known |
| 4 | Real insight | Changes how I think about one of the concepts |
| 5 | Breakthrough | Surprising, multi-perspective, genuinely new framing |

**Why keep the slider if the binaries are enough?** The classifier eventually needs to learn the 3/4 boundary — the hardest discrimination in the system. The binary questions can't distinguish "real mechanism but trivial" from "real mechanism that changes my thinking." A grade-3 can have all three binaries as yes (real mechanism, not heard before, but doesn't actually change thinking in a meaningful way). The continuous signal from the slider lets the classifier learn this boundary even if individual labels are noisy.

---

## 3. How Binaries + Slider Work Together

| Q1 (Thinking?) | Q2 (Real?) | Q3 (Heard?) | Likely Grade | Interpretation |
|:-:|:-:|:-:|:-:|---|
| No | No | - | 1 | Nonsense connection |
| No | Yes | - | 2-3 | Real mechanism but doesn't illuminate anything |
| Yes | No | - | 2 | Feels insightful but mechanism is wrong |
| Yes | Unsure | No | 3-4 | Interesting, can't verify mechanism |
| Yes | Yes | Yes | 3 | Valid but already known |
| Yes | Yes | No | 4-5 | The target: real, novel, changes thinking |

The binaries give fast, honest signal. The slider captures nuance the binaries miss. Together they produce richer training data than either alone.

---

## 4. Mapping to Implementation

### Storage Format

Each graded result stores 6 fields in ChromaDB metadata:

```
changes_thinking: "True" / "False" / ""
mechanism_real:   "yes" / "no" / "unsure" / ""
heard_before:     "True" / "False" / ""
human_grade:      "1"-"5"
graded_at:        ISO timestamp
grade_notes:      free text
```

### FP/FN Classification (unchanged)

| Grade | Classification | Effect |
|-------|---------------|--------|
| 1-3 | Invalid (FP if accepted) | Counts toward FP rate |
| 4-5 | Valid | Counts toward valid rate |

The 3/4 boundary remains the decision threshold for classifier training: grades 1-3 = negative class, grades 4-5 = positive class.

### Future: Binary-Derived Labels

Once enough data accumulates, the binary screening questions provide separate training signals:

- **Structural classifier** (detect): trained on Q1 + Q2 labels
- **Novelty filter** (subtract): validated against Q3 labels + FAISS corpus overlap

---

## 5. Audit Operations

### Grading Workflow

1. Open Synthesis tab, select "Accepted (ungraded)" view
2. Read the connection claim (generator labels hidden for blind review)
3. Answer the three binary questions quickly
4. Set the slider to your gut feel
5. Optionally add notes
6. Submit

### Threshold Calibration

- If **FN rate** (grade 4-5 found in composite rejects) rises, lower the composite threshold
- If **FP rate** (grade 1-3 found in accepted) exceeds 50% over 10+ graded items, the system **auto-halts** synthesis dreaming

### Auto-Halt Mechanism

```
if fp_rate > SYNTHESIS_AUDIT_FP_HALT_THRESHOLD
   AND total_graded >= SYNTHESIS_AUDIT_MIN_GRADED:
    -> skip synthesis, log warning
```

### Classifier Bootstrap

Target: **300+ graded examples** (balanced valid/invalid) to train SVM/KNN on binary features + composite score components. The binary screening questions are the primary training features; the 1-5 slider provides the decision boundary.

---

## 6. Configuration

```yaml
# config/config.yaml
synthesis_audit:
  enabled: true
  fp_halt_threshold: 0.50
  min_graded: 10

# Generators (currently disabled)
synthesis:
  enabled: false
synthesis_generator:
  enabled: false
synthesis_retrieval:
  enabled: false
graph_walk:
  enabled: false
```

```python
# config/app_config.py
SYNTHESIS_AUDIT_ENABLED = True
SYNTHESIS_AUDIT_FP_HALT_THRESHOLD = 0.50
SYNTHESIS_AUDIT_MIN_GRADED = 10
```
