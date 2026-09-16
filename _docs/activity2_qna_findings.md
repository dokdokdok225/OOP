# Activity 2 Q&A Findings — Baseline Model and Decision Tree

This document organizes the findings explained during the Activity 2 Q&A session.
It covers the baseline model, the comparison decision tree, evaluation metrics,
error analysis patterns, and how to inspect the model's decisions.

---

## 1. The two models compared

| | DummyClassifier (baseline) | DecisionTreeClassifier (comparison) |
|---|---|---|
| **Purpose** | Reference point; shows the minimum useful performance | First real model that learns from features |
| **How it predicts** | Always predicts the majority class (Success) | Learns if/then rules from mission features |
| **Uses features?** | No | Yes |
| **Handles imbalance?** | No (ignores minority class) | Yes, via `class_weight="balanced"` |
| **Not-Success F1** | 0.0000 | 0.3463 |
| **Accuracy** | ≈ 0.8985 | ≈ 0.7635 |

The decision tree configuration:

```python
DecisionTreeClassifier(
    max_depth=5,
    class_weight="balanced",
    random_state=42
)
```

---

## 2. Why the dummy baseline looks "perfect" but is not

- About **90%** of missions are Successful, so predicting "Success" every time
  gives roughly 90% **accuracy**.
- This high accuracy comes from **class imbalance**, not predictive ability.
- The baseline detects **zero** Not-Success missions:
  - Not-Success recall = 0.0000
  - Not-Success precision = 0.0000
  - Not-Success F1 = 0.0000

Conceptual confusion matrix for the dummy baseline:

```text
                 Predicted
              Not-Success  Success
Actual Not-Success     0       94
Actual Success         0      832
```

**Lesson:** accuracy alone is misleading on imbalanced data.

---

## 3. The four evaluation metrics

For this project, **Not-Success (0)** is the positive class, because detecting
unsuccessful missions is the main goal.

### Confusion-matrix terms

| Term | Meaning in this project |
|---|---|
| **True Positive (TP)** | Not-Success correctly predicted as Not-Success |
| **True Negative (TN)** | Success correctly predicted as Success |
| **False Positive (FP)** | Success incorrectly predicted as Not-Success |
| **False Negative (FN)** | Not-Success incorrectly predicted as Success |

### The metrics

1. **Accuracy** — proportion of all predictions that are correct.
   `(TP + TN) / all predictions`
   Misleading here because the majority class dominates.

2. **Precision** — of the missions predicted Not-Success, how many truly were?
   `TP / (TP + FP)`
   High precision = few false alarms. Tree precision ≈ **0.2407**.

3. **Recall** — of all actual Not-Success missions, how many were detected?
   `TP / (TP + FN)`
   High recall = few missed failures. Tree recall ≈ **0.6170**.

4. **F1 score** — harmonic mean of precision and recall.
   `2 × (precision × recall) / (precision + recall)`
   High only when both are reasonably high. This is the **primary metric**
   (Not-Success F1) because the dataset is imbalanced.

### Results summary

| Model | Not-Success F1 | Precision | Recall | Accuracy |
|---|---:|---:|---:|---:|
| DummyClassifier | 0.0000 | 0.0000 | 0.0000 | ≈ 0.8985 |
| Decision tree | 0.3463 | ≈ 0.2407 | ≈ 0.6170 | ≈ 0.7635 |

The tree has **lower accuracy** but is far more useful for the actual objective:
it detects about **61.7%** of failures that the baseline misses entirely.

---

## 4. How each model "thinks"

### Dummy baseline process

```text
Look at class counts
        │
        ▼
Success is most common (~90%)
        │
        ▼
Predict Success for EVERY mission
(features are ignored completely)
```

### Decision tree process

```text
Look at training features and outcomes
        │
        ▼
Find feature splits that separate the classes
        │
        ▼
Build a limited-depth tree of if/then rules
        │
        ▼
Follow one path per new mission
        │
        ▼
Predict Success or Not-Success
```

Simplified illustration of learned rules (not the exact tree):

```text
                Prior failure rate > threshold?
                     /                    \
                   Yes                    No
                   /                      \
      Few prior company launches?       Success
             /              \
           Yes              No
           /                \
    Not-Success            Success
```

### Why the two models predict differently

- The dummy model uses **only the class frequency**.
- The tree uses **individual mission features** (company history, prior failure
  rate, launch year, country, rocket frequency, seasonality).
- `class_weight="balanced"` makes Not-Success mistakes **more costly**, so the
  tree is willing to flag borderline missions as Not-Success instead of
  defaulting to the majority class.

---

## 5. Overfitting controls

The tree is deliberately constrained to reduce overfitting:

- **`max_depth=5`** — limits tree complexity so it learns broad patterns rather
  than memorizing individual missions.
- **Fixed `random_state=42`** — reproducible results.
- **Untouched test set** — honest evaluation of generalization.

`class_weight="balanced"` does **not** reduce overfitting; it addresses class
imbalance only.

---

## 6. Part F — Error analysis findings

Part F inspects misclassified held-out test observations. Because the saved
Activity 1 arrays do not include original CSV row IDs, observations are
identified by test-row position and their strongest encoded feature values.

### Confusion matrix of the tree (labels [Not-Success, Success])

```text
                 Predicted
              Not-Success  Success
Actual Not-Success     58       36
Actual Success        183      649
```

- **58** true Not-Success correctly detected
- **36** false negatives (missed failures)
- **183** false positives (successful missions flagged as failures)
- **649** true Success correctly detected

### Patterns found

1. **False positives dominate the errors (183 vs 36).**
   The class-weighted tree intentionally accepts more false alarms in exchange
   for higher Not-Success recall.

2. **No single feature explains all mistakes.**
   The inspected errors showed a mix of higher `Company_Prior_Failure_Rate`
   values, certain `Launch_Year` ranges, and seasonal (`Month_Sin`/`Month_Cos`)
   values — but missions with these "risky-looking" patterns still succeeded.

3. **Prior failure rate is useful but not deterministic.**
   Several false positives had high encoded prior failure rates
   (e.g., 2.821, 3.374) yet were successful missions. Historical reliability
   signals risk; it does not guarantee failure.

4. **The classes overlap substantially.**
   Similar feature patterns can lead to either outcome, so the model cannot
   cleanly separate Success from Not-Success with the current features.

5. **The model trades precision for recall.**
   It catches 61.7% of actual failures, but only ~24% of its failure alerts are
   correct. Useful evidence for the next modeling step; not production-ready.

6. **The shallow tree may be too limited.**
   `max_depth=5` prevents overfitting but may miss more complex interactions
   between company, rocket, country, and time.

### Interpretation of the two error types

- **False negatives** (actual Not-Success → predicted Success) are the most
  consequential misses for the stated objective; they are why recall matters.
- **False positives** (actual Success → predicted Not-Success) reduce precision
  and could trigger unnecessary failure alerts or investigations.

---

## 7. How to see the model's decisions

### a) Export the learned rules

```python
from sklearn.tree import export_text
print(export_text(tree_model, feature_names=feature_names))
```

Indentation shows depth; `class:` lines are the leaf predictions.

### b) Visualize the tree

```python
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(24, 12))
plot_tree(tree_model,
          feature_names=feature_names,
          class_names=["Not-Success", "Success"],
          filled=True, rounded=True, proportion=True)
plt.show()
```

Each box shows the split feature and threshold, sample proportion, class
distribution, and predicted class.

### c) Trace one mission's decision path

```python
row = X_test[673].reshape(1, -1)
node_ids = tree_model.decision_path(row).indices
```

Walk from the root node to the leaf, checking each feature threshold against the
mission's actual value, to see exactly why a prediction was made.

### d) Feature importance (global view)

```python
importance_table = (
    pd.DataFrame({
        "feature": feature_names,
        "importance": tree_model.feature_importances_
    })
    .sort_values("importance", ascending=False)
)
importance_table.head(10)
```

Shows which features were most useful across the whole tree — not per-prediction
explanations.

**Limitation:** tree rules are descriptive, not causal. A split on
`Company_Prior_Failure_Rate` means the feature helped separate classes in this
dataset, not that it causes mission failure.

---

## 8. Known caveats and next steps

- The split is **random** although the data spans 1957–2022; a **chronological
  holdout** (train on earlier missions, test on later ones) is the single most
  important next check.
- The saved matrices lack original CSV row IDs, so error analysis reports
  test-row positions rather than mission identifiers; a future pipeline should
  preserve IDs.
- Researched future models (not part of the Activity 2 submission):
  Random Forest (CV F1 ≈ 0.349, heldout ≈ 0.384) and HistGradientBoosting
  (heldout ≈ 0.371).
- No SMOTE/resampling was used; imbalance was handled with stratification and
  class weighting only.
