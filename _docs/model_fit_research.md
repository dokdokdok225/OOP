# Model-Fit Research — Space Missions Activity 2

## Research question

Which model families are appropriate for the current Space Missions dataset, and which
one should be tested next after the Activity 2 Decision Tree baseline?

## Current data profile

- **Task:** binary supervised classification.
- **Target:** `MissionStatus`, encoded as `1 = Success` and `0 = Not-Success`.
- **Observations:** 4,628 total; 3,702 training and 926 held-out test observations.
- **Class balance:** approximately 90% Success and 10% Not-Success.
- **Features:** 46 final numeric columns after Activity 1 preprocessing:
  - five numeric/engineered features;
  - one-hot encoded company and country columns;
  - frequency-encoded rocket information.
- **Primary metric:** F1-score for Not-Success (`pos_label=0`). Recall and precision
  must also be reported; accuracy is secondary because of the class imbalance.
- **Temporal concern:** missions span many years. A chronological holdout is needed
  before making claims about performance on future missions.

## Research conclusions

### Best fit for the current feature matrix

The strongest practical candidates are **Random Forest** and
**Histogram Gradient Boosting**. Both fit dense tabular data, can represent nonlinear
relationships and feature interactions, and support class imbalance through class
weights or sample weights. They are more suitable than ordinary linear regression for
this target.

### Best linear-style candidate

Use **Logistic Regression**, not `LinearRegression`. Logistic Regression is a
classification model that produces class probabilities and can use
`class_weight="balanced"`. It is a useful interpretable benchmark, especially because
Activity 1 standardized the numeric features.

### Best interpretable baseline

The existing shallow, class-weighted **DecisionTreeClassifier** is appropriate for
Activity 2. It is simple enough to explain and provides a meaningful comparison with
the majority-class DummyClassifier. It should remain the official Activity 2 model,
not be replaced by a more complex ensemble.

## Leakage-safe empirical comparison

The comparison below used only the existing `X_train`/`y_train` split for 5-fold
stratified cross-validation. The existing `X_test`/`y_test` split was kept untouched
until the final test evaluation. The models used fixed, reasonable settings rather
than an extensive hyperparameter search.

| Model | CV Not-Success F1 | CV recall | CV precision | CV accuracy | Held-out Not-Success F1 | Held-out recall | Held-out precision | Held-out accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression (`class_weight="balanced"`) | 0.312 | 0.647 | 0.206 | 0.712 | 0.322 | 0.670 | 0.212 | 0.714 |
| Decision Tree (`max_depth=5`, balanced) | 0.322 | 0.578 | 0.226 | 0.751 | 0.346 | 0.617 | 0.241 | 0.763 |
| Random Forest (`n_estimators=300`, balanced) | **0.349** | 0.356 | **0.344** | 0.866 | **0.384** | 0.404 | **0.365** | 0.868 |
| Extra Trees (`n_estimators=300`, balanced) | 0.246 | 0.187 | 0.370 | **0.885** | 0.264 | 0.202 | 0.380 | 0.886 |
| Histogram Gradient Boosting (balanced) | 0.348 | 0.462 | 0.279 | 0.825 | 0.371 | 0.521 | 0.288 | 0.821 |
| RBF SVC (`class_weight="balanced"`) | 0.341 | **0.634** | 0.233 | 0.751 | 0.344 | 0.649 | 0.234 | 0.748 |

### Interpretation

1. **Random Forest is the best next candidate by current F1.** It improves both
   minority-class precision and held-out F1 relative to the Activity 2 Decision Tree.
   It also produces a much better accuracy/F1 balance than the class-weighted linear
   and shallow-tree models.
2. **Histogram Gradient Boosting is nearly tied with Random Forest in cross-validation.**
   It obtains higher recall than Random Forest while producing more false alarms. It
   is a strong alternative when catching more Not-Success missions is more important
   than alert precision.
3. **Logistic Regression is a valuable interpretability benchmark, not the likely
   final performer.** Its linear boundary appears too restrictive for the nonlinear
   relationships in the current feature set.
4. **Extra Trees is not preferred for this objective in the current run.** Its accuracy
   is high, but its Not-Success recall and F1 are lower, which suggests it is too
   conservative about failure predictions under the tested settings.
5. **RBF SVC catches many failures but produces low precision.** It is less attractive
   than Random Forest or HistGradientBoosting here because it is less interpretable
   and requires more careful scaling/threshold decisions.

These numbers are a screening result, not a final model-selection claim. The random
split can make the estimates optimistic for a future-prediction use case.

## Recommendation

### For Activity 2 submission

Keep the current design:

```python
DummyClassifier(strategy="most_frequent")
DecisionTreeClassifier(
    max_depth=5,
    class_weight="balanced",
    random_state=42,
)
```

This satisfies the assignment's request for a naive baseline plus one simple,
interpretable model without excessive tuning.

### For the next modeling activity

Test **RandomForestClassifier** first, then compare it with
**HistGradientBoostingClassifier** using:

- a chronological train/validation/test design;
- `StratifiedKFold` only when the evaluation question is cross-sectional;
- Not-Success F1 as the primary selection metric;
- Not-Success recall and precision as guardrails;
- average precision / precision-recall curves to evaluate probability ranking;
- threshold selection performed on validation data only;
- final evaluation exactly once on an untouched chronological test set.

Do not use `LinearRegression` for this target. If a linear comparison is needed, use
`LogisticRegression(class_weight="balanced")`.

## Important modeling caveat: time

The feature `Company_Prior_Launches` and `Company_Prior_Failure_Rate` are historical
features and are conceptually valid at prediction time. However, the current random
split mixes earlier and later missions across train and test. A model can therefore
learn patterns from later-era missions while being evaluated on earlier-era missions.

For a model intended to predict future launches, rebuild or preserve row IDs and use a
chronological split. Any preprocessing and any history-based feature construction must
be fitted or generated in a way that does not use information from the future.

## Sources consulted

- scikit-learn ensemble methods and gradient-boosted trees:
  <https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosted-trees>
- scikit-learn model evaluation and classification metrics:
  <https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics>
- scikit-learn cross-validation guidance, including stratification and time-aware
  validation:
  <https://scikit-learn.org/stable/modules/cross_validation.html>
- scikit-learn linear models, including the distinction between regression and
  classification:
  <https://scikit-learn.org/stable/modules/linear_model.html>
