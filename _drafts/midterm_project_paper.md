# MIDTERM PROJECT

**Space Missions Launch-Outcome Classifier — Improving, Comparing, and Explaining a Mission-Success Prediction Model**

Prepared by: Von Mangui-ob
Track: Data Scientist
Course: DSC 2135 — Object Oriented Programming
Date: September 2026

---

## Table of Contents

1. Project Overview
2. Data and Project Inputs
3. Project Methodology and Development Process
4. Final Project Output
5. Results and Key Findings
6. Discussion
7. Conclusion
8. Recommendations and Future Improvements
9. References
10. Appendices

---

## 1. Project Overview

**Background.** Space launch is a high-stakes, high-cost activity, and launch outcomes vary systematically with the organisation performing the launch, the era in which it flies, and the vehicle used. Historical launch data therefore offers a realistic tabular machine-learning problem: can a model, built only from information known before a launch, meaningfully screen missions for risk?

**Main problem.** Predict whether a given space mission will end in **Success** or **Not-Success** (failure, partial failure, or prelaunch failure) using pre-launch characteristics of the mission, the organisation, and the rocket.

**Objectives.**

1. Build a clean, leakage-safe tabular dataset from the raw Space Missions CSV (Phase 1 / Activity 1).
2. Establish a defensible baseline classifier with an honest evaluation design (Phase 2 / Activity 2).
3. Improve on the baseline by comparing alternative candidate models, performing limited hyperparameter tuning on training data only, and explaining what drives the final model (Phase 3 / Activity 3).

**Scope.** One dataset (4,630 raw launch records, 4,628 after cleaning), classical scikit-learn models only, one binary classification task. Preprocessing was locked in Phase 1 and reused unchanged throughout all later phases so that every metric is comparable.

**Limitations.** The dataset spans 1957–2022 but was split randomly rather than chronologically; per-mission technical details (launch cost, payload mass, weather, vehicle maturity) are absent; and the model is an academic prototype, not a production or safety system.

---

## 2. Data and Project Inputs

**Primary input:** `space_missions.csv` — 4,630 rows × 9 columns. Each row is one launch attempt.

| Original column | Meaning | Final treatment |
|---|---|---|
| `Company` | Launching organisation | Kept; one-hot encoded |
| `Location` | Launch-site string | Used to extract `Country`, then dropped |
| `Date` | Launch date | Derived `Launch_Year`, cyclic `Month_Sin`/`Month_Cos`, then dropped |
| `Time` | Launch time of day | Dropped — missingness plus weak signal |
| `Rocket` | Rocket family | Kept; frequency encoded |
| `Mission` | Mission name | Dropped — ~98.4% unique, identifier-like (leakage vector) |
| `RocketStatus` | Active/Retired at time of record | Dropped — temporal leakage risk in a random split |
| `Price` | Launch cost | Dropped — 72.68% missing, above the drop threshold |
| `MissionStatus` | Outcome label (4 classes) | Collapsed into binary target |

**Target:** `1 = Success`, `0 = Not-Success` (Failure, Partial Failure, Prelaunch Failure combined). The collapse reflects the practical decision-support question ("will this launch succeed?") and leaves an approximately **90/10 class imbalance** (4,160 Success vs 468 Not-Success).

**Engineered features (46 total):** launch year and cyclic month features; `Company_Prior_Launches` and `Company_Prior_Failure_Rate` computed strictly from prior rows; one-hot company and country indicators; frequency-encoded rocket names.

**Derived artefacts:** `X_train.npy` / `X_test.npy` (3,702 / 926 rows × 46 features), `y_train.npy` / `y_test.npy`, `feature_names.txt`, `space_missions_engineered.csv` (human-readable table keyed by `Source_Row_ID`).

**Supporting notebooks:** `oop.ipynb` (preprocessing), `activity_2_baseline_model.ipynb` (baseline), `activity_3_improve_compare_explain.ipynb` (this phase's model work).

---

## 3. Project Methodology and Development Process

The project followed three phases. Each phase built on, and never re-opened, the decisions of the previous one — which is what keeps every reported number comparable.

### Phase 1 — Preparation and Foundation (Activity 1)

- **Cleaning:** one true duplicate row removed; no imputation anywhere — every missing column was either dropped (`Price`, `Time`) or used before dropping (`Location` → `Country`).
- **Leakage control:** `Mission` (identifier-like) and `RocketStatus` (post-hoc status) were dropped explicitly as leakage vectors; all history features (`Company_Prior_Launches`, `Company_Prior_Failure_Rate`) were computed from **prior launches only**, never including the row itself.
- **Feature engineering:** date decomposition into year and cyclic month encodings (so December–January wrap correctly); one-hot and frequency encodings for categorical variables.
- **Split:** stratified 80/20 train/test, `random_state=42` → **3,702 training / 926 held-out test rows**. The split was saved as arrays and treated as locked thereafter.

### Phase 2 — Development and Analysis (Activity 2)

- **Evaluation design:** primary metric = **F1-score of the Not-Success class** (`pos_label=0`), chosen because accuracy is misleading at 90/10 imbalance; accuracy, precision, and recall retained as supporting metrics.
- **Baseline pair:** `DummyClassifier` (majority class) as the honest floor, and `DecisionTreeClassifier(max_depth=5, class_weight='balanced', random_state=42)` as the simple model — a testable, interpretable classifier that acknowledges the imbalance.
- **Baseline result:** Dummy F1 = 0.0000 (never predicts the minority class); Decision Tree held-out F1 = **0.3463** (precision 0.2407, recall 0.6170, accuracy 0.7635).
- **Class-weighting experiment and a documented negative result:** `LinearRegression` was tested empirically and rejected — modelling a binary outcome with unrestricted real predictions failed badly (recall 0.096, F1 0.159, predictions outside [0,1]). The valid linear classifier, `LogisticRegression`, was deferred to Phase 3.
- **Exploratory appendix:** RandomForest and HistGradientBoosting were benchmarked informally against the held-out set (clearly stronger) — explicitly flagged as exploration for a future phase, not as the official submission.

### Phase 3 — Final Technical Development (Activity 3)

Phase 3 re-used the locked split and preprocessing and added disciplined model selection:

- **Part A — Re-established the design:** same objective, task type, target, split sizes, and baseline restated; the random-vs-chronological split question was documented as a known limitation.
- **Part B — Candidate comparison with identical preprocessing:** Dummy (floor), Activity 2's Decision Tree, `LogisticRegression` (balanced), `RandomForestClassifier` (300 trees, balanced), and `HistGradientBoostingClassifier` (max_iter=200, balanced).
- **Part C — Training-only model selection:** 5-fold stratified cross-validation on the 3,702 training rows only, scored with the primary metric. Test set untouched. Results:

| Candidate | CV mean Not-Success F1 | CV std |
|---|---|---|
| RandomForest (300 trees) | **0.3489** | 0.0642 |
| HistGradientBoosting | 0.3480 | 0.0437 |
| Decision Tree (Act 2) | 0.3223 | 0.0346 |
| LogisticRegression | 0.3130 | 0.0192 |
| DummyClassifier | 0.0000 | 0.0000 |

- **Part D — Limited tuning:** the RandomForest was tuned with a deliberately small `GridSearchCV` grid (3 hyperparameters × modest value ranges, training data only). The winning configuration was **`n_estimators=500, max_depth=12, min_samples_leaf=2`**, lifting mean validation F1 from 0.3489 to **0.3842** (+0.0352).
- **Part E — Final held-out evaluation (test set used once):** fitted the tuned model on the full training set and evaluated a single time; comparison against the Activity 2 baseline is in Section 5.
- **Part F — Model explanation:** feature-importance analysis of the tuned forest (Section 5).
- **Part G — Final error analysis:** the held-out membership was re-derived from the engineered CSV with the identical split call and verified label-for-label against the saved test arrays before any row was inspected; six misclassified missions were reviewed by name.
- **Part H — Updated model card and reflection**, including a responsible-use statement (Section 5 / 7).

---

## 4. Final Project Output

The final technical product is **`activity_3_improve_compare_explain.ipynb`** — a fully executed, reproducible notebook containing the evaluation design, the comparison, the tuning search, the single held-out evaluation, the explanation, the error analysis, and the model card, all with `random_state = 42` and no reliance on manual cell-order.

Its centrepiece is the **tuned RandomForestClassifier** (`n_estimators=500, max_depth=12, min_samples_leaf=2, class_weight='balanced'`) selected through training-only cross-validation, with the Activity 2 decision tree retained as evidence of improvement.

---

## 5. Results and Key Findings

**Final held-out performance (single evaluation, after all selection decisions):**

| Metric | Activity 2 baseline (Tree) | Phase 3 final (tuned RF) |
|---|---|---|
| **Not-Success F1 (primary)** | 0.3463 | **0.4215** |
| Not-Success precision | 0.2407 | 0.3293 |
| Not-Success recall | 0.6170 | 0.5851 |
| Accuracy | 0.7635 | 0.8369 |

The tuned random forest **improved the primary metric by +0.0752** (≈22% relative), and the improvement is non-trivial: the ensemble led the tree in four of five validation folds on mean F1 (0.3489 vs 0.3223 — the highest mean, with noticeable fold variability rather than a clean sweep), precision rose 36%, and accuracy climbed by 0.073. The confusion matrix moved from 58 caught / 36 missed / 183 false alarms (tree) to 55 caught / 39 missed / 112 false alarms / 720 correct successes (forest).

**Expected trade-off (stated, not hidden):** the tuned forest gives up a little recall (0.6170 → 0.5851) to buy precision (0.2407 → 0.3293). It raises ~25% fewer noisy failure alerts but misses 39 real failures where the tree missed 36.

**What drives the model (feature importance; the top four features carry 72.5% of total reliance):**

| Feature | Importance | Pattern |
|---|---|---|
| `Company_Prior_Failure_Rate` | 0.2305 | Operator's own historical failure rate dominates |
| `Company_Prior_Launches` | 0.1771 | Operator experience level |
| `Launch_Year` | 0.1662 | Technology/safety era effects |
| `Rocket_Freq` | 0.1517 | Rare, low-flight-history vehicles are riskier |
| `Month_Sin`/`Month_Cos` | 0.0529 / 0.0533 | Mild seasonality |
| Company/Country one-hots | ≤0.0238 each | Historical operator regimes visible at low per-feature weight |

**Error analysis (six named missions reviewed).** The two error types have opposite signatures:

- *False alarms* (predicted Not-Success, actually flew) — missions against the operator's own history: JAXA's small Epsilon PBS (2019), MHI's H-IIA (2002), a 1964 General Dynamics Atlas. Early or exotic vehicles make the history features look risky.
- *Missed failures* (predicted Success, actually failed) — tail failures of historically reliable systems: two RVSN USSR Cosmos missions (1971, 1980) and a Lockheed Athena I (1995). An excellent track record lets rare failures slip through — exactly where the binary collapse of partial failures bites.

**What the errors suggest.** Missing features (launch cost, payload mass, weather, vehicle maturity — no first-flight indicator) lower the ceiling; the 90/10 imbalance keeps a few misses costly; and history-split features make the model partially memorise era clusters, so genuinely far-tail failures are hardest to catch.

---

## 6. Discussion

The results should be read through the evaluation design, not just the metrics. Five points frame what the improvement actually means.

**6.1 The improvement reflects model capacity, not a new imbalance remedy.** Every candidate retained the same Phase 1 preprocessing and the same `class_weight="balanced"` treatment as the Activity 2 tree. The tuned forest improved mainly by modelling nonlinear interactions and by reducing the variance of a single tree through averaging — not because resampling or a different class-balance strategy was introduced. This matters for interpretation: the forest is telling the same *"operator track record plus era"* story the tree told, but more reliably. No SMOTE or synthetic resampling was used anywhere, which kept the evaluation honest — a resampled training set would have inflated apparent success without adding information.

**6.2 The primary gain is a precision–recall rebalancing.** Not-Success precision rose from 0.2407 to 0.3293 while recall declined modestly from 0.6170 to 0.5851; the F1 increase (0.3463 → 0.4215) shows the precision gain outweighed the recall loss. Had the priority instead been *never miss a failure* (safety-review posture), the Activity 2 tree or a threshold-shifted forest would remain more appropriate — the choice of final model embeds a judgment about the relative cost of false alarms versus missed failures.

**6.3 Accuracy only became informative alongside minority-class metrics.** Accuracy rose from 0.7635 to 0.8369, but at a ~90/10 imbalance a model can score high accuracy while never identifying a failure — the `DummyClassifier` does exactly this with accuracy 0.8985 and F1 = 0.0000. The project's success criterion therefore remained Not-Success F1 throughout, and accuracy is meaningful only when read next to precision and recall on the minority class.

**6.4 The model is organically a historical risk screen.** Four features — `Company_Prior_Failure_Rate` (0.2305), `Company_Prior_Launches` (0.1771), `Launch_Year` (0.1662), and `Rocket_Freq` (0.1517) — account for approximately 72.5% of the forest's reliance. The model is effectively an organisational/historical risk-screening tool rather than a detailed engineering assessment, and it makes systematic mistakes with exactly that character: false alarms occur where a mission runs against its operator's risk history, and missed failures occur when historically reliable systems (e.g. the Soviet Cosmos family, a mature ULA-class lineage) suffer rare tail failures. Historical reliability is useful but structurally insufficient for low-frequency tail events.

**6.5 Disciplined process, defensible result.** Five candidates were compared with identical preprocessing via training-only cross-validation; one model was tuned on a deliberately limited grid; the held-out test set was evaluated exactly once after selection. The final improvement is therefore evidence of generalisation rather than of an uncontrolled search — a more complex model was adopted only because it was justified by training-set evidence, and its limits are stated rather than hidden. Three explicit trade-offs bound the claim:

- *Failure detection vs alert precision:* the forest raises more trustworthy failure alerts but catches slightly fewer real failures than the tree (39 vs 36 missed).
- *Complexity vs interpretability:* the forest predicts better but is slower and harder to explain per decision; importances describe predictive reliance, not causation.
- *Retrospective performance vs forward validity:* the random stratified split enables comparability across phases but means the reported performance is retrospective prototype evidence, not proof of future deployment performance, because later-era information can inform training while earlier-era missions sit in testing.

---

## 7. Conclusion

The project achieved all three of its stated objectives. Phase 1 produced a documented, leakage-safe dataset and a locked, reproducible split. Phase 2 established a defensible baseline (Decision Tree, Not-Success F1 0.3463) with a deliberately honest evaluation design that even named its own negative result (LinearRegression). Phase 3 improved the model through training-only comparison and limited tuning to a final held-out **Not-Success F1 of 0.4215** — a clear, non-trivial improvement — while retaining full interpretability of what drives the prediction and an explicit statement of its trade-offs and limits.

The final model remains an **academic prototype for retrospective risk classification**, not a production launch-safety system. It demonstrates how historical mission, operator, country, rocket-frequency, and era features relate to binary outcomes in this dataset, but it does not establish causation, does not model engineering or operational conditions, collapses several failure types into one class, and has not been validated on a strictly future chronological holdout. It must not be used alone to approve a mission, certify a vehicle, make crew-safety decisions, or price insurance. What it does demonstrate — and what this project's central finding is — is that disciplined improvement (evaluation-first design, training-only selection, one held-out evaluation) is worth ~22% relative F1, without letting model complexity outrun the evidence.

---

## 8. Recommendations and Future Improvements

1. **Chronological holdout.** Re-run with a time-based split (train ≤ cutoff year, test after) and measure F1 drift. This directly tests whether 0.4215 reflects genuine forward-predictive ability or partly retrospective memory of past eras.
2. **Vehicle-maturity features.** Add a first-flight/vehicle-age indicator: the error analysis shows both false alarms and missed failures concentrate on debut or low-history vehicles that current features cannot describe.
3. **Operating-threshold tuning and calibration analysis.** Replace reliance on the default 0.5 threshold with precision–recall curve analysis, selecting a threshold by the practical cost of false alarms versus missed failures, and evaluate calibration so predicted probabilities become meaningful risk estimates rather than mere labels.
4. **Longer term:** collect richer mission data (cost, payload, weather) so the features stop being proxies; combine with the calibration work above so alerts carry calibrated risk, not just labels.

---

## 9. References

1. Space Missions dataset (Kaggle community dataset; 4,630 global launch records, 1957–2022).
2. Scikit-learn documentation: `RandomForestClassifier`, `HistGradientBoostingClassifier`, `DecisionTreeClassifier`, `LogisticRegression`, `DummyClassifier`, `GridSearchCV`, `StratifiedKFold`.
3. Course materials: DSC 2135 lecture references (`oop_rules.md`) and the Phase 1 Data Preprocessing Activity guide.
4. Project internal documentation: `_docs/project_data_cheat_sheet.md` (full data-flow reference), `_docs/model_fit_research.md` (candidate-model research).

---

## 10. Appendices

- **Appendix A — Data-flow reference:** `_docs/project_data_cheat_sheet.md` documents every column decision from raw CSV to the final 46-feature arrays.
- **Appendix B — Activity 2 notebook:** `activity_2_baseline_model.ipynb` (baseline establishment, split justification, leakage checks, confusion matrices, initial error analysis, mini model card, exploratory four-model appendix).
- **Appendix C — Activity 3 notebook:** `activity_3_improve_compare_explain.ipynb` (CV comparison table, GridSearchCV results table, final comparison table, confusion matrix pair, importance chart, reviewed error missions, updated model card, reflection).
- **Appendix D — Artefact inventory:** `X_train.npy` (3702×46), `X_test.npy` (926×46), `y_train.npy`, `y_test.npy`, `feature_names.txt` (46 names), `space_missions_engineered.csv` (4,628 rows keyed by `Source_Row_ID`).
- **Appendix E — Repository:** github.com/dokdokdok225/OOP (commits 57cae57, 6ed3738, 7525bc4).
