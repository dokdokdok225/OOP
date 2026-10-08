# Midterm Document — Screenshot & Copy-Paste Guide

Project: Space Missions classification (`oop.ipynb`) — dataset inspection and
preprocessing, Parts A–H.
Target docx: `Midterm Project Document Template (2).docx`

How to use:
- **Screenshot** = open `oop.ipynb` in Jupyter, run all cells first
  (`python execute_notebook.py` or Kernel → Restart & Run All), then screenshot
  the listed cell number with its OUTPUT visible. Crop to the cell, not the
  whole browser.
- **Paste to docx** = text you can type/paste directly into the document body.

---

## 1. Project Overview

Copy to docx:

> This project performs end-to-end dataset inspection and preprocessing of the
> Space Missions dataset (4,630 launches, 9 original features) to prepare a
> machine-learning-ready classification matrix for predicting launch success.
> The work is organized as Parts A–H in a single Jupyter notebook: dataset
> inspection (A), problem definition (B), data cleaning (C), feature
> transformation (D), feature engineering (E), data-leakage auditing (F),
> feature/target separation (G), and train/test preparation with leak-safe
> preprocessing (H). The final output is a saved set of model-ready NumPy
> arrays (`X_train.npy`, `X_test.npy`, `y_train.npy`, `y_test.npy` and
> `feature_names.txt`).
>
> Limitation: `Price` was 72.68% missing and was dropped rather than imputed,
> and preprocessing deliberately excludes post-launch information to avoid
> target leakage.

---

## 2. Data and Project Inputs

Paste to docx (adapt course/track fields):

> Input: `space_missions.csv` — 4,630 rows × 9 columns. Columns: Company,
> Location, Date, Time, Rocket, Mission, RocketStatus, MissionStatus, Price.
> The dataset spans launches from 1957-10-04 to 2022-07-29. Missing values:
> Price 72.68%, Time 2.72%; 1 exact duplicate row. The raw file is never
> written to — all cleaning happens on a working copy (`df`).

Screenshots:

| # | Cell | What it shows | Caption suggestion |
|---|------|---------------|--------------------|
| 1 | Cell 2 | Raw dimensions (4,630 × 9), missing-value counts, duplicate check | "Figure X: Raw dataset inspection (Part A)" |
| 2 | Cell 18 | Cleaned dataset preview (`df.head()`) + final column types | "Figure X: Cleaned dataset preview" |
| 3 | Cell 15 | Cleaning summary: 2 rows removed, 9 → 8 columns | "Figure X: Cleaning summary and integrity checks" |

---

## 3. Project Methodology and Development Process

### Phase 1 — Preparation and Foundation (Parts A–C: inspection & cleaning)

Paste to docx:

> Phase 1 inspected the raw CSV and cleaned it using eight documented rules:
> Date parsed to real datetime (0 unparseable, range 1957–2022), 1 exact
> duplicate removed (4,630 → 4,629), 1 business duplicate removed requiring a
> missing Time as the duplicate signature (→ 4,628), Price repaired from
> text-to-float with 0 values lost, Price column dropped (72.68% missing,
> above the 50–70% drop threshold), Country extracted from Location (22
> distinct countries), and Time deliberately left unimputed and excluded from
> features.

Screenshots:

| # | Cell | Shows | Caption |
|---|------|-------|---------|
| 4 | Cell 7 | Exact duplicate found and removed | "Duplicate removal (Rule 1)" |
| 5 | Cell 10 | Price text→float repair, 0 values lost | "Price repair (Rule 4)" |
| 6 | Cell 11 | Price dropped at 72.68% missing | "Column drop decision (Rule 5)" |
| 7 | Cell 17 | Missingness % chart / discussion | "Missing-value threshold analysis" |
| 8 | Cell 20 | Cleaning visual before/after | "Visual: what cleaning removed" |

### Phase 2 — Development and Analysis (Parts D–F: transformation, engineering, leakage)

Paste to docx:

> Phase 2 created the feature space. Part D derived Launch_Year and cyclical
> month encoding (Month_Sin / Month_Cos, verified so December–January measure
> as adjacent), demonstrated why scaling matters for distance-based models but
> not for trees, and showed one-hot encoding of Rocket (370 distinct values →
> 370 columns) is infeasible, motivating frequency encoding.
>
> Part E implemented OOP feature builders — an abstract `FeatureBuilder` base
> class (ABC, cannot instantiate without implementing the interface) and a
> concrete `OperationalExperienceBuilder` producing `Company_Prior_Launches`
> and `Company_Prior_Failure_Rate`. Leak-free experience raises F1 vs a leaky
> version found to inflate F1 by +49.6%.
>
> Part F audited leakage with a `LeakageGuard` class and proved three latent
> leaks in the raw columns: RocketStatus is temporally encoded (all 1,837
> pre-1980 launches are 'Retired'), Mission names encode outcome ('test'
> missions), and Price is recorded retroactively (33/774 priced in the 1960s
> vs 528/676 in the 2010s).

Screenshots:

| # | Cell | Shows | Caption |
|---|------|-------|---------|
| 9 | Cell 25 | New time features created | "Date/time feature extraction (D.2)" |
| 10 | Cell 28 | Cyclical encoding chart | "Visual: sin/cos month encoding vs naive integer" |
| 11 | Cell 37 | One-hot infeasibility evidence | "Evidence: Rocket must not be one-hot encoded" |
| 12 | Cell 44 | Concrete FeatureBuilder applied | "OperationalExperienceBuilder applied" |
| 13 | Cell 46 | Leaky feature inflates F1 by +49.6% | "Proof leakage would mislead the model" |
| 14 | Cells 42–43 | `FeatureBuilder` ABC + concrete class code | "OOP: abstract base class and concrete builder" |
| 15 | Cell 56 | `LeakageGuard` class code | "OOP: LeakageGuard class" |
| 16 | Cell 59 | RocketStatus temporal leak evidence | "Leakage evidence 1: RocketStatus" |
| 17 | Cell 61 | Mission-name leak evidence | "Leakage evidence 2: Mission names" |
| 18 | Cell 68 | Part F summary chart (3 leak findings) | "Visual: the three leak findings" |

### Phase 3 — Final Technical Development (Parts G–H)

Paste to docx:

> Phase 3 separated the target (Part G): y has 4,628 values with a 89.89%
> positive rate (8.89:1 imbalance, hence Part B rejected accuracy as the
> metric); X has 4,628 rows × 8 columns, all 7 excluded columns
> leak-verified. Part H used stratified train/test split (80/20 →
> 3,702/926) performed BEFORE any fitting, then a leak-safe sklearn
> `ColumnTransformer`+`Pipeline` (imputers, StandardScaler on 5 numeric,
> one-hot on 2 low-cardinality, custom `FrequencyEncoder` on Rocket) fitted
> on training data only. Four leakage checks passed (scaler stats from train
> only; unseen rocket categories routed to 'infrequent'; encoder vocabulary
> learned from train; integer feature counts pass 100:1). Final artifacts
> saved: `X_train.npy` (3,702 × 46), `X_test.npy` (926 × 46),
> `y_train.npy`, `y_test.npy`, `feature_names.txt`.

Screenshots:

| # | Cell | Shows | Caption |
|---|------|-------|---------|
| 19 | Cell 71 | X/y separation summary | "Feature/target separation (Part G)" |
| 20 | Cell 72 | Target distribution, 89.89% positive | "Class imbalance in the target" |
| 21 | Cell 74 | LeakageGuard PASS on X | "Guard verdict: no leakage in X" |
| 22 | Cell 87 | Pipeline definition code | "Leak-safe preprocessing pipeline (H.8)" |
| 23 | Cell 88 | Fit on train only, shapes (3702,46)/(926,46) | "Pipeline fitted on training data only" |
| 24 | Cells 90 + 91 + 92 | The four leakage checks (may crop into 2 shots) | "Leakage verification checks H.10a–d" |
| 25 | Cell 96 | Persistence check PASS, rows preserved end-to-end | "Before/after validation" |
| 26 | Cell 97 | Saved .npy artifacts confirmation | "Saved model-ready artifacts" |
| 27 | Cell 98 | Split composition chart | "Visual: stratified split composition" |
| 28 | Cell 100 | Feature-origins chart (46 features from 8 columns) | "Visual: where the 46 features come from" |

---

## 4. Final Project Output

Paste to docx:

> The final output is the completed notebook `oop.ipynb` (Parts A–H) and its
> saved model-ready artifacts: `X_train.npy`, `X_test.npy`, `y_train.npy`,
> `y_test.npy`, and `feature_names.txt` — a 46-feature, leakage-free,
> train/test-split classification matrix ready for any scikit-learn model.

Screenshots:

| # | What | Caption |
|---|------|---------|
| 29 | Final state of the notebook: full file listing in Jupyter or the last cells run successfully | "Figure X: Final delivered notebook and artifacts" |
| 30 | Folder/file listing in your repo (VS Code/File Explorer) |
| 31 | Terminal proof of a headless run: `python execute_notebook.py` (output: no errors, saved lines) | "Headless execution confirming end-to-end reproducibility" |

---

## 5. Results and Key Findings

Paste to docx (bullets):

> - Cleaned data: 4,630 → 4,628 rows, 9 → 8 columns (+Country), 0 lost values.
> - Final feature matrix: 46 engineered features from 8 source columns
>   (5 numeric, 40 one-hot, 1 frequency-encoded); rows:features = 100:1.
> - Class imbalance: 89.89% success rate → 8.89:1 ratio; accuracy would be
>   misleading as a metric (Part B finding).
> - Frequency encoding on Rocket saved ~324 columns versus one-hot (370
>   distinct rockets).
> - Leakage: three post-launch/information-leak channels found and excluded —
>   RocketStatus (temporal proxy), Mission names (outcome naming), Price
>   (retroactive availability). A leaky experience feature inflated F1 by
>   +49.6% versus the honest version.
> - Stratified split preserved class balance: positive rates 0.899/0.899/0.899
>   across full/train/test.
> - First launches fail at 41.9% vs 5.1% after 500+ cumulative launches — an
>   8.2× experience effect motivating the engineered features.

Screenshots:

| # | Cell | Shows | Caption |
|---|------|-------|---------|
| 32 | Cell 15 | Cleaning summary | "Result: cleaning summary" |
| 33 | Cell 47 | 41.9% vs 5.1% failure-rate chart | "Finding: operational experience effect" |
| 34 | Cell 51 | Leak-free vs leaky model comparison chart | "Finding: leakage inflates performance by 50%" |
| 35 | Cell 77 | Class balance chart | "Finding: 89.89% positive class" |
| 36 | Cell 101 | Part H summary with saved artifacts | "Result: pipeline output summary" |

---

## 6. Discussion

Paste to docx (short paragraph):

> The pipeline's behavior is consistent with its purpose: preprocessing is
> fitted on training data only, and every excluded column has a documented,
> evidence-backed reason. The results show why leakage matters dataset-wide —
> several raw columns (RocketStatus, Mission, Price) look informative but
> would not be computable at predict-time, which would have made any model
> trained on them unusable. The engineered experience features are the most
> informative honest signal discovered. The 89.89% class imbalance means
> future evaluation should use F1/precision-recall, not accuracy — a decision
> carried through Part B.

---

## 7. Conclusion

Paste to docx:

> All Phases 1–3 objectives were achieved: the raw CSV was inspected and
> documented; three latent leakage vectors were identified and proven
> quantitative; `FeatureBuilder` and `LeakageGuard` classes exercise
> inheritance, interfaces, and validation (OOP rules); a leak-safe sklearn
> pipeline produced and persisted a clean, split, 46-feature model-ready
> dataset with zero values lost and rows fully preserved end-to-end.

---

## 8. Recommendations and Future Improvements

Paste to docx (bullets):

> - Train and evaluate actual classifiers (logistic regression, random
>   forest, gradient boosting) on the saved arrays; report F1, not accuracy.
> - Address the 8.9:1 imbalance (class weights, SMOTE, or threshold tuning).
> - Substitute or impute a sparse, era-independent proxy for Price (e.g.
>   rocket-class tiers) instead of dropping it.
> - Tune the FrequencyEncoder / 'infrequent' bucket threshold on validation
>   folds.
> - Extend the OOP layer with additional concrete builders (e.g. a country
>   history builder) behind the same abstract interface.

---

## 9. References

Paste to docx:

> - Space Missions dataset (Kaggle, `space_missions.csv`, 4,630 launches × 9
>   features).
> - Scikit-learn documentation: ColumnTransformer, Pipeline, StandardScaler,
>   train_test_split, OneHotEncoder. https://scikit-learn.org/stable/
> - Python `abc` module documentation (abstract base classes).
>   https://docs.python.org/3/library/abc.html
> - Course lecture notes: OOP rules and architecture reference
>   (`oop_rules.md`); Data preprocessing notes and activity guide.
> - Course repository: https://github.com/dokdokdok225/OOP

---

## 10. Appendices (supporting screenshots NOT in the body)

| # | What to screenshot | Purpose |
|---|--------------------|---------|
| A1 | Cell 42 — `FeatureBuilder` ABC full source | Code snippet: OOP abstract class |
| A2 | Cell 43 — `OperationalExperienceBuilder` full source | Code snippet: concrete class |
| A3 | Cell 56 — `LeakageGuard` full source | Code snippet: validation class |
| A4 | Cell 82 — `FrequencyEncoder` full source | Code snippet: custom transformer |
| A5 | Cell 87 — ColumnTransformer/Pipeline full source | Code snippet: preprocessing pipeline |
| A6 | Cell 92 — FrequencyEncoder vocabulary evidence | Encoder train/test vocabulary proof |
| A7 | Cell 100 — feature origins breakdown output | Where 46 features came from |
| A8 | GitHub repo page (browser) showing commit history | Evidence of development process across phases |
| A9 | Repo file tree (VS Code/Explorer) with `X_train.npy`, `feature_names.txt` visible | Saved artifacts proof |

---

## Quick checklist (do in this order)

1. `python execute_notebook.py` — ensure 0 errors; capture terminal output
   (screenshot #31).
2. Open `oop.ipynb` in Jupyter; confirm all cells show outputs.
3. Screenshot cells in the tables above, in order; save as
   `fig_01_raw_inspection.png` … `fig_36_pipeline_summary.png`.
4. Paste the prose blocks into the matching docx sections.
5. Match each screenshot to its caption from the table.
