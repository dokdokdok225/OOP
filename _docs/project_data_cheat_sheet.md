# Space Missions Project — Data Flow Cheat Sheet

This document is a quick reference for the complete journey from the original CSV
to the final machine-learning arrays used by Activity 2.

> **Short version:**
> `space_missions.csv` → cleaned table → engineered features → separated `X`/`y`
> → train/test split → leakage-safe transformation → `.npy` arrays → baseline model.

---

## 1. Project files at a glance

| File/path | Purpose | Use in Activity 2? |
|---|---|---:|
| `space_missions.csv` | Original raw Space Missions dataset | Reference/raw source |
| `oop.ipynb` | Main Activity 1 preprocessing notebook | Reference |
| `oop_refinement.ipynb` | Refined Activity 1 notebook | Reference |
| `X_train.npy` | Transformed training feature matrix | **Load** |
| `X_test.npy` | Transformed held-out test feature matrix | **Load** |
| `y_train.npy` | Training target vector | **Load** |
| `y_test.npy` | Held-out test target vector | **Load** |
| `feature_names.txt` | Names of the 46 final columns in `X_train`/`X_test` | **Load** |
| `activity_2_baseline_model.ipynb` | Activity 2 implementation notebook | **Run/submit** |
| `_docs/project_data_cheat_sheet.md` | This data-flow reference | **Reference** |
| `_docs/model_fit_research.md` | Research on suitable future models | Reference/future work |

---

## 2. Original raw dataset

**File:** `space_missions.csv`

**Original size:** 4,630 rows × 9 columns

Each row represents one space mission / launch attempt.

| Original column | Meaning | Final treatment |
|---|---|---|
| `Company` | Organization conducting the launch | Kept; one-hot encoded |
| `Location` | Full launch-site description | Used to extract `Country`, then dropped |
| `Date` | Launch date | Used to derive year/month features, then dropped |
| `Time` | Launch time | Dropped; missing values and weak signal |
| `Rocket` | Rocket family/name | Kept; frequency encoded |
| `Mission` | Mission name | Dropped; approximately 98.4% unique and identifier-like |
| `RocketStatus` | Rocket status such as Active/Retired | Dropped; temporal leakage risk |
| `Price` | Launch cost | Dropped; approximately 72.68% missing |
| `MissionStatus` | Original mission result | Becomes target `y` |

### Raw target values

The original `MissionStatus` has four labels:

| Original value | Meaning | Binary target |
|---|---|---:|
| `Success` | Mission succeeded | `1` |
| `Failure` | Mission failed | `0` |
| `Partial Failure` | Mission partly failed | `0` |
| `Prelaunch Failure` | Mission failed before launch | `0` |

The four-level target is collapsed because `Prelaunch Failure` has only a very small
number of observations. The Activity 2 question is therefore:

> **Will this mission be successful or not successful?**

---

## 3. Cleaning stage

Cleaning happens before model preparation.

### Main cleaning actions

1. Work on a copy so the raw CSV remains preserved.
2. Remove confirmed duplicate records.
3. Parse `Date` as a date type.
4. Extract the final location component as `Country`.
5. Convert date information into usable calendar fields.
6. Drop `Price` because most values are missing.
7. Drop `Time` because it contains missing values and is not used as a model input.
8. Keep the original `MissionStatus` labels long enough to create the binary target.
9. Remove columns that are identifiers, redundant, or leakage risks.

### Important distinction

Cleaning does **not** mean scaling or encoding. Cleaning makes the table valid and
usable. Model preparation later converts the cleaned columns into the numerical matrix
required by scikit-learn.

---

## 4. Feature and target separation

After cleaning and feature engineering, the data is separated as:

```python
X = input features
 y = target labels
```

### `y`: target vector

```text
y = MissionStatus collapsed to binary
```

| Encoded value | Label | Meaning | Approximate count |
|---:|---|---|---:|
| `1` | Success | Mission succeeded | 4,160 |
| `0` | Not-Success | Failure, Partial Failure, or Prelaunch Failure | 468 |

The target is **binary classification**, not regression.

### `X`: eight conceptual source features

Before encoding, `X` contains these eight source features:

| Source feature | Type | Why it is used |
|---|---|---|
| `Company` | Categorical | Organization conducting the mission |
| `Country` | Categorical | Broad geographic launch context |
| `Rocket` | Categorical/high-cardinality | Rocket family information |
| `Launch_Year` | Numeric | Technology/industry era |
| `Month_Sin` | Numeric | Cyclic month representation |
| `Month_Cos` | Numeric | Cyclic month representation |
| `Company_Prior_Launches` | Numeric/engineered | Company experience before current mission |
| `Company_Prior_Failure_Rate` | Numeric/engineered | Historical company reliability before current mission |

---

## 5. Engineered features

### 5.1 Calendar features

The raw `Date` is not sent directly to the model.

| Raw source | Engineered feature | Explanation |
|---|---|---|
| `Date` year | `Launch_Year` | Captures the launch era and industry maturity |
| `Date` month | `Month_Sin` | Represents one side of the month cycle |
| `Date` month | `Month_Cos` | Represents the other side of the month cycle |

Sine and cosine are used because December and January are adjacent in a calendar,
although the numbers 12 and 1 are far apart numerically.

### 5.2 Company-history features

These features use only missions that occurred **before the current mission**.

| Engineered feature | Definition | Interpretation |
|---|---|---|
| `Company_Prior_Launches` | Number of earlier launches by the same company | Company experience |
| `Company_Prior_Failure_Rate` | Prior failures ÷ prior launches | Historical company reliability |

The current mission outcome is excluded from its own history. Otherwise, the feature
would leak the answer into the input data.

`Company_Prior_Failures` was created during exploration but was not retained because
it was highly correlated with `Company_Prior_Launches` and added mostly duplicate
information.

---

## 6. Encoding and scaling map

The conceptual eight source features become 46 numerical columns.

| Source column(s) | Transformation | Final result |
|---|---|---:|
| `Launch_Year`, `Month_Sin`, `Month_Cos` | Median imputation → `StandardScaler` | 3 numeric columns |
| `Company_Prior_Launches`, `Company_Prior_Failure_Rate` | Median imputation → `StandardScaler` | 2 numeric columns |
| `Company` | One-hot encoding; rare categories grouped | 29 columns |
| `Country` | One-hot encoding; rare categories grouped | 12 columns |
| `Rocket` | Frequency encoding using training frequencies | 1 column |
| **Total** | — | **46 columns** |

### Why frequency encode `Rocket`?

`Rocket` has high cardinality. One-hot encoding it would create many columns. Frequency
encoding replaces each rocket value with how common it is in the training data, keeping
one numerical column instead of creating a large sparse block.

### Unknown categories

Categories that occur in the test set but were not present in training are handled
without refitting on the test set:

- unseen one-hot categories go to the infrequent bucket;
- unseen rocket values receive frequency `0.0`.

---

## 7. Leakage-safe train/test flow

The correct order is:

```text
1. Clean and engineer the dataset
2. Separate X and y
3. Split X/y into train and test
4. Fit imputers, scalers, and encoders on X_train only
5. Transform X_train using fitted training parameters
6. Transform X_test using the same frozen training parameters
7. Save the resulting arrays
8. Train Activity 2 models on X_train/y_train
9. Evaluate once on X_test/y_test
```

### Current split

| Array | Shape | Purpose |
|---|---:|---|
| `X_train` | `(3702, 46)` | Training features |
| `X_test` | `(926, 46)` | Held-out test features |
| `y_train` | `(3702,)` | Training labels |
| `y_test` | `(926,)` | Held-out test labels |

The split is:

- 80% training / 20% testing;
- stratified to preserve the approximately 90/10 Success/Not-Success ratio;
- reproducible with `random_state=42`.

### What “fit on training only” means

The training set learns:

- imputation values;
- scaling means and standard deviations;
- one-hot category vocabulary;
- rocket-frequency values.

The test set is only transformed using those already-learned values. It must not
calculate its own statistics or update the encoder vocabulary.

---

## 8. Final saved artifacts

### `X_train.npy` and `X_test.npy`

These contain only numerical feature values. They do **not** contain the target.

```python
X_train.shape  # (3702, 46)
X_test.shape   # (926, 46)
```

### `y_train.npy` and `y_test.npy`

These contain only the binary target labels.

```python
y_train.shape  # (3702,)
y_test.shape   # (926,)
```

The encoding is:

```python
1 = Success
0 = Not-Success
```

### `feature_names.txt`

This lists the 46 columns in the same order as the columns in both feature matrices.
The order must not be changed when interpreting feature importance or error cases.

---

## 9. Activity 2 model inputs

Activity 2 does **not** redo cleaning, engineering, splitting, scaling, or encoding.
It loads the saved arrays:

```python
import numpy as np

X_train = np.load("X_train.npy")
X_test  = np.load("X_test.npy")
y_train = np.load("y_train.npy")
y_test  = np.load("y_test.npy")
```

Then it compares:

```python
DummyClassifier(strategy="most_frequent")
DecisionTreeClassifier(
    max_depth=5,
    class_weight="balanced",
    random_state=42,
)
```

The main evaluation target is:

```python
f1_score(y_test, predictions, pos_label=0)
```

That evaluates the **Not-Success** class, which is the minority and more important
class for failure detection.

---

## 10. Columns deliberately excluded from `X`

| Excluded column | Reason |
|---|---|
| `MissionStatus` | It is the target; including it would be circular |
| `Mission` | Mostly unique identifier; poor generalization and possible name leakage |
| `Time` | Missing values and weak demonstrated signal |
| `RocketStatus` | Temporal leakage risk; status may reflect later information |
| `Date` | Replaced by `Launch_Year`, `Month_Sin`, and `Month_Cos` |
| `Price` | Too many missing values to serve as a reliable feature |
| `Company_Prior_Failures` | Near-redundant with prior launches and not retained |
| `Location` | Replaced by the simpler `Country` feature |

---

## 11. One-minute explanation for Q&A

> The original CSV contains raw launch information and the final mission outcome. We
> cleaned the data without changing the raw file, extracted country and calendar
> information, and built two historical company features using only earlier launches.
> We then separated the binary target `y` from the input features `X`. The categorical
> company and country fields were one-hot encoded, the high-cardinality rocket field
> was frequency encoded, and the numeric fields were standardized. The data was split
> into stratified training and testing sets before any preprocessing was fitted. The
> final saved matrices have 3,702 training rows and 926 test rows, with 46 numerical
> feature columns. Activity 2 loads those arrays and compares a majority-class dummy
> classifier with a class-weighted decision tree.

---

## 12. Important limitations

1. The current split is random rather than chronological, even though missions span
   multiple decades.
2. The saved arrays do not contain original CSV row IDs, so Activity 2 error analysis
   identifies test-row positions rather than original mission names.
3. The target is strongly imbalanced, so accuracy alone cannot show whether failures
   are being detected.
4. The Activity 2 decision tree is a baseline, not a production-ready model.
5. Random Forest and HistGradientBoosting are possible future models, not replacements
   for the required Activity 2 baseline.
