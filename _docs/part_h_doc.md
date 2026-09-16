### Part H — Training and Testing Preparation

The guide states the requirement precisely:

> Partition the data (e.g., 80% Training, 20% Testing; adjust and justify if
> dataset scale requires another ratio).
>
> **Leakage Control Requirement:** Compute all imputation, scaling, and encoding
> parameters using the **training set only**, then transform the test set using
> those pre-fit values.

That second sentence is the whole point of Part H. Everything else is
bookkeeping.

---

#### H.1 — The order of operations is the entire lesson

There are two possible orders, and only one is correct:

| | Wrong order | Correct order |
| :--- | :--- | :--- |
| **Step 1** | Fit scaler on ALL data | **Split** first |
| **Step 2** | Split | Fit scaler on **train only** |
| **Step 3** | Transform both sets | Transform **both** with those frozen parameters |
| **Result** | Test statistics leak into training | Honest evaluation |

The wrong order is easy to fall into because it is more convenient: you scale
once, split once, and everything looks fine. The damage is invisible — the model
scores *better* than it deserves, and you find out only when it meets real data.

Concretely, fitting a scaler on the full dataset means the mean and standard
deviation are computed partly from test rows. The model then "knows" something
about data it is supposed to be evaluated on. The guide names this explicitly in
§15.6: *"Scaling Before Splitting."*

**The fix is structural, not disciplinary.** We do not rely on remembering to fit
only on train. We use a scikit-learn `Pipeline`, which makes it impossible to fit
on anything but the data passed to `fit()`.

---

#### H.2 — Split ratio: 80/20, stratified

| Consideration | Reasoning |
| :--- | :--- |
| **Ratio** | 80/20 → **3,702** training rows, **926** test rows |
| **Minority rows in test** | ~**94** failures — enough to compute minority-class F1 and PR-AUC meaningfully |
| **Stratification** | Mandatory. At a 9:1 imbalance, an unstratified split can produce a test set with a materially different positive rate, making scores unstable between runs |
| **No separate validation set** | With only 468 failures total, carving out a third split would leave too few minority rows to evaluate on. Cross-validation on the training set serves that purpose instead |

`stratify=y` preserves the ~89.89% / 10.11% balance in both partitions. This
follows directly from the imbalance measured in Part B — it is a consequence of
that number, not a default setting.

---

#### H.3 — What actually needs transforming

Part D decided the encoding strategy; Part H implements it. The split matters here
because the cardinalities are what force the choices:

| Column | Type | Cardinality | Treatment |
| :--- | :--- | :--- | :--- |
| `Company_Prior_Launches`, `Company_Prior_Failure_Rate`, `Launch_Year`, `Month_Sin`, `Month_Cos` | numeric | — | Median impute → **StandardScaler** |
| `Country` | categorical | 22 | One-hot (`min_frequency=10`) |
| `Company` | categorical | 62 | One-hot (`min_frequency=10`) |
| `Rocket` | categorical | **370** | **Frequency encoding** |

**Why `Rocket` gets frequency encoding rather than one-hot.** Part D committed to
this, and the measured numbers confirm it was right. One-hot encoding with
`min_frequency=10` — which already collapses every rocket seen fewer than ten
times — *still* produces **86 columns** for `Rocket` alone:

| Encoding | Columns produced |
| :--- | :--- |
| `Country` one-hot | 12 |
| `Company` one-hot | 29 |
| `Rocket` one-hot | **86** |
| `Rocket` **frequency** | **1** |

Frequency encoding compresses the entire column into a single numeric feature:
each rocket is replaced by how often it appears. That captures "common,
well-established vehicles behave differently from one-off experimental ones"
without exploding the feature space.

---

#### H.4 — Handling categories that appear only in the test set

This is a subtle failure mode worth understanding, because it is common in real
data and easy to miss in coursework.

With an 80/20 split, some categories appear in the test set but never in training:

| Column | Categories present only in test |
| :--- | :--- |
| `Rocket` | **18** |
| `Company` | **1** |
| `Country` | 0 |

A naive encoder crashes on these (`ValueError: Found unknown categories`), because
it was fitted on a vocabulary that does not contain them. This is not a bug in the
data — it is exactly what happens when a new operator or vehicle appears after
your training period.

Two settings handle it correctly:

- `OneHotEncoder(handle_unknown="infrequent_if_exist")` routes unseen categories
  to the "infrequent" bucket instead of raising.
- The `FrequencyEncoder` maps unseen categories to `0.0` rather than `NaN`.

Both mean the test set is transformed using **training-derived vocabulary only**,
which is the required behaviour.

---

#### H.5 — A note on imputation

After Part C, `X` contains **no missing values at all**. Every gap was resolved
during cleaning: `Time` was excluded, `Price` was dropped, and the remaining
columns are complete.

Imputers are still included in the pipeline, for two honest reasons:

1. **Robustness.** If the pipeline is later applied to fresh data with gaps, it
   will not break.
2. **Correctness of the contract.** The guide requires imputation parameters to be
   fit on train only. Wiring the imputer in now means it is fitted on train even
   when it has nothing to do.

We document that they are **idle** rather than pretending they do work.
