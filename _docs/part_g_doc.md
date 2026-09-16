### Part G — Feature and Target Separation

The guide gives the pattern directly:

```python
X = df.drop(columns=["target_variable_name"])
y = df["target_variable_name"]
```

The code is trivial. The actual work is deciding **what belongs in `X`** and being
able to justify every exclusion — because a leaked or redundant column silently
degrades the model rather than raising an error.

---

#### G.1 — The separation decision

Everything in the cleaned dataset falls into exactly one of three buckets:

| Bucket | Columns | Count |
| :--- | :--- | :--- |
| **Target (`y`)** | `MissionStatus` → binarised | 1 |
| **Features (`X`)** | `Company`, `Rocket`, `Country`, `Company_Prior_Launches`, `Company_Prior_Failure_Rate`, `Launch_Year`, `Month_Sin`, `Month_Cos` | 8 |
| **Excluded** | `Mission`, `Time`, `RocketStatus`, `Date`, `Company_Prior_Failures` | 5 |

---

#### G.2 — Justifying every exclusion

Each excluded column is excluded for a specific, documented reason. This table is
the defence:

| Excluded column | Reason | Established in |
| :--- | :--- | :--- |
| `MissionStatus` | **It is the target.** Training on it would be circular. | Part B |
| `Mission` | 98.4% unique identifier — cannot generalise. Also carries a latent name-based leak (`"test"` names fail 52.2%). | Parts A, F |
| `Time` | Verified flat: failure rate 7.6%–10.2% across hour buckets. No signal, and 127 missing values. | Parts C, F |
| `RocketStatus` | **Severe temporal leakage.** Zero of 1,837 pre-1980 launches are `Active`; reverse causation (Active 93.2% vs Retired 89.0% success). | Part F |
| `Date` | The raw timestamp is **superseded** by the derived calendar features (`Launch_Year`, `Month_Sin`, `Month_Cos`). Keeping both would duplicate information. | Part D |
| `Company_Prior_Failures` | Near-redundant: correlates **0.952** with `Company_Prior_Launches`. | Part E |

Two of these deserve emphasis because they are the ones a marker would question:

**Why drop `Date` when it is clearly informative?** Because its information is
already captured. `Launch_Year` is the era signal, and `Month_Sin`/`Month_Cos`
are the seasonal signal. A raw `datetime64` column cannot be consumed by a tree
anyway, and leaving it in would duplicate what the engineered features express.
This is the "avoid over-processing" principle from the guide applied in reverse:
we do not keep a redundant column just because it was in the original file.

**Why drop `Company_Prior_Failures` when Part E built it?** Because Part E
measured it as redundant (0.952 correlation) and rejected it there. Part G simply
honours that decision. Keeping a column that duplicates `Company_Prior_Launches`
adds a dimension without adding information, and correlated inputs make a tree's
feature importances harder to interpret.

---

#### G.3 — The target vector

`MissionStatus` has four levels, and Part B decided to collapse it to binary:

| Original label | Count | Share | Binary `y` |
| :--- | :--- | :--- | :--- |
| `Success` | 4,161 | 89.89% | 1 |
| `Failure` | 357 | 7.71% | 0 |
| `Partial Failure` | 107 | 2.31% | 0 |
| `Prelaunch Failure` | **4** | 0.09% | 0 |

**Why binarise?** `Prelaunch Failure` has only **4 observations**. A class that
small cannot be stratified reliably and cannot be learned. Collapsing to
`Success` vs `Not-Success` preserves the business question — *did the mission
succeed?* — while making the split viable.

**We keep the original labels too.** The 4-level text is retained as
`outcome_labels` for reference and for the Part I summary, even though it is not
used as a target.

---

#### G.4 — Scale check

| Quantity | Value |
| :--- | :--- |
| Observations | 4,628 |
| Features in `X` | 8 |
| Categorical features | 3 (`Company`, `Rocket`, `Country`) |
| Numeric features | 5 (`Company_Prior_Launches`, `Company_Prior_Failure_Rate`, `Launch_Year`, `Month_Sin`, `Month_Cos`) |
| Positive rate | 89.89% |
| **Ratio of rows to features** | **579 : 1** |

That row-to-feature ratio is healthy. For contrast, one-hot encoding `Rocket`
alone would have added 370 columns and dropped the ratio to about 12:1 — which is
precisely the justification for frequency encoding established in Part D.
