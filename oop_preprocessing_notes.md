# OOP Preprocessing — Study Notes

**Course:** DSC 2135 – Object Oriented Programming
**Institution:** Ateneo de Davao University
**Project:** Phase 1 Data Preprocessing — *All Space Missions from 1957*
**Motto:** *Fortes-in-Fide*

---

## How to use this file

One section per activity part. Each section has the same shape:

| Subsection | Purpose |
| :--- | :--- |
| **What the guide asks** | The literal requirement, quoted from the activity guide. |
| **The concept** | The idea explained plainly. |
| **What we did** | What goes in the notebook. |
| **Why** | The justification, in your words. |
| **Defend it** | The sentence to say if a panel asks. |

---

# Table of Contents

1. [Part A — Dataset Inspection & Profiling](#part-a)
2. [Part B — Define the Machine-Learning Problem](#part-b)
3. [Part C — Data Cleaning](#part-c) *(pending)*
4. [Part D — Feature Transformation](#part-d) *(pending)*
5. [Part E — Feature Engineering](#part-e) *(pending)*
6. [Part F — Check for Data Leakage](#part-f) *(pending)*
7. [Part G — Feature and Target Separation](#part-g) *(pending)*
8. [Part H — Training and Testing Preparation](#part-h) *(pending)*
9. [Part I — Preprocessing Summary Table](#part-i) *(pending)*

---
---

# Part A — Dataset Inspection & Profiling

*Complete. Summarised here so Part B has a starting point.*

## What the guide asks

Determine dimensions; list columns and data types; classify variables as
numerical or categorical; quantify missing values and duplicates; identify
outliers and invalid values; identify a candidate target (`y`); identify
candidate features (`X`).

## Evidence — what profiling found

| Finding | Value | Consequence |
| :--- | :--- | :--- |
| Dimensions | 4,630 rows × 9 columns | ~926 rows available for testing at an 80/20 split |
| All dtypes | `object` (text) | Every field needs deliberate parsing |
| `Price` missing | 3,365 rows (**72.68%**) | Above the guide's 50–70% drop threshold |
| `Time` missing | 127 rows (2.74%) | Low, but see Part B |
| Exact duplicates | **1** row | Remove during cleaning |
| `Mission` unique | 4,556 of 4,630 (**98.4%**) | An identifier, not a feature |
| `Rocket` unique | 370 | Too many for one-hot encoding |
| Target balance | Success **89.89%** vs not-success **10.11%** | Imbalanced — accuracy is misleading |
| Date span | 1957-10-04 → 2022-07-29 | 65 years; strong era effect likely |

## The most important Part A finding

Every column loaded as `object`. Pandas cannot know that `"1957-10-04"` is a
date or that `"1,160.00"` is a number, so it falls back to text. Three
consequences run through the whole project:

1. **No arithmetic** — `df["Price"].sum()` would concatenate strings, not add numbers.
2. **No chronology** — text dates sort alphabetically, not chronologically.
3. **Missing values stay masked** — a `0` might mean "free" or "unknown."

> **Habit to build:** after every type conversion, re-count the missing values.
> If the count changes, the conversion did something you did not intend.

---
---

# Part B — Define the Machine-Learning Problem

## What the guide asks

> Identify the most suitable modeling framework for the dataset:
> **Classification** (predicting discrete class labels), **Regression**
> (predicting a continuous numeric quantity), **Clustering** (unsupervised
> grouping of unlabelled records), or **Other Analytical Task**.
> *Justification:* Explain why the dataset attributes fit this problem type.

**This is a decision + justification. It is not a modelling exercise.** No model
is trained in Part B.

## The concept

The four options are not interchangeable — they are decided by **what kind of
column you are trying to predict, and whether that column exists at all.**

| Framework | You predict | The target column is… |
| :--- | :--- | :--- |
| **Regression** | A number | Continuous (price, temperature, duration) |
| **Classification** | A category | Discrete and **already labelled** in your data |
| **Clustering** | Nothing — you find groups | **Absent.** No label exists. |
| **Other** | Anomalies, sequences, rules | Varies |

The deciding question for our dataset: **do we have a labelled outcome column?**

We do — `MissionStatus` records the actual result of each launch. That is a
discrete label with four levels. So **classification** is available.

But our dataset also supports an exploratory **clustering** step first, and the
reason is worth understanding rather than skipping.

### Why clustering can come before classification

Supervised learning can only exploit differences **you already suspect**. If you
hand a classifier all the columns and ask it to predict success, it will find
whatever correlates with the outcome — including columns that leak the answer
(see Part F). It will not tell you *how missions naturally group*.

Clustering never sees the target. It groups records purely by similarity. That
makes it a genuinely independent check: if the groups it finds turn out to differ
in failure rate, then the structure is real rather than an artefact of the label.

This maps onto the guide's analysis types (§2):

- **Exploratory** — *"What might be hidden?"* → preparation focus is flexible
  inspection, and *"without aggressively deleting context or unusual points too early."*
- **Predictive** — *"What is likely next?"* → preparation focus is a clean target,
  a disciplined train/test split, and strict leakage prevention.

Our project does exploratory first, then predictive.

### Why K-Means, and where a decision tree fits

This part of the justification matters, because the two stages use different
algorithms for different reasons.

**K-Means for the clustering stage.** It is the standard choice when you want to
group records by similarity across numeric features, and when you do not know the
groups in advance. It works by repeatedly assigning each record to its nearest
cluster centre and then recomputing those centres. Two properties drive the choice:

- It needs **numeric, scaled** features, because it measures Euclidean distance.
  That is why Part D's scaling decision is not optional — it is a precondition.
- It requires you to **choose the number of clusters (k)** in advance, which is
  why we compare several values of k rather than assuming one.

**Decision tree for the classification stage.** A tree splits the data with
if/else rules on individual features. It suits our problem well because:

- It handles **mixed feature types** — numeric and categorical — without needing
  every feature to be scaled or one-hot encoded.
- It is **interpretable**. Each path from root to leaf reads as a plain rule
  ("operator had fewer than 20 prior launches AND launch year before 1975 →
  higher failure risk"), which is exactly the kind of explanation this domain wants.
- It captures **non-linear relationships and interactions** — for example, that
  failure risk depends on era *and* operator maturity together, not on either alone.
- It is **robust to irrelevant features**, since it simply will not split on a
  column that adds no separation.

**Why this pairing is sensible.** The clustering stage tells us *which structural
factors* appear to matter. The tree stage then tests whether those factors
actually predict the outcome on held-out data, and gives us a readable rule set
if they do. The tree is not chosen because it is the most powerful classifier
available — it is chosen because it is the most **defensible and explainable** one
for a risk question.

## What goes in the notebook

A short markdown section stating:
1. The framework: clustering (exploratory) → classification (predictive).
2. The target: `MissionStatus`, binarised.
3. The features: which columns and why.
4. The justification, including why K-Means and why a tree.

## Why — your wording

*(to be filled in — this is where your own phrasing goes)*

## Defend it

*(to be filled in)*

---
---

# Part C — Data Cleaning

## What the guide asks

> Inspect and handle data quality issues using explicit, rule-based logic:
> missing values, duplicate observations (**exact vs. business**), incorrect
> storage types, inconsistent categorical entries, impossible domain entries,
> and outliers.

## The concept

The single idea that makes Part C hard is the difference between an **exact
duplicate** and a **business duplicate**.

| | Exact duplicate | Business duplicate |
| :--- | :--- | :--- |
| **Definition** | Every column is identical | Different in some columns, but the *same real-world event* |
| **How to detect** | `df.duplicated()` | Requires **domain logic** |
| **Risk** | Almost none — obviously redundant | **High** — a bad rule deletes real data |

Our dataset contains both, and the second one is the interesting case.

### The trap we avoided

The obvious rule is *"same company + same rocket + same date = duplicate."*
That rule is **wrong for this dataset**, and it would have silently deleted 6
real launches.

Six pairs share Company + Rocket + Date. But only **one** is a true duplicate:

| Pair | Times | Payloads | Verdict |
| :--- | :--- | :--- | :--- |
| US Air Force 1968-08-06 | 16:33 and 18:30 | `OPS 5187` / `OPS 5247` | **Two launches** |
| RVSN USSR 1976-07-27 | 05:21 and 12:00 | `Cosmos 845` / `Intercosmos-16` | **Two launches** |
| MHI 1990-02-07 | 01:33 and 01:33 | `Orizuru…` / `MOS-1B…` | **Two payloads, one launch** |
| ExPace 2019-12-07 | 02:55 and 08:52 | `Jilin-1…` / `HEAD-2A/B…` | **Two launches** |
| **ISRO 2012-04-26** | **00:17 and *missing*** | `RISAT-1` / `PSLV- C19 RISAT-1` | **DUPLICATE** |

Look at the ISRO row. Its mission name `"PSLV- C19 RISAT-1"` simply prefixes the
launch vehicle onto the payload name `"RISAT-1"`. It shares the date, rocket,
price (31) and outcome with the row above it, and it is **missing its launch
time**.

That missing time is the fingerprint. The narrow rule we use is:

> Same Company + Rocket + Date + Price + MissionStatus, **and** one row lacks
> a launch Time → the time-less row is a duplicate re-entry.

This rule matches **exactly one row** and leaves all 6 co-launches intact.

## What we did

A **decision register** — a table stating every problem, its evidence, the
decision, and the reason — written *before* the code, so the code only applies
decisions rather than making them.

| # | Problem | Decision |
| :-: | :--- | :--- |
| 1 | 1 exact duplicate | Drop one copy |
| 2 | 1 business duplicate (ISRO) | Drop the time-less row |
| 3 | 6 co-launch pairs | **Keep all** |
| 4 | `Price` as text with commas | Strip `,`, cast to float |
| 5 | `Price` 72.68% missing | Drop the column |
| 6 | `Location` composite string | Split on `,`, take last part as `Country`, **then drop `Location`** |
| 7 | All columns `object` | Parse `Date` → `datetime64` |
| 8 | `Time` 2.74% missing | Leave; exclude from features |
| 9 | 4-level target, one class has 4 rows | Collapse to binary *(done in Part G)* |

### Also important: what we refused to do

- **No outliers deleted.** Extreme values here are valid domain extremes, not errors.
- **No imputation anywhere.** Every missing value is either in a dropped column
  or intentionally excluded.
- **No rows dropped just for being incomplete** — only the one true duplicate.
- **Raw file never modified.**

## A bug worth remembering (rule ordering)

My first version parsed `Date` *after* the other rules. The co-launch verification
calls `dt.date()`, which only exists on a datetime — so it crashed with
`AttributeError: 'str' object has no attribute 'date'`.

The lesson: **rule order is part of the logic, not an implementation detail.**
Because the duplicate rule keys on `Date` and the co-launch check groups by it,
`Date` must be parsed first. That makes the remaining rules order-independent.

## Evidence

```
Raw file shape      : (4630, 9)
Exact duplicates found   : 1
Rows matching the duplicate key : 10
Rows dropped as business dups   : 1
Rows before -> after            : 4629 -> 4628

Rows sharing Company+Rocket+Date: 8
PASS: all 8 co-launch rows preserved.

Non-null Price before parse : 1263
Non-null Price after parse  : 1263
Values lost                 : 0
PASS: every non-null Price value survived as a real number.

Distinct countries extracted: 22
Blank country values: 0

Unparseable dates : 0

Time missing values    : 126 (2.72%)

Final shape        : (4628, 9)
PASS: exactly 2 rows removed, both documented duplicates.
```

**Final shape: 4,628 rows × 9 columns.**

## Why — your wording

*(to be filled in — your own phrasing)*

## Defend it

*(to be filled in)*

---

# Part D — Feature Transformation

## What the guide asks

Apply transformations converting variables into model-ready representations, and
answer one **required discussion question**:

> Explicitly evaluate whether **normalization** or **standardization** is best
> suited for your numerical attributes and target problem, and explain your
> technical justification.

## The concept

### Why month needs cyclical encoding

This is the most counter-intuitive idea in Part D, and it is worth sitting with.

Leave month as a raw integer and you tell the model:

```
January = 1     December = 12
```

So December and January are **11 apart** — the widest gap in the year. But they
are consecutive months. The representation states something false.

The fix is to put each month on a circle:

$$\text{Month\_Sin} = \sin\!\left(\frac{2\pi \cdot m}{12}\right), \qquad
\text{Month\_Cos} = \cos\!\left(\frac{2\pi \cdot m}{12}\right)$$

Both are needed. A single sine value is ambiguous — two different months can share
one sine — so sine *and* cosine together give each month a unique position.

Measured result:

| Pair | Distance in sin/cos space | Correct? |
| :--- | :--- | :--- |
| January ↔ December | **0.5176** | Adjacent |
| January ↔ February | **0.5176** | Adjacent — identical, as it should be |
| January ↔ July | **2.0** | Opposite |

### Normalization vs standardization

The textbook answer is "use standardization for distance-based algorithms." That
is true but incomplete for us, and the incomplete version is worth understanding
because it leads to a lazy justification.

| | Min-Max | Z-Score |
| :--- | :--- | :--- |
| Formula | $\frac{X - X_{min}}{X_{max} - X_{min}}$ | $\frac{X - \mu}{\sigma}$ |
| Range | Bounded $[0,1]$ | Unbounded, ~$[-3,3]$ |
| Outlier sensitivity | High (max dictates everything) | Lower |

**The deciding question is: which algorithm consumes the feature?** And our two
stages answer differently.

#### Stage 1 — K-Means: scaling is mandatory

K-Means uses Euclidean distance, so feature scale determines influence. Our two
clustering features:

- `Launch_Year`: span **65**
- `Company_Prior_Launches`: span **1,776** — **27.3× wider**

Since distance squares each difference, the wider feature dominates:

| Feature | Share of max squared distance (unscaled) |
| :--- | :--- |
| `Company_Prior_Launches` | **99.9%** |
| `Launch_Year` | 0.1% |

Look at the cluster centres:

| Scaling | Year centres | Prior centres |
| :--- | :--- | :--- |
| Unscaled | `1994, 1984, 1973` | `105, 1440, 770` |
| Standardized | `-1.0, 1.0, -0.5` | `0, -1, 2` |

Both rows contain real numbers — which is exactly why this is subtle. The problem
is not that unscaled values are *wrong*, it is **weighting**: a 1,000-launch
difference counts vastly more than a 40-year difference, purely because of units
chosen. The clustering becomes almost entirely about volume.

#### Stage 2 — Decision Tree: scaling is irrelevant

A tree splits on thresholds ("is year ≤ 1972.5?"). Rescaling cannot change the
**order** of values, so every split point remains available and the tree learns
an identical model.

Verified:

| Metric | Raw | Standardized |
| :--- | :--- | :--- |
| `Launch_Year` importance | 0.669 | **0.669** |
| `Company_Prior_Launches` importance | 0.297 | **0.297** |
| `Company_Prior_Failures` importance | 0.034 | **0.034** |
| Cross-validated F1 | 0.2052 | 0.2015 |

Identical importances. F1 differs by −0.004, which is noise.

#### Our decision: Standardization

1. **Stage 1 needs it; Stage 2 is indifferent.** One choice serves both.
2. **The distribution is right-skewed.** `Company_Prior_Launches` spans 0–1,776
   with most values near 0. Min-Max depends entirely on the maximum, so one prolific
   operator would squash everything else into a sliver of $[0,1]$.
3. **Zero-centred, unit-variance is the right assumption for Euclidean methods.**

**When Min-Max would be correct:** bounded inputs — neural networks with
sigmoid/tanh activations, or image pixels in $[0,255]$. Neither applies.

#### Where it is applied

**Not here.** Part D decides *what*; **Part H** applies it inside a
`ColumnTransformer` fitted on the training split only. Fitting a scaler before
splitting lets test statistics influence training parameters — the leakage the
guide warns about in §15.6.

### Categorical encoding

Encoding is driven by **cardinality**, because the wrong choice has a real cost.

| Column | Cardinality | Encoding | Reason |
| :--- | :--- | :--- | :--- |
| `Country` | 22 | One-hot | Nominal, no order; 22 columns manageable |
| `Company` | 62 | One-hot + rare collapse | Nominal; grouping controls width |
| `Rocket` | **370** | **Frequency encoding** | One-hot would add 370 sparse columns |

**Why `Rocket` gets frequency encoding.** 370 categories on 4,628 rows means one
column per 12.5 rows. Worse, **83 rockets appear exactly once** and 123 appear at
most twice — those columns could not generalise at all, and a tree would happily
split on the noise. Frequency encoding compresses it to one numeric column.

**The caveat we accept.** Two different rockets appearing equally often get the
*identical* value, so the distinction is lost. With 370 mostly-singleton categories,
that trade is worth it.

**Ordinal encoding is used nowhere** — the dataset has no genuinely ordered
categorical column, and using ordinal on nominal data invents a false order.

## What we did

| Transformation | Output |
| :--- | :--- |
| Date extraction | `Launch_Year`, `Month_Sin`, `Month_Cos` |
| Scaling | **Decided here, applied in Part H** |
| Encoding | **Strategy decided here, implemented in Part H** |

## What we deliberately did NOT do

- **No log/power transformation.** The guide lists it for features spanning orders
  of magnitude, but after Part C no such continuous feature remains (`Price` was
  dropped). We do not apply a technique just because it appears in a table.
- **No scaling applied to the dataset here** — deferred to Part H to avoid leakage.
- **No feature engineering** — that is Part E, kept separate deliberately.

## Evidence

```
Existing numeric columns: (NONE)
-> There are no numeric features yet. Everything must be derived or encoded.

January <-> December   distance = 0.5176   (adjacent)
January <-> February   distance = 0.5176   (adjacent)
January <-> July       distance = 2.0000   (opposite)
PASS: adjacent months are close, opposite months are far.

The prior-launch count spans 27.3x wider than the year.

Share of maximum squared distance, UNSCALED:
  Company_Prior_Launches :   99.9%
  Launch_Year            :    0.1%

Feature importances identical: 0.669 / 0.297 / 0.034
Tree CV F1 unscaled=0.2052  standardized=0.2015  (-0.0037)

One-hot would add 370 columns (12.5 rows per column)
Rockets appearing once: 83
```

## Why — your wording

*(to be filled in — your own phrasing)*

## Defend it

*(to be filled in)*

---

# Part E — Feature Engineering

## What the guide asks

> Derive at least **one** new feature that adds domain value.
> *Explanation Requirement:* Detail why this derived feature provides stronger
> predictive or analytical signal than the raw attributes alone.

## The concept

### Where the idea comes from

Part B's clustering found a **mid-era cluster with the lowest failure rate of
all three groups (4.2%)** and a median of **1,270 prior launches**. If calendar
year alone drove reliability, that cluster would sit *between* the early and
modern groups. It does not.

Something other than time separates it — and that something is **accumulated
operator experience**. Part E turns that observation into actual features.

### The leakage trap

This is the most important idea in Part E, because getting it wrong looks like
*success*.

The natural way to compute failure history:

```python
df.groupby("Company")["is_fail"].transform("sum")
```

That one line **leaks the target**. It gives every row the operator's *total*
failures across the whole dataset — including the outcome of the mission being
predicted, and of every later mission. At liftoff, none of that exists.

The correct version uses only strictly earlier missions:

```python
prior_launches = group.cumcount()                          # rows before this one
prior_failures = group["is_fail"].cumsum() - own_is_fail   # subtract own row
```

**We measured the difference:**

| Features | Cross-validated F1 |
| :--- | :--- |
| Honest (causal) | **0.1805** |
| Leaky (company totals) | **0.2700** |

The leaky version scores **~50% higher** and is worthless. This is what makes
leakage dangerous: it flatters your metrics instead of announcing itself.

## The OOP design

Part E implements `oop_rules.md` §14 properly, unlike my earlier abandoned
attempt which was procedural code wearing class syntax.

| Rule (§13) | How it is satisfied |
| :--- | :--- |
| 1. Single responsibility | `FeatureBuilder` derives; `OperationalExperienceBuilder` owns one derivation |
| 2. No naked mutable state | `self._name`, `self._group_col`, `self._history` |
| 3. Validate early | `_validate_text()` + `isinstance` checks in `__init__` and `fit_transform` |
| 4. Preserve invariants | Validation happens **before** assignment, so a failed call leaves state unchanged |
| 5. `@property` accessors | `name`, `group_col`, `is_fitted`, `feature_names` are read-only |
| 6. Defend collections | `history` returns a `tuple`, not the live list |
| 7. Instance vs class state | `REQUIRES_SORTED_CHRONOLOGY` is a class attribute |
| 8. Cooperative `super()` | `super().__init__(...)` in the subclass |
| 9. Respect MRO | Verified with `.mro()` |
| 10. ABC contracts | `FeatureBuilder(ABC)` with `@abstractmethod` |

### The template method pattern

`fit_transform()` lives in the **base** class and owns the whole sequence:

1. Validate the input is a DataFrame
2. Validate required columns exist
3. Call the subclass's `_build()`
4. Check the promised features were produced
5. Record history

Subclasses supply **only** `_build()`. This means every future feature builder
gets validation and logging for free — it cannot skip them. That is the practical
payoff of the ABC pattern, not just tidiness.

### Proof the rules hold

The notebook demonstrates each one rather than claiming it:

```
1. ABSTRACT CONTRACT  -> FeatureBuilder() refused (TypeError)
2. VALIDATION (§5)    -> empty/non-string group_col rejected
3. INPUT TYPE (§5)    -> non-DataFrame rejected at transform time
4. DEFENSIVE COPIES   -> history is a tuple; append refused (AttributeError)
5. READ-ONLY (§6)     -> assignment to .name refused
6. CLASS STATE (§7)   -> REQUIRES_SORTED_CHRONOLOGY on the class
7. MRO (§10)          -> OperationalExperienceBuilder -> FeatureBuilder -> ABC -> object
```

## What we did

### Features kept and rejected

| Feature | Decision | Reason |
| :--- | :--- | :--- |
| `Company_Prior_Launches` | **Keep** | Primary experience measure; monotonic signal |
| `Company_Prior_Failure_Rate` | **Keep** | Adds the *quality* dimension |
| `Company_Prior_Failures` | **Reject** | Correlates **0.952** with launches — near-redundant |
| `Is_First_Company_Launch` | **Reject** | Derivable from `prior_launches == 0` |

### Why these beat the raw columns

| Raw column | What it cannot express |
| :--- | :--- |
| `Company` (62-value label) | Cannot represent *change over time* within one operator; a one-hot column can't answer "has this org improved?" |
| `Date` | Conflates industry-wide maturation with operator-specific learning |

The engineered count also **generalises** — it works for operators with only 2–3
records, where a one-hot column would be nearly empty.

## Evidence

```
Causal check (SpaceX trace):
          Date  _is_fail  prior_launches  prior_failures
    2009-07-14     0          4               3
    2010-06-04     0          5               3   <- own failure NOT counted

PASS: the caller's dataframe was not mutated (defensive copy worked).

Leakage inflates F1 by +0.0895 (+49.6%)

A first launch fails at 41.9%, versus 5.1% after 500+ launches.
That is a 8.2x difference.
```

| Experience bucket | Missions | Failure rate |
| :--- | :--- | :--- |
| 1st launch | 62 | **41.9%** |
| 2–6 | 224 | 32.1% |
| 7–21 | 395 | 19.5% |
| 22–101 | 1,238 | 9.0% |
| 102–501 | 1,434 | 8.2% |
| 502+ | 1,276 | **5.1%** |

Global rate: **10.1%**. An operator's first launch is roughly **8×** more
failure-prone than their 500th.

## Honest caveat

Repeated cross-validation (5-fold × 10) showed **all four** candidate features
scoring marginally best (0.1880) versus the two we kept (0.1805). But that gap is
~0.005 against a standard deviation of ~0.05 — **inside the noise**.

We keep two because they are defensible and non-redundant, not because two won a
single CV run. Claiming the two-feature set is "better" would be overclaiming.

## The charts

**Part E (4 charts):**
1. Failure rate by experience bucket — the headline signal (41.9% → 5.1%)
2. Causal proof — the first SpaceX failure is *not* counted in its own row
3. Honest vs leaky F1 — visualises why leakage is seductive
4. Raw (decade) vs engineered (experience) signal, side by side

**Added to earlier parts (4 charts):**
5. Part C — missingness bar chart (why `Price` was dropped)
6. Part C — rows removed vs co-launches preserved
7. Part D — month on a line vs on a circle
8. Part D — raw scales vs identical tree importances

Total: **8 charts**, each tied to a decision already made rather than decoration.

## Why — your wording

*(to be filled in — your own phrasing)*

## Defend it

*(to be filled in)*

---

# Part F — Check for Data Leakage

## What the guide asks

> Does any feature directly or indirectly expose information about the target?
> Could future information leak into past predictions?
> Provide an evaluation of your dataset's leakage risks.

## The concept

### The one test that governs everything

For every column, ask:

> **Would this value be knowable in the mission-control room, before liftoff?**

If no, it is a leak. That is the whole test.

### Why leakage is uniquely dangerous

Other data problems announce themselves. Missing values show up as `NaN`.
Duplicates inflate row counts. Bad types throw errors.

**Leakage does none of that.** It makes your model look *better*. In Part E we
measured this: leaky features scored F1 **0.2700** against honest features at
**0.1805** — a ~50% inflation. The leaky model is not obviously broken; it looks
like an improvement. That is what makes it the most dangerous problem in the
whole pipeline.

## What we found

### The audit table

| Column | Knowable pre-launch? | Verdict |
| :--- | :--- | :--- |
| `MissionStatus` | — | The target itself |
| `RocketStatus` | **No** | **SEVERE — temporal leakage** |
| `Mission` | Yes, but useless | Identifier + **latent leak** |
| `Price` | Partially | Retroactive recording bias |
| `Time` | Yes | Verified no signal |
| `Date`, `Company`, `Rocket`, `Country` | Yes | Safe |

### Finding 1 — `RocketStatus` (severe)

This field records whether a vehicle was operational **as of August 2022**, when
the dataset was compiled. It is not a pre-flight condition.

**Evidence A — the temporal tell:**

| Decade | Active | Retired |
| :--- | :--- | :--- |
| 1950s | **0** | 51 |
| 1960s | **0** | 774 |
| 1970s | **0** | 1,012 |
| 1980s | 8 | 623 |
| 2010s | 471 | 204 |
| 2020s | 365 | 4 |

**Zero pre-1980 launches are `Active`, across 1,837 records.** That is not a fact
about rockets — anything launched 40+ years ago is retired *now* by definition.

**Evidence B — reverse causation:** `Active` vehicles succeed **93.2%** of the
time vs **89.0%** for `Retired`. Causation runs backwards: programmes end and
unreliable vehicles get withdrawn, so the *outcome partly causes the status*.

A model trained on this column learns "old missions used retired rockets." True
by construction, and useless for predicting a new launch.

### Finding 2 — `Mission` names carry risk (latent)

`Mission` was already being dropped as a 98.4%-unique identifier. But auditing it
revealed the name itself encodes outcome information:

| Name pattern | Missions | Failure rate | Median year |
| :--- | :--- | :--- | :--- |
| contains `"test"` | 23 | **52.2%** | **2018** |
| contains `"demo"` | 20 | 25.0% | 2014 |
| all missions | 4,628 | 10.1% | 1987 |

**The key check: is this just an era proxy?** No — and that is what makes it
interesting. Test-named missions have a median year of **2018**, far *later* than
the dataset median of 1987, yet still fail at 5× the base rate. Restricted to
pre-1970 alone: **75.0%** vs **23.6%**.

So the name signals a genuinely higher-risk mission *type* (first flights,
unproven vehicles). We drop `Mission` anyway, which removes the vector — but the
lesson generalises:

> **An identifier is not automatically inert.** "It is just an ID" is not a
> reason to stop auditing a column.

*Note:* the mission name **is** known at liftoff, so a `name_contains_test`
feature would be *legitimate*, not leaky. We do not build one (23 rows would
generalise poorly), but the distinction between "known at liftoff" and "leaky"
is exactly the test being applied.

### Finding 3 — `Price` missingness is systematic (moderate)

| Decade | With price | % recorded |
| :--- | :--- | :--- |
| 1960s | 33 / 774 | 4.3% |
| 1990s | 132 / 642 | 20.6% |
| 2010s | 528 / 676 | 78.1% |
| 2020s | 312 / 369 | 84.6% |

Pricing is recorded mainly for **modern** missions. Since era correlates with the
target, imputing `Price` would have smuggled era information into a column that
looks like a neutral financial measure. Dropping it was right on two independent
grounds.

### Finding 4 — `Time` is provably harmless

| Launch hour | Failure rate |
| :--- | :--- |
| 00:00–05:59 | 7.6% |
| 06:00–11:59 | 10.2% |
| 12:00–17:59 | 9.8% |
| 18:00–23:59 | 10.1% |

Flat around the 10.1% base rate. Part C excluded `Time` on reasoning; Part F
*confirms* it with measurement. Excluding it costs nothing.

## The OOP design: `LeakageGuard`

A guard that never fails is worthless, so the notebook feeds it known-bad
matrices and confirms it rejects them.

| Check | Catches |
| :--- | :--- |
| **Blocklist** (`FORBIDDEN_COLUMNS`, a class attribute) | Columns known to be unsafe |
| **Near-restatement scan** | A numeric column that *is* the target in disguise |

The second check exists because of a **real mistake during development**: a
`_is_fail` helper column built for Part E silently entered `X`, and the blocklist
did not catch it because nobody had listed it.

The fix is a general property rather than another blocklist entry:

> Any binary numeric column correlating ≥ 0.999 with the target is a leak,
> regardless of what it is named.

Demonstrated in the notebook:

```
Test 1 - clean matrix          : PASS
Test 2 - blocklisted column    : correctly REJECTED
          Blocklisted column(s) present in X: ['RocketStatus']
Test 3 - disguised target      : correctly REJECTED
          Column '_hidden_flag' restates the target (binary, |corr| = 1.0000).
Test 4 - guard reusable        : PASS
```

`LeakageGuard` applies the §13 rules: protected `_target`/`_violations`, class-level
`FORBIDDEN_COLUMNS` and `MAX_TARGET_CORRELATION`, read-only `@property` accessors,
defensive `tuple()` copies, and `isinstance` validation before assignment.

## Evidence summary

```
Launches before 1980 labelled 'Active' : 0
Total launches before 1980             : 1837

contains "test"   n=23   failure=0.5217   median_year=2018
all missions      n=4628 failure=0.1011   median_year=1987

  test-named, pre-1970 : 0.750 (n=4)
  all others, pre-1970 : 0.236 (n=821)

Price % recorded: 4.3% (1960s) -> 78.1% (2010s)

Guard verdict: PASS / PASS
```

## The charts

Part F adds **3 charts** (11 total in the notebook):

1. `RocketStatus` temporal tell — the stacked decade chart with zero Active bars
2. Reverse causation — Active 93.2% vs Retired 89.0% success
3. `Price` availability by decade — climbing from ~0% to ~85%

## Why — your wording

*(to be filled in — your own phrasing)*

## Defend it

*(to be filled in)*

---

# Part G — Feature and Target Separation

## What the guide asks

> Separate your preprocessed dataset into feature matrices and target vectors:
> ```python
> X = df.drop(columns=["target_variable_name"])
> y = df["target_variable_name"]
> ```

## The concept

The code is **two lines**. There is nothing technically hard about Part G.

The whole difficulty is *deciding what belongs in `X`*, because the failure mode
is silent: a leaked or redundant column does not raise an error. It just quietly
degrades the model, and you find out much later.

So Part G is really a **justification exercise**. Every column must be assigned to
one of three buckets, and every exclusion needs a reason traceable to an earlier
part.

## What we did

| Bucket | Columns | Count |
| :--- | :--- | :--- |
| **Target (`y`)** | `MissionStatus` → binarised | 1 |
| **Features (`X`)** | `Company`, `Rocket`, `Country`, `Company_Prior_Launches`, `Company_Prior_Failure_Rate`, `Launch_Year`, `Month_Sin`, `Month_Cos` | 8 |
| **Excluded** | `MissionStatus`, `Mission`, `Time`, `RocketStatus`, `Date`, `Company_Prior_Failures`, `_is_fail` | 7 |

### Every exclusion, with its reason

| Excluded | Reason | From |
| :--- | :--- | :--- |
| `MissionStatus` | It **is** the target | Part B |
| `_is_fail` | Binary restatement of the target (a Part E helper) | Part F |
| `Mission` | 98.4% unique identifier + latent name leak | Parts A, F |
| `Time` | Verified flat: 7.6–10.2% failure across hour buckets | Parts C, F |
| `RocketStatus` | **Severe temporal leakage** | Part F |
| `Date` | Superseded by the derived calendar features | Part D |
| `Company_Prior_Failures` | 0.952 correlation — redundant | Part E |

### The two exclusions a marker would question

**"Why drop `Date`? It is obviously informative."**

Because its information is **already captured**. `Launch_Year` carries the era
signal; `Month_Sin`/`Month_Cos` carry the seasonal signal. A raw `datetime64`
cannot be consumed by a tree anyway, and keeping it would duplicate what the
engineered features already express.

This is the guide's "avoid over-processing" rule applied in reverse: **do not
keep a redundant column just because it was in the original file.**

**"Why drop `Company_Prior_Failures`? Part E literally built it."**

Because Part E *measured* it as redundant (0.952 correlation with
`Company_Prior_Launches`) and rejected it there. Part G honours that decision.
Keeping a near-duplicate adds a dimension without adding information, and
correlated inputs make feature importances harder to interpret.

## The target vector

| Original label | Count | Binary `y` |
| :--- | :--- | :--- |
| `Success` | 4,160 | **1** |
| `Failure` | 357 | 0 |
| `Partial Failure` | 107 | 0 |
| `Prelaunch Failure` | **4** | 0 |

**Why binarise?** `Prelaunch Failure` has **4 observations**. A class that small
cannot be stratified reliably and cannot be learned. Collapsing to
Success vs Not-Success preserves the business question while making the split
viable.

We keep the original 4-level text as `outcome_labels` for reference, even though
it is not used as a target.

## Validation, not assumption

Separating `X` and `y` is a claim, so it is checked. Two layers:

**1. The `LeakageGuard` from Part F** (reused, which is why it was written as a class):

```
No blocklisted columns in X                       PASS
No column restates the target (|corr| >= 0.999)   PASS
```

**2. Five independent structural checks:**

```
1. Index alignment      : PASS (4628 rows, aligned)
2. Target absent from X : PASS
3. No constant columns  : PASS
4. Class balance        : PASS (0.8989 == raw 0.8989)
5. Feature count        : PASS (8 features)
```

Check 4 is worth noting: it confirms the binarisation did not accidentally
shift the positive rate. Check 3 catches a subtle failure — a constant column
carries no information but still occupies a dimension.

## Scale check

| Quantity | Value |
| :--- | :--- |
| Observations | 4,628 |
| Features | 8 |
| Rows : features | **579 : 1** |

That ratio is healthy. One-hot encoding `Rocket` would have added 370 columns and
dropped the ratio to ~12:1 — exactly the justification for frequency encoding
established in Part D.

## A correction we made along the way

`Location` was still present in the dataframe. Part C had extracted `Country`
from it but never dropped the source column, so the same information existed
twice: 158 raw composite strings alongside 22 clean country labels.

We fixed this **in Part C** rather than patching it in Part G, so the extraction
and the drop are one atomic step. Part D's encoding table was updated to match.
Leaving it to Part G would have meant carrying a redundant column through two
whole parts.

## The charts

Part G adds **2 charts** (13 total in the notebook):

1. **Class balance** — 89.89% vs 10.11%, annotated with the trivial-predictor
   trap, plus the original 4-level distribution showing why binarisation was
   necessary.
2. **The separation map** — a visual of all three buckets: what went into `X`,
   what became `y`, and every excluded column with its reason beside it.

## Why — your wording

*(to be filled in — your own phrasing)*

## Defend it

*(to be filled in)*

---

# Part H — Training and Testing Preparation

## What the guide asks

> Partition the data (e.g., 80% Training, 20% Testing).
> **Leakage Control Requirement:** Compute all imputation, scaling, and encoding
> parameters using the **training set only**, then transform the test set using
> those pre-fit values.

## The concept

### The order of operations IS the lesson

Two possible orders, only one correct:

| | ❌ Wrong | ✅ Correct |
| :--- | :--- | :--- |
| 1 | Fit scaler on **all** data | **Split** first |
| 2 | Split | Fit on **train only** |
| 3 | Transform both | Transform both with **frozen** params |
| | Test stats leak into training | Honest evaluation |

The wrong order is tempting because it is *more convenient* — scale once, split
once, done. The damage is invisible: the model scores better than it deserves.

Concretely, fitting a scaler on all data means its mean and standard deviation
are partly computed from test rows. The model then "knows" something about the
data it will be evaluated on. The guide names this in §15.6: **"Scaling Before
Splitting."**

### The fix is structural, not disciplinary

We do not rely on remembering to fit only on train. We use a scikit-learn
`Pipeline`, and the pipeline is only ever given `X_train` to `fit()`. Leakage
becomes **impossible by construction**, not by vigilance.

## What we did

### The split: 80/20, stratified

| | Value |
| :--- | :--- |
| Training rows | **3,702** |
| Test rows | **926** |
| Minority rows in test | **94** failures |
| Train positive rate | 0.8990 |
| Test positive rate | 0.8985 |

**Why stratified?** At a 9:1 imbalance, an unstratified split can produce a test
set with a materially different positive rate, making scores unstable between
runs. `stratify=y` preserves the balance. This follows directly from the
imbalance measured in Part B — it is a consequence of that number, not a default.

**Why no separate validation set?** With only 468 failures total, a third split
would leave too few minority rows to evaluate on. Cross-validation on the
training set serves that purpose instead.

### The feature space: 46 columns

| Source | Columns | Method |
| :--- | :--- | :--- |
| Numeric (5) | 5 | Median impute → `StandardScaler` |
| `Country` | 12 | One-hot (`min_frequency=10`) |
| `Company` | 28 | One-hot (`min_frequency=10`) |
| `Rocket` | **1** | **Frequency encoding** |
| **Total** | **46** | rows:features = **100 : 1** |

### Why `Rocket` is frequency-encoded, not one-hot

Part D committed to this, and Part H shows the saving concretely:

| Encoding | Columns for `Rocket` |
| :--- | :--- |
| One-hot (`min_frequency=10`) | **86** |
| **Frequency encoding** | **1** |

Even with rare rockets collapsed, one-hot produces **86 columns for a single
feature** — it would dominate the matrix. Frequency encoding captures "common,
well-established vehicles differ from one-off experimental ones" in one column.

## The subtle failure mode: test-only categories

This is the part worth understanding properly, because it bites in real data.

With an 80/20 split, some categories appear in the **test set but never in
training**:

| Column | Categories only in test |
| :--- | :--- |
| `Rocket` | **18** |
| `Company` | **1** |
| `Country` | 0 |

A naive encoder **crashes** on these (`ValueError: Found unknown categories`),
because its vocabulary came from training and does not contain them.

This is not a data bug. It is exactly what happens when a new operator or vehicle
appears after your training period — which is precisely the situation a deployed
model faces.

**How we handle it:**

- `OneHotEncoder(handle_unknown="infrequent_if_exist")` routes unseen categories
  to the "infrequent" bucket rather than raising.
- `FrequencyEncoder` maps unseen categories to `0.0` rather than `NaN`.

Both mean the test set is transformed using **training-derived vocabulary only** —
the required behaviour.

## Verifying no leakage occurred

Four independent numeric checks. A claim of "no leakage" is worthless untested.

### Check 1 — scaler means match TRAIN, not full data

We compare the scaler's stored means against **both** the training means and the
full-data means. This proves *which one it actually learned*:

| Column | Scaler (as fitted) | Actual TRAIN mean | Actual FULL mean |
| :--- | :--- | :--- | :--- |
| `Launch_Year` | 1989.422474 | 1989.422474 | 1989.607390 |
| `Company_Prior_Launches` | 395.006213 | 395.006213 | (differs) |

**Max deviation from training means: 0.000000000000**

The deviations from the *full-data* means are non-zero — which is the proof. Had
the scaler seen the test rows, it would match the full means instead.

### Check 2 — encoder vocabulary excludes test-only categories

```
Company : 61 train categories, 1 test-only, vocabulary 61
Country : 22 train categories, 0 test-only, vocabulary 22
PASS: no test-only category appears in the fitted vocabulary.
```

### Check 3 — frequency mapping came from training

| Rocket | Learned from train | Actual train freq | Full-data freq |
| :--- | :--- | :--- | :--- |
| `Cosmos-3M (11K65M)` | 0.099406 | 0.099406 | 0.096370 |
| `Voskhod` | 0.064290 | 0.064290 | 0.064607 |

The learned values match **train** frequencies exactly and differ from full-data
frequencies. 351 rockets learned from train, versus 370 in the full dataset.

### Check 4 — matrix integrity

```
Training shape (3702, 46) | Test shape (926, 46)
NaNs: 0 in both
Numeric block mean (train) : -0.0000000000
Numeric block std  (train) :  1.0000000000
```

The numeric block is correctly standardized (mean 0, std 1) on **training** data.

### Plus: the `LeakageGuard` re-run

```
No blocklisted columns in X                       PASS
No column restates the target (|corr| >= 0.999)   PASS
Feature-name sources: 5 numeric + 40 one-hot + 1 frequency = 46
```

## On imputation being idle

After Part C, `X` has **no missing values at all** — `Time` was excluded, `Price`
dropped, and everything else is complete.

Imputers are still wired into the pipeline, for two honest reasons:

1. **Robustness** — if the pipeline meets fresh data with gaps, it will not break.
2. **Contract correctness** — the guide requires imputation parameters be fit on
   train only; wiring it in now means it is fitted on train even when idle.

We **document that they are idle** rather than pretending they do work.

## A bug worth remembering: `categories_` ≠ output width

Part H initially crashed with `AssertionError: Feature-name count mismatch!` —
89 generated names against 46 actual columns.

The cause: `OneHotEncoder.categories_` lists **every category seen during fit**,
but `min_frequency` collapses the rare ones into a *single* `infrequent` column.
So `len(categories_)` does **not** equal the output width.

The authoritative source of names is `get_feature_names_out()`:

```
categories_ stored : ['Chad', 'Fiji', 'Russia', 'USA']   <- 4
actual output cols : 3
feature names      : ['Country_Russia', 'Country_USA', 'Country_infrequent_sklearn']
```

**Lesson:** when `min_frequency` is in play, never infer width from the learned
category list. This is a genuine API subtlety, not a typo.

## Evidence summary

```
SPLIT COMPLETE - before any fitting
  Training : 3,702 rows (80.0%)
  Test     :   926 rows (20.0%)
  Minority-class rows in TEST : 94 failures

Transformed training matrix : (3702, 46)
Transformed test matrix     : (926, 46)

Max deviation from TRAINING means : 0.000000000000
Categories appearing only in the test set: 1
FrequencyEncoder: 351 rockets learned from train (370 in full data)

PASS: 4,628 rows preserved end-to-end (3,702 + 926)
PASS: 4,160 positives and 468 negatives preserved
```

## The charts

Part H adds **3 charts** (16 total in the notebook):

1. **Split composition** — 80/20 sizes plus the stratification proof (three
   positive rates nearly identical)
2. **The leakage proof** — scaler means vs train means vs full means, with a
   log-scale deviation panel showing zero deviation against train
3. **Feature provenance** — where all 46 columns came from, and the 86-vs-1
   `Rocket` encoding comparison

## Artefacts saved

`X_train.npy`, `X_test.npy`, `y_train.npy`, `y_test.npy`, `feature_names.txt`

## Why — your wording

*(to be filled in — your own phrasing)*

## Defend it

*(to be filled in)*

---

# Part I — Preprocessing Summary Table

*(pending)*
