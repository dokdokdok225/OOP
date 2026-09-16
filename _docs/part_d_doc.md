### Part D — Feature Transformation

Part C left the dataset clean but still entirely **textual**: after cleaning, there
are *no numeric columns at all* (only `Date`, which is a datetime). Part D converts
those columns into representations a model can actually consume.

---

#### D.1 — What actually needs transforming

An inventory of the cleaned dataset, grouped by what each column needs:

| Column | Type | Cardinality | Transformation needed |
| :--- | :--- | :--- | :--- |
| `Date` | datetime | 4,628 distinct | **Date/time extraction** → numeric calendar features |
| `Country` | text | 22 | **One-hot encoding** (low cardinality, nominal) |
| `Company` | text | 62 | **One-hot with rare collapse** (moderate cardinality, nominal) |
| `Rocket` | text | 370 | **Frequency encoding** (high cardinality — see D.4) |
| `RocketStatus` | text | 2 | *None — excluded in Part F as temporal leakage* |
| `MissionStatus` | text | 4 | *None — this is the target (Part G)* |
| `Time` | text | — | *None — excluded (no causal bearing on reliability)* |
| `Location` | text | 158 | *None — decomposed into `Country` in Part C* |
| `Mission` | text | 4,555 | *None — an identifier, not a feature* |

---

#### D.2 — Date/time extraction

Two kinds of feature are derived from `Date`, for different reasons.

**(a) Ordinal year — `Launch_Year`**

The dataset spans 1957–2022. Part A established that failure rates fall sharply
after the early experimental era, so the year carries real signal.

**(b) Cyclical month — `Month_Sin`, `Month_Cos`**

Month needs careful handling. Left as a raw integer, December (12) and January (1)
appear **maximally distant** — 11 units apart — when in reality they are adjacent
months. That is backwards, and it would mislead any distance-based algorithm.

The fix is to project the month onto a unit circle:

$$\text{Month\_Sin} = \sin\!\left(\frac{2\pi \cdot \text{month}}{12}\right), \qquad
\text{Month\_Cos} = \cos\!\left(\frac{2\pi \cdot \text{month}}{12}\right)$$

Both components are required. A single sine value is ambiguous — two different
months can share one sine value — so sine and cosine together give each month a
unique position on the circle.

**Verified numerically.** Measuring Euclidean distance in sine/cosine space:

| Pair | Distance | Meaning |
| :--- | :--- | :--- |
| January ↔ December | **0.5176** | Adjacent — correct |
| January ↔ February | **0.5176** | Adjacent — identical, as it should be |
| January ↔ July | **2.0** | Opposite — correct |

If month were left as a raw integer, January↔December would be the *largest* gap
(11) rather than tied for the smallest.

> **Why not quarter?** `Launch_Quarter` is a coarser restatement of month. Once
> cyclical month is present, quarter adds no information and would only add a
> redundant column. The guide warns against over-processing.

---

#### D.3 — REQUIRED DISCUSSION: Normalization vs. Standardization

This is the discussion the activity guide explicitly requires. The honest answer
for this project is **standardization (Z-score)**, but the reasoning is more
interesting than the textbook version — because the two stages of our pipeline
need different things.

##### The two methods

| | Min-Max Normalization | Z-Score Standardization |
| :--- | :--- | :--- |
| **Formula** | $X_{norm} = \dfrac{X - X_{min}}{X_{max} - X_{min}}$ | $Z = \dfrac{X - \mu}{\sigma}$ |
| **Output range** | Bounded, exactly $[0, 1]$ | Unbounded, typically $\approx [-3, 3]$ |
| **Uses** | Min and max | Mean and standard deviation |
| **Outlier sensitivity** | **High** — one extreme value squashes everything else | **Lower** — an outlier shifts $\mu$ modestly |
| **Removes units?** | Yes | Yes |

##### The deciding factor: which algorithm consumes the features?

We measured this rather than assuming it, and the two stages behave completely
differently.

**Stage 1 — K-Means: scaling is mandatory.**

K-Means assigns each record to the nearest cluster centre using **Euclidean
distance**. Distance is only meaningful if each feature is on a comparable scale.
Our two clustering features have very different ranges:

- `Launch_Year`: 1957 → 2022 (a span of 65)
- `Company_Prior_Launches`: 0 → 1,776 (a span of 1,776)

The prior-launch count spans **27.3× wider** than the year. Because Euclidean
distance squares each difference, the max squared distance is dominated almost
entirely by that one feature:

| Feature | Share of maximum squared distance (unscaled) |
| :--- | :--- |
| `Company_Prior_Launches` | **99.9%** |
| `Launch_Year` | 0.1% |

The consequence is visible in the cluster centres. Unscaled, the algorithm is
really clustering on launch volume alone and the year coordinate barely
distinguishes the groups:

| Scaling applied | Year centres | Prior-launch centres |
| :--- | :--- | :--- |
| **Unscaled** | `1994, 1984, 1973` | `105, 1440, 770` |
| **Standardized** | `-1.0, 1.0, -0.5` | `0, -1, 2` |

Both rows contain real numbers, so what is the problem? The issue is **weighting,
not validity**. Unscaled, a 1,000-launch difference between two operators counts
for vastly more than a 40-year difference between two eras, purely because of the
units chosen. The clusters are therefore driven almost entirely by launch volume,
and the year feature contributes almost nothing — 0.1% of the distance.
After standardization both features contribute comparably, which is what we want
when the goal is to group by *era and maturity together*.

**Stage 2 — Decision Tree: scaling is irrelevant.**

A decision tree splits a feature at a threshold ("is `Launch_Year` ≤ 1972.5?").
Rescaling a feature cannot change the **order** of its values, so every possible
split point remains available. The tree therefore learns an identical model.

We verified this by training the same tree on raw and standardized features and
comparing what it actually learned:

| Metric | Raw | Standardized |
| :--- | :--- | :--- |
| Importance of `Launch_Year` | 0.669 | 0.669 |
| Importance of `Company_Prior_Launches` | 0.297 | 0.297 |
| Importance of `Company_Prior_Failures` | 0.034 | 0.034 |
| Cross-validated F1 | 0.2052 | 0.2015 |

The feature importances are **identical**. The F1 difference of −0.004 is noise.
This confirms the theory: for a tree, scaling changes nothing about what is learned.

##### Decision: Standardization

**We use Z-score standardization**, for three reasons:

1. **Stage 1 requires it and Stage 2 is indifferent to it.** Standardization is
   necessary for K-Means and harmless for the tree, so one choice serves both.
   Min-Max would also work for K-Means, so this alone is not decisive — reason 2 is.

2. **The distribution is right-skewed and unbounded.** `Company_Prior_Launches`
   spans 0 to 1,776 with most values near 0. Min-Max depends entirely on the
   maximum, so a single prolific operator would squash every other value into a
   sliver of $[0, 1]$. Standardization uses the mean and standard deviation, so one
   extreme organisation shifts the centre slightly rather than destroying the scale.

3. **Standardization is the better default for distance-based methods.** It
   produces zero-centred, unit-variance features, which is the assumption behind
   Euclidean-distance algorithms like K-Means.

**When Min-Max would be correct instead:** when inputs must be *bounded* — for
example a neural network with sigmoid or tanh activations, or image pixel values
that are naturally in $[0, 255]$. Neither applies here.

##### Where the scaling is actually applied

The scaling is **not** performed in this section. It is implemented inside the
`ColumnTransformer` pipeline in **Part H**, where it is fitted on the training
split only.

This matters. Fitting a scaler on the full dataset before splitting lets
test-set statistics influence the training parameters — the classic leakage
pattern the guide warns about in §15.6 ("scaling before splitting"). Part D
decides *what* to do; Part H applies it in a leak-free way.

---

#### D.4 — Categorical encoding strategy

Encoding choice is driven by cardinality, because the wrong choice has a concrete
cost.

| Encoding | Mechanism | Use when | Cost of misuse |
| :--- | :--- | :--- | :--- |
| **One-hot** | One binary column per category | Low-cardinality nominal | Curse of dimensionality at high cardinality |
| **Ordinal** | Integers reflecting a real rank | Genuinely ordered categories | Implies false order on nominal data |
| **Frequency** | Replace category with its count | High-cardinality nominal | Categories with equal counts become identical |

**Our decisions:**

| Column | Cardinality | Encoding | Reason |
| :--- | :--- | :--- | :--- |
| `Country` | 22 | **One-hot** | Nominal with no order; 22 columns is manageable |
| `Company` | 62 | **One-hot with rare collapse** | Nominal; rare organisations grouped to control width |
| `Rocket` | 370 | **Frequency encoding** | See below |

**Why `Rocket` gets frequency encoding rather than one-hot.**

370 categories on 4,628 rows means one-hot would add 370 sparse columns — roughly
one column per 12 observations. Most of those columns would be almost entirely
zeros, and the tree would happily split on noise. Worse, 83 rockets appear exactly
once and 123 appear at most twice, so those columns could not generalise at all.

Frequency encoding compresses the whole column into a single numeric feature: each
rocket is replaced by how often it appears. That captures the intuition that
*common, well-established vehicles behave differently from one-off experimental
ones*, without exploding the feature space.

**A caveat we accept.** Frequency encoding has a known weakness the guide names
explicitly: two different rockets that happen to appear equally often receive the
*identical* value, so any distinction between them is lost. For `Rocket` this is an
acceptable trade — with 370 categories and mostly single appearances, the
alternative is unusable sparsity.

**Ordinal encoding is not used anywhere**, because the dataset contains no
genuinely ordered categorical column. The guide warns that applying ordinal
encoding to nominal data invents a mathematical order that does not exist.

---

#### D.5 — What we are deliberately NOT doing

- **No log/power transformation.** The guide lists it for features spanning
  several orders of magnitude. After cleaning, no such continuous feature remains
  in the modelling set — `Price` was dropped in Part C. We are not applying a
  technique merely because it appears in the table.
- **No scaling in this section.** Deferred to Part H to avoid leakage (§15.6).
- **No feature engineering beyond date extraction.** That is Part E's job, kept
  separate so each part answers only its own question.
- **No encoding of the target.** `MissionStatus` is handled in Part G.