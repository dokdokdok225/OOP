# Space Missions Project — Full Study Guide (Activities 1–3)

**Course context:** DSC 2135 / IS 4-513 · Data Scientist Track
**Goal of this document:** one connected story you can read from start to finish and
then **present confidently** — what we built, *why* we made each decision, and what it
means — written so a beginner can follow every step.

**How to read it:** the three activities are one connected story, not three separate tasks.
Activity 1 *prepares the data*, Activity 2 *establishes a fair starting bar*, Activity 3
*beats that bar honestly*. Each section ends with "Why we did this" and "Say it like
this" boxes you can reuse when presenting.

> **The one-sentence version of the whole project:**
> We turned a messy CSV of 4,628 space launches into a leak-free, machine-readable
> dataset, proved our first simple model genuinely detects failed missions better than
> a no-knowledge baseline, then improved that score ~22% with a tuned Random Forest —
> explaining every decision and admitted limitation along the way.

---

# THE STORY ARC (read this first)

```text
ACTIVITY 1 — BUILD THE FOUNDATION
   raw CSV → clean it → engineer features → split train/test → save sealed arrays
   (no model yet — this is the data factory)

ACTIVITY 2 — SET A FAIR BAR
   load arrays → no-knowledge baseline (Dummy) → one simple model (Decision Tree)
   → evaluate imbalanced-aware → inspect mistakes honestly
   (this defines what "better than nothing" means)

ACTIVITY 3 — IMPROVE, COMPARE, EXPLAIN
   race 5 models fairly (cross-validation) → tune the winner → one final sealed
   exam → explain what drives it → inspect remaining errors by name
   (the bar rises: 0.3463 → 0.4215, with honesty preserved)
```

Every later activity *trusts* the earlier one. That is why discipline (no leakage,
test set untouched) matters more in Activity 1 than in any later step — it is the
foundation the whole bridge stands on — actually, more than foundation: it is the
*house rules*. If Activity 1 leaks, every later score is quietly wrong.

---

# PART 1 — ACTIVITY 1: FROM RAW FILE TO ML-READY DATA

## 1.1 The raw material and what we found

**Input:** `space_missions.csv` — 4,630 rows × 9 columns. One row = one launch attempt.

| Column | What it holds | What we decided | Why |
|---|---|---|---|
| `Company` | Who launched it | **Keep**, one-hot encode | Operator identity is a real risk signal |
| `Location` | Full site text | Extract `Country`, **drop the rest** | Country is generalizable; full site text is mostly noise |
| `Date` | Launch date | Derive **Launch_Year, Month_Sin/Cos**, then drop | Era and season carry signal after extraction |
| `Time` | Time of day | **Drop** | 127 missing (2.74%), weak signal anyway |
| `Rocket` | Rocket family | **Keep**, frequency encode (370 families!) | One-hot would explode to 370 near-empty columns |
| `Mission` | Mission name | **Drop** | ~98.4% unique → behaves like an ID; can't generalize |
| `RocketStatus` | Active/Retired | **Drop** | Temporal leakage: status at *scrape time*, not launch time |
| `Price` | Launch cost | **Drop** | 72.68% missing — imputation would fabricate 3 of 4 values |
| `MissionStatus` | Result | **Becomes the target** `y` | This is what we predict |

**Key data findings:**

- **4,630 raw rows → 4,628 final rows:** removed one exact duplicate (indices
  3530/3531 — CASC, Long March 2D, 2008-11-05) AND one documented business duplicate
  (the same launch re-entered with incomplete information). Legitimate co-launches
  (same date/company/rocket, separate payloads) were deliberately kept.
- Class imbalance: Success ~**89.89%** vs Not-Success ~**10.11%** — this single fact shapes
  the whole project (it is why accuracy lies and why we need F1).
- Outliers are *valid history, not errors*: the 1957 Sputnik-era launches, $5,000M
  Energiya/Buran, $2.5M Astra Rocket 3, 1,777 rows of RVSN USSR vs one-off companies.

> **Why we did this (say it like this):**
> "We profiled before we cleaned — duplicates, missingness, cardinality, and leakage
> risk — and every column decision has a written reason. We dropped `Price` not because
> cost is unimportant, but because 72.68% missingness means any imputed values would be
> invented data. We dropped `Mission` because identifiers can't generalize."

## 1.2 Leak-free features ("prior" thinking)

Two causal company-history features — computed **from earlier missions only**:

- `Company_Prior_Launches` — how many launches *(before this one)* by this company.
- `Company_Prior_Failure_Rate` — prior failures ÷ prior launches (historical reliability).
- We dropped `Company_Prior_Failures` (raw count) because it correlated ~0.952 with
  prior launches — nearly duplicated information.

> **Why "prior" is sacred:** if a mission's own outcome sneaked into its own feature,
> the feature would BE the answer (target leakage). Using only *earlier* rows makes
> these features something you could honestly compute before a launch happens.
>
> This is also why `RocketStatus` was dropped: "Active" vs "Retired" describes the
> rocket today, 2026 — not what anyone knew at launch time.

## 1.3 Encoding: converting words into honest numbers

| Technique | Applied to | What it does | Why this one |
|---|---|---|---|
| **Standardization** (z-score) | Numeric features | centers at 0, unit spread | Needed by distance-based methods; fitted on train only. |
| **Month_Sin / Month_Cos** | Month | puts months on a circle | December (12) and January (1) are *neighbours* — raw numbers 12 vs 1 would lie about distance. |
| **One-hot encoding** | Company, Country | one 0/1 column per category | Company labels have no true *order* — label-encoding 1,2,3 would invent order. |
| **Frequency encoding** | Rocket | replaces name with training-set frequency | 370 one-hot columns is too many; frequency answers "how proven is this rocket?" |

For the target, four raw outcomes collapse into two (on the final 4,628-row table):

```text
Success          → 1        (4,160 rows, ~89.89%)
Failure          → 0
Partial Failure  → 0        (468 rows, ~10.11% — "not a full success")
Prelaunch Fail.  → 0
```

(Trade-off accepted: partial failures lose their nuance — remembered in Part G of
Activity 3 as a data-quality insight.)

## 1.4 The split (Part H) — the golden rule moment

- **Stratified 80/20 split**, `random_state=42` → 3,702 train / 926 test, both keeping
  the ~90/10 mix (stratification prevents a training set that accidentally has almost
  no failures to learn from).
- **All preprocessing (scalers, encoders) fitted on X_train only**, then *applied* to
  X_test. Fitting on all data first would let test statistics (means, frequencies)
  sneak into training — leakage.
- Saved artifacts: `X_train.npy`, `X_test.npy`, `y_train.npy`, `y_test.npy`,
  `feature_names.txt` (46 final columns).
- Known gap, honestly recorded: **row IDs were not saved** with the arrays (fixed in
  Activity 3 Part G).

> **Say it like this:**
> "The test set is a sealed envelope. We built the entire data factory without opening
> it — scaling and encoding rules were learned from training data alone, then *applied*
> to the test rows."

---

# PART 2 — ACTIVITY 2: THE FAIR BASELINE AND THE FIRST REAL MODEL

Notebook: `activity_2_baseline_model.ipynb` (Parts A–G).

## 2.1 The two contenders and why

| | DummyClassifier | DecisionTreeClassifier |
|---|---|---|
| Role | **The floor** — the best a knowing-nothing model can do | **The first real learner** |
| Predicts | "Success" for everything | if/then rules from features |
| Config | `strategy="most_frequent"` | `max_depth=5, class_weight="balanced", random_state=42` |

**Why a dummy at all?** Because it answers a vital question: *"Can a model with zero
knowledge reach this score?"* If our model can't beat it, the features hold no signal.

**Why these exact tree settings:**
- `max_depth=5` — anti-overfitting: a deeper tree memorizes; a shallow one generalizes.
- `class_weight="balanced"` — anti-imbalance: mistakes on the rare class (failures)
  cost more, so the tree doesn't just ignore them. *(This does not reduce overfitting —
  it fixes a different problem.)*
- `random_state=42` — reproducibility.

## 2.2 The four metrics (learn these cold)

With **Not-Success = the positive class** (`pos_label=0` — because detecting failures
is the point; "positive" here means *the class we're focusing on*, not a good outcome):

| Metric | Question it answers | Tree result |
|---|---|---|
| **Accuracy** | Overall, how often right? — *misleading under imbalance* | 0.7635 |
| **Precision** | Of missions flagged as failures, how many really failed? | 0.2407 |
| **Recall** | Of all real failures, how many caught? | 0.6170 |
| **F1** | The balance of precision and recall (harmonic mean) | **0.3463** ← primary |

**Confusion matrix** (rows = truth, columns = guess), test set, 926 missions:

```text
                 Predicted
              Not-Success  Success
Actual Not-S       58 (TP)     36 (FN)   → 94 real failures
Actual  Success   183 (FP)    649 (TN)   → 832 successes
```

## 2.3 The dummy's "90% trap"

Dummy accuracy ≈ **0.8985** — looks nearly perfect — because predicting "Success" always
is right 90% of the time **when 90% of missions succeed**. But it catches **zero**
failures: F1 = 0.0000. *It looks better than the tree on accuracy and is infinitely
worse on the metric that matters.* This is THE argument for choosing F1 over accuracy,
and it comes with numbers.

## 2.4 Results and the improvement statement

```text
Primary metric — Not-Success F1
  Dummy baseline: 0.0000
  Decision tree : 0.3463   (+0.3463 over the floor)
```

- The tree detects **58 of 94** failures (recall 0.6170) that the dummy misses entirely.
- Cost: **183 false alarms** and lower accuracy (0.7635). The tree improves failure
  recall, but many successful missions are incorrectly flagged as Not-Success —
  a model that *reports*, not one ready to cancel missions.

## 2.5 Part F error analysis (learn the two error types)

- **False negatives** (FN, 36): real failures predicted "Success" — the costly kind
  for our objective. Reduce with **higher recall**.
- **False positives** (FP, 183): successes flagged as failures — wasted alarms
  (low precision, 0.2407).
- Pattern found: classes **overlap** — no single feature separates them. Risky-looking
  missions (high prior failure rate, early/exotic rockets) often still fly.
- Hidden limitation: arrays lacked row IDs → we could only say "test row position N."

## 2.6 Reflection conclusions (agreed answers)

1. **Better than baseline?** Yes on the metric that matters (F1 0→0.3463). Fitting on
   train only and scoring once on untouched test means the win is *generalization,
   not memorization*. Trade-off (low precision) acknowledged.
2. **Biggest decision?** Choosing Not-Success F1 over accuracy, plus class weighting —
   without them the 90% majority class would have made the dummy look like a genius.
3. **Fail-on-unseen evidence?** 183 noisy false alarms; no CV or train-vs-test gap measured;
   random split mixes 1957–2022 eras making the task easier than reality.
4. **Next step?** **Chronological holdout** — train on the past, test on the future —
   the real deployment question.

---

# PART 3 — ACTIVITY 3: IMPROVE, COMPARE, EXPLAIN

Notebook: `activity_3_improve_compare_explain.ipynb` (Parts A–H).

## 3.1 Part A — same rules, new ambition

Restate (identical to Activity 2): 3,702/926 split, ~90/10 imbalance, Not-Success F1 as
primary, and the bars to beat — Dummy 0.0000 and the Act-2 tree 0.3463.

## 3.2 Part B — Five candidates line up

```text
Dummy                (the floor)
Decision Tree        (re-enters for a fair race)
LogisticRegression   (the valid *linear* classifier — LinearRegression is WRONG for
                      a categorical outcome)
RandomForest         (300 trees in the race; tuned to 500 later in Part D)
HistGradientBoosting (trees built in sequence, each fixing the last one's errors)
```

## 3.3 Part C — Fair judging: 5-fold cross-validation

**Why not one test?** One exam can be lucky. Instead:

```text
Split training rows into 5 piles. Round k: pile k = exam, other 4 = study.
Each row tested exactly once. Score = average ± spread.
```

- All folds **stratified** (same 90/10 mix per fold).
- **Training data only** — CV is for *choosing*; the sealed test is for *final judgment*.

```text
CV scores (Not-Success F1, train-only):
Random Forest        0.3489 ± 0.0642   ← winner
HistGradientBoosting 0.3480 ± 0.0437
Decision Tree        0.3223 ± 0.0346
LogisticRegression   0.3130 ± 0.0192   (most consistent, least strong)
Dummy                0.0
```

## 3.4 Part D — Tuning: GridSearchCV

18 knob combinations × 5 folds each; winner on CV (never on test):

```text
n_estimators=500 (500 trees vote together), max_depth=12, min_samples_leaf=2
class_weight="balanced" kept, random_state=42, no resampling (no SMOTE)
```

**Where overfitting is handled — four layers:**
1. `max_depth` + `min_samples_leaf` cap every individual tree.
2. 500 voting trees **average away** individual quirks (the ensemble safety net).
3. Selection via CV prevents "tuning-to-the-exam" overfitting.
4. Cautious evidence: CV (0.3842) → test (0.4215) are *consistent with* genuine
   generalization, but not proof — no train-vs-test gap was reported, and the
   chronological holdout remains the definitive check.

## 3.5 Part E — The one sealed exam (used exactly once)

```text
                       Act-2 tree → Tuned forest
Not-Success F1:         0.3463 → 0.4215  (+0.0752, ~22% relative)  ✅ new champion
Accuracy:               0.7635 → 0.8369
Precision:              0.2407 → 0.3293  (112 vs 183 false alarms)
Recall:                 0.6170 → 0.5851  (slight deliberate loss)

Forest confusion matrix (labels [Not-Success, Success]):
  caught + flagged correctly: 55   |  false alarms: 112
  missed failures:            39   |  correct successes: 720
```

**Story:** *fewer false alarms, slightly lower detection.* It trades a little recall for
much better precision — the same trade-off dial from Activity 2, now deliberately
rebalanced.

## 3.6 Part F — What drives the model

| Rank | Feature | Weight | Plain meaning |
|---|---|---|---|
| 1 | Company_Prior_Failure_Rate | 0.2305 | The operator's own track record |
| 2 | Company_Prior_Launches | 0.1771 | Operator experience |
| 3 | Launch_Year | 0.1662 | Technology era |
| 4 | Rocket_Freq | 0.1517 | Rare/unproven rockets look risky |

Top 4 carry **72.5%** of reliance. The model reads *operator track record + era* —
the same story the Activity 2 tree told, resolved with more votes. **Caution:
importance = reliance, not causation** (and no cost/payload/vehicle-maturity data
exists in the dataset).

## 3.7 Part G — Errors, now with names

Part G re-derived test-row identities from the engineered CSV — **verified
label-identical to the saved `y_test` before trusting the join** (leakage discipline
again) — fixing Activity 2's "test row 673" limitation.

- **False alarms = missions flying against the operator's own good record:**
  JAXA Epsilon PBS (2019), MHI H-IIA (2002), GD Atlas (1964). History says "risky";
  they flew fine.
- **Missed failures = reliable systems' rare tail failures:** RVSN USSR Cosmos-3M (1980),
  Cosmos-2I (1971), Lockheed Athena I (1995). Spotless records let one bad day through
  — like a 20-year accident-free driver's first crash.
- **Data limits exposed:** no first-flight indicator, no vehicle maturity, several
  "failures" are actually *partial* failures of mature systems — the binary collapse
  makes them hard to separate.

## 3.8 Part H — Model card and honest limits

- **Use case:** decision-support risk screen, never a standalone go/no-go call.
- **F1 = 0.4215** held-out; +0.0752 over the Activity 2 bar, ~22% relative improvement.
- **Limitations:** no chronological split (splash of future-era style into training),
  importance ≠ causation, missing technical features.
- **Next step unchanged from Act-2 reflection:** the chronological holdout.

---

# PART 4 — THE CONNECTED THREAD (for presenting)

Present the project as **one chain of disciplined decisions**, each fixing the risk
the last one exposed:

```text
1. Raw data had hidden traps (dupes, leakage-prone columns, imbalance)
      → fix: profile first, written reasoning for every column decision.

2. "Prior" features look powerful but could secretly leak
      → fix: compute history from earlier missions only.

3. Encodings could invent fake order/massive sparsity
      → fix: one-hot for Company/Country, frequency for Rocket, sin/cos for months.

4. A single 80/20 split could be lucky — and if scaling leaked, all scores lie
      → fix: stratified split, fit-preprocess-on-train-only, saved arrays.

5. Accuracy under imbalance is meaningless
      → fix: Dummy baseline exposed it (90% accuracy, 0.0000 failure detection),
        Not-Success F1 became the primary metric.

6. But one tree's score could be fluctuation — is 0.3463 real?
      → fix: 5-fold CV to choose fairly among 5 candidates (Act 3).

7. Chosen on CV, tuned via CV — but does tuning "study for the exam"?
      → fix: selection never touched the test set; test opened exactly once (Part E).

8. A better score could still be luck or memorization
      → fix: CV→test consistency (0.3842→0.4215), then explain + name the errors (G).

9. Even a passing score isn't deployment-ready
      → fix: model card + chronological holdout as the declared next step.
```

Each link answers the suspicious question about the previous one — that is the whole
methodology in 9 lines.

---

# PART 5 — Q&A QUICK ANSWERS (one per likely question)

- **Classification or regression? Why not LinearRegression?**
  `MissionStatus` is a **category** (Success/Not-Success), not a number. Regression
  predicts an unrestricted continuous value. If a linear model is wanted, the correct
  one is **LogisticRegression** (tested in Act 3; F1 0.3223).

- **Why F1 and not accuracy?**
  90/10 imbalance makes accuracy lie: the Dummy scored ~0.8985 accuracy with **zero**
  failures detected. F1 demands both catching failures AND not over-alarming.

- **Why is the dummy "not perfect" despite ~90% accuracy?**
  That accuracy is the *imbalance talking*. It detects zero failures (recall/precision
  /F1 all 0.0000). It exists to prove our improvement is real signal, not luck.

- **What is 5-fold CV, in one breath?**
  Cut training data into 5 piles; each takes one turn as the exam while the rest are
  study material; average the five scores — a fairer, luck-resistant estimate that
  never touches the sealed test set.

- **How was overfitting handled?**
  Act 2: depth-5 cap + untouched test set. Act 3: depth/leaf caps + 500-tree voting +
  CV-based tuning selection + CV→test score consistency (0.3842→0.4215).

- **Why class_weight instead of SMOTE/resampling?**
  Reweighting teaches the loss function that failures matter without inventing synthetic
  rows — simpler, no duplication risk, and both activities kept it consistently.

- **Why random split — and what's the caveat?**
  Stratified random split is standard and preserves the class mix, but data spans
  1957–2022: a random mix lets the model see "the era" of rows it must predict.
  Honest deployment = chronological holdout. This is declared the top next step.

- **Precision vs recall trade-off, in one line?**
  Class weighting made the model willing to raise more alarms; the tuned forest
  rebalanced to *fewer, sharper* alarms (precision 0.24→0.33, recall 0.62→0.59).

- **Which features drive prediction — and what's the caution?**
  Operator history, launch year, rocket frequency (top 4 = 72.5%). Caution: that is
  *model reliance*, not proof of causation.

- **What do the errors tell us?**
  False alarms = good operators doing something unusual; missed failures = reliable
  systems' rare breakdowns. Both trace to missing features (vehicle maturity, cost)
  and the 90/10 imbalance — not to a coding bug.

- **How can you see why one mission was predicted?**
  Tree rules (`export_text`), the plotted tree, `decision_path` for one mission, and
  global `feature_importances_` — with the reminder that rules describe, they don't
  cause.

- **What would you do next?**
  1. Chronological holdout (the deployment-honest test). 2. Calibration to steer
  precision/recall at deployment. 3. Collect real per-mission features (cost, payload).

---

# PART 6 — FILE MAP

| File | Role |
|---|---|
| `space_missions.csv` | Raw source (4,630 × 9) |
| `oop.ipynb` / `oop_refinement.ipynb` | Activity 1: cleaning, features, leakage checks, split (Parts A–I) |
| `X_train.npy`, `X_test.npy`, `y_train.npy`, `y_test.npy`, `feature_names.txt` | Sealed split artifacts (3,702×46 / 926×46) |
| `activity_2_baseline_model.ipynb` | Activity 2: Dummy vs Decision Tree, metrics, error analysis |
| `activity_3_improve_compare_explain.ipynb` | Activity 3: 5-model CV race, tuning, final eval, explanation |
| `_docs/` | All reference docs, including this guide |
| `_docs/model_fit_research.md` | Research backing the model-race choices |
