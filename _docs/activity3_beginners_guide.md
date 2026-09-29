# Activity 3 Beginner's Guide — Improve, Compare, and Explain

This guide explains the Activity 3 notebook (`activity_3_improve_compare_explain.ipynb`)
in simple, everyday language, part by part. Read it alongside the Activity 2 Q&A
findings (`activity2_qna_findings.md`) — everything learned there still applies.

---

## The Big Picture

Activity 2 gave us a starting point:

- A **dumb baseline** (DummyClassifier) that always predicts "Success" → catches **zero** failures.
- A **simple model** (shallow Decision Tree, depth 5) → Not-Success F1 = **0.3463**.

Activity 3 asks: **"Can we do better?"** It answers in three moves:
1. **Improve** — try stronger models.
2. **Compare** — pick the winner fairly (not by luck).
3. **Explain** — show what the winner looked at and where it still fails.

Headline result:

```text
Activity 2 tree:            Not-Success F1 = 0.3463
Activity 3 tuned forest:    Not-Success F1 = 0.4215   (+0.0752, ~22% better)
```

---

## Part A — Re-establish the evaluation design

A quick restatement so the notebook is self-contained:

- Same data: **3,702 train / 926 test** missions (the same saved arrays from Activity 1).
- Same target: **Not-Success (0) vs Success (1)**, about 90/10 imbalanced.
- Same primary metric: **Not-Success F1** (failures are the "positive" class we focus on).
- The Activity 2 numbers are **recomputed exactly** as the bar to beat.

> Beginner point: Activity 3 changes nothing about the data or the split. It only
> improves the *model* on top of it.

---

## Part B — The Candidates

Five models line up for a fair race:

| Candidate | What it is in plain words |
|---|---|
| **DummyClassifier** | The no-brainer: always predicts "Success". The floor everyone must beat. |
| **DecisionTreeClassifier** | One expert reading if/then rules (our Activity 2 tree, in the race for fairness). |
| **LogisticRegression** | A simple "risk score" weighted sum — the valid *linear* classifier (not LinearRegression). |
| **RandomForestClassifier** | Many mini-trees **voting together** (300 trees in the initial comparison; tuned to 500 in Part D). One tree = one opinion; a forest = trees averaged. |
| **HistGradientBoosting (HGB)** | Trees built one after another, each fixing the previous ones' mistakes. |

All five see **identical data** and are judged by the **same scorer** (Not-Success F1),
so the comparison is fair.

---

## Part C — Fair Judging: 5-Fold Cross-Validation

**The problem:** one single exam can go luckily or unluckily.

**The solution:** cut the 3,702 training rows into **5 equal piles**. Take 5 rounds.
In each round, one different pile is the "exam" and the other four are "study material":

```text
Pile:      1      2      3      4      5
Round 1   💥     study  study  study  study
Round 2   study  💥     study  study  study
Round 3   study  study  💥     study  study
Round 4   study  study  study  💥     study
Round 5   study  study  study  study  💥
```

- Every row is tested **exactly once** and used for study in the other four rounds.
- Final score = **average of the 5 exam scores** (± the spread, which shows stability).

**Two rules:**
1. CV uses **training data only** — the 926 test rows stay sealed for the final exam (Part E).
2. Folds are **stratified** — each exam keeps the same ~90/10 class mix.

**Results (training-only, 5-fold CV):**

```text
Random Forest        0.3489  ± 0.0642   ← practice-test winner
HistGradientBoosting 0.3480  ± 0.0437   ← very close second
Decision Tree        0.3223  ± 0.0346
LogisticRegression   0.3130  ± 0.0192   ← most consistent, least strong
Dummy                0.0000             ← the floor
```

> Beginner point: a single exam can flatter a model. Averaging five exams exposes
> luck. That is why Part C judges candidates, and Part E judges the final choice.

---

## Part D — Tuning the Winner (GridSearchCV)

The Random Forest won, so we fine-tuned its knobs:

| Knob | Winner | What it means |
|---|---|---|
| `n_estimators=500` | 500 trees vote on every mission instead of one tree's opinion. |
| `max_depth=12` | Each tree may be a bit deeper than Activity 2's depth-5 tree — but still capped, so it can't memorize missions. |
| `min_samples_leaf=2` | A rule must cover at least 2 missions — no conclusions from a single fluke row. |
| `class_weight="balanced"` | Kept from Activity 2: failures matter more in the scoring. |

How it chose: `GridSearchCV` tried **18 combinations** of these knobs. Each combination
was judged by 5-fold CV **on training data only**. Best average won:

```text
Untuned RF CV score:  0.3489
Tuned RF CV score:    0.3842
```

**Where overfitting is handled:**
1. `max_depth=12` and `min_samples_leaf=2` cap each tree's complexity directly.
2. 500 trees voting **average out** individual trees' quirks — the forest's built-in safety net.
3. Selecting the knobs on CV (not on the test set) prevents "tuning to the exam" —
   a sneaky form of overfitting where you optimize against the final exam itself.
4. Evidence of real generalization: the CV score (0.3842) **held up** on the sealed
   test set later (0.4215). An overfit model would collapse between the two.

---

## Part E — The One Real Exam

The sealed test set (926 never-seen missions) is opened **exactly once**:

```text
                          Activity 2 tree → Tuned forest
Not-Success F1:              0.3463    →    0.4215  ← new champion
Accuracy:                    0.7635    →    0.8369
Precision (failure alerts):  0.2407    →    0.3293
Recall (failures caught):    0.6170    →    0.5851  (slight loss)

Confusion matrix (labels [Not-Success, Success]):
  caught failures      = 55
  missed failures      = 39
  false alarms         = 112   (Activity 2 tree had 183)
  correct successes    = 720
```

The story: **fewer false warnings with similar, though slightly lower, detection.** Precision jumped because
the forest raises ~30% fewer false alarms while still catching about 55 of the
94 real failures. The small recall loss is the *deliberate* precision/recall trade-off
— the statement the assignment expects.

---

## Part F — Explaining the Model (Feature Importance)

What the forest relies on (this is **model reliance, not causation**):

| Rank | Feature | Importance | Plain meaning |
|---|---|---|---|
| 1 | Company_Prior_Failure_Rate | 0.2305 | How often has this operator failed before? |
| 2 | Company_Prior_Launches | 0.1771 | How experienced is the operator? |
| 3 | Launch_Year | 0.1662 | Which technology/safety era? |
| 4 | Rocket_Freq | 0.1517 | Rare, low-flight-history rockets look risky. |

- Top 4 features = **72.5%** of the model's reliance; top 15 of 46 = **92.5%**.
- The model is essentially reading **operator track record + era** — the same story
  the Activity 2 tree told, but resolved more reliably through many trees.

> Beginner caution: "important feature" ≠ "causes failure". Prior failure rate may
> predict failure because risky organisations keep failing — not because history
> itself makes a mission fail.

---

## Part G — Error Analysis, Now With Real Names

Activity 2 could only say "test row position 673" because IDs were not saved.
Part G re-derives the test rows from the engineered CSV (verified label-identical
against the saved `y_test` before joining) — so errors now have **Company, Rocket, Date**.

**False alarms (predicted failure, actually succeeded) — type: risky-looking but fine:**
- JAXA's small Epsilon PBS (2019)
- MHI's H-IIA in 2002
- General Dynamics Atlas (1964)
- Pattern: missions flown *against the operator's own good history* (early/exotic vehicle,
  high-variance era) — the history features panic, the flight succeeds.

**Missed failures (predicted success, actually failed) — the costliest kind:**
- RVSN USSR Cosmos-3M (1980) and Cosmos-2I (1971)
- Lockheed Athena I (1995)
- Historically **reliable** systems with rare tail failures — a spotless record lets
  the odd failure slip through, exactly like a 20-year accident-free driver who
  finally crashes.

---

## Part H — Updated Model Card (summary)

- **Objective:** pre-launch risk screen — decision support, never a stand-alone go/no-go call.
- **Method:** tuned RandomForest, class-weighted, `random_state=42`.
- **Performance:** held-out Not-Success F1 = **0.4215** (bar was 0.3463).
- **Drivers:** operator history, launch year, rocket frequency.
- **Limitations:** random (not chronological) split; no cost/payload/vehicle-maturity
  features in the dataset; importance ≠ causation.
- **Next step:** a **chronological holdout** — train on older missions, test on newer ones —
  to prove the score survives true forward prediction.

---

## Final Reflection (Activity 3 version)

1. **Chosen model?** Tuned RandomForest — led the 5-fold training-only comparison and
   confirmed the edge on the sealed test set (+0.0752 F1).
2. **Is the complexity worth it?** Yes, with a caveat: robust across folds and clearly
   better on the primary metric; costs are slower fitting, lower per-tree interpretability,
   and a small recall sacrifice to buy precision.
3. **Biggest constraint?** No per-mission technical context (cost, payload mass, weather,
   vehicle maturity) — the model leans on blunt proxies like history and era.
4. **Next test:** chronological holdout and performance-drift analysis first; then
   probability calibration; then collecting the missing real-world features.

---

## Concepts Introduced in Activity 3 (vs Activity 2)

| Concept | Activity 2 | Activity 3 |
|---|---|---|
| Model choice | One tree vs dummy | 5 candidates raced |
| Validation | One random 80/20 test split | + 5-fold CV on training data |
| Tuning | Hand-set `max_depth=5` | GridSearchCV over 18 configs |
| Ensembling | No | 500 trees voting (Random Forest) |
| Error analysis | Test-row positions only | Real mission identities via `Source_Row_ID` |
| Test-set discipline | Used once at the end | Same — still used once at the end |

## Quick glossary

- **Cross-validation (CV):** split training data into 5 piles; each pile takes its turn
  as an exam; average the scores for a luck-free estimate.
- **Hyperparameter:** a setting chosen *before* training (like `max_depth`), not learned
  *from* the data.
- **GridSearchCV:** a robot that tries every combination of settings and keeps the best
  CV performer.
- **Ensemble:** many models combined by voting; averaging cancels individual mistakes.
- **Feature importance:** how much the model *uses* each feature — reliance, not causation.
- **Chronological holdout:** train on the past, test on the future — the honest
  simulation of real deployment.
