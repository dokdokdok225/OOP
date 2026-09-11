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

*(pending)*

---

# Part D — Feature Transformation

*(pending)*

---

# Part E — Feature Engineering

*(pending)*

---

# Part F — Check for Data Leakage

*(pending)*

---

# Part G — Feature and Target Separation

*(pending)*

---

# Part H — Training and Testing Preparation

*(pending)*

---

# Part I — Preprocessing Summary Table

*(pending)*
