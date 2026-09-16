### Part F — Check for Data Leakage

The guide asks three questions:

> Does any feature directly or indirectly expose information about the target?
> Could future information leak into past predictions?
> *Explanation:* Provide an evaluation of your dataset's leakage risks.

Leakage is dangerous because it **flatters your metrics instead of announcing
itself**. A leaked model looks excellent in validation and fails in production,
because the information it learned on does not exist at prediction time.

The governing test applied to every column:

> **Would this value be knowable in the mission-control room, before liftoff?**

---

#### F.1 — The column-by-column audit

| Column | Knowable pre-launch? | Verdict | Action |
| :--- | :--- | :--- | :--- |
| `MissionStatus` | — | **The target itself** | Excluded from `X`; becomes `y` in Part G |
| `RocketStatus` | **No** | **SEVERE — temporal leakage** | **Dropped** |
| `Mission` | Yes, but useless | Identifier + **latent leak** | Dropped as identifier |
| `Price` | Partially | Retroactive recording bias | Dropped (72.7% missing) |
| `Time` | Yes | Verified no signal | Excluded (no causal bearing) |
| `Date` | Yes | Safe when used causally | Kept → engineered in Parts D/E |
| `Company` | Yes | Safe | Kept |
| `Rocket` | Yes | Safe | Kept (frequency encoded) |
| `Country` | Yes | Safe | Kept |

---

#### F.2 — `RocketStatus`: the severe leak

This is the one unambiguous leak, and it is worth demonstrating rather than
asserting. `RocketStatus` records whether a vehicle was operational **as of
August 2022**, the date the dataset was compiled. It is not a pre-flight
condition at all.

Three pieces of evidence:

**1. The temporal tell.** Every launch before 1980 is labelled `Retired` — with
**zero exceptions** across 1,837 launches. That is not a fact about rockets; it is
an artefact of recording status in 2022. Anything launched 40+ years ago is
retired *now* by definition.

**2. Reverse causation.** `Active` vehicles show a **93.2%** success rate versus
**89.0%** for `Retired`. The direction of causation is inverted: programmes end
and unreliable vehicles are withdrawn, so the *outcome partly causes the status*
rather than the reverse. A model would exploit this backwards relationship.

**3. It is useless even where correlated.** A model trained on this column learns
"old missions used retired rockets." That statement is true by construction and
carries no predictive information about a *new* launch.

---

#### F.3 — `Mission`: a latent leak inside an identifier

`Mission` is 98.4% unique, so it was always going to be dropped as an identifier.
But auditing it revealed something more interesting: **the name itself encodes
outcome information**.

| Name pattern | Missions | Failure rate |
| :--- | :--- | :--- |
| Contains `"test"` | 23 | **52.2%** |
| Contains `"demo"` | 20 | 25.0% |
| All missions | 4,628 | 10.1% |

This is **not merely an era proxy**, which is what makes it worth documenting.
Test-named missions have a median launch year of **2018**, far *later* than the
dataset median of 1987 — yet they still fail at more than five times the base
rate. And restricting to the pre-1970 era alone, test-named missions fail at
**75.0%** versus **23.6%** for everything else.

So the pattern reflects a genuinely higher-risk mission *type* (first flights,
unproven vehicles), not just age.

We already drop `Mission`, which removes this vector. But it is documented
because it illustrates a general lesson: **an identifier can carry target
information**, and "it is just an ID" is not a sufficient reason to stop
auditing a column.

*Note:* because the mission name is known at liftoff, a `name_contains_test`
feature would be legitimate rather than leaky. We do not build one — it would
generalise poorly from a 23-row pattern — but the distinction between "known at
liftoff" and "leaky" is exactly the test being applied.

---

#### F.4 — `Price`: retroactive recording bias

`Price` was dropped in Part C for 72.7% missingness. The audit reveals the
missingness is **not random**, which is a leakage-adjacent concern.

| Decade | Launches with a price | Total launches |
| :--- | :--- | :--- |
| 1950s | 0 | 51 |
| 1960s | 33 | 774 |
| 1970s | 6 | 1,012 |
| 1990s | 132 | 642 |
| 2010s | **528** | 676 |
| 2020s | 312 | 369 |

Pricing is recorded mostly for **modern** missions: 528 of 676 in the 2010s versus
33 of 774 in the 1960s. Launch era correlates with the target, so had we imputed
`Price`, we would have injected era information into a column that looks like a
neutral financial measure. Dropping it was correct on two independent grounds.

---

#### F.5 — `Time`: verified to be harmless

Part C excluded `Time` on the reasoning that launch time-of-day has no causal
bearing on vehicle reliability. The audit confirms it rather than assuming it:

| Launch hour | Failure rate | Missions |
| :--- | :--- | :--- |
| 00:00–05:59 | 7.6% | 984 |
| 06:00–11:59 | 10.2% | 1,269 |
| 12:00–17:59 | 9.8% | 1,149 |
| 18:00–23:59 | 10.1% | 1,100 |

The spread is 7.6%–10.2% around a 10.1% base rate — essentially flat. No signal,
so excluding it costs nothing.

---

#### F.6 — The leakage guard

Preventing leakage is a claim, so it is enforced programmatically. A
`LeakageGuard` class asserts two things on every feature matrix:

1. **No forbidden column is present** — a maintained blocklist, checked at runtime.
2. **No numeric column is a near-perfect restatement of the target** — this
   catches a binary flag that is really `y` in disguise.

The second check is the important one, because a blocklist can only catch columns
someone remembered to list.
