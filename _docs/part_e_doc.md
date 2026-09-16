### Part E — Feature Engineering

The guide asks for at least **one** new feature with domain value, plus an
explanation of *why it beats the raw columns*. This section builds two, and
deliberately rejects two others after testing them.

---

#### E.1 — What we are engineering, and why

Part B's clustering found something specific: a mid-era cluster with the
**lowest** failure rate of all three groups (4.2%) and a median of **1,270 prior
launches**. If calendar year alone drove reliability, that cluster would sit
*between* the early and modern groups. It does not. Something other than time
separates it — and that something is **accumulated operator experience**.

So the engineered features encode experience. Two questions have to be answered
carefully:

1. **What exactly do we measure?** (how many launches, how many failures, a rate?)
2. **How do we avoid leaking the answer?** (see the trap below)

---

#### E.2 — The leakage trap (the most important idea in Part E)

This is the part that is easy to get wrong, and getting it wrong looks like
*success* rather than failure.

The natural way to compute "this operator's failure history" is:

```python
df["prior_failures"] = df.groupby("Company")["is_fail"].transform("sum")
```

That one line **leaks the target**. It gives every row the operator's *total*
failure count across the whole dataset — including the outcome of the very
mission being predicted, and of every mission that happened *after* it. In
production, none of that information exists at liftoff.

The correct computation uses only strictly earlier missions:

```python
prior_failures = group["is_fail"].cumsum() - own_is_fail   # subtract own row
prior_launches = group.cumcount()                          # counts before this row
```

**We measured the difference.** Training the same decision tree on honest versus
leaky features:

| Features | Cross-validated F1 |
| :--- | :--- |
| Honest (causal) | **0.1805** |
| Leaky (company totals) | **0.2700** |

The leaky version scores roughly **50% higher** — and is worthless. This is why leakage
is dangerous: it does not announce itself, it flatters your metrics. A student who
did not check would submit the leaky version believing it was better.

---

#### E.3 — Why these features beat the raw columns

| Raw column | What it cannot express | What the engineered feature adds |
| :--- | :--- | :--- |
| `Company` | A 62-value label. The model must learn each operator's reliability separately, and cannot represent *change over time* within one operator. | `Company_Prior_Launches` is continuous, so it generalises to operators with only 2–3 records. |
| `Date` | Captures the *global* era, but conflates two separate causes: industry-wide technological maturation (shared by everyone) and operator-specific learning (varies by organisation). | `Company_Prior_Failure_Rate` isolates the operator-specific component. |

A raw one-hot `Company` column also cannot answer "has this organisation improved?"
An experience count can.

---

#### E.4 — The two features we keep, and the two we rejected

| Feature | Decision | Reason |
| :--- | :--- | :--- |
| `Company_Prior_Launches` | **Keep** | The primary experience measure. Failure rate falls monotonically across its buckets. |
| `Company_Prior_Failure_Rate` | **Keep** | Adds the *quality* dimension — an operator with 50 launches and 10 failures differs from one with 50 and 0. |
| `Company_Prior_Failures` | **Reject** | Correlates **0.952** with `Company_Prior_Launches` — near-redundant. Adds a column without adding information. |
| `Is_First_Company_Launch` | **Reject** | Derivable from `prior_launches == 0`; the count already carries it. |

*Note on honesty:* repeated cross-validation showed all four together scoring
marginally best (0.1880 vs 0.1805 for two), but the gap is ~0.005 against a
standard deviation of ~0.05 — inside the noise. We keep two because they are
defensible and non-redundant, not because two scored highest in one run.

---

#### E.5 — The signal, measured

Failure rate by how many launches the operator had already flown:

| Experience bucket | Missions | Failure rate |
| :--- | :--- | :--- |
| 1st launch | 62 | **41.9%** |
| 2–6 | 224 | 32.1% |
| 7–21 | 395 | 19.5% |
| 22–101 | 1,238 | 9.0% |
| 102–501 | 1,434 | 8.2% |
| 502+ | 1,276 | **5.1%** |

Against a global rate of **10.1%**, an operator's first launch is roughly **eight
times** more failure-prone than their 500th. That is a strong, monotonic signal —
and it is not visible in any single raw column.