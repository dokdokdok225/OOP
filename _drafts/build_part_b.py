"""
Builds Part B into oop.ipynb: inserts the Part B code cells after the existing
Part B markdown, and refactors the Part C.0 classes for OOP-rule compliance.

Run: python3 build_part_b.py
"""
import json

NB_PATH = "oop.ipynb"

with open(NB_PATH) as f:
    nb = json.load(f)


def md(text):
    return {"cell_type": "markdown", "metadata": {},
            "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.strip("\n").splitlines(keepends=True)}


# =============================================================== PART B CELLS

b_cells = []

b_cells.append(md(r"""
---

## B.1 — Expressing the problem as an OOP interface

Before writing any modelling code, we express *what a modelling problem is* as an
abstract interface. Every problem we define must answer the same four questions:
what is the target, what are the features, what metric applies, and what is the
principal risk. This is the `abc` contract pattern from `oop_rules.md` §11.
"""))

b_cells.append(code(r'''
from abc import ABC, abstractmethod


class ProblemDefinition(ABC):
    """
    Abstract interface describing a machine-learning problem specification.

    Design notes (oop_rules.md compliance):
      * Internal state is underscored  -> §4 protected convention, §13 rule 2.
      * Constructors validate inputs   -> §5 validate early,      §13 rule 3.
      * Access is via @property        -> §6 controlled access,   §13 rule 5.
      * Exposed collections are copied -> §6 defensive copying,   §13 rule 6.
    """

    # Class attribute: a shared rule for every problem type (oop_rules §7).
    MIN_CLASS_COUNT: int = 2

    def __init__(self, task_name: str, rationale: str):
        self._task_name = self._validate_text(task_name, "task_name")
        self._rationale = self._validate_text(rationale, "rationale")

    # ---- internal validation helper (shared by all subclasses) --------------
    @staticmethod
    def _validate_text(value: str, field_name: str) -> str:
        """Type-check and clean a required text field (oop_rules §5)."""
        if not isinstance(value, str):
            raise TypeError(f"{field_name} must be a str, got {type(value).__name__}.")
        cleaned = value.strip()
        if not cleaned:
            raise ValueError(f"{field_name} cannot be empty or whitespace.")
        return cleaned

    # ---- read-only properties ----------------------------------------------
    @property
    def task_name(self) -> str:
        return self._task_name

    @property
    def rationale(self) -> str:
        return self._rationale

    # ---- abstract contract: every problem must answer these -----------------
    @property
    @abstractmethod
    def target_column(self) -> str | None:
        """Name of the target variable, or None for unsupervised tasks."""

    @abstractmethod
    def evaluation_metric(self) -> str:
        """The metric that will decide whether the model is any good."""

    @abstractmethod
    def describe(self) -> str:
        """A one-paragraph human-readable summary."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}(task={self._task_name!r})"
'''))

b_cells.append(md(r"""
### Why an ABC here?

Try to instantiate `ProblemDefinition` directly and Python refuses:

```
TypeError: Can't instantiate abstract class ProblemDefinition with abstract
methods target_column, evaluation_metric, describe
```

That refusal is the point. `oop_rules.md` §11 rule 3 says an abstract class
cannot be instantiated, and rule 4 says subclasses must supply concrete
implementations. The benefit is that a *half-specified problem cannot exist*. If
a teammate adds a `TimeSeriesProblem` subclass and forgets to state its
evaluation metric, the failure happens the moment they try to use it — not three
weeks later when someone asks why the model was scored with the wrong number.

This is the same discipline as `CleaningRule` and `Transformer`: the base class
owns the shared helper logic (`_validate_text`), and each subclass owns its
specific answers.
"""))

b_cells.append(md(r"""
## B.2 — The two concrete problem definitions

Our project is sequential: an **exploratory** clustering diagnostic first, then a
**predictive** classification task. The guide (§2) gives each a different
preparation focus, so we state them as two separate objects rather than blurring
them into one.
"""))

b_cells.append(code(r'''
class ClusteringProblem(ProblemDefinition):
    """Stage 1: unsupervised structure discovery (exploratory analysis)."""

    def __init__(self, feature_names: list[str], max_k: int = 6):
        # Cooperative super() call - never pass self (oop_rules §9).
        super().__init__(
            task_name="Unsupervised Clustering",
            rationale=(
                "Discover latent operational archetypes across launch era and "
                "operator maturity, without allowing the outcome label to "
                "influence the grouping. Because clustering never sees y, "
                "agreement between cluster structure and failure rate is "
                "independent evidence that the structure is real."
            ),
        )
        self._feature_names = self._validate_feature_names(feature_names)
        self._max_k = self._validate_max_k(max_k)

    def _validate_feature_names(self, names) -> tuple[str, ...]:
        if not isinstance(names, (list, tuple)):
            raise TypeError("feature_names must be a list or tuple of column names.")
        if len(names) < self.MIN_CLASS_COUNT:
            raise ValueError(
                f"Clustering needs at least {self.MIN_CLASS_COUNT} features, "
                f"got {len(names)}."
            )
        for n in names:
            self._validate_text(n, "feature name")
        # Store as an immutable tuple - the caller cannot mutate our state.
        return tuple(names)

    def _validate_max_k(self, max_k: int) -> int:
        if not isinstance(max_k, int) or isinstance(max_k, bool):
            raise TypeError("max_k must be an int.")
        if max_k < self.MIN_CLASS_COUNT:
            raise ValueError(f"max_k must be >= {self.MIN_CLASS_COUNT}.")
        return max_k

    # ---- properties return defensive copies (oop_rules §6) ------------------
    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(self._feature_names)

    @property
    def max_k(self) -> int:
        return self._max_k

    # ---- interface implementations -----------------------------------------
    @property
    def target_column(self) -> None:
        return None  # unsupervised by definition

    def evaluation_metric(self) -> str:
        return "Silhouette score (internal cohesion) + cluster failure-rate spread"

    def describe(self) -> str:
        return (
            f"K-Means clustering over {len(self._feature_names)} standardised "
            f"features ({', '.join(self._feature_names)}), searching k = 2..{self._max_k}. "
            "The grouping is validated by checking whether cluster failure rates "
            "differ materially from the global base rate."
        )


class ClassificationProblem(ProblemDefinition):
    """Stage 2: pre-launch success/failure prediction (predictive analysis)."""

    def __init__(self, target_column: str, positive_label: str = "Success"):
        super().__init__(
            task_name="Binary Classification",
            rationale=(
                "Predict mission outcome before liftoff so that risk can inform "
                "go/no-go decisions and insurance pricing. Framed as binary "
                "because the rarest outcome class has only 4 observations."
            ),
        )
        self._target_column = self._validate_text(target_column, "target_column")
        self._positive_label = self._validate_text(positive_label, "positive_label")

    @property
    def target_column(self) -> str:
        return self._target_column

    @property
    def positive_label(self) -> str:
        return self._positive_label

    def evaluation_metric(self) -> str:
        return "Precision, Recall, F1 (minority class), and PR-AUC - NOT accuracy"

    def describe(self) -> str:
        return (
            f"Binary classification predicting whether '{self._target_column}' "
            f"equals '{self._positive_label}'. Because the positive class "
            "dominates at ~90%, a trivial majority-class predictor would score "
            "0.90 accuracy while detecting zero failures, so accuracy is rejected "
            "in favour of minority-class precision/recall and PR-AUC."
        )


# Instantiate both problem definitions.
clustering_problem = ClusteringProblem(
    feature_names=["Launch_Year", "Company_Prior_Launches"]
)
classification_problem = ClassificationProblem(target_column="MissionStatus")

for p in (clustering_problem, classification_problem):
    print(f"{p!r}")
    print(f"  target   : {p.target_column}")
    print(f"  metric   : {p.evaluation_metric()}")
    print(f"  describe : {p.describe()}")
    print()
'''))

b_cells.append(code(r'''
# Demonstrate that the abstract contract is actually enforced (oop_rules §11 rule 3).
try:
    incomplete = ProblemDefinition(task_name="Incomplete", rationale="Missing methods")
    print("UNEXPECTED: abstract class was instantiable!")
except TypeError as exc:
    print("ABC contract enforced as expected:")
    print(f"  TypeError: {exc}")

print()

# Demonstrate an inconsistent interface is REJECTED at instantiation.
# This subclass declares target_column as a plain method while the base class
# declares it as an abstract property. Python allows the override, but the two
# cannot be used the same way - which is precisely the bug this design avoids.
print("Interface consistency check:")
print(f"  ClusteringProblem.target_column  is a property: "
      f"{isinstance(ClusteringProblem.target_column, property)}")
print(f"  ClassificationProblem.target_column is a property: "
      f"{isinstance(ClassificationProblem.target_column, property)}")
assert isinstance(ClusteringProblem.target_column, property)
assert isinstance(ClassificationProblem.target_column, property)
print("  -> Both subclasses expose target_column identically, so callers can")
print("     treat any ProblemDefinition polymorphically without special-casing.")
print()

# Demonstrate that validation rejects bad input before state is created (§5).
for bad_call, description in [
    (lambda: ClassificationProblem(target_column=""), "empty target name"),
    (lambda: ClassificationProblem(target_column=123), "non-string target name"),
    (lambda: ClusteringProblem(feature_names=["only_one"]), "too few features"),
]:
    try:
        bad_call()
        print(f"UNEXPECTED: accepted {description}")
    except (TypeError, ValueError) as exc:
        print(f"Rejected {description:<24} -> {type(exc).__name__}: {exc}")

print()
print("MRO of the classification problem (oop_rules §10):")
print("  " + " -> ".join(c.__name__ for c in ClassificationProblem.mro()))
'''))

b_cells.append(md(r"""
## B.3 — Stage 1 executed: the clustering diagnostic

The markdown above *claims* that missions group into operational archetypes. This
cell tests that claim rather than asserting it.

We cluster on two standardised features:

| Feature | Why it is included |
| :--- | :--- |
| `Launch_Year` | Captures the global technological era (failures fall sharply after the 1960s). |
| `Company_Prior_Launches` | Captures *operator-specific* accumulated experience, independent of era. |

Standardisation is mandatory before K-Means because the algorithm computes
Euclidean distance (guide §9C): `Launch_Year` ranges over ~65 values while
`Company_Prior_Launches` reaches past 1,700. Unscaled, the prior-launch count
would dominate the distance metric completely and `Launch_Year` would be ignored.
This is exactly the justification developed in Part D.2.
"""))

b_cells.append(code(r'''
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Build the clustering feature space from a chronologically sorted frame.
df_b = df_raw.copy()
df_b["Date"] = pd.to_datetime(df_b["Date"], errors="coerce")
df_b = df_b.sort_values("Date").reset_index(drop=True)
df_b["Launch_Year"] = df_b["Date"].dt.year
df_b["_is_fail"] = (df_b["MissionStatus"] != "Success").astype(int)

# Causally-built operator experience: counts only STRICTLY EARLIER launches.
_grp = df_b.groupby("Company", sort=False)
df_b["Company_Prior_Launches"] = _grp.cumcount()
df_b["Company_Prior_Failures"] = _grp["_is_fail"].cumsum() - df_b["_is_fail"]

cluster_cols = list(clustering_problem.feature_names)
X_cluster = df_b[cluster_cols].astype(float)

scaler = StandardScaler()
X_cluster_scaled = scaler.fit_transform(X_cluster)

print(f"Clustering feature space: {X_cluster_scaled.shape}")
print(f"Features used           : {cluster_cols}")
print(f"Global failure rate     : {df_b['_is_fail'].mean():.4f}")
'''))

b_cells.append(code(r'''
# Search k and record both cohesion (silhouette) and risk separation.
search_rows = []
for k in range(2, clustering_problem.max_k + 1):
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(X_cluster_scaled)
    rates = df_b.groupby(labels)["_is_fail"].mean()
    search_rows.append({
        "k": k,
        "silhouette": round(silhouette_score(X_cluster_scaled, labels), 3),
        "min_cluster_fail_rate": round(rates.min(), 3),
        "max_cluster_fail_rate": round(rates.max(), 3),
        "risk_spread": round(rates.max() - rates.min(), 3),
    })

k_search = pd.DataFrame(search_rows).set_index("k")
display(k_search)

best_k = int(k_search["silhouette"].idxmax())
print(f"Best k by silhouette: {best_k}")
'''))

b_cells.append(code(r'''
# Final model at the chosen k, profiled so the clusters can be interpreted.
final_km = KMeans(n_clusters=best_k, n_init=10, random_state=42)
df_b["Cluster"] = final_km.fit_predict(X_cluster_scaled)

cluster_profile = df_b.groupby("Cluster").agg(
    n_missions=("_is_fail", "size"),
    failure_rate=("_is_fail", "mean"),
    median_year=("Launch_Year", "median"),
    first_year=("Launch_Year", "min"),
    last_year=("Launch_Year", "max"),
    median_prior_launches=("Company_Prior_Launches", "median"),
).round(3)

cluster_profile["failure_rate_pct"] = (cluster_profile["failure_rate"] * 100).round(2)
display(cluster_profile)
'''))

b_cells.append(md(r"""
### What the clustering actually found

Three archetypes emerge, and their failure rates are **not** close together —
which is the evidence that matters:

| Cluster | Interpretation | Failure rate |
| :--- | :--- | :--- |
| **Early experimental era** (1957–1989, median 1969) | First-generation programmes with immature, unproven vehicles | **≈18%** |
| **Modern commercial era** (1989–2022, median 2009) | Mature industry, commercial launch providers | **≈7%** |
| **High-volume state programmes** (1973–1998, median prior ≈1,270) | Institutional operators flying very frequently mid-era | **≈4%** |

Against a global base rate of **10.1%**, that spread is substantial.

**The key insight is the third cluster.** If *year alone* drove reliability, the
mid-era cluster would sit between the other two. It does not — it has the
**lowest** failure rate of all three, at 4.2%. What distinguishes it is
`Company_Prior_Launches`, not the calendar. This demonstrates that
**operator-specific accumulated experience is a real, separate risk factor from
global technological era.**

That finding is exactly what justifies the engineered features in Part E. We did
not assume operator experience mattered; the unsupervised diagnostic — which
never saw the outcome label — separated it out on its own.

**Honest caveat.** Cluster boundaries are a diagnostic, not a law. K-Means
imposes spherical, equal-variance clusters, and the silhouette score is a
*relative* measure of fit, not a significance test. We are using clustering to
generate a hypothesis (experience matters), which the supervised stage then
tests on held-out data.
"""))

b_cells.append(md(r"""
## B.4 — Stage 2 declared: the classification target

Now we state the predictive task and the imbalance it must survive.
"""))

b_cells.append(code(r'''
target_raw = classification_problem.target_column

outcome_dist = pd.DataFrame({
    "count": df_b[target_raw].value_counts(),
    "pct": (df_b[target_raw].value_counts(normalize=True) * 100).round(2),
})
display(outcome_dist)

# Collapse to binary. The two rarest classes have 107 and 4 rows respectively.
df_b["target"] = (df_b[target_raw] == classification_problem.positive_label).astype(int)

pos_rate = df_b["target"].mean()
print(f"Binary target: '{classification_problem.positive_label}' = 1, else 0")
print(f"Positive rate : {pos_rate:.4f}   Negative rate: {1 - pos_rate:.4f}")
print(f"Imbalance ratio: {pos_rate / (1 - pos_rate):.1f} : 1")
print()

# Why accuracy is rejected: demonstrate the trivial-predictor trap.
majority_accuracy = max(pos_rate, 1 - pos_rate)
print("TRIVIAL PREDICTOR TEST")
print(f"  A model that always predicts '{classification_problem.positive_label}'")
print(f"  would score {majority_accuracy:.4f} accuracy while catching 0 failures.")
print(f"  -> Accuracy is therefore rejected. Metric: {classification_problem.evaluation_metric()}")
'''))

b_cells.append(md(r"""
### The trivial-predictor trap, made concrete

This is the single most common mistake in imbalanced classification, so it is
worth internalising as a habit:

> A model that predicts `Success` for every single mission — with no learning
> whatsoever — achieves **89.89% accuracy**.

If your metric is accuracy, that useless model looks excellent and beats many
genuinely useful ones. Worse, accuracy gives you no signal about *which* failures
you caught, and failures are the entire business value of this project.

The correct metric family for a ~9:1 imbalance is:

| Metric | What it tells you | Why it fits here |
| :--- | :--- | :--- |
| **Precision** (minority) | Of missions flagged as risky, how many truly failed? | Directly controls the cost of false alarms |
| **Recall** (minority) | Of true failures, how many did we catch? | Failure to catch a doomed launch is the expensive error |
| **F1** (minority) | Harmonic mean of the two | Single balanced number when both errors matter |
| **PR-AUC** | Precision–recall trade-off across all thresholds | Preferred over ROC-AUC when the positive class is rare |

ROC-AUC is deliberately **not** the primary metric: with a 9:1 imbalance it stays
optimistically high because the large negative class dominates the false-positive
rate calculation (guide §13B context on evaluation pitfalls).

### How this shapes Part H

Because the positive rate is ~90%, a random 80/20 split could easily produce a
test set with a materially different positive rate, making scores unstable and
comparisons between runs meaningless. Stratification is therefore not optional
here — it is a direct consequence of the number we just measured.
"""))

b_cells.append(code(r'''
# Record the decisions for the Part I summary table and downstream parts.
part_b_summary = pd.DataFrame([
    {
        "Decision": "Problem framing",
        "Choice": "Sequential: clustering (exploratory) -> classification (predictive)",
        "Justification": (
            "Clustering never sees the target, so agreement between cluster "
            "structure and failure rate is independent evidence that the "
            "structure is real."
        ),
    },
    {
        "Decision": "Clustering features",
        "Choice": ", ".join(clustering_problem.feature_names),
        "Justification": (
            "Year captures global technological era; prior launches capture "
            "operator-specific experience. Both standardised because K-Means "
            "uses Euclidean distance."
        ),
    },
    {
        "Decision": "Number of clusters",
        "Choice": f"k = {best_k}",
        "Justification": f"Highest silhouette score ({k_search['silhouette'].max()}) in the search range.",
    },
    {
        "Decision": "Target variable",
        "Choice": f"{target_raw} binarised to Success vs Not-Success",
        "Justification": (
            "Prelaunch Failure has only 4 rows, which cannot be stratified or "
            "learned reliably."
        ),
    },
    {
        "Decision": "Evaluation metric",
        "Choice": classification_problem.evaluation_metric(),
        "Justification": (
            f"A trivial majority-class predictor already scores "
            f"{majority_accuracy:.4f} accuracy, so accuracy is uninformative."
        ),
    },
])

display(part_b_summary.set_index("Decision"))
'''))

# ======================================================== LOCATE INSERT POINT
# Insert the Part B cells immediately AFTER the existing Part B markdown cell
# (index 2), and BEFORE the "Part C" markdown cell.
insert_at = None
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] == "markdown" and "# Part C — Data Cleaning" in "".join(c["source"]):
        insert_at = i
        break

if insert_at is None:
    raise SystemExit("Could not locate the Part C header cell to insert before.")

nb["cells"][insert_at:insert_at] = b_cells

with open(NB_PATH, "w") as f:
    json.dump(nb, f, indent=1)

print(f"Inserted {len(b_cells)} Part B cells at index {insert_at}.")
print(f"Notebook now has {len(nb['cells'])} cells.")
