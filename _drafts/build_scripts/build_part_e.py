"""
Inserts Part E (Feature Engineering) into oop.ipynb.

Applies OOP fundamentals per oop_rules.md §13:
  - abstract base class + concrete subclass   (§11, §13 rule 10)
  - protected attributes with leading _       (§4,  §13 rule 2)
  - isinstance validation before assignment   (§5,  §13 rule 3)
  - @property accessors                       (§6,  §13 rule 5)
  - defensive copies of collections           (§6,  §13 rule 6)
  - class attributes for shared config        (§7,  §13 rule 7)
  - cooperative super()                       (§9,  §13 rule 8)

Run: python3 build_part_e.py
"""
import json

NB_PATH = "oop.ipynb"
DOC_PATH = "_docs/part_e_doc.md"

with open(NB_PATH) as f:
    nb = json.load(f)
with open(DOC_PATH) as f:
    part_e_doc = f.read()


def md(text):
    return {"cell_type": "markdown", "metadata": {},
            "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.strip("\n").splitlines(keepends=True)}


cells = [md(part_e_doc)]

# ------------------------------------------------------- E.6 plots setup + ABC
cells.append(md(r"""
---

#### E.6 — The feature builder as an OOP class

Per `oop_rules.md` §14, the pipeline is built from small classes with clear
responsibilities. A feature builder must answer the same questions regardless of
what it derives, so that shared shape is an **abstract interface** (`abc` §11).
"""))

cells.append(code(r'''
from abc import ABC, abstractmethod
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier

# NOTE: matplotlib and the shared chart styling are set up at the top of the
# notebook, before the Part C charts, so they are not re-imported here.
'''))

cells.append(code(r'''
class FeatureBuilder(ABC):
    """
    Abstract interface for deriving new columns from existing ones.

    OOP rules exercised (oop_rules.md):
      §4  protected attributes use a leading underscore
      §5  inputs validated before state is assigned
      §6  exposed collections are copied, not handed out live
      §7  shared configuration lives in class attributes
      §11 abstract contract enforced by ABC
    """

    # Class attribute: a shared rule for every builder (oop_rules §7).
    REQUIRES_SORTED_CHRONOLOGY: bool = True

    def __init__(self, name: str, group_col: str, date_col: str):
        # Validate BEFORE assigning, so a bad call cannot leave a half-built
        # object behind (oop_rules §5 rule 2: leave state unchanged on failure).
        self._name = self._validate_text(name, "name")
        self._group_col = self._validate_text(group_col, "group_col")
        self._date_col = self._validate_text(date_col, "date_col")
        self._history: list[str] = []
        self._is_fitted = False

    # ---- internal validation helper (shared by all subclasses) -------------
    @staticmethod
    def _validate_text(value: str, field_name: str) -> str:
        """Type-check and clean a required text field (oop_rules §5)."""
        if not isinstance(value, str):
            raise TypeError(f"{field_name} must be a str, got {type(value).__name__}.")
        cleaned = value.strip()
        if not cleaned:
            raise ValueError(f"{field_name} cannot be empty or whitespace.")
        return cleaned

    # ---- read-only properties (oop_rules §6) ------------------------------
    @property
    def name(self) -> str:
        return self._name

    @property
    def group_col(self) -> str:
        return self._group_col

    @property
    def date_col(self) -> str:
        return self._date_col

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def history(self) -> tuple[str, ...]:
        """Defensive copy - callers cannot mutate our internal list."""
        return tuple(self._history)

    # ---- abstract contract -------------------------------------------------
    @property
    @abstractmethod
    def feature_names(self) -> tuple[str, ...]:
        """Names of the columns this builder produces."""

    @abstractmethod
    def _build(self, df: pd.DataFrame) -> pd.DataFrame:
        """Concrete derivation logic. Must not mutate the input frame."""

    # ---- template method: shared validation, then subclass work -----------
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate, derive, and record. This is a 'template method': the base
        class owns the sequence, subclasses supply only the derivation.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"df must be a DataFrame, got {type(df).__name__}.")
        for col in (self._group_col, self._date_col):
            if col not in df.columns:
                raise KeyError(f"Required column {col!r} missing from the dataframe.")

        # Work on a copy so the caller's frame is never mutated.
        out = self._build(df.copy())

        produced = self.feature_names
        missing = [c for c in produced if c not in out.columns]
        if missing:
            raise RuntimeError(f"{type(self).__name__} failed to produce {missing}.")

        self._history.append(f"built {produced}")
        self._is_fitted = True
        return out

    def __repr__(self) -> str:
        state = "fitted" if self._is_fitted else "not fitted"
        return f"{type(self).__name__}(name={self._name!r}, {state})"
'''))

# ------------------------------------------------------------ concrete builder
cells.append(code(r'''
class OperationalExperienceBuilder(FeatureBuilder):
    """
    Derives accumulated operator experience features, CAUSALLY.

    The causal requirement is the whole point of this class. Every count
    excludes the current row and every later row, so nothing about the mission
    being predicted can enter its own features.
    """

    def __init__(self, group_col: str = "Company", date_col: str = "Date",
                 outcome_col: str = "MissionStatus", success_label: str = "Success"):
        super().__init__(name="OperationalExperience", group_col=group_col,
                         date_col=date_col)
        self._outcome_col = self._validate_text(outcome_col, "outcome_col")
        self._success_label = self._validate_text(success_label, "success_label")
        self._excluded: tuple[str, ...] = ()

    @property
    def feature_names(self) -> tuple[str, ...]:
        # Only the two non-redundant features are kept (see E.4).
        return ("Company_Prior_Launches", "Company_Prior_Failure_Rate")

    @property
    def excluded_reason(self) -> dict[str, str]:
        """Documents the candidates we rejected, and why."""
        return dict(self._excluded)

    def _build(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.REQUIRES_SORTED_CHRONOLOGY:
            df = df.sort_values(self._date_col).reset_index(drop=True)

        # Binary failure indicator for the running history.
        is_fail = (df[self._outcome_col] != self._success_label).astype(int)

        grouped = df.groupby(self._group_col, sort=False)

        # --- CAUSAL counts: strictly BEFORE the current row ---
        # cumcount() counts preceding rows; the shifted cumsum excludes this
        # row's own outcome. Both therefore see only the past.
        prior_launches = grouped.cumcount()
        prior_failures = (
            is_fail.groupby(df[self._group_col], sort=False).cumsum() - is_fail
        )

        df["Company_Prior_Launches"] = prior_launches
        df["Company_Prior_Failures"] = prior_failures

        # Historical failure rate, guarded against divide-by-zero.
        df["Company_Prior_Failure_Rate"] = np.where(
            prior_launches > 0, prior_failures / prior_launches, 0.0
        )

        # Record the rejected candidates for the write-up.
        self._excluded = {
            "Company_Prior_Failures": (
                "Correlates 0.952 with Company_Prior_Launches - near-redundant."
            ),
            "Is_First_Company_Launch": (
                "Derivable from prior_launches == 0; adds no information."
            ),
        }
        return df


experience_builder = OperationalExperienceBuilder()
print(repr(experience_builder))
print(f"Produces features : {experience_builder.feature_names}")
print(f"Never sees        : the current row's own outcome, or any later row")
'''))

cells.append(code(r'''
# Apply the builder.
# NOTE: `df` already carries `Company_Prior_Launches` / `Company_Prior_Failures`
# from Part D's scaling discussion. The builder is idempotent for those columns,
# so rebuilding them here is safe and keeps Part E self-contained.
df_eng = experience_builder.fit_transform(df)

print(repr(experience_builder))
print(f"history: {experience_builder.history}\n")
display(df_eng[["Company", "Date", "MissionStatus"] + list(experience_builder.feature_names)].head(8))

# The builder must not mutate the frame handed to it. We test that directly by
# snapshotting the frame before the call and comparing after.
snapshot = df.copy(deep=True)
_ = experience_builder.fit_transform(df)
after = sorted(c for c in df.columns if c not in snapshot.columns)

assert after == [], f"build() added columns to the caller's frame: {after}"
assert df.equals(snapshot), "build() mutated the caller's dataframe!"
print("\nPASS: the caller's dataframe was not mutated (defensive copy worked).")
'''))

# ---------------------------------------------------------------- OOP proofs
cells.append(code(r'''
# =====================================================================
# E.7 - Demonstrating the OOP rules actually hold
# =====================================================================

print("1. ABSTRACT CONTRACT (oop_rules §11)")
try:
    FeatureBuilder(name="x", group_col="Company", date_col="Date")
    print("   UNEXPECTED: abstract class was instantiable!")
except TypeError as exc:
    print(f"   FeatureBuilder() refused: {str(exc)[:70]}...")

print("\n2. VALIDATION BEFORE STATE (§5)")
for bad, label in [
    (lambda: OperationalExperienceBuilder(group_col=""), "empty group_col"),
    (lambda: OperationalExperienceBuilder(group_col=123), "non-string group_col"),
]:
    try:
        bad()
        print(f"   UNEXPECTED: accepted {label}")
    except (TypeError, ValueError) as exc:
        print(f"   rejected {label:<22} -> {type(exc).__name__}")

print("\n3. VALIDATION AT TRANSFORM TIME (§5)")
try:
    experience_builder.fit_transform([1, 2, 3])
except TypeError as exc:
    print(f"   non-DataFrame rejected -> {type(exc).__name__}: {exc}")

print("\n4. DEFENSIVE COPIES (§6)")
h = experience_builder.history
print(f"   history type is immutable: {type(h).__name__}")
try:
    h.append("tampering")
    print("   UNEXPECTED: external code mutated internal state!")
except AttributeError:
    print("   external append refused -> internal state is protected")

print("\n5. READ-ONLY PROPERTIES (§6)")
try:
    experience_builder.name = "hacked"
    print("   UNEXPECTED: property was writable!")
except AttributeError:
    print("   assignment to .name refused -> read-only property")

print("\n6. CLASS vs INSTANCE STATE (§7)")
print(f"   class attribute  REQUIRES_SORTED_CHRONOLOGY = {FeatureBuilder.REQUIRES_SORTED_CHRONOLOGY}")
print(f"   instance attribute _name                   = {experience_builder.name!r}")

print("\n7. MRO (§10)")
print("   " + " -> ".join(c.__name__ for c in OperationalExperienceBuilder.mro()))
'''))

# ------------------------------------------------------------------- leakage
cells.append(code(r'''
# =====================================================================
# E.8 - PROVING the leak would have mattered
# =====================================================================
# The whole point of the causal design is that the LAZY version leaks the
# target and inflates scores. We build the lazy version deliberately and
# compare, so the danger is measured rather than asserted.

grouped = df.groupby("Company", sort=False)

# LEAKY: the operator's TOTAL outcome across the entire dataset. Every row
# inherits information about its own fate and about all later missions.
df_leak = df.copy()
df_leak["Leaky_Company_Failures"] = grouped["_is_fail"].transform("sum")
df_leak["Leaky_Company_Failure_Rate"] = grouped["_is_fail"].transform("mean")
df_leak["Launch_Year"] = df_leak["Date"].dt.year

df_honest = df_eng.copy()
df_honest["Launch_Year"] = df_honest["Date"].dt.year

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)

def cv_f1(frame: pd.DataFrame, cols: list[str]) -> tuple[float, float]:
    scores = cross_val_score(
        DecisionTreeClassifier(max_depth=5, random_state=42),
        frame[cols], frame["_is_fail"], cv=cv, scoring="f1",
    )
    return scores.mean(), scores.std()

honest_cols = ["Launch_Year", "Company_Prior_Launches", "Company_Prior_Failure_Rate"]
leak_cols = ["Launch_Year", "Leaky_Company_Failures", "Leaky_Company_Failure_Rate"]

honest_f1, honest_sd = cv_f1(df_honest, honest_cols)
leak_f1, leak_sd = cv_f1(df_leak, leak_cols)

leak_comparison = pd.DataFrame([
    {"feature_set": "Honest (causal)", "features": ", ".join(honest_cols),
     "cv_f1": round(honest_f1, 4), "cv_sd": round(honest_sd, 4)},
    {"feature_set": "LEAKY (company totals)", "features": ", ".join(leak_cols),
     "cv_f1": round(leak_f1, 4), "cv_sd": round(leak_sd, 4)},
]).set_index("feature_set")

display(leak_comparison)
print(f"Leakage inflates F1 by {(leak_f1 - honest_f1):+.4f} "
      f"({(leak_f1/honest_f1 - 1)*100:+.1f}%)")
print("\nThe leaky model looks BETTER while being unusable in production:")
print("at liftoff, no one knows the outcome of the mission being predicted.")
'''))

# ------------------------------------------------------------------- plots
cells.append(code(r'''
# =====================================================================
# E.9 - CHART 1: the experience signal
# =====================================================================
# The headline result: failure rate falls monotonically with experience.

bins = pd.cut(
    df_eng["Company_Prior_Launches"],
    bins=[-1, 0, 5, 20, 100, 500, 100000],
    labels=["1st\nlaunch", "2-6", "7-21", "22-101", "102-501", "502+"],
)
signal = df_eng.groupby(bins, observed=True)["_is_fail"].agg(["count", "mean"])
signal["pct"] = signal["mean"] * 100

fig, ax = plt.subplots()
bars = ax.bar(signal.index.astype(str), signal["pct"],
              color="#4C72B0", edgecolor="white", linewidth=1.2)

for bar, pct, n in zip(bars, signal["pct"], signal["count"]):
    ax.text(bar.get_x() + bar.get_width() / 2, pct + 1.0,
            f"{pct:.1f}%\n(n={n:,})", ha="center", va="bottom", fontsize=9)

global_rate = df_eng["_is_fail"].mean() * 100
ax.axhline(global_rate, ls="--", color="#C44E52", linewidth=1.5,
           label=f"Global failure rate ({global_rate:.1f}%)")

ax.set_xlabel("Operator's prior launches before this mission", fontsize=11)
ax.set_ylabel("Failure rate (%)", fontsize=11)
ax.set_title("Failure rate falls monotonically with operator experience",
             fontsize=13, fontweight="bold")
ax.set_ylim(0, signal["pct"].max() * 1.25)
ax.legend()
plt.tight_layout()
plt.show()

print(f"A first launch fails at {signal['pct'].iloc[0]:.1f}%, versus "
      f"{signal['pct'].iloc[-1]:.1f}% after 500+ launches.")
print(f"That is a {signal['pct'].iloc[0] / signal['pct'].iloc[-1]:.1f}x difference.")
'''))

cells.append(code(r'''
# =====================================================================
# E.10 - CHART 2: the causal proof
# =====================================================================
# Shows WHY the feature is safe: the running count never includes the
# current row's outcome, unlike the leaky company total.

trace = df_eng[df_eng["Company"] == "SpaceX"].sort_values("Date").head(16).reset_index(drop=True)
leaky_total = int(df.groupby("Company")["_is_fail"].sum().loc["SpaceX"])

fig, ax = plt.subplots()
x = np.arange(len(trace))

ax.plot(x, trace["Company_Prior_Failures"], "o-", color="#4C72B0",
        linewidth=2, markersize=6, label="Causal running total (used)")
ax.axhline(leaky_total, ls="--", color="#C44E52", linewidth=2,
           label=f"Leaky company total ({leaky_total})")

# Mark which missions actually failed, to show the count does NOT include them.
fail_idx = trace.index[trace["_is_fail"] == 1]
ax.scatter(fail_idx, trace.loc[fail_idx, "Company_Prior_Failures"],
           s=180, facecolors="none", edgecolors="#55A868", linewidths=2.2,
           zorder=5, label="Mission FAILED (its outcome is not counted yet)")

ax.set_xlabel("SpaceX launch number (chronological)", fontsize=11)
ax.set_ylabel("Accumulated failure count", fontsize=11)
ax.set_title("Causal proof: a row never sees its own outcome",
             fontsize=13, fontweight="bold")
ax.legend(loc="center right", fontsize=9)
plt.tight_layout()
plt.show()

print("Read the first marker: SpaceX's 1st launch FAILED, yet the blue line")
print("starts at 0 — the count does not yet include that failure. Only the")
print("NEXT mission sees it. This is what makes the feature production-safe.")
'''))

cells.append(code(r'''
# =====================================================================
# E.11 - CHART 3: honest vs leaky performance
# =====================================================================
# Visualises why leakage is dangerous: it looks like an improvement.

fig, ax = plt.subplots(figsize=(8, 5))
labels = ["Honest\n(causal)", "LEAKY\n(company totals)"]
values = [honest_f1, leak_f1]
errors = [honest_sd, leak_sd]
colors = ["#55A868", "#C44E52"]

bars = ax.bar(labels, values, yerr=errors, capsize=8,
              color=colors, edgecolor="white", linewidth=1.2, alpha=0.9)

for bar, v in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.012,
            f"{v:.4f}", ha="center", fontsize=12, fontweight="bold")

ax.set_ylabel("Cross-validated F1 (minority class)", fontsize=11)
ax.set_title("Leakage inflates the score while destroying validity",
             fontsize=13, fontweight="bold")
ax.set_ylim(0, max(values) * 1.3)
ax.annotate("", xy=(1, leak_f1), xytext=(0, honest_f1),
            arrowprops=dict(arrowstyle="->", color="gray", lw=1.5))
ax.text(0.5, (honest_f1 + leak_f1) / 2 + 0.015,
        f"+{(leak_f1/honest_f1 - 1)*100:.0f}%", ha="center",
        fontsize=12, color="gray", fontweight="bold")
plt.tight_layout()
plt.show()

print(f"The leaky model looks {(leak_f1/honest_f1 - 1)*100:.0f}% better and is worthless — at liftoff")
print("nobody knows how the mission ends, so the feature could never be computed.")
'''))

cells.append(code(r'''
# =====================================================================
# E.12 - CHART 4: engineered vs raw, on the same axis
# =====================================================================
# Compares the engineered experience measure against the raw year signal,
# showing they capture different things.

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: failure rate by era (the RAW signal)
era = df_eng.groupby(df_eng["Date"].dt.year // 10 * 10)["_is_fail"].agg(["mean", "count"])
axes[0].bar(era.index.astype(str), era["mean"] * 100, color="#8172B2",
            edgecolor="white", linewidth=1.2)
axes[0].axhline(global_rate, ls="--", color="#C44E52", linewidth=1.5)
axes[0].set_xlabel("Decade of launch", fontsize=11)
axes[0].set_ylabel("Failure rate (%)", fontsize=11)
axes[0].set_title("RAW signal: failure rate by decade", fontsize=12, fontweight="bold")
axes[0].tick_params(axis="x", rotation=45)

# Right: failure rate by experience (the ENGINEERED signal)
axes[1].bar(signal.index.astype(str), signal["pct"], color="#4C72B0",
            edgecolor="white", linewidth=1.2)
axes[1].axhline(global_rate, ls="--", color="#C44E52", linewidth=1.5,
                label=f"global {global_rate:.1f}%")
axes[1].set_xlabel("Operator's prior launches", fontsize=11)
axes[1].set_ylabel("Failure rate (%)", fontsize=11)
axes[1].set_title("ENGINEERED signal: failure rate by experience",
                  fontsize=12, fontweight="bold")
axes[1].legend()

plt.tight_layout()
plt.show()

print("Both axes show declining risk, but they measure different things:")
print("  LEFT  = industry-wide era effect (shared by all operators)")
print("  RIGHT = operator-specific learning (varies by organisation)")
print("The Part B clustering found a mid-era cluster with the LOWEST failure")
print("rate, which is only explainable by the right-hand signal.")
'''))

# ---------------------------------------------------------------- summary
cells.append(code(r'''
# =====================================================================
# E.13 - Part E summary
# =====================================================================

summary_e = pd.DataFrame([
    {"feature": "Company_Prior_Launches", "kept": "Yes",
     "why": "Primary experience measure; monotonic 41.9% -> 5.1% signal."},
    {"feature": "Company_Prior_Failure_Rate", "kept": "Yes",
     "why": "Adds the quality dimension (50 launches with 10 failures != 50 with 0)."},
    {"feature": "Company_Prior_Failures", "kept": "No",
     "why": "0.952 correlation with prior launches - near-redundant."},
    {"feature": "Is_First_Company_Launch", "kept": "No",
     "why": "Derivable from prior_launches == 0."},
]).set_index("feature")

display(summary_e)
print(f"Final engineered features: {list(experience_builder.feature_names)}")
print(f"Dataframe shape           : {df_eng.shape}")
'''))

nb["cells"].extend(cells)

with open(NB_PATH, "w") as f:
    json.dump(nb, f, indent=1)

print(f"Appended {len(cells)} Part E cells.")
print(f"Notebook now has {len(nb['cells'])} cells.")
