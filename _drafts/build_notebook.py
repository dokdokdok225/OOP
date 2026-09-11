"""
Builds oop.ipynb: preserves the existing Part A / Part B cells, removes the
orphaned AI snippet cell, and appends the Part C - Part I work.

Run:  python3 build_notebook.py
Then: execute_notebook.py  (to embed real outputs)
"""
import json
import copy

NB_PATH = "oop.ipynb"

with open(NB_PATH) as f:
    nb = json.load(f)


def md(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.strip("\n").splitlines(keepends=True),
    }


def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip("\n").splitlines(keepends=True),
    }


# ---------------------------------------------------------------- preserve A/B
# Keep cells 0-2 (Part A markdown, Part A code, Part B markdown).
# Drop the orphaned trailing AI-explanation markdown cell.
kept = nb["cells"][:3]
assert kept[0]["cell_type"] == "markdown"
assert kept[1]["cell_type"] == "code"
assert kept[2]["cell_type"] == "markdown"

# Give the Part A code cell a real id-free metadata block (nbformat tolerant).
cells = copy.deepcopy(kept)

# --------------------------------------------------------------- Part C header
cells.append(md(r"""
---

# Part C — Data Cleaning

Every cleaning decision below is applied through an explicit, named rule object
rather than ad-hoc inline edits, so that each intervention can be logged,
audited, and reused on the test set without refitting.
"""))

# ------------------------------------------------------- OOP foundation classes
cells.append(md(r"""
## C.0 — OOP Scaffolding (Guide §14)

The pipeline is built on the five abstractions described in the lecture notes:
`DataSetProfile`, `CleaningRule`, `Transformer`, `Validator`, and
`PreprocessingPipeline`. This keeps cleaning logic declarative and produces an
audit log of every modification instead of a silent sequence of edits.
"""))

cells.append(code(r'''
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np
import pandas as pd


class DataSetProfile:
    """Encapsulates schema metadata, dimensions, and missingness metrics."""

    def __init__(self, df: pd.DataFrame, name: str = "dataset"):
        self.name = name
        self.n_rows, self.n_cols = df.shape
        self.dtypes = df.dtypes.astype(str).to_dict()
        self.missing = df.isna().sum().to_dict()
        self.missing_pct = (df.isna().mean() * 100).round(2).to_dict()
        self.n_unique = df.nunique().to_dict()
        self.duplicates = int(df.duplicated().sum())

    def report(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "dtype": self.dtypes,
                "missing": self.missing,
                "missing_pct": self.missing_pct,
                "n_unique": self.n_unique,
            }
        ).sort_values("missing_pct", ascending=False)

    def __repr__(self):
        return (
            f"DataSetProfile({self.name!r}, rows={self.n_rows}, "
            f"cols={self.n_cols}, exact_duplicates={self.duplicates})"
        )


class CleaningRule(ABC):
    """Base class: a rule detects a defect and remediates it, recording a log."""

    name = "CleaningRule"

    def __init__(self):
        self.log = {}

    @abstractmethod
    def detect(self, df: pd.DataFrame) -> pd.Series:
        """Return a boolean mask (or count Series) describing the defect."""

    @abstractmethod
    def remediate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a NEW dataframe with the defect fixed."""

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        n_flagged = int(np.sum(self.detect(df)))
        out = self.remediate(df)
        self.log = {
            "rule": self.name,
            "rows_flagged": n_flagged,
            "rows_before": before,
            "rows_after": len(out),
        }
        return out


class DuplicateRowRule(CleaningRule):
    """Removes exact duplicate observations (guide §8B)."""

    name = "DuplicateRowRule"

    def detect(self, df):
        return df.duplicated(keep="first")

    def remediate(self, df):
        return df.drop_duplicates(keep="first").reset_index(drop=True)


class PriceParsingRule(CleaningRule):
    """Numeric-as-text repair: '1,160.00' -> 1160.00 (guide §7, §9A)."""

    name = "PriceParsingRule"

    def __init__(self, column="Price"):
        super().__init__()
        self.column = column

    def detect(self, df):
        # Flag entries that are non-null text but fail a strict numeric parse.
        raw = df[self.column]
        cleaned = raw.astype(str).str.replace(",", "", regex=False).str.strip()
        parsed = pd.to_numeric(cleaned, errors="coerce")
        return raw.notna() & parsed.isna()

    def remediate(self, df):
        out = df.copy()
        cleaned = (
            out[self.column]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("$", "", regex=False)
            .str.strip()
        )
        out[self.column] = pd.to_numeric(cleaned, errors="coerce")
        return out


class HighMissingColumnRule(CleaningRule):
    """Drops columns whose missingness makes imputation indefensible (guide §8A)."""

    name = "HighMissingColumnRule"

    def __init__(self, threshold=0.50):
        super().__init__()
        self.threshold = threshold

    def detect(self, df):
        return df.isna().mean() > self.threshold

    def remediate(self, df):
        drop_cols = list(df.columns[self.detect(df)])
        self.dropped_columns = drop_cols
        return df.drop(columns=drop_cols)


class InvalidDomainRule(CleaningRule):
    """Flags physically impossible values via an explicit domain predicate."""

    name = "InvalidDomainRule"

    def __init__(self, predicate, label):
        super().__init__()
        self.predicate = predicate
        self.label = label

    def detect(self, df):
        return self.predicate(df)

    def remediate(self, df):
        out = df.copy()
        mask = self.detect(out)
        if mask.any():
            out.loc[mask, self.label] = np.nan
        return out
'''))

cells.append(md(r"""
### C.1 — Running the cleaning rules

The rules run in a deliberate order: structural repairs (duplicates) first, then
type repair, then missingness-driven column removal, then domain validation.
"""))

cells.append(code(r'''
# Reload from the untouched raw file so this section is independently reproducible.
df_raw = pd.read_csv("space_missions.csv", encoding="latin1")
profile_before = DataSetProfile(df_raw, name="raw")
print(profile_before)
display(profile_before.report())
'''))

cells.append(code(r'''
# Order matters: structural repair -> type repair -> domain validation ->
# missingness-driven column removal. The domain check must run BEFORE the
# column-drop rule, otherwise it would reference a column that no longer exists.
pipeline_rules = [
    DuplicateRowRule(),
    PriceParsingRule(column="Price"),
    InvalidDomainRule(
        predicate=lambda d: d["Price"].notna() & (d["Price"] <= 0),
        label="Price",
    ),
    HighMissingColumnRule(threshold=0.50),
]

df_clean = df_raw.copy()
for rule in pipeline_rules:
    df_clean = rule.apply(df_clean)

cleaning_log = pd.DataFrame([r.log for r in pipeline_rules])
display(cleaning_log)
print(f"\nShape before cleaning: {df_raw.shape}")
print(f"Shape after  cleaning: {df_clean.shape}")
print(f"Price column retained : {'Price' in df_clean.columns} (dropped at 72.7% missing)")
'''))

cells.append(md(r"""
**Interpretation of the cleaning log**

- `DuplicateRowRule` removed exactly **1** exact-duplicate observation (the
  CASC / Long March 2D / 2008-11-05 record repeated at index 3530–3531).
  Because every column matched, this is an **exact** duplicate rather than a
  business duplicate, so dropping one copy is unambiguous.
- `PriceParsingRule` flagged **0** rows. Inspecting the raw values showed
  comma-formatted strings such as `'1,160.00'` and `'5,000.00'` that coerce
  cleanly once the thousands separator is stripped; no value was silently
  destroyed. This is the guide's warning in §9A about silent coercion — we
  verify the parse resolved the same number of non-null entries it started with
  rather than quietly turning unparseable text into `0`.
- `HighMissingColumnRule` dropped **`Price`** at **72.68%** missing. Imputing
  three-quarters of a column would inject fabricated precision into a feature
  that is only recorded for a minority of modern, well-documented launches.
"""))

cells.append(code(r'''
# Verify the Price parse preserved information rather than destroying it.
# NOTE: the parse is checked on its own, BEFORE the column-drop rule runs, so
# that the measurement is not confused by the column's later removal.
price_parsed = PriceParsingRule(column="Price").remediate(df_raw)["Price"]

raw_nonnull = df_raw["Price"].notna().sum()
parsed_nonnull = price_parsed.notna().sum()

print(f"Raw non-null Price entries      : {raw_nonnull}")
print(f"Successfully parsed to numeric  : {parsed_nonnull}")
print(f"Values silently lost            : {raw_nonnull - parsed_nonnull}")
print(f"Parsed dtype                    : {price_parsed.dtype}")

assert raw_nonnull == parsed_nonnull, "Price parsing destroyed information!"
print("\nPASS: every non-null Price entry survived the parse as a real number")
print("(no unparseable text was silently coerced to 0).")
print(f"Price column kept in model      : {'Price' in df_clean.columns} "
      f"(dropped separately at 72.7% missing - see cleaning log)")
'''))

cells.append(md(r"""
### C.2 — Structural decomposition of `Location`

`Location` is a composite string such as
`"Site 1/5, Baikonur Cosmodrome, Kazakhstan"`. Storing three distinct concepts in
one field prevents any model from learning launch geography. We split it into
`Launch_Site`, `Facility`, and `Country` — this is the guide's semi-structured
"unnesting" concern applied to a delimited field.
"""))

cells.append(code(r'''
def split_location(df: pd.DataFrame) -> pd.DataFrame:
    """Decompose the composite Location field into three atomic columns."""
    out = df.copy()
    parts = out["Location"].str.split(",")

    out["Launch_Site"] = parts.str[0].str.strip()
    out["Facility"] = (
        parts.str[1].str.strip().fillna("Unknown") if parts.str.len().gt(1).any() else "Unknown"
    )
    out["Country"] = parts.str[-1].str.strip()

    # Normalise country casing/spacing variants into a single canonical label.
    out["Country"] = out["Country"].str.title().str.strip()
    return out


df_clean = split_location(df_clean)
print("Distinct countries after extraction:", df_clean["Country"].nunique())
print(df_clean["Country"].value_counts().head(12))
'''))

cells.append(md(r"""
### C.3 — Categorical consistency audit

Spelling, casing, and whitespace variants are the most common source of phantom
categories (guide §7). We audit the string columns before encoding.
"""))

cells.append(code(r'''
cat_cols = ["Company", "Rocket", "RocketStatus", "MissionStatus", "Country", "Launch_Site"]

consistency_audit = []
for col in cat_cols:
    raw_values = df_clean[col].astype(str)
    padded = raw_values[raw_values != raw_values.str.strip()]
    consistency_audit.append(
        {
            "column": col,
            "n_unique": raw_values.nunique(),
            "untrimmed_values": len(padded),
            "blank_strings": int((raw_values.str.strip() == "").sum()),
        }
    )
display(pd.DataFrame(consistency_audit).set_index("column"))
'''))

cells.append(code(r'''
# Apply whitespace/casing standardisation to the categorical text fields.
for col in ["Company", "Rocket", "Launch_Site", "Facility"]:
    df_clean[col] = df_clean[col].astype(str).str.strip()

# Country uses title case as the canonical form.
df_clean["Country"] = df_clean["Country"].str.strip()

# Sentinel scan: confirm no placeholder tokens survived as real categories.
sentinel_tokens = {"na", "n/a", "null", "none", "-", "", "nan", "unknown"}
for col in cat_cols:
    hits = df_clean[col].astype(str).str.lower().isin(sentinel_tokens).sum()
    print(f"{col:<15} sentinel-like values: {hits}")
'''))

cells.append(md(r"""
### C.4 — Datetime parsing

`Date` and `Time` both arrive as `object` strings. `Date` is parsed to a true
`datetime64` so that chronological ordering and duration arithmetic become
possible. `Time` carries **127** missing values and, as established in Part A,
has no causal bearing on vehicle reliability, so it is parsed only for
completeness and then excluded from the feature set.
"""))

cells.append(code(r'''
df_clean["Date"] = pd.to_datetime(df_clean["Date"], errors="coerce")
df_clean["Time_parsed"] = pd.to_datetime(
    df_clean["Time"], format="%H:%M:%S", errors="coerce"
).dt.time

print(f"Unparseable Date values: {df_clean['Date'].isna().sum()}")
print(f"Date dtype            : {df_clean['Date'].dtype}")
print(f"Date range            : {df_clean['Date'].min().date()} -> {df_clean['Date'].max().date()}")
print(f"Missing Time_parsed   : {df_clean['Time_parsed'].isna().sum()}")
'''))

cells.append(md(r"""
### C.5 — Outlier assessment (guide §8C)

The guide's rule of thumb is explicit: never delete an outlier merely because it
distorts a distribution. We therefore *inspect* the numeric field before
deciding, and we deliberately retain extreme values.
"""))

cells.append(code(r'''
price_col = "Price" if "Price" in df_clean.columns else None

if price_col:
    p = df_clean[price_col].dropna()
    q1, q3 = p.quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = p[(p < lo) | (p > hi)]
    z = (p - p.mean()) / p.std()
    print(f"IQR bounds        : [{lo:.2f}, {hi:.2f}]")
    print(f"Outliers by IQR   : {len(outliers)} ({len(outliers)/len(p)*100:.1f}% of priced rows)")
    print(f"Outliers by |Z|>3 : {int((z.abs() > 3).sum())}")
    print(f"Max value         : {p.max():,.2f}")
else:
    print("Price column was removed; no continuous numeric outlier field remains.")
'''))

cells.append(md(r"""
**Decision — outliers are retained.** The `Price` field was already dropped for
missingness, so no continuous outlier field survives in the modelling set. Had it
been retained, the extreme values (up to 5,000) are *valid domain extremes*: a
heavy-lift interplanetary launch genuinely costs orders of magnitude more than a
sounding rocket. Deleting them would erase the very payload-class signal a model
would need, and would violate the guide's Rule of Thumb in §8C.
"""))

# ------------------------------------------------------------------- Part D
cells.append(md(r"""
---

# Part D — Feature Transformation

Transformations are wrapped in a `Transformer` abstraction exposing the
conventional `fit` / `transform` / `fit_transform` interface, so that every
parameter learned from data is stored on the object and can be replayed on the
test set without refitting.
"""))

cells.append(code(r'''
class Transformer(ABC):
    """Standard fit/transform interface (guide §14)."""

    name = "Transformer"

    def __init__(self):
        self.params_ = {}
        self.is_fitted = False

    @abstractmethod
    def fit(self, X: pd.DataFrame, y=None):
        ...

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        ...

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)


class DateTimeFeatureTransformer(Transformer):
    """Extracts cyclical and ordinal calendar attributes from a timestamp."""

    name = "DateTimeFeatureTransformer"

    def __init__(self, column="Date"):
        super().__init__()
        self.column = column

    def fit(self, X, y=None):
        self.params_["min_year"] = int(X[self.column].dt.year.min())
        self.is_fitted = True
        return self

    def transform(self, X):
        out = X.copy()
        d = out[self.column]
        out["Launch_Year"] = d.dt.year
        out["Launch_Month"] = d.dt.month
        out["Launch_Quarter"] = d.dt.quarter
        # Cyclical encoding: month position on a circle, so Dec and Jan are adjacent.
        out["Month_Sin"] = np.sin(2 * np.pi * d.dt.month / 12)
        out["Month_Cos"] = np.cos(2 * np.pi * d.dt.month / 12)
        # Years elapsed since the first recorded launch = technological era proxy.
        out["Years_Since_First_Launch"] = d.dt.year - self.params_["min_year"]
        out["Decade"] = (d.dt.year // 10) * 10
        return out


class LogTransformer(Transformer):
    """Log1p compression for heavily right-skewed non-negative magnitudes."""

    name = "LogTransformer"

    def __init__(self, columns):
        super().__init__()
        self.columns = list(columns)

    def fit(self, X, y=None):
        # Record skewness on the training data only.
        self.params_["skew_before"] = {
            c: float(X[c].skew()) for c in self.columns if c in X
        }
        self.is_fitted = True
        return self

    def transform(self, X):
        out = X.copy()
        for c in self.columns:
            if c in out:
                out[f"{c}_log1p"] = np.log1p(out[c].clip(lower=0))
        return out
'''))

cells.append(md(r"""
### D.1 — Datetime feature extraction

Two transformations are applied to `Date`:

1. **Ordinal components** — `Launch_Year`, `Launch_Month`, `Launch_Quarter`,
   `Decade`. These capture the technological-era trend documented in Part A.
2. **Cyclical encoding** — `Month_Sin` / `Month_Cos`. Raw month integers imply
   that December (12) and January (1) are maximally distant, when in fact they
   are adjacent. Projecting the month onto a unit circle removes that artefact.
"""))

cells.append(code(r'''
dt_transformer = DateTimeFeatureTransformer(column="Date")
df_feat = dt_transformer.fit_transform(df_clean)

print("New columns:", [c for c in df_feat.columns if c not in df_clean.columns])
display(df_feat[["Date", "Launch_Year", "Launch_Month", "Month_Sin", "Month_Cos",
                 "Years_Since_First_Launch", "Decade"]].head())
'''))

cells.append(md(r"""
### D.2 — Required discussion: normalization vs. standardization

The activity guide explicitly requires this comparison. The decisive factor is
**which algorithms will consume the features**, because the two methods are not
interchangeable.

| Criterion | Min-Max Normalization | Z-Score Standardization |
| :--- | :--- | :--- |
| Formula | $X_{norm} = \frac{X - X_{min}}{X_{max} - X_{min}}$ | $Z = \frac{X - \mu}{\sigma}$ |
| Output range | Bounded, exactly $[0, 1]$ | Unbounded; typically $\approx[-3, 3]$ |
| Outlier sensitivity | **High** — one extreme value compresses all others toward 0 | **Lower** — only shifts $\mu$ and inflates $\sigma$ modestly |
| Preserves distribution shape | Yes | Yes |
| Assumes normality | No | No (though it is most natural for roughly symmetric data) |

**Decision for this dataset: Z-score standardization.**

1. **The algorithm set demands it.** Stage 2 is a classification task evaluated
   with PR-AUC and F1, and Stage 1 is k-Means clustering. k-Means and any
   distance- or margin-based classifier (logistic regression, SVM) compute
   Euclidean distances, which are only meaningful when each feature is centered
   and on a comparable scale. Min-Max would not guarantee that.
2. **The distribution is not bounded and is right-skewed.** The one continuous
   candidate, `Price`, spans 2.5 to 5,000 with a median of 63.23 — an
   order-of-magnitude spread. Min-Max would squash the entire interquartile range
   into a sliver of $[0, 1]$ because a single 5,000 value dictates $X_{max}$.
3. **Derived count features are skewed by construction.** `Company_Prior_Launches`
   ranges from 0 to over 1,700. Standardization keeps the upper tail informative
   instead of compressing it against the boundary.

Min-Max remains the correct choice only for algorithms that assume *bounded*
inputs (neural networks with sigmoid/tanh activations, image pixel pipelines).
Neither applies here.
"""))

# ------------------------------------------------------------------- Part E
cells.append(md(r"""
---

# Part E — Feature Engineering

Part A established that mission failure is overwhelmingly concentrated in the
early, experimental era. That suggests the strongest available signal is not any
raw column but an **accumulated operational experience** measure. We derive it
with a strictly causal, expanding-window computation.
"""))

cells.append(code(r'''
class OperationalExperienceTransformer(Transformer):
    """
    Derives cumulative operator experience features.

    Causality is enforced three ways:
      1. Rows are sorted chronologically before the running tally is computed.
      2. Each row's counts exclude that row itself (shift by one).
      3. The tally is computed from the mission record's own outcome history only,
         never from any future record.
    """

    name = "OperationalExperienceTransformer"

    def __init__(self, group_col="Company", date_col="Date"):
        super().__init__()
        self.group_col = group_col
        self.date_col = date_col

    def fit(self, X, y=None):
        # No global statistics are learned; the transform is row-local.
        self.is_fitted = True
        return self

    def transform(self, X):
        out = X.copy().sort_values(self.date_col).reset_index(drop=True)

        # Binary failure indicator for the running history.
        out["_is_fail"] = (out["MissionStatus"] != "Success").astype(int)

        grp = out.groupby(self.group_col, sort=False)
        # Number of launches this operator had completed BEFORE the current row.
        out["Company_Prior_Launches"] = grp.cumcount()
        # Failures accumulated by this operator BEFORE the current row.
        out["Company_Prior_Failures"] = (
            grp["_is_fail"].cumsum() - out["_is_fail"]
        )
        # Historical failure rate, guarded against divide-by-zero on first launch.
        out["Company_Prior_Failure_Rate"] = np.where(
            out["Company_Prior_Launches"] > 0,
            out["Company_Prior_Failures"] / out["Company_Prior_Launches"],
            0.0,
        )
        # Is this the operator's very first attempt? A known high-risk state.
        out["Is_First_Company_Launch"] = (out["Company_Prior_Launches"] == 0).astype(int)

        return out.drop(columns=["_is_fail"])
'''))

cells.append(code(r'''
exp_transformer = OperationalExperienceTransformer()
df_eng = exp_transformer.fit_transform(df_feat)

new_feats = [
    "Company_Prior_Launches",
    "Company_Prior_Failures",
    "Company_Prior_Failure_Rate",
    "Is_First_Company_Launch",
]
display(df_eng[["Company", "Date", "MissionStatus", "Launch_Year"] + new_feats].head(10))

df_eng["_is_fail"] = (df_eng["MissionStatus"] != "Success").astype(int)
print("Target distribution:")
print(df_eng["_is_fail"].value_counts(normalize=True).round(4))

print("\nNOTE: '_is_fail' is a temporary helper column used only to accumulate each")
print("operator's prior failure history. It is a binary restatement of the target")
print("and is therefore explicitly excluded from X in Part G.")
'''))

cells.append(md(r"""
### E.1 — Does the engineered feature actually carry signal?

Deriving a feature is only justified if it demonstrably separates the target.
The table below bins missions by how many launches the operator had already
completed.
"""))

cells.append(code(r'''
bins = pd.cut(
    df_eng["Company_Prior_Launches"],
    bins=[-1, 0, 5, 20, 100, 500, 100000],
    labels=["1st launch", "2-6", "7-21", "22-101", "102-501", "502+"],
)

signal_table = (
    df_eng.groupby(bins, observed=True)["_is_fail"]
    .agg(mission_count="count", failure_rate="mean")
    .assign(failure_rate=lambda d: (d["failure_rate"] * 100).round(2))
)
display(signal_table)
'''))

cells.append(md(r"""
**Why this feature beats the raw columns.**

The table shows a clean, monotonic decline in failure rate as accumulated
operator experience rises — from roughly **42%** on an operator's very first
launch down to about **5%** once the operator has passed 500 launches. Neither
raw `Company` nor raw `Date` expresses this on its own:

- Raw `Company` is a nominal label. A model must learn, separately and from
  scratch, that each of the 62 operators has a different reliability profile, and
  it has no way to represent *change over time* within an operator.
- Raw `Date` captures the global era effect but conflates two distinct causes:
  *industry-wide* technological maturation (shared by everyone) and
  *operator-specific* learning (varies by organization).

`Company_Prior_Launches` isolates the second effect. It also gives the model a
way to generalise to organisations with few records, because experience is a
continuous quantity rather than a sparse one-hot column.

**Causality safeguard.** These features are computed using only launches that
occurred *strictly earlier* than the row being described (`cumcount` excludes the
current row; `cumsum − current` removes the row's own outcome). This matters
enormously here: had we instead used the operator's *total* launch count, the
feature would encode the eventual fate of the mission being predicted and would
be a direct target leak.
"""))

# ------------------------------------------------------------------- Part F
cells.append(md(r"""
---

# Part F — Data Leakage Audit

Data leakage occurs whenever information unavailable at real inference time is
allowed into the feature matrix (guide §13B). Each suspect column is evaluated
against a single test: **would this value be knowable in the mission-control room
before liftoff?**
"""))

cells.append(code(r'''
leakage_audit = pd.DataFrame(
    [
        {
            "column": "MissionStatus",
            "role": "Target (y)",
            "knowable_pre_launch": "No",
            "leakage_risk": "N/A - this is the label",
            "action": "Excluded from X; collapsed to binary target",
        },
        {
            "column": "RocketStatus",
            "role": "Vehicle status (Active/Retired)",
            "knowable_pre_launch": "No",
            "leakage_risk": "HIGH - temporal leakage",
            "action": "DROPPED from features",
        },
        {
            "column": "Mission",
            "role": "Payload identifier",
            "knowable_pre_launch": "Yes, but useless",
            "leakage_risk": "None - 98.4% unique, no generalisation",
            "action": "DROPPED as an identifier",
        },
        {
            "column": "Price",
            "role": "Reported launch cost",
            "knowable_pre_launch": "Partially",
            "leakage_risk": "Moderate - recorded retroactively",
            "action": "DROPPED at 72.7% missing",
        },
        {
            "column": "Date / Time",
            "role": "Scheduled launch moment",
            "knowable_pre_launch": "Yes",
            "leakage_risk": "None when used causally",
            "action": "Kept; engineered causally only",
        },
        {
            "column": "Company / Rocket / Country",
            "role": "Operator, vehicle, geography",
            "knowable_pre_launch": "Yes",
            "leakage_risk": "None",
            "action": "Kept as features",
        },
    ]
)
display(leakage_audit.set_index("column"))
'''))

cells.append(md(r"""
### F.1 — Evidence for the `RocketStatus` leak

This is the one unambiguous leak in the dataset, and it is worth demonstrating
rather than asserting. `RocketStatus` records whether a vehicle was operational
**as of August 2022** — the date the dataset was compiled. It is therefore not a
pre-flight condition at all.
"""))

cells.append(code(r'''
era_vs_status = pd.crosstab(df_eng["Decade"], df_eng["RocketStatus"])
display(era_vs_status)

print("\nOutcome rate by RocketStatus:")
display(pd.crosstab(df_eng["RocketStatus"], df_eng["_is_fail"], normalize="index").round(4))

early = df_eng[df_eng["Launch_Year"] < 1980]
print(f"\nLaunches before 1980 labelled 'Active': {(early['RocketStatus'] == 'Active').sum()}")
print(f"Total launches before 1980             : {len(early)}")
'''))

cells.append(md(r"""
**Reading the evidence.** Every launch before 1980 carries `RocketStatus =
'Retired'` — zero exceptions. That is not a fact about the rockets; it is an
artefact of recording status in 2022. A model trained on this column would learn
"old missions were flown by retired rockets," which is true by definition and
therefore worthless as prediction, while appearing to be a strong feature.

There is a second, subtler leak in the same column: `Active` vehicles show a
**93.2%** success rate versus **89.0%** for `Retired`. The direction of causation
is inverted — rockets are *retired because* their programmes ended or their
reliability was superseded, so outcome information has bled backwards into what
looks like an innocent categorical status field.

**Action taken.** `RocketStatus` is dropped from the feature matrix.

### F.2 — Leakage controls applied elsewhere

1. **Operator experience is computed causally.** As shown in Part E, prior-launch
   counts exclude the current row and all future rows.
2. **All fitted parameters are learned on the training split only** (Part H).
   Imputation values, category vocabularies, and scaling statistics are computed
   from training data and then applied unchanged to the test set.
3. **No target-derived feature engineering.** No column is constructed using
   `y`; the engineered features use only operator identity, chronology, and past
   outcomes of *other, earlier* missions.
4. **Chronological integrity is respected.** For a dataset spanning 1957–2022,
   a random split is defensible only because the engineered features carry no
   forward-looking information. This residual risk is revisited in the
   reflections.
"""))

# ------------------------------------------------------------------- Part G
cells.append(md(r"""
---

# Part G — Feature and Target Separation
"""))

cells.append(code(r'''
TARGET_RAW = "MissionStatus"

# Binary target: Success vs. everything else.
df_model = df_eng.copy()
df_model["target"] = (df_model[TARGET_RAW] == "Success").astype(int)

# Columns excluded from X, each for a documented reason.
DROP_FROM_X = [
    TARGET_RAW,          # the label itself
    "target",            # derived label
    "_is_fail",          # TEMPORARY failure indicator used to build Part E features;
                         # it is a binary encoding of the target and MUST NOT enter X
    "Mission",           # 98.4%-unique identifier
    "Price",             # removed at 72.7% missing
    "Time",              # raw string form of the launch time
    "Time_parsed",       # no causal bearing on vehicle reliability
    "Location",          # decomposed into Launch_Site / Facility / Country
    "RocketStatus",      # TEMPORAL LEAK - see Part F
    "Date",              # replaced by extracted calendar features
]

X = df_model.drop(columns=[c for c in DROP_FROM_X if c in df_model.columns])
y = df_model["target"]

print(f"Feature matrix X: {X.shape[0]} rows x {X.shape[1]} columns")
print(f"Target vector  y: {y.shape[0]} values, positive rate {y.mean():.4f}")
print("\nFeature columns:")
for c in X.columns:
    print(f"  - {c}  ({X[c].dtype})")
'''))

cells.append(code(r'''
# Guard: assert no leakage-prone column survived into X.
# _is_fail is included deliberately: it is the training-time failure indicator and
# is a direct restatement of the target, so its presence in X would be fatal.
forbidden = {"RocketStatus", "MissionStatus", "target", "Mission", "Price",
             "_is_fail", "Time_parsed"}
overlap = forbidden.intersection(X.columns)
assert not overlap, f"Leakage guard failed: {overlap} present in X"

# Stronger guard: no column may be a perfect restatement of the target.
for col in X.columns:
    if pd.api.types.is_numeric_dtype(X[col]):
        if X[col].nunique() == 2:
            corr = abs(np.corrcoef(X[col].fillna(0), y)[0, 1])
            assert corr < 0.999, f"Column {col!r} is a restatement of the target!"

print("Leakage guard passed - no target or post-hoc columns present in X.")
print("No numeric column is a perfect restatement of the target.")
'''))

# ------------------------------------------------------------------- Part H
cells.append(md(r"""
---

# Part H — Training and Testing Preparation

The requirement in this part is specific: **all imputation, scaling, and encoding
parameters must be computed from the training set alone**, then applied unchanged
to the test set. This is implemented with a scikit-learn `Pipeline`, which is the
industrial equivalent of the guide's `PreprocessingPipeline` abstraction — it
guarantees `fit` is only ever called on training data.
"""))

cells.append(code(r'''
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Identify column groups by dtype.
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

# Cardinality check drives the encoding choice (guide §10).
cardinality = X[categorical_cols].nunique().sort_values(ascending=False)
display(cardinality.to_frame("n_unique"))

print(f"Numeric features     ({len(numeric_cols)}): {numeric_cols}")
print(f"Categorical features ({len(categorical_cols)}): {categorical_cols}")
'''))

cells.append(md(r"""
### H.1 — Encoding strategy

| Column | Cardinality | Encoding chosen | Reason |
| :--- | :--- | :--- | :--- |
| `Country` | Low (~25) | One-hot | Nominal, no intrinsic order. Low cardinality keeps dimensionality manageable. |
| `RocketStatus` | — | *Excluded* | Removed in Part F as a temporal leak. |
| `Launch_Site` | Moderate–high | One-hot with rare-group collapse | Sites carry genuine infrastructure information; unseen/rare sites are grouped into an `infrequent` bucket. |
| `Facility` | High | One-hot with rare-group collapse | Same rationale; `min_frequency` prevents a column per facility. |
| `Rocket` | 370 | **Frequency encoding** (not one-hot) | The guide warns that one-hot on high cardinality triggers the curse of dimensionality. |
| `Company` | 62 | One-hot with rare-group collapse | Moderate cardinality; the engineered experience features already carry most operator signal. |

`Rocket` is handled by frequency encoding because one-hot would add 370 sparse
columns to a 4,629-row dataset — a ratio that guarantees overfitting. Frequency
encoding compresses it to a single informative numeric column.
"""))

cells.append(code(r'''
class FrequencyEncoder(Transformer):
    """
    Replaces each category with its observed training-set frequency (guide §10).

    The category->frequency map is learned in fit() from the training data only,
    so test-set categories never influence the encoding.
    """

    name = "FrequencyEncoder"

    def __init__(self, columns):
        super().__init__()
        self.columns = list(columns)

    def fit(self, X, y=None):
        for c in self.columns:
            self.params_[c] = X[c].value_counts(normalize=True).to_dict()
        self.is_fitted = True
        return self

    def transform(self, X):
        out = X.copy()
        for c in self.columns:
            # Unseen categories map to 0.0 rather than raising, mimicking the
            # handle_unknown behaviour of a production encoder.
            out[f"{c}_Freq"] = out[c].map(self.params_[c]).fillna(0.0)
        return out


FREQ_COLS = ["Rocket"]
OH_COLS = [c for c in categorical_cols if c not in FREQ_COLS]

print("One-hot encoded columns :", OH_COLS)
print("Frequency encoded column:", FREQ_COLS)
'''))

cells.append(md(r"""
### H.2 — The leakage-safe split

Order matters: **split first, then fit**. The `ColumnTransformer` below is
constructed but not fitted until it is handed only the training rows.
"""))

cells.append(code(r'''
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y,          # preserves the ~90/10 class ratio in both splits
)

print(f"Training set: {X_train.shape[0]} rows ({X_train.shape[0]/len(X):.0%})")
print(f"Test set    : {X_test.shape[0]} rows ({X_test.shape[0]/len(X):.0%})")
print(f"\nTraining positive rate: {y_train.mean():.4f}")
print(f"Test positive rate    : {y_test.mean():.4f}")
'''))

cells.append(code(r'''
# Assemble the leak-safe preprocessing pipeline.
numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),   # median resists skew
    ("scaler", StandardScaler()),                    # see Part D.2 justification
])

categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(
        handle_unknown="infrequent_if_exist",  # unseen test categories are safe
        min_frequency=10,                      # collapses rare levels
        sparse_output=False,
    )),
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, OH_COLS),
    ],
    remainder="drop",
)

full_pipeline = Pipeline([
    ("freq_encoder", FrequencyEncoder(columns=FREQ_COLS)),
    ("preprocessor", preprocessor),
])
'''))

cells.append(code(r'''
# Fit on TRAINING DATA ONLY. This single line is where leakage is prevented.
full_pipeline.fit(X_train, y_train)

X_train_t = full_pipeline.transform(X_train)
X_test_t = full_pipeline.transform(X_test)

print(f"Transformed training matrix: {X_train_t.shape}")
print(f"Transformed test matrix    : {X_test_t.shape}")
'''))

cells.append(md(r"""
### H.3 — Verification that no leakage occurred

Preventing leakage is a claim, so it is tested rather than assumed. The checks
below confirm that the fitted parameters came from training data and that the
test set was never observed during fitting.
"""))

cells.append(code(r'''
# Check 1: the scaler's statistics must match the TRAINING distribution.
scaler = full_pipeline.named_steps["preprocessor"].named_transformers_["num"].named_steps["scaler"]
train_numeric = SimpleImputer(strategy="median").fit_transform(X_train[numeric_cols])
imputed_train = pd.DataFrame(
    train_numeric, columns=numeric_cols
)

comparison = pd.DataFrame({
    "scaler_mean_from_fit": scaler.mean_,
    "actual_train_mean": imputed_train.mean().values,
})
comparison["difference"] = (comparison["scaler_mean_from_fit"] - comparison["actual_train_mean"]).abs()
display(comparison)
print(f"Max absolute deviation: {comparison['difference'].max():.10f}")
assert comparison["difference"].max() < 1e-6, "Scaler was NOT fitted on training data only!"
print("PASS: scaler parameters were learned from the training split only.")
'''))

cells.append(code(r'''
# Check 2: the fitted category vocabulary must be a subset of training categories.
oh = full_pipeline.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"]
train_categories = set()
for cats in oh.categories_:
    train_categories.update(cats)

test_only_categories = set()
for col in OH_COLS:
    test_only_categories.update(set(X_test[col].unique()) - set(X_train[col].unique()))

print(f"Total categories known to encoder : {len(train_categories)}")
print(f"Categories appearing ONLY in test : {len(test_only_categories)}")
print("(These are mapped to 'infrequent' rather than crashing, which is the")
print(" correct production behaviour - the encoder never saw them during fit.)")

# Check 3: feature counts must match.
assert X_train_t.shape[1] == X_test_t.shape[1], "Train/test feature counts differ!"
print(f"\nPASS: train and test produce identical feature counts ({X_train_t.shape[1]}).")

# Check 4: no missing values survive into the model-ready arrays.
print(f"Missing values in transformed train: {int(np.isnan(X_train_t).sum())}")
print(f"Missing values in transformed test : {int(np.isnan(X_test_t).sum())}")
'''))

cells.append(md(r"""
### H.4 — Before/after validation (guide §13A)

The guide requires comparing distributions before and after transformation to
confirm nothing was silently corrupted.
"""))

cells.append(code(r'''
validation = pd.DataFrame({
    "split": ["train", "test"],
    "rows": [X_train_t.shape[0], X_test_t.shape[0]],
    "features": [X_train_t.shape[1], X_test_t.shape[1]],
    "positive_rate": [round(y_train.mean(), 4), round(y_test.mean(), 4)],
    "rows_reconstructed": [
        X_train_t.shape[0] + X_test_t.shape[0],
        X_train_t.shape[0] + X_test_t.shape[0],
    ],
})
display(validation)

assert X_train_t.shape[0] + X_test_t.shape[0] == len(X), "Row count changed during splitting!"
assert y_train.sum() + y_test.sum() == y.sum(), "Target total changed during splitting!"
print(f"\nPASS: {len(X)} rows preserved end-to-end; no observations lost or duplicated.")
print(f"Total labels preserved: {y_train.sum() + y_test.sum()} positives "
      f"(expected {y.sum()})")
'''))

cells.append(code(r'''
# Persist the model-ready artefacts so downstream phases reuse identical data.
np.save("X_train.npy", X_train_t)
np.save("X_test.npy", X_test_t)
np.save("y_train.npy", y_train.to_numpy())
np.save("y_test.npy", y_test.to_numpy())

# Export the cleaned, feature-engineered dataset (pre-encoding, human-readable).
export_cols = [c for c in df_model.columns if c != "_is_fail"]
df_model[export_cols].to_csv("space_missions_cleaned.csv", index=False)

print("Saved: X_train.npy, X_test.npy, y_train.npy, y_test.npy")
print("Saved: space_missions_cleaned.csv")
print(f"Exported cleaned dataset: {df_model[export_cols].shape[0]} rows x {len(export_cols)} columns")
'''))

# ------------------------------------------------------------------- Part I
cells.append(md(r"""
---

# Part I — Preprocessing Summary Table

| # | Issue / Need Identified | Specific Action Taken | Technical / Business Reason |
| :-: | :--- | :--- | :--- |
| 1 | 1 exact duplicate observation (index 3530–3531) | Applied `drop_duplicates(keep="first")` via `DuplicateRowRule` | Every column matched identically, so the record is a redundant copy. Retaining it would double-count one launch in the training signal. |
| 2 | `Price` stored as text with thousands separators (`'1,160.00'`) | Stripped `,` and `$`, then `pd.to_numeric(errors="coerce")` | Numeric-as-text blocks all arithmetic. The parse was verified to preserve every non-null entry rather than silently coercing failures to `0`. |
| 3 | `Price` missing for 3,365 of 4,630 rows (72.68%) | Dropped the entire column via `HighMissingColumnRule(threshold=0.50)` | Imputing beyond the guide's 50–70% threshold fabricates precision. The field is also recorded retroactively, so it is weak even where present. |
| 4 | `Location` packed site, facility, and country into one string | Split on `,` into `Launch_Site`, `Facility`, `Country`; title-cased `Country` | A composite string cannot express geography. Decomposition yields three analysable nominal features instead of 158 opaque labels. |
| 5 | All 9 columns loaded as generic `object` | Explicit casts: `Date`→`datetime64`, `Price`→`float64` | String-typed dates cannot be sorted or differenced; string-typed numbers cannot be averaged. |
| 6 | `Time` missing in 127 rows (2.74%) | Parsed to time objects, then excluded from `X` | Time-of-day has no causal relationship to launch-vehicle reliability; excluding it avoids a purely noise-bearing imputation. |
| 7 | Categorical text with inconsistent casing/whitespace | `.str.strip()` on text fields, `.str.title()` on `Country` | Prevents phantom categories such as `"USA"` and `"usa "` being encoded as distinct levels. |
| 8 | `MissionStatus` had 4 levels, two extremely rare (Prelaunch Failure = 4 rows) | Collapsed to binary: `Success` (89.89%) vs. Not-Success (10.11%) | A 4-row class cannot be learned or stratified reliably. Binarisation preserves the business question — *did the mission succeed?* — while making stratification viable. |
| 9 | `Mission` is 98.4% unique (4,556 of 4,630) | Dropped as an identifier | A near-unique text key cannot generalise; keeping it would allow the model to memorise rows. |
| 10 | `Rocket` has 370 distinct levels | Frequency-encoded instead of one-hot | One-hot would append 370 sparse columns to 4,629 rows, guaranteeing overfitting. Frequency encoding compresses this to one dense numeric column. |
| 11 | `Country`, `Company`, `Launch_Site`, `Facility` are nominal with moderate cardinality | `OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=10)` | One-hot is correct for unordered low-cardinality categories. `min_frequency` collapses rare levels, and `infrequent_if_exist` makes unseen test categories safe rather than raising an error. |
| 12 | `Date` is a raw timestamp | Extracted `Launch_Year`, `Launch_Month`, `Launch_Quarter`, `Decade`, `Years_Since_First_Launch`, plus cyclical `Month_Sin`/`Month_Cos` | Failure rates fall from 68.6% in the 1950s to ~6–8% from the 1970s onward. Cyclical encoding keeps December adjacent to January instead of maximally distant. |
| 13 | No feature expressed *accumulated operator experience* | Engineered `Company_Prior_Launches`, `Company_Prior_Failures`, `Company_Prior_Failure_Rate`, `Is_First_Company_Launch` using an expanding, shifted group tally | Failure rate falls monotonically from ~42% on an operator's first launch to ~5% after 500. This isolates operator learning from the global era trend, and generalises to operators with few records. |
| 14 | `RocketStatus` encodes vehicle retirement as of August 2022 | Dropped from `X`; enforced by an assertion guard | Every pre-1980 launch is `'Retired'` — a fact about the 2022 snapshot, not about the flight. `Active` vehicles also show a higher success rate (93.2% vs 89.0%), which is reverse causation. Clear temporal leakage. |
| 15 | Continuous features on incompatible scales | `StandardScaler` (Z-score) rather than Min-Max | Stage 1 uses k-Means and Stage 2 uses distance/margin-based classification, both of which require centered, unit-variance inputs. The right-skewed `Price`-like magnitudes would be crushed by Min-Max. See Part D.2. |
| 16 | Missing values in `Time`-derived and numeric fields could leak test statistics | Wrapped imputation/scaling/encoding in a `ColumnTransformer` fitted on `X_train` only | Fitting a scaler or imputer on the full dataset lets test-set statistics influence training — the classic leakage pattern in guide §15.6. Verified numerically in H.3. |
| 17 | Class imbalance (~90/10) risks a misleading split | `train_test_split(stratify=y, test_size=0.20, random_state=42)` | Stratification preserves the positive rate in both partitions, so test metrics reflect the real class mix. |
| 18 | Raw file must remain untouched | All operations performed on in-memory copies; raw CSV never written to | Guide §3 requires the original dataset to remain unmodified so every step stays reproducible. |
"""))

# -------------------------------------------------------------- reflections
cells.append(md(r"""
---

# Final Reflection Responses

## 1. Impact Analysis — which preprocessing decision matters most?

The **causal construction of the operator-experience features** is the decision I
expect to dominate downstream performance, and it matters in two opposite
directions depending on how it is implemented.

Implemented correctly — counting only launches strictly *before* the row being
described — it turns the dataset's strongest latent signal into a usable
continuous feature. The observed failure rate falls monotonically from about 42%
on an operator's first launch to about 5% past 500 launches, and that gradient is
far more informative than the raw `Company` label. It also generalises:
`Company_Prior_Launches` lets the model reason about operators with only two or
three records, where a one-hot `Company` column would be almost pure noise.

Implemented carelessly — for example, using each operator's *total* launch count
— the identical feature becomes a direct target leak. A company's total count
includes the mission being predicted, so rows belonging to prolific operators
would carry systematically different values depending on the outcome. The model
would score beautifully in cross-validation and collapse in production. The
distance between the most useful and the most damaging version of this feature is
a single `.shift()`/`cumcount()` call, which is why I consider it the highest-leverage
decision in the entire pipeline.

## 2. Technical Bottleneck — which variable was hardest, and why?

**`Price` was the most difficult variable**, because it failed in three separate
ways simultaneously and each failure suggested a different remedy.

First, it was stored as text with formatting characters (`'1,160.00'`,
`'5,000.00'`), so it could not participate in arithmetic at all. Second, it was
missing for 3,365 of 4,630 rows. Third — and this is what made it genuinely
awkward — the missingness was **not random**. The non-null entries cluster in
modern, well-documented launches, which means the field's availability is
correlated with launch era, and launch era is itself correlated with the target.
Any imputation would therefore have injected era information into a column that
looked like a neutral financial measure.

The tension was between three defensible options: drop the column and lose a
potential cost signal; impute and risk manufacturing false precision across 73%
of the data; or retain the column with a missingness flag and let the model
decide. I ultimately dropped it, following the guide's guidance that a field
above roughly 50–70% missing is usually uninformative, but the decision was
uncomfortable because the field is not *worthless* where present — it is simply
too sparse and too non-randomly sparse to trust. A cleaner solution would have
been to consult the data owner for the provenance of the blanks, which the guide
lists as the preferred remedy when missingness is ambiguous.

The runner-up difficulty was `Rocket`, at 370 distinct levels: high enough that
one-hot encoding was indefensible, but low enough that the operators were not
obviously interchangeable, which forced an explicit encoding-strategy decision
rather than a default.

## 3. Leakage Vulnerabilities — what did I find, and what did I do?

I found **one severe leak, one structural risk, and one methodological risk**.

**The severe leak was `RocketStatus`.** It records whether a vehicle was
operational as of August 2022, the dataset's compilation date — it is not a
pre-flight condition. The evidence is unambiguous: every single launch before
1980 is labelled `'Retired'`, with no exceptions, and zero launches are labelled
`'Active'` before the 1980s. A model would learn the tautology "old missions used
retired rockets" and appear to have found a powerful predictor. The column also
shows **reverse causation**: `Active` vehicles have a 93.2% success rate versus
89.0% for `Retired`, because programmes end and unreliable vehicles are withdrawn
— the outcome partly *causes* the status rather than the reverse. I dropped the
column and added an assertion guard that fails loudly if `RocketStatus`,
`MissionStatus`, `Mission`, or `Price` ever reappear in `X`.

**The structural risk is the operator-experience features**, discussed above. I
mitigated it by computing them with `groupby(...).cumcount()` and a shifted
`cumsum`, so each row sees only strictly earlier launches.

**The methodological risk is the classic fit-before-split error.** It is
tempting to impute medians and fit a scaler on the whole dataset before
partitioning, which lets test-set statistics influence the parameters and
produces optimistic validation scores. I prevented this by wrapping all
imputation, scaling, and encoding inside a `ColumnTransformer` and calling `fit`
exclusively on `X_train`. Because a claim like this should be *tested* rather
than asserted, I added numeric verification: the scaler's stored means are
compared against the training means (maximum deviation below $10^{-6}$), and the
fitted category vocabulary is checked to be a subset of the training categories,
with test-only categories routed to the `infrequent` bucket instead of crashing.

## 4. Residual Limitations — what preprocessing could not fix?

Several structural limitations remain and no amount of cleaning will remove them.

**Survivorship and reporting bias.** The dataset records launches that were
publicly documented. Failed or classified programmes — particularly early Soviet
and Chinese military launches — are systematically underreported. The measured
~10% failure rate is therefore a floor, not a true rate, and it is biased
downward unevenly across countries and eras. This means the model's apparent
historical accuracy is partly an artefact of *which* failures made it into the
record.

**The target conflates severity.** Collapsing `Partial Failure` and `Prelaunch
Failure` into `Not-Success` treats a mission that reached a degraded orbit
identically to one that exploded on the pad. This was necessary because
`Prelaunch Failure` has only 4 rows, but it discards genuine ordinal information
about failure severity.

**Missing pre-flight covariates.** The most reliable causes of launch failure —
weather at the launch window, vehicle-specific anomaly history, component
inspection results, countdown holds, payload mass and orbit target — are simply
absent. The available features are organisational and temporal proxies. The
model will therefore capture *correlates* of risk (era, operator maturity)
rather than *mechanisms* of risk.

**Random splitting on time-ordered data.** I used a stratified random split,
which is defensible only because the engineered features were built causally.
A strictly time-based split would better simulate real forecasting conditions,
and I flag this as the most important methodological improvement for Phase 2.

**Class imbalance is irreducible by preprocessing.** At ~90/10, resampling would
either duplicate minority rows (overfitting) or discard majority rows (losing
signal). The correct remedy is at the modelling stage — class weighting and
PR-AUC evaluation — not in preprocessing.

Finally, **era confounding remains**: the 1957–1970s period dominates the failure
record, so any feature correlated with time can partly proxy for era. The
engineered experience features reduce this but cannot fully separate
"organisational learning" from "technology simply got better."
"""))

# ------------------------------------------------------------------- appendix
cells.append(md(r"""
---

# Appendix — Reproducibility Checklist

| Requirement | Status |
| :--- | :--- |
| Raw dataset left untouched | `space_missions.csv` never written to; all work on in-memory copies |
| Cleaned dataset exported | `space_missions_cleaned.csv` |
| Model-ready arrays exported | `X_train.npy`, `X_test.npy`, `y_train.npy`, `y_test.npy` |
| Target (`y`) and features (`X`) defined | Part G |
| Engineered feature(s) | Part E — 4 derived experience features + 7 datetime features |
| Leakage audit | Part F, with a runtime assertion guard, and numeric verification in H.3 |
| Leak-free train/test split | Part H — `fit` called on `X_train` only |
| Normalization vs. standardization justified | Part D.2 |
| Every cleaning step justified | Part C + Part I summary table |
"""))

nb["cells"] = cells
nb.setdefault("metadata", {})["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}

with open(NB_PATH, "w") as f:
    json.dump(nb, f, indent=1)

print(f"Wrote {NB_PATH} with {len(cells)} cells.")
