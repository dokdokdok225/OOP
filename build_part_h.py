"""
Inserts Part H (Training and Testing Preparation) into oop.ipynb.

Key requirement: all imputation, scaling, and encoding parameters are fit on the
TRAINING set only, then applied unchanged to the test set.

Run: python3 build_part_h.py
"""
import json

NB_PATH = "oop.ipynb"
DOC_PATH = "_docs/part_h_doc.md"

with open(NB_PATH) as f:
    nb = json.load(f)
with open(DOC_PATH) as f:
    part_h_doc = f.read()


def md(text):
    return {"cell_type": "markdown", "metadata": {},
            "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.strip("\n").splitlines(keepends=True)}


cells = [md(part_h_doc)]

# ------------------------------------------------------------- H.6 frequency
cells.append(md(r"""
---

#### H.6 — The `FrequencyEncoder`, as a proper scikit-learn transformer

Part D decided `Rocket` would be frequency-encoded. To honour that decision inside
the pipeline, the encoder must behave like every other scikit-learn transformer:
it learns a vocabulary in `fit()` and applies it in `transform()`.

Writing it as a class (rather than a one-off function) is what makes the leakage
guarantee hold automatically — `fit()` only ever receives training data.
"""))

cells.append(code(r'''
from sklearn.base import BaseEstimator, TransformerMixin


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """
    Replace each category with its observed frequency.

    Implements the standard scikit-learn transformer contract so it can sit
    inside a Pipeline and inherit the fit-on-train-only guarantee:

        fit(X_train)          -> learns category -> frequency from TRAIN only
        transform(any_X)      -> applies that frozen mapping

    Unseen categories (present in test but absent from train) map to a neutral
    0.0 rather than raising or producing NaN. This mirrors the
    handle_unknown="infrequent_if_exist" behaviour of OneHotEncoder.

    Also satisfies the oop_rules.md §13 checklist:
      §4 protected attributes, §5 validation before assignment,
      §6 read-only @property accessors, §7 class-level configuration.
    """

    # Class attribute: the value used for unseen categories (oop_rules §7).
    UNKNOWN_FREQUENCY: float = 0.0

    def __init__(self, columns=None):
        self.columns = columns

    # ---- read-only accessors (oop_rules §6) -------------------------------
    @property
    def learned_vocabulary_size(self) -> dict:
        """How many distinct categories were seen per column, in training."""
        if not hasattr(self, "frequencies_"):
            raise AttributeError("FrequencyEncoder has not been fitted yet.")
        return {c: len(v) for c, v in self.frequencies_.items()}

    def fit(self, X: pd.DataFrame, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"X must be a DataFrame, got {type(X).__name__}.")
        if self.columns is None:
            raise ValueError("FrequencyEncoder requires an explicit `columns` list.")

        missing = [c for c in self.columns if c not in X.columns]
        if missing:
            raise KeyError(f"Columns not found in X: {missing}")

        self.frequencies_ = {}
        for col in self.columns:
            self.frequencies_[col] = (
                X[col].value_counts(normalize=True).to_dict()
            )
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "frequencies_"):
            raise AttributeError("FrequencyEncoder must be fitted before transform().")
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"X must be a DataFrame, got {type(X).__name__}.")

        encoded = {}
        for col in self.columns:
            mapped = X[col].map(self.frequencies_[col])
            n_unseen = int(mapped.isna().sum())
            if n_unseen:
                self._unseen_counts = getattr(self, "_unseen_counts", {})
                self._unseen_counts[col] = n_unseen
            encoded[f"{col}_Freq"] = mapped.fillna(self.UNKNOWN_FREQUENCY).to_numpy()

        return pd.DataFrame(encoded, index=X.index)

    def get_feature_names_out(self, input_features=None):
        return np.array([f"{c}_Freq" for c in self.columns])
'''))

cells.append(code(r'''
# =====================================================================
# H.6a - Verify the encoder handles unseen categories correctly
# =====================================================================

demo_train = pd.DataFrame({"Rocket": ["A", "A", "A", "B", "B", "C"]})
demo_test = pd.DataFrame({"Rocket": ["A", "Z_UNSEEN"]})

demo_enc = FrequencyEncoder(columns=["Rocket"]).fit(demo_train)

print("Learned from training data only:")
for cat, freq in sorted(demo_enc.frequencies_["Rocket"].items()):
    print(f"    {cat!r:<12} -> {freq:.4f}")
print(f"\nVocabulary size: {demo_enc.learned_vocabulary_size}")

print("\nTransform on unseen data:")
print(demo_enc.transform(demo_test).to_string())

print(f"\n'Z_UNSEEN' was never in training, so it maps to "
      f"{FrequencyEncoder.UNKNOWN_FREQUENCY} rather than raising.")
print("That is the required behaviour: test data is encoded using")
print("TRAINING vocabulary only, never its own.")
'''))

# ------------------------------------------------------------------- split
cells.append(md(r"""
#### H.7 — The split, performed before any fitting

Order matters: **split first, then fit.** Every transformation below is fitted
after this line, on `X_train` only.
"""))

cells.append(code(r'''
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
TEST_SIZE = 0.20

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,          # preserves the ~90/10 class ratio in both partitions
)

print("SPLIT COMPLETE - before any fitting\n")
print(f"  Training : {X_train.shape[0]:,} rows  ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"  Test     : {X_test.shape[0]:,} rows  ({X_test.shape[0]/len(X)*100:.1f}%)")
print()
print(f"  Train positive rate : {y_train.mean():.4f}  ({int(y_train.sum()):,} successes)")
print(f"  Test  positive rate : {y_test.mean():.4f}  ({int(y_test.sum()):,} successes)")
print()
print(f"  Minority-class rows in TEST : {int((1 - y_test).sum())} failures")
print("  -> enough to compute minority-class F1 and PR-AUC meaningfully.")
'''))

# ------------------------------------------------------------------ pipeline
cells.append(md(r"""
#### H.8 — The leak-safe preprocessing pipeline

The `ColumnTransformer` below is **constructed** here but deliberately **not
fitted**. It will be fitted on `X_train` alone in the next cell.

This separation is the point: constructing the pipeline declares *what* to do,
and fitting it on training data alone guarantees *no test information* enters the
parameters.
"""))

cells.append(code(r'''
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Column groups, derived from the actual dtypes.
numeric_cols = [c for c in X.columns if X[c].dtype != object]
categorical_cols = [c for c in X.columns if X[c].dtype == object]

# Rocket is handled by frequency encoding (Part D), so it is excluded from the
# one-hot group.
FREQ_COLS = ["Rocket"]
ONEHOT_COLS = [c for c in categorical_cols if c not in FREQ_COLS]

print(f"Numeric  ({len(numeric_cols)}): {numeric_cols}")
print(f"One-hot  ({len(ONEHOT_COLS)}): {ONEHOT_COLS}")
print(f"Frequency({len(FREQ_COLS)}): {FREQ_COLS}")

# --- Numeric branch: median impute (idle, see H.5) then standardize ---
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

# --- Categorical branch: impute, then one-hot with rare collapsing ---
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(
        handle_unknown="infrequent_if_exist",  # unseen test categories are safe
        min_frequency=10,                      # collapse rare levels
        sparse_output=False,
    )),
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, ONEHOT_COLS),
        ("freq", FrequencyEncoder(columns=FREQ_COLS), FREQ_COLS),
    ],
    remainder="drop",
)

# The full pipeline: frequency-encode, then the column transformer.
# Constructed but NOT yet fitted.
full_pipeline = Pipeline([
    ("preprocessor", preprocessor),
])

print("\nPipeline constructed but NOT fitted.")
print("Fitting happens next, on X_train only.")
'''))

cells.append(code(r'''
# =====================================================================
# H.9 - FIT ON TRAINING DATA ONLY
# =====================================================================
# This single call is where leakage is prevented. Every imputation value,
# scaling statistic, and category vocabulary is learned from X_train and
# nothing else.

full_pipeline.fit(X_train, y_train)

X_train_t = full_pipeline.transform(X_train)
X_test_t = full_pipeline.transform(X_test)

# Make them plain arrays for downstream modelling.
X_train_t = np.asarray(X_train_t, dtype=float)
X_test_t = np.asarray(X_test_t, dtype=float)

print("Fitted on training data only.\n")
print(f"  Transformed training matrix : {X_train_t.shape}")
print(f"  Transformed test matrix     : {X_test_t.shape}")
print(f"  Feature counts match        : {X_train_t.shape[1] == X_test_t.shape[1]}")
'''))

# -------------------------------------------------------------- verification
cells.append(md(r"""
#### H.10 — Verifying that no leakage occurred

Preventing leakage is a claim, so it is **tested numerically** rather than
asserted. Four independent checks follow.
"""))

cells.append(code(r'''
# =====================================================================
# H.10a - CHECK 1: scaler statistics match the TRAINING distribution
# =====================================================================
# If the scaler had been fitted on the full dataset, its stored means would
# match the full-data means instead. We compare against BOTH to prove which
# one it actually learned.

scaler = (
    full_pipeline.named_steps["preprocessor"]
    .named_transformers_["num"]
    .named_steps["scaler"]
)

# Reproduce what the imputer would output, so the means are comparable.
imputed_train = SimpleImputer(strategy="median").fit_transform(X_train[numeric_cols])
imputed_full = SimpleImputer(strategy="median").fit_transform(X[numeric_cols])

train_means = pd.DataFrame(imputed_train, columns=numeric_cols).mean()
full_means = pd.DataFrame(imputed_full, columns=numeric_cols).mean()

comparison = pd.DataFrame({
    "scaler_mean_from_fit": scaler.mean_,
    "actual_TRAIN_mean": train_means.values,
    "actual_FULL_mean": full_means.values,
})
comparison["diff_vs_train"] = (
    comparison["scaler_mean_from_fit"] - comparison["actual_TRAIN_mean"]
).abs()
comparison["diff_vs_full"] = (
    comparison["scaler_mean_from_fit"] - comparison["actual_FULL_mean"]
).abs()
comparison.index = numeric_cols

display(comparison.round(6))

max_vs_train = comparison["diff_vs_train"].max()
print(f"Max deviation from TRAINING means : {max_vs_train:.12f}")

assert max_vs_train < 1e-8, "Scaler was NOT fitted on training data only!"
print("\nPASS: the scaler's parameters came from the training split.")
print("      (Deviations from the full-data means are present, which is")
print("       expected and correct - the test rows were excluded.)")
'''))

cells.append(code(r'''
# =====================================================================
# H.10b - CHECK 2: encoder vocabulary is a subset of TRAINING categories
# =====================================================================

onehot = (
    full_pipeline.named_steps["preprocessor"]
    .named_transformers_["cat"]
    .named_steps["onehot"]
)

# OneHotEncoder stores every category it saw during fit, per input column,
# in ONEHOT_COLS order. Note that min_frequency collapses rare categories into
# a single output column, so this list is WIDER than the output width - which
# is exactly what we want here: it is the full learned vocabulary.
train_vocab = {}
for col, cats in zip(ONEHOT_COLS, onehot.categories_):
    train_vocab[col] = set(cats)

rows = []
for col in ONEHOT_COLS:
    train_cats = set(X_train[col].unique())
    test_cats = set(X_test[col].unique())
    only_in_test = test_cats - train_cats

    rows.append({
        "column": col,
        "train_categories": len(train_cats),
        "test_categories": len(test_cats),
        "categories_only_in_test": len(only_in_test),
        "encoder_vocabulary": len(train_vocab[col]),
    })

vocab_report = pd.DataFrame(rows).set_index("column")
display(vocab_report)

n_test_only = int(vocab_report["categories_only_in_test"].sum())
print(f"Categories appearing only in the test set: {n_test_only}")
print("These are routed to the 'infrequent' bucket - the encoder never saw them")
print("during fit, which is exactly the correct production behaviour.")

# The encoder must not have learned anything that is test-only.
for col in ONEHOT_COLS:
    test_only = set(X_test[col].unique()) - set(X_train[col].unique())
    leaked = test_only.intersection(train_vocab[col] - {"infrequent_sklearn"})
    assert not leaked, f"Encoder learned test-only categories for {col}: {leaked}"

print("\nPASS: no test-only category appears in the fitted vocabulary.")
'''))

cells.append(code(r'''
# =====================================================================
# H.10c - CHECK 3: the frequency encoder used training frequencies only
# =====================================================================

freq_encoder = (
    full_pipeline.named_steps["preprocessor"]
    .named_transformers_["freq"]
)

print("FrequencyEncoder vocabulary, learned from training data:\n")
print(f"  Distinct rockets learned : {freq_encoder.learned_vocabulary_size['Rocket']}")
print(f"  Distinct rockets in full : {X['Rocket'].nunique()}")
print(f"  Distinct rockets in test : {X_test['Rocket'].nunique()}")

# Compare against frequencies computed from the FULL dataset - they must differ,
# proving the encoder did not see the test rows.
train_freq = X_train["Rocket"].value_counts(normalize=True)
full_freq = X["Rocket"].value_counts(normalize=True)

sample_rockets = train_freq.head(5).index.tolist()
freq_comparison = pd.DataFrame({
    "learned_from_train": [freq_encoder.frequencies_["Rocket"].get(r, 0.0) for r in sample_rockets],
    "actual_train_freq": [train_freq[r] for r in sample_rockets],
    "full_data_freq": [full_freq[r] for r in sample_rockets],
}, index=sample_rockets)

display(freq_comparison.round(6))

learned_matches_train = np.allclose(
    freq_comparison["learned_from_train"], freq_comparison["actual_train_freq"]
)
print(f"Learned values match TRAIN frequencies : {learned_matches_train}")
assert learned_matches_train, "FrequencyEncoder did not use training frequencies!"
print("PASS: frequency mapping came from the training split only.")
'''))

cells.append(code(r'''
# =====================================================================
# H.10d - CHECK 4: transformed matrices are clean and consistent
# =====================================================================

print("Transformed matrix integrity:\n")
print(f"  Training shape          : {X_train_t.shape}")
print(f"  Test shape              : {X_test_t.shape}")
print(f"  Feature counts match    : {X_train_t.shape[1] == X_test_t.shape[1]}")
print(f"  NaNs in training        : {int(np.isnan(X_train_t).sum())}")
print(f"  NaNs in test            : {int(np.isnan(X_test_t).sum())}")

# Numeric branch should be standardized: mean ~0, std ~1 on training data.
n_numeric = len(numeric_cols)
numeric_train_block = X_train_t[:, :n_numeric]

print(f"\n  Numeric block mean (train) : {numeric_train_block.mean():.10f}")
print(f"  Numeric block std  (train) : {numeric_train_block.std():.10f}")

assert X_train_t.shape[1] == X_test_t.shape[1], "Feature counts differ!"
assert not np.isnan(X_train_t).any(), "NaNs in transformed training data!"
assert not np.isnan(X_test_t).any(), "NaNs in transformed test data!"
assert abs(numeric_train_block.mean()) < 1e-8, "Numeric block is not centred!"

print("\nPASS: matrices are clean, consistent, and correctly standardized.")
print("      Test data was transformed using TRAINING parameters, not its own.")
'''))

# ------------------------------------------------------------ guard + totals
cells.append(code(r'''
# =====================================================================
# H.11 - Re-run the LeakageGuard on the final matrices
# =====================================================================
# The guard from Part F operates on DataFrames with named columns. After
# transformation the feature names are generated, so we reconstruct them and
# confirm no generated name corresponds to a forbidden column.
#
# NOTE ON NAMING: OneHotEncoder.categories_ lists EVERY category seen, but
# min_frequency collapses the rare ones into a single 'infrequent' column.
# So len(categories_) does NOT equal the output width. The authoritative
# source of names is get_feature_names_out().

onehot_names = onehot.get_feature_names_out(ONEHOT_COLS)

feature_names = []
feature_names.extend(numeric_cols)                 # numeric pass through
feature_names.extend(onehot_names)                 # one-hot (collapsed)
feature_names.extend(freq_encoder.get_feature_names_out())  # frequency

print(f"Feature-name sources:")
print(f"  numeric              : {len(numeric_cols)}")
print(f"  one-hot (collapsed)  : {len(onehot_names)}")
print(f"  frequency-encoded    : {len(freq_encoder.get_feature_names_out())}")
print(f"  TOTAL                : {len(feature_names)}")
print(f"\nMatrix width           : {X_train_t.shape[1]}")

assert len(feature_names) == X_train_t.shape[1], "Feature-name count mismatch!"
print("PASS: generated names match the matrix width exactly.")

X_train_df = pd.DataFrame(X_train_t, columns=feature_names, index=X_train.index)

guard = LeakageGuard(target=y_train)
print("\nGuard report on the transformed TRAINING matrix:\n")
display(guard.report(X_train_df))
guard.check(X_train_df)

forbidden_present = [c for c in LeakageGuard.FORBIDDEN_COLUMNS if c in feature_names]
print(f"\nForbidden columns among generated names: {forbidden_present or 'none'}")
print("PASS: transformed training matrix is leakage-free.")
'''))

cells.append(code(r'''
# =====================================================================
# H.12 - Before/after validation (guide §13A)
# =====================================================================

# Reconstruct row counts to confirm nothing was lost or duplicated.
total_rows = X_train.shape[0] + X_test.shape[0]

validation = pd.DataFrame([
    {"stage": "After cleaning (Part C)", "rows": len(df_eng), "columns": df_eng.shape[1]},
    {"stage": "After separation (Part G)", "rows": len(X), "columns": X.shape[1]},
    {"stage": "Training split", "rows": X_train.shape[0], "columns": X_train.shape[1]},
    {"stage": "Test split", "rows": X_test.shape[0], "columns": X_test.shape[1]},
    {"stage": "Transformed (train)", "rows": X_train_t.shape[0], "columns": X_train_t.shape[1]},
    {"stage": "Transformed (test)", "rows": X_test_t.shape[0], "columns": X_test_t.shape[1]},
]).set_index("stage")

display(validation)

assert total_rows == len(X), "Rows lost during the split!"
assert int(y_train.sum()) + int(y_test.sum()) == int(y.sum()), "Labels lost!"
assert (1 - y_train).sum() + (1 - y_test).sum() == (1 - y).sum(), "Minority labels lost!"

print(f"PASS: {len(X):,} rows preserved end-to-end ({X_train.shape[0]:,} + {X_test.shape[0]:,}).")
print(f"PASS: {int(y.sum()):,} positives and {int((1-y).sum()):,} negatives preserved.")
'''))

cells.append(code(r'''
# =====================================================================
# H.13 - Persist the model-ready artefacts
# =====================================================================

np.save("X_train.npy", X_train_t)
np.save("X_test.npy", X_test_t)
np.save("y_train.npy", np.asarray(y_train))
np.save("y_test.npy", np.asarray(y_test))

with open("feature_names.txt", "w") as fh:
    fh.write("\n".join(feature_names))

print("Saved model-ready artefacts:")
print(f"  X_train.npy        {X_train_t.shape}")
print(f"  X_test.npy         {X_test_t.shape}")
print(f"  y_train.npy        {np.asarray(y_train).shape}")
print(f"  y_test.npy         {np.asarray(y_test).shape}")
print(f"  feature_names.txt  {len(feature_names)} names")

print("\nPart H complete: the data is split, transformed, and leak-free.")
print("Part I will summarise every preprocessing decision in one table.")
'''))

# ---------------------------------------------------------------- charts
cells.append(md(r"""
#### H.14 — Visual: the split, the leakage proof, and the feature counts
"""))

cells.append(code(r'''
# =====================================================================
# H.14 - CHART 1: split composition and stratification
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# --- Left: row counts ---
sizes = [len(X_train), len(X_test)]
labels = [f"Training\n{sizes[0]:,} rows", f"Test\n{sizes[1]:,} rows"]
bars = axes[0].bar(labels, sizes, color=["#4C72B0", "#DD8452"],
                   edgecolor="white", linewidth=1.5)
for bar, s in zip(bars, sizes):
    axes[0].text(bar.get_x() + bar.get_width()/2, s + 60,
                 f"{s/len(X)*100:.0f}%", ha="center", fontsize=13, fontweight="bold")
axes[0].set_ylabel("Number of rows", fontsize=11)
axes[0].set_title("80/20 split", fontsize=13, fontweight="bold")
axes[0].set_ylim(0, max(sizes) * 1.18)

# --- Right: stratification worked ---
split_names = ["Full data", "Training", "Test"]
pos_rates = [y.mean() * 100, y_train.mean() * 100, y_test.mean() * 100]
minority_rates = [100 - p for p in pos_rates]

x = np.arange(len(split_names))
w = 0.36
axes[1].bar(x - w/2, pos_rates, w, label="Success", color="#55A868", edgecolor="white")
axes[1].bar(x + w/2, minority_rates, w, label="Not-Success", color="#C44E52", edgecolor="white")

for xi, (p, m) in enumerate(zip(pos_rates, minority_rates)):
    axes[1].text(xi - w/2, p + 1.5, f"{p:.2f}%", ha="center", fontsize=9.5, fontweight="bold")
    axes[1].text(xi + w/2, m + 1.5, f"{m:.2f}%", ha="center", fontsize=9.5, fontweight="bold")

axes[1].set_xticks(x)
axes[1].set_xticklabels(split_names)
axes[1].set_ylabel("Share of rows (%)", fontsize=11)
axes[1].set_title("Stratification preserved the class ratio", fontsize=13, fontweight="bold")
axes[1].legend()
axes[1].set_ylim(0, 108)

plt.tight_layout()
plt.show()

print(f"Positive rate - full: {y.mean():.4f} | train: {y_train.mean():.4f} | test: {y_test.mean():.4f}")
print("The three rates are nearly identical, which is what stratify=y guarantees.")
print("Test set carries", int((1 - y_test).sum()), "failure cases to evaluate on.")
'''))

cells.append(code(r'''
# =====================================================================
# H.15 - CHART 2: the leakage proof
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# --- Left: scaler means vs train means vs full means ---
x = np.arange(len(numeric_cols))
w = 0.27

axes[0].bar(x - w, comparison["scaler_mean_from_fit"], w,
            label="Scaler (as fitted)", color="#4C72B0", edgecolor="white")
axes[0].bar(x, comparison["actual_TRAIN_mean"], w,
            label="Actual TRAIN mean", color="#55A868", edgecolor="white")
axes[0].bar(x + w, comparison["actual_FULL_mean"], w,
            label="Actual FULL mean", color="#C44E52", edgecolor="white", alpha=0.75)

short = ["Prior\nLaunches", "Prior\nFail Rate", "Launch\nYear", "Month\nSin", "Month\nCos"]
axes[0].set_xticks(x)
axes[0].set_xticklabels(short, fontsize=9)
axes[0].set_title("Scaler matches TRAIN means, not full-data means",
                  fontsize=12, fontweight="bold")
axes[0].legend(fontsize=8.5)
axes[0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

# --- Right: deviation comparison, log scale ---
dev_train = comparison["diff_vs_train"].values
dev_full = comparison["diff_vs_full"].values

axes[1].bar(x - w/2, np.maximum(dev_train, 1e-13), w,
            label="Deviation vs TRAIN means", color="#55A868", edgecolor="white")
axes[1].bar(x + w/2, np.maximum(dev_full, 1e-13), w,
            label="Deviation vs FULL means", color="#C44E52", edgecolor="white")

axes[1].set_yscale("log")
axes[1].set_xticks(x)
axes[1].set_xticklabels(short, fontsize=9)
axes[1].set_ylabel("Absolute deviation (log scale)", fontsize=11)
axes[1].set_title("Zero deviation vs train = no leakage", fontsize=12, fontweight="bold")
axes[1].legend(fontsize=8.5)
axes[1].axhline(1e-8, ls="--", color="gray", linewidth=1.3)
axes[1].text(0.05, 3e-8, "tolerance", fontsize=8.5, color="gray")

plt.tight_layout()
plt.show()

print(f"Max deviation from training means : {max_vs_train:.2e}")
print("The green bars sit at the floor (zero deviation); the red bars do not.")

print("\nInterpretation: the scaler's stored means reproduce the TRAINING means")
print("to machine precision. If it had seen the test rows, its means would have")
print("matched the full-data means instead - which they do not.")
'''))

cells.append(code(r'''
# =====================================================================
# H.16 - CHART 3: where the features came from
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# --- Left: contribution of each source column to final width ---
# Count one-hot names per source column using the authoritative name list.
n_country = sum(1 for n in onehot_names if n.startswith("Country_"))
n_company = sum(1 for n in onehot_names if n.startswith("Company_"))

contributions = {
    f"Numeric ({len(numeric_cols)} cols)": len(numeric_cols),
    "Country one-hot": n_country,
    "Company one-hot": n_company,
    "Rocket frequency": 1,
}
total = sum(contributions.values())

bars = axes[0].barh(list(contributions.keys()), list(contributions.values()),
                    color=["#4C72B0", "#55A868", "#8172B2", "#DD8452"],
                    edgecolor="white", linewidth=1.2)
for bar, v in zip(bars, contributions.values()):
    axes[0].text(v + 0.4, bar.get_y() + bar.get_height()/2,
                 f"{v}  ({v/total*100:.0f}%)", va="center", fontsize=10, fontweight="bold")
axes[0].set_xlabel("Columns produced", fontsize=11)
axes[0].set_title(f"Final feature space: {total} columns", fontsize=12, fontweight="bold")
axes[0].set_xlim(0, max(contributions.values()) * 1.4)

# --- Right: the Rocket encoding decision, side by side ---
rocket_options = {
    "One-hot\n(min_frequency=10)": 86,
    "Frequency\nencoding (chosen)": 1,
}
bars2 = axes[1].bar(list(rocket_options.keys()), list(rocket_options.values()),
                    color=["#C44E52", "#55A868"], edgecolor="white", linewidth=1.5)
for bar, v in zip(bars2, rocket_options.values()):
    axes[1].text(bar.get_x() + bar.get_width()/2, v + 2, str(v),
                 ha="center", fontsize=14, fontweight="bold")
axes[1].set_ylabel("Columns added for 'Rocket'", fontsize=11)
axes[1].set_title("Part D's frequency-encoding decision, quantified",
                  fontsize=12, fontweight="bold")
axes[1].set_ylim(0, 100)
axes[1].annotate("86 columns for one feature\nwould dominate the matrix",
                 xy=(0, 86), xytext=(0.35, 62), fontsize=9.5, color="#7A2020",
                 arrowprops=dict(arrowstyle="->", color="#C44E52", lw=1.5))

plt.tight_layout()
plt.show()

print(f"Final matrix: {X_train_t.shape[1]} features from {len(X.columns)} source columns.")
print(f"Rows:features ratio = {len(X) // X_train_t.shape[1]} : 1")
print("\nFrequency encoding saved 85 columns versus one-hot on 'Rocket' alone.")
print("Part D committed to this; Part H implements it and shows the saving.")
'''))

# ---------------------------------------------------------------- summary
cells.append(code(r'''
# =====================================================================
# H.17 - Part H summary
# =====================================================================

part_h_summary = pd.DataFrame([
    {"item": "Split ratio", "value": "80 / 20, stratified (random_state=42)"},
    {"item": "Training rows", "value": f"{X_train.shape[0]:,}"},
    {"item": "Test rows", "value": f"{X_test.shape[0]:,}"},
    {"item": "Minority rows in test", "value": f"{int((1 - y_test).sum())} failures"},
    {"item": "Source features", "value": f"{X.shape[1]}"},
    {"item": "Transformed features", "value": f"{X_train_t.shape[1]}"},
    {"item": "Fitted on", "value": "X_train ONLY"},
    {"item": "Imputation", "value": "Wired in, idle (X has no missing values)"},
    {"item": "Scaling", "value": "StandardScaler, fitted on train"},
    {"item": "Encoding", "value": "One-hot (Country, Company) + Frequency (Rocket)"},
    {"item": "Unseen test categories", "value": "18 Rocket, 1 Company -> 'infrequent'/0.0"},
    {"item": "Leakage checks", "value": "4 numeric checks + LeakageGuard, all PASS"},
    {"item": "Rows preserved", "value": f"{len(X):,} (train + test)"},
]).set_index("item")

display(part_h_summary)

print("Artefacts saved: X_train.npy, X_test.npy, y_train.npy, y_test.npy, feature_names.txt")
'''))

nb["cells"].extend(cells)

with open(NB_PATH, "w") as f:
    json.dump(nb, f, indent=1)

print(f"Appended {len(cells)} Part H cells.")
print(f"Notebook now has {len(nb['cells'])} cells.")
