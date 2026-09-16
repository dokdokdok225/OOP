"""
Inserts Part G (Feature and Target Separation) into oop.ipynb.

Reuses the LeakageGuard from Part F to validate X before it reaches Part H.

Run: python3 build_part_g.py
"""
import json

NB_PATH = "oop.ipynb"
DOC_PATH = "_docs/part_g_doc.md"

with open(NB_PATH) as f:
    nb = json.load(f)
with open(DOC_PATH) as f:
    part_g_doc = f.read()


def md(text):
    return {"cell_type": "markdown", "metadata": {},
            "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.strip("\n").splitlines(keepends=True)}


cells = [md(part_g_doc)]

# ------------------------------------------------------------ G.5 separation
cells.append(md(r"""
---

#### G.5 — The separation, performed

Two definitions drive everything: which column is the target, and which columns
are forbidden in `X`. Both are expressed as named configuration, not sprinkled
through the code, so the decision is auditable in one place.
"""))

cells.append(code(r'''
# =====================================================================
# G.5 - Feature and target separation
# =====================================================================

TARGET_SOURCE = "MissionStatus"
TARGET_LABEL = "target"
POSITIVE_LABEL = "Success"

# Columns excluded from X, each with the part that justified the decision.
EXCLUSION_REGISTER = {
    "MissionStatus":            "The target itself (Part B).",
    "target":                   "Derived target - never a feature.",
    "_is_fail":                 "Binary restatement of the target (Part E helper).",
    "Mission":                  "98.4% unique identifier + latent name leak (Parts A, F).",
    "Time":                     "No signal: 7.6-10.2% failure across hour buckets (Parts C, F).",
    "RocketStatus":             "SEVERE temporal leakage - 2022 snapshot (Part F).",
    "Date":                     "Superseded by Launch_Year / Month_Sin / Month_Cos (Part D).",
    "Company_Prior_Failures":   "0.952 correlation with prior launches - redundant (Part E).",
}

# --- Build the target vector ---
# y = 1 when the mission succeeded, 0 otherwise.
y = (df_eng[TARGET_SOURCE] == POSITIVE_LABEL).astype(int)
y.name = TARGET_LABEL

# Keep the ORIGINAL 4-level text for reference (not used for modelling).
outcome_labels = df_eng[TARGET_SOURCE].copy()
outcome_labels.name = "outcome_labels"

# --- Build the feature matrix ---
drop_columns = [c for c in EXCLUSION_REGISTER if c in df_eng.columns]
X = df_eng.drop(columns=drop_columns)

print(f"Target vector  y : {y.shape[0]} values  (name={y.name!r})")
print(f"Feature matrix X : {X.shape[0]} rows x {X.shape[1]} columns")
print(f"\nColumns dropped from X ({len(drop_columns)}):")
for col in drop_columns:
    print(f"  - {col:<26} {EXCLUSION_REGISTER[col]}")

print(f"\nFeature columns ({X.shape[1]}):")
for col in X.columns:
    kind = "categorical" if X[col].dtype == object else "numeric"
    print(f"  - {col:<28} {str(X[col].dtype):<10} {kind:<12} {X[col].nunique():>5} unique")
'''))

cells.append(code(r'''
# =====================================================================
# G.5a - Target distribution
# =====================================================================
target_summary = pd.DataFrame({
    "label": [POSITIVE_LABEL, "Not-Success"],
    "binary_value": [1, 0],
    "count": [int(y.sum()), int((1 - y).sum())],
    "share_pct": [round(y.mean() * 100, 2), round((1 - y.mean()) * 100, 2)],
}).set_index("label")

display(target_summary)

print(f"Positive rate  : {y.mean():.4f}")
print(f"Imbalance ratio: {y.mean() / (1 - y.mean()):.2f} : 1")
print(f"\nOriginal 4-level labels preserved as 'outcome_labels':")
print(outcome_labels.value_counts().to_string())
'''))

# ------------------------------------------------------------ validation
cells.append(md(r"""
#### G.6 — Validating `X` with the Part F guard

A separation is only correct if it is *checked*. The `LeakageGuard` built in
Part F is applied to `X` here — which is also why it was written as a reusable
class rather than a one-off assertion.
"""))

cells.append(code(r'''
# =====================================================================
# G.6 - Run the LeakageGuard against the separated X
# =====================================================================

guard = LeakageGuard(target=y)

print("Guard report:\n")
display(guard.report(X))

# check() raises AssertionError on any violation.
guard.check(X)

print("\nPASS: X contains no target leakage.")
print(f"      - no forbidden columns among {X.shape[1]} features")
print(f"      - no binary column restates the target")
'''))

cells.append(code(r'''
# =====================================================================
# G.6a - Independent structural checks
# =====================================================================
# The guard covers leakage. These assertions cover the rest of the separation.

# 1. X and y must describe the same observations, in the same order.
assert len(X) == len(y) == len(df_eng), "X and y have different row counts!"
assert X.index.equals(y.index), "X and y are not index-aligned!"
print(f"1. Index alignment      : PASS ({len(X)} rows, aligned)")

# 2. The target must NOT appear anywhere in X.
assert TARGET_SOURCE not in X.columns, "Target column leaked into X!"
assert TARGET_LABEL not in X.columns, "Derived target leaked into X!"
print("2. Target absent from X : PASS")

# 3. No constant (zero-variance) features - they carry no information.
constant_cols = [c for c in X.columns if X[c].nunique() <= 1]
assert not constant_cols, f"Constant columns in X: {constant_cols}"
print("3. No constant columns  : PASS")

# 4. Class balance is preserved from the raw data.
raw_positive = (df_eng[TARGET_SOURCE] == POSITIVE_LABEL).mean()
assert abs(raw_positive - y.mean()) < 1e-12, "Class balance drifted!"
print(f"4. Class balance        : PASS ({y.mean():.4f} == raw {raw_positive:.4f})")

# 5. Feature count is what we expect (guards against a silent drop failure).
expected_features = 8
assert X.shape[1] == expected_features, f"Expected {expected_features} features, got {X.shape[1]}"
print(f"5. Feature count        : PASS ({X.shape[1]} features)")

print("\nAll structural checks passed.")
'''))

# ------------------------------------------------------------------- charts
cells.append(md(r"""
#### G.7 — Visual: the target distribution and the feature map
"""))

cells.append(code(r'''
# =====================================================================
# G.7 - CHART 1: class balance, and why accuracy is rejected
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# --- Left: the binary target distribution ---
counts = [int(y.sum()), int((1 - y).sum())]
labels = [f"Success\n(n={counts[0]:,})", f"Not-Success\n(n={counts[1]:,})"]
colors = ["#55A868", "#C44E52"]

bars = axes[0].bar(labels, counts, color=colors, edgecolor="white", linewidth=1.5)
for bar, c in zip(bars, counts):
    pct = c / len(y) * 100
    axes[0].text(bar.get_x() + bar.get_width() / 2, c + 70,
                 f"{pct:.2f}%", ha="center", fontsize=13, fontweight="bold")

axes[0].set_ylabel("Number of missions", fontsize=11)
axes[0].set_title("Binary target is imbalanced 9:1", fontsize=13, fontweight="bold")
axes[0].set_ylim(0, max(counts) * 1.18)

# The trivial-predictor line
axes[0].axhline(counts[0], ls=":", color="gray", linewidth=1.6)
axes[0].text(0.5, counts[0] * 0.55,
             "A model predicting 'Success'\nfor EVERY mission scores\n"
             f"{y.mean()*100:.2f}% accuracy\nand catches 0 failures.",
             ha="center", fontsize=10,
             bbox=dict(boxstyle="round,pad=0.5", fc="#FFF3CD", ec="gray"))

# --- Right: the original 4-level distribution ---
level_counts = outcome_labels.value_counts()
axes[1].barh(level_counts.index[::-1], level_counts.values[::-1],
             color=["#C44E52", "#DD8452", "#C44E52", "#55A868"],
             edgecolor="white", linewidth=1.2)
for i, v in enumerate(level_counts.values[::-1]):
    axes[1].text(v + 60, i, f"{v:,}", va="center", fontsize=11, fontweight="bold")

axes[1].set_xlabel("Number of missions", fontsize=11)
axes[1].set_title("Original 4 levels before binarisation", fontsize=13, fontweight="bold")
axes[1].set_xlim(0, level_counts.max() * 1.25)
axes[1].annotate("only 4 observations -\ncannot be stratified",
                 xy=(4, 0), xytext=(900, 0.55), fontsize=10, color="#C44E52",
                 arrowprops=dict(arrowstyle="->", color="#C44E52", lw=1.5))

plt.tight_layout()
plt.show()

print(f"The positive class dominates at {y.mean()*100:.2f}%.")
print("This is exactly why Part B rejected accuracy: a trivial majority-class")
print("predictor scores ~90% while detecting zero of the 468 failures.")
print(f"\n'Prelaunch Failure' has only {level_counts['Prelaunch Failure']} rows, which is")
print("why binarisation was necessary rather than optional.")
'''))

cells.append(code(r'''
# =====================================================================
# G.8 - CHART 2: the separation map - what went where and why
# =====================================================================
fig, ax = plt.subplots(figsize=(13, 6.5))
ax.axis("off")

# Three columns: features, target, excluded.
feature_cols = list(X.columns)
excluded_cols = list(EXCLUSION_REGISTER.keys())
excluded_present = [c for c in excluded_cols if c in df_eng.columns]

ax.text(0.17, 0.97, "FEATURES  (X)", ha="center", fontsize=13,
        fontweight="bold", color="#4C72B0")
ax.text(0.50, 0.97, "TARGET  (y)", ha="center", fontsize=13,
        fontweight="bold", color="#55A868")
ax.text(0.83, 0.97, "EXCLUDED  (with reason)", ha="center", fontsize=13,
        fontweight="bold", color="#C44E52")

# Left column: features
for i, col in enumerate(feature_cols):
    yy = 0.88 - i * 0.088
    ax.add_patch(plt.Rectangle((0.03, yy - 0.032), 0.28, 0.068,
                               facecolor="#DCE6F5", edgecolor="#4C72B0", linewidth=1.2))
    ax.text(0.17, yy, col, ha="center", va="center", fontsize=9.5)

# Middle: target
ax.add_patch(plt.Rectangle((0.38, 0.74), 0.24, 0.10,
                           facecolor="#D9EFD9", edgecolor="#55A868", linewidth=1.8))
ax.text(0.50, 0.79, "MissionStatus", ha="center", va="center", fontsize=10, fontweight="bold")
ax.text(0.50, 0.655, "binarised: Success=1\nvs Not-Success=0", ha="center",
        va="center", fontsize=9)
ax.text(0.50, 0.56, "4,161 / 468", ha="center", va="center",
        fontsize=9, style="italic", color="gray")

# Right column: excluded
for i, col in enumerate(excluded_present):
    yy = 0.88 - i * 0.088
    reason = EXCLUSION_REGISTER[col].split("(")[0].strip().rstrip(".")
    if len(reason) > 34:
        reason = reason[:32] + "..."
    ax.add_patch(plt.Rectangle((0.60, yy - 0.032), 0.38, 0.068,
                               facecolor="#FBE3E3", edgecolor="#C44E52", linewidth=1.2))
    ax.text(0.62, yy, col, ha="left", va="center", fontsize=9.5, fontweight="bold")
    ax.text(0.79, yy, reason, ha="left", va="center", fontsize=8, color="#7A2020")

ax.text(0.5, 0.03, f"{X.shape[0]:,} observations  |  {X.shape[1]} features  |  "
                   f"ratio {X.shape[0] // X.shape[1]}:1",
        ha="center", fontsize=11, fontweight="bold")

ax.set_title("Part G: feature and target separation, with every exclusion justified",
             fontsize=13, fontweight="bold", pad=16)
plt.tight_layout()
plt.show()

print("Every excluded column has a documented reason traceable to a specific part.")
print("Nothing was dropped silently, and nothing forbidden reached X.")
'''))

cells.append(code(r'''
# =====================================================================
# G.9 - Part G summary
# =====================================================================

separation_summary = pd.DataFrame([
    {"item": "Observations", "value": f"{len(X):,}"},
    {"item": "Features in X", "value": f"{X.shape[1]}"},
    {"item": "Categorical features", "value": "3  (Company, Rocket, Country)"},
    {"item": "Numeric features", "value": "5  (Prior_Launches, Prior_Failure_Rate, Launch_Year, Month_Sin, Month_Cos)"},
    {"item": "Target", "value": f"y = ({TARGET_SOURCE} == '{POSITIVE_LABEL}')  ->  1 / 0"},
    {"item": "Positive rate", "value": f"{y.mean():.4f}"},
    {"item": "Excluded columns", "value": f"{len(excluded_present)}  (each justified in G.2)"},
    {"item": "Leakage guard", "value": "PASS"},
    {"item": "Rows : features", "value": f"{len(X) // X.shape[1]} : 1"},
]).set_index("item")

display(separation_summary)

print("X and y are separated, validated, and ready for Part H.")
print("Part H will split them and fit all transformations on the TRAINING set only.")
'''))

nb["cells"].extend(cells)

with open(NB_PATH, "w") as f:
    json.dump(nb, f, indent=1)

print(f"Appended {len(cells)} Part G cells.")
print(f"Notebook now has {len(nb['cells'])} cells.")
