"""
Adds the 4 visual charts to Parts C and D:
  Part C: (1) missingness bar chart, (2) rows before/after cleaning
  Part D: (3) cyclical month circle, (4) unscaled vs standardized comparison

Inserted as new cells immediately after the relevant existing cells, without
modifying any working cell.

Run: python3 add_charts_cd.py
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


def find_cell(predicate):
    for i, c in enumerate(nb["cells"]):
        if predicate("".join(c["source"])):
            return i
    raise SystemExit("Target cell not found.")


# ============================================================ PART C CHARTS

# Matplotlib must be imported before the Part C charts. The setup is placed
# here (the first chart in the notebook) so later parts inherit the styling.
nb["cells"].insert(0, code(r'''
import matplotlib.pyplot as plt

# Consistent chart styling for the whole notebook. Declared once, here, so
# every chart in Parts C, D and E renders with the same conventions.
plt.rcParams.update({
    "figure.figsize": (9, 5),
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})
'''))

c_missing_md = md(r"""
---

#### C.5 — Visual: the missingness that drove our decisions

The table above states the numbers; this chart makes the *scale* of the problem
visible. `Price` is not merely incomplete — it is absent for roughly three
quarters of all records, which is why it could not be repaired.
""")

c_missing_code = code(r'''
missing_pct = (df_raw.isna().mean() * 100).sort_values(ascending=False)
missing_pct = missing_pct[missing_pct > 0]

fig, ax = plt.subplots(figsize=(8, 4.5))
bars = ax.barh(missing_pct.index, missing_pct.values,
               color=["#C44E52" if v > 50 else "#4C72B0" for v in missing_pct.values],
               edgecolor="white", linewidth=1.2)

for bar, (col_name, v) in zip(bars, missing_pct.items()):
    n_missing = int(df_raw[col_name].isna().sum())
    ax.text(v + 1.0, bar.get_y() + bar.get_height() / 2,
            f"{v:.2f}%  ({n_missing:,} rows)", va="center", fontsize=10)

ax.axvline(50, ls="--", color="gray", linewidth=1.4)
ax.text(50.8, -0.45, "50% threshold\n(guide §8A)", fontsize=9, color="gray")
ax.set_xlabel("Percentage of values missing (%)", fontsize=11)
ax.set_title("Missingness before cleaning: Price was the decisive case",
             fontsize=12, fontweight="bold")
ax.set_xlim(0, 88)
plt.tight_layout()
plt.show()

print(f"Price was {missing_pct['Price']:.2f}% missing - above the guide's 50-70% drop")
print("threshold, so imputation would have fabricated more than it recovered.")
print(f"Time was only {missing_pct['Time']:.2f}% missing - low enough to keep, but with")
print("no causal bearing on reliability, so it was excluded rather than imputed.")
''')

c_rows_md = md(r"""
#### C.6 — Visual: what cleaning actually removed

Cleaning should be **conservative**. This chart shows how little was removed —
and, more importantly, that the co-launch cases were *not* removed.
""")

c_rows_code = code(r'''
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

# Left: row counts through the cleaning steps
steps = ["Raw", "After exact\ndedup", "After business\ndedup", "Final"]
counts = [4630, 4629, 4628, len(df)]
colors = ["#8172B2", "#4C72B0", "#4C72B0", "#55A868"]
bars = axes[0].bar(steps, counts, color=colors, edgecolor="white", linewidth=1.2)
for bar, v in zip(bars, counts):
    axes[0].text(bar.get_x() + bar.get_width() / 2, v + 0.6,
                 f"{v:,}", ha="center", fontsize=10, fontweight="bold")
axes[0].set_ylim(4620, 4635)
axes[0].set_ylabel("Number of rows", fontsize=11)
axes[0].set_title("Cleaning removed only 2 rows", fontsize=12, fontweight="bold")
axes[0].tick_params(axis="x", labelsize=9)

# Right: what we deliberately KEPT
kept = pd.Series({
    "Legitimate\nco-launches\n(KEPT)": 8,
    "Business\nduplicate\n(removed)": 1,
    "Exact\nduplicate\n(removed)": 1,
})
bars2 = axes[1].bar(kept.index, kept.values,
                    color=["#55A868", "#C44E52", "#C44E52"],
                    edgecolor="white", linewidth=1.2)
for bar, v in zip(bars2, kept.values):
    axes[1].text(bar.get_x() + bar.get_width() / 2, v + 0.25, str(v),
                 ha="center", fontsize=11, fontweight="bold")
axes[1].set_ylabel("Number of rows", fontsize=11)
axes[1].set_title("Rows sharing Company+Rocket+Date", fontsize=12, fontweight="bold")
axes[1].set_ylim(0, 10)
axes[1].tick_params(axis="x", labelsize=9)

plt.tight_layout()
plt.show()

print("A naive 'same company + rocket + date = duplicate' rule would have")
print("deleted all 10 rows. Requiring a MISSING TIME as the duplicate signature")
print("removed only the 1 true re-entry and preserved 8 real launches.")
''')

# ============================================================ PART D CHARTS

d_month_md = md(r"""
---

#### D.7 — Visual: why month needs cyclical encoding

This is the clearest way to see the problem. Both panels show the twelve months,
but positioned differently:

- **Left** — months as raw integers on a line. December sits 11 units from
  January: the *widest* gap in the year, despite being consecutive.
- **Right** — months on a circle. December and January are neighbours, exactly
  as they are in reality.
""")

d_month_code = code(r'''
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# --- Left: the WRONG representation (raw integers on a line) ---
months = np.arange(1, 13)
labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
axes[0].scatter(months, np.zeros_like(months), s=260, color="#4C72B0", zorder=3)
for m, lab in zip(months, labels):
    axes[0].annotate(lab, (m, 0), textcoords="offset points", xytext=(0, 14),
                     ha="center", fontsize=10)
axes[0].annotate("", xy=(12, 0), xytext=(1, 0),
                 arrowprops=dict(arrowstyle="<->", color="#C44E52", lw=2))
axes[0].text(6.5, -0.30, "distance = 11\n(the LARGEST gap)", ha="center",
             fontsize=11, color="#C44E52", fontweight="bold")
axes[0].set_xlim(0, 13)
axes[0].set_ylim(-0.6, 0.5)
axes[0].set_yticks([])
axes[0].set_xlabel("Month as a raw integer", fontsize=11)
axes[0].set_title("WRONG: raw integers imply Dec and Jan are farthest apart",
                  fontsize=12, fontweight="bold")

# --- Right: the CORRECT representation (unit circle) ---
theta = np.linspace(0, 2 * np.pi, 300)
axes[1].plot(np.cos(theta), np.sin(theta), color="lightgray", lw=1.5, zorder=1)
for m, lab in zip(months, labels):
    angle = 2 * np.pi * m / 12
    x, y = np.cos(angle), np.sin(angle)
    color = "#C44E52" if lab in ("Dec", "Jan") else "#4C72B0"
    axes[1].scatter([x], [y], s=240, color=color, zorder=3)
    axes[1].annotate(lab, (x, y), textcoords="offset points", xytext=(x * 13, y * 13),
                     ha="center", fontsize=10, fontweight="bold" if lab in ("Dec", "Jan") else "normal")

# Highlight the Dec-Jan adjacency
for lab in ("Dec", "Jan"):
    m = labels.index(lab) + 1
    angle = 2 * np.pi * m / 12
    axes[1].scatter([np.cos(angle)], [np.sin(angle)], s=380, facecolors="none",
                    edgecolors="#C44E52", linewidths=2, zorder=4)

axes[1].set_aspect("equal")
axes[1].set_xlim(-1.5, 1.5)
axes[1].set_ylim(-1.5, 1.5)
axes[1].set_title("CORRECT: sin/cos places Dec next to Jan",
                  fontsize=12, fontweight="bold")
axes[1].annotate("distance = 0.52\n(same as Jan-Feb)",
                 xy=(0.85, 0.35), fontsize=10, color="#C44E52", fontweight="bold")

plt.tight_layout()
plt.show()

print(f"Measured in sin/cos space: Jan<->Dec = {month_distance(1, 12):.4f}")
print(f"                          Jan<->Feb = {month_distance(1, 2):.4f}  (identical)")
print(f"                          Jan<->Jul = {month_distance(1, 7):.4f}  (opposite)")
print("\nThe raw-integer version states something false about the data.")
''')

d_scale_md = md(r"""
#### D.8 — Visual: what standardization changes (and what it does not)

The required discussion argued that scaling matters for K-Means but not for a
tree. These two panels show the evidence directly.
""")

d_scale_code = code(r'''
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# --- Left: raw feature scales are wildly different ---
Xc = df[["Launch_Year", "Company_Prior_Launches"]].astype(float)
axes[0].scatter(Xc["Launch_Year"], Xc["Company_Prior_Launches"],
                s=6, alpha=0.25, color="#4C72B0")
axes[0].set_xlabel("Launch_Year  (span 65)", fontsize=11)
axes[0].set_ylabel("Company_Prior_Launches  (span 1,776)", fontsize=11)
axes[0].set_title("Raw scales: one feature dwarfs the other", fontsize=12, fontweight="bold")
axes[0].text(1958, 1500,
             f"prior-launches drives\n{share_prior:.1f}% of the squared distance",
             fontsize=11, color="#C44E52", fontweight="bold")

# --- Right: the tree is unaffected by scaling ---
importance_plot = pd.DataFrame({
    "raw": tree_raw.feature_importances_,
    "standardized": tree_std.feature_importances_,
}, index=tree_features)

x = np.arange(len(tree_features))
width = 0.36
axes[1].bar(x - width / 2, importance_plot["raw"], width,
            label="Unscaled", color="#4C72B0", edgecolor="white")
axes[1].bar(x + width / 2, importance_plot["standardized"], width,
            label="Standardized", color="#55A868", edgecolor="white")
axes[1].set_xticks(x)
axes[1].set_xticklabels(["Launch\nYear", "Prior\nLaunches", "Prior\nFailures"], fontsize=10)
axes[1].set_ylabel("Decision tree feature importance", fontsize=11)
axes[1].set_title("Tree importances are IDENTICAL either way",
                  fontsize=12, fontweight="bold")
axes[1].legend()

plt.tight_layout()
plt.show()

print("Left  : K-Means needs scaling, because distance is dominated by the")
print(f"        wider feature ({share_prior:.1f}% of squared distance).")
print("Right : the tree's importances are byte-for-byte identical, because")
print("        rescaling preserves the ORDER of values and thus every split.")
print("\nThis is why we standardize: Stage 1 requires it, Stage 2 ignores it.")
print("One choice serves both, and the right-skewed distribution favours")
print("standardization over Min-Max (see D.3).")
''')

# ================================================================ insertions
# Part C chart 1: after the "C.4 - Cleaning summary" code cell
idx_c_summary = find_cell(lambda s: "C.4 - Cleaning summary" in s)
nb["cells"][idx_c_summary + 1:idx_c_summary + 1] = [c_missing_md, c_missing_code]

# Part C chart 2: after the "Preview the cleaned dataset" code cell
idx_c_preview = find_cell(lambda s: "Preview the cleaned dataset" in s)
nb["cells"][idx_c_preview + 1:idx_c_preview + 1] = [c_rows_md, c_rows_code]

# Part D chart 1: after the D.2b verification cell
idx_d_verify = find_cell(lambda s: "D.2b - VERIFY the cyclical encoding" in s)
nb["cells"][idx_d_verify + 1:idx_d_verify + 1] = [d_month_md, d_month_code]

# Part D chart 2: after the D.3d CV evidence cell
idx_d_cv = find_cell(lambda s: "D.3d - EVIDENCE: does scaling affect tree" in s)
nb["cells"][idx_d_cv + 1:idx_d_cv + 1] = [d_scale_md, d_scale_code]

with open(NB_PATH, "w") as f:
    json.dump(nb, f, indent=1)

print(f"Added 4 charts. Notebook now has {len(nb['cells'])} cells.")
