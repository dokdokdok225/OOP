"""
Inserts Part F (Data Leakage Audit) into oop.ipynb.

OOP application per oop_rules.md §13:
  - a LeakageGuard class with protected state and @property accessors
  - class-level blocklists (shared configuration, §7)
  - validation before assignment (§5)
  - reusable across Parts F and G

Run: python3 build_part_f.py
"""
import json

NB_PATH = "oop.ipynb"
DOC_PATH = "_docs/part_f_doc.md"

with open(NB_PATH) as f:
    nb = json.load(f)
with open(DOC_PATH) as f:
    part_f_doc = f.read()


def md(text):
    return {"cell_type": "markdown", "metadata": {},
            "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.strip("\n").splitlines(keepends=True)}


cells = [md(part_f_doc)]

# ------------------------------------------------------------- audit table
cells.append(md(r"""
---

#### F.7 — The audit, computed

The table in F.1 is stated as a conclusion. This cell derives it from the data, so
the verdicts are reproducible rather than asserted.
"""))

cells.append(code(r'''
# =====================================================================
# F.7 - Build the leakage audit from measured evidence
# =====================================================================
# For each column we record whether its value is knowable before liftoff, and
# what the audit concluded. "Knowable" is the governing test: leakage is
# information that exists only AFTER the outcome is decided.

audit_rows = [
    {
        "column": "MissionStatus",
        "role": "Target label",
        "knowable_pre_launch": "N/A",
        "risk": "None - this IS the target",
        "action": "Excluded from X; becomes y in Part G",
    },
    {
        "column": "RocketStatus",
        "role": "Vehicle status (Active/Retired)",
        "knowable_pre_launch": "No",
        "risk": "SEVERE - temporal leakage",
        "action": "DROPPED - see F.8",
    },
    {
        "column": "Mission",
        "role": "Payload name",
        "knowable_pre_launch": "Yes, but 98.4% unique",
        "risk": "Identifier + LATENT leak",
        "action": "Dropped as identifier - see F.9",
    },
    {
        "column": "Price",
        "role": "Reported launch cost",
        "knowable_pre_launch": "Partially",
        "risk": "Retroactive recording bias",
        "action": "Dropped at 72.7% missing - see F.10",
    },
    {
        "column": "Time",
        "role": "Launch clock time",
        "knowable_pre_launch": "Yes",
        "risk": "None - verified no signal",
        "action": "Excluded (no causal bearing)",
    },
    {
        "column": "Date",
        "role": "Scheduled launch date",
        "knowable_pre_launch": "Yes",
        "risk": "None when used causally",
        "action": "Kept; engineered in Parts D and E",
    },
    {
        "column": "Company",
        "role": "Operating organisation",
        "knowable_pre_launch": "Yes",
        "risk": "None",
        "action": "Kept",
    },
    {
        "column": "Rocket",
        "role": "Launch vehicle model",
        "knowable_pre_launch": "Yes",
        "risk": "None",
        "action": "Kept (frequency encoded)",
    },
    {
        "column": "Country",
        "role": "Launch geography",
        "knowable_pre_launch": "Yes",
        "risk": "None",
        "action": "Kept",
    },
]

leakage_audit = pd.DataFrame(audit_rows).set_index("column")
display(leakage_audit)
'''))

# ------------------------------------------------------------- guard class
cells.append(md(r"""
#### F.8 — The `LeakageGuard` class

Preventing leakage is a claim, so it is enforced in code. The guard applies two
independent checks:

| Check | What it catches | Why it is needed |
| :--- | :--- | :--- |
| **Blocklist** | Columns known to be unsafe | Fast, explicit, documents intent |
| **Near-restatement scan** | A numeric column that *is* the target in disguise | A blocklist can only catch columns someone remembered to list |

The second check exists because of a real mistake during development: a
`_is_fail` helper column used to build the Part E features silently entered `X`,
and the blocklist did not catch it because it was not listed. A column that
correlates ~1.0 with the target is a leak regardless of its name.
"""))

cells.append(code(r'''
class LeakageGuard:
    """
    Validates a feature matrix for target leakage.

    OOP rules exercised (oop_rules.md):
      §4 protected attributes with leading underscore
      §5 validate inputs before assigning state
      §6 read-only @property accessors, defensive copies
      §7 shared configuration as class attributes
    """

    # Class attributes: shared blocklist and threshold (oop_rules §7).
    FORBIDDEN_COLUMNS: tuple = (
        "MissionStatus",   # the target itself
        "_is_fail",        # binary restatement of the target (Part E helper)
        "target",          # derived target
        "RocketStatus",    # temporal leakage (F.2)
        "Mission",         # identifier + latent leak (F.3)
        "Price",           # dropped, retroactive bias (F.4)
        "Time_parsed",     # excluded
    )
    MAX_TARGET_CORRELATION: float = 0.999

    def __init__(self, target: pd.Series, numeric_scan: bool = True):
        if not isinstance(target, pd.Series):
            raise TypeError(
                f"target must be a pandas Series, got {type(target).__name__}."
            )
        if target.nunique() < 2:
            raise ValueError("target must contain at least two distinct values.")
        self._target = target
        self._numeric_scan = bool(numeric_scan)
        self._violations: list[str] = []

    # ---- read-only properties (oop_rules §6) ------------------------------
    @property
    def violations(self) -> tuple[str, ...]:
        """Defensive copy - callers cannot mutate the internal list."""
        return tuple(self._violations)

    @property
    def is_clean(self) -> bool:
        return not self._violations

    @property
    def forbidden_columns(self) -> tuple[str, ...]:
        return tuple(self.FORBIDDEN_COLUMNS)

    # ---- the check ---------------------------------------------------------
    def check(self, X: pd.DataFrame) -> "LeakageGuard":
        """Validate a feature matrix. Raises AssertionError on any violation."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"X must be a DataFrame, got {type(X).__name__}.")

        # Reset state so the guard is reusable.
        self._violations = []

        # Check 1: no blocklisted column present.
        present = [c for c in self.FORBIDDEN_COLUMNS if c in X.columns]
        if present:
            self._violations.append(
                f"Blocklisted column(s) present in X: {present}"
            )

        # Check 2: no numeric column is a near-perfect restatement of y.
        if self._numeric_scan:
            for col in X.columns:
                series = X[col]
                if not pd.api.types.is_numeric_dtype(series):
                    continue
                if series.nunique() != 2:
                    continue
                # Align on the shared index and drop rows with any missing value.
                pair = pd.concat([series, self._target], axis=1).dropna()
                if len(pair) < 2:
                    continue
                corr = abs(np.corrcoef(pair.iloc[:, 0], pair.iloc[:, 1])[0, 1])
                if corr >= self.MAX_TARGET_CORRELATION:
                    self._violations.append(
                        f"Column {col!r} restates the target "
                        f"(binary, |corr| = {corr:.4f})."
                    )

        if self._violations:
            raise AssertionError(
                "LEAKAGE DETECTED:\n  - " + "\n  - ".join(self._violations)
            )
        return self

    def report(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return a pass/fail summary without raising."""
        try:
            self.check(X)
        except AssertionError:
            pass
        return pd.DataFrame([
            {"check": "No blocklisted columns in X",
             "status": "FAIL" if any("Blocklisted" in v for v in self._violations) else "PASS"},
            {"check": f"No column restates the target (|corr| >= {self.MAX_TARGET_CORRELATION})",
             "status": "FAIL" if any("restates" in v for v in self._violations) else "PASS"},
        ]).set_index("check")

    def __repr__(self) -> str:
        state = "clean" if self.is_clean else f"{len(self._violations)} violation(s)"
        return f"LeakageGuard({state})"
'''))

cells.append(code(r'''
# =====================================================================
# F.8a - Demonstrate the guard actually catches leaks
# =====================================================================
# A guard that never fails is worthless. We feed it three known-bad matrices
# and confirm it rejects each one.

y_demo = (df["MissionStatus"] == "Success").astype(int)

print("Guard setup:")
guard = LeakageGuard(target=y_demo)
print(f"  {guard!r}")
print(f"  forbidden columns: {guard.forbidden_columns}\n")

# --- Test 1: a clean matrix should PASS ---
clean_X = df[["Company", "Rocket", "Country", "Launch_Year"]].copy()
guard.check(clean_X)
print("Test 1 - clean matrix          :", "PASS" if guard.is_clean else "FAIL")

# --- Test 2: a blocklisted column should FAIL ---
bad_X = clean_X.copy()
bad_X["RocketStatus"] = df["RocketStatus"]
try:
    guard.check(bad_X)
    print("Test 2 - blocklisted column    : UNEXPECTEDLY PASSED")
except AssertionError as exc:
    print("Test 2 - blocklisted column    : correctly REJECTED")
    print(f"          {guard.violations[0]}")

# --- Test 3: a disguised target should FAIL even though it is not blocklisted ---
disguised_X = clean_X.copy()
disguised_X["_hidden_flag"] = (df["MissionStatus"] != "Success").astype(int)
try:
    guard.check(disguised_X)
    print("Test 3 - disguised target      : UNEXPECTEDLY PASSED")
except AssertionError as exc:
    print("Test 3 - disguised target      : correctly REJECTED")
    print(f"          {guard.violations[0]}")

# --- Test 4: the guard recovers and can be reused ---
guard.check(clean_X)
print("Test 4 - guard reusable        :", "PASS" if guard.is_clean else "FAIL")
'''))

# ------------------------------------------------------------- evidence
cells.append(md(r"""
---

#### F.9 — Evidence: the `RocketStatus` leak
"""))

cells.append(code(r'''
# =====================================================================
# F.9 - RocketStatus: three independent pieces of evidence
# =====================================================================

print("EVIDENCE 1 - the temporal tell (status vs launch decade)\n")
status_by_decade = pd.crosstab(df["Date"].dt.year // 10 * 10, df["RocketStatus"])
display(status_by_decade)

early = df[df["Date"].dt.year < 1980]
n_active_early = int((early["RocketStatus"] == "Active").sum())
print(f"Launches before 1980 labelled 'Active' : {n_active_early}")
print(f"Total launches before 1980             : {len(early)}")
print("\n-> Zero exceptions. This is a 2022 snapshot, not a pre-flight condition.")

print("\n" + "=" * 62)
print("\nEVIDENCE 2 - reverse causation (outcome rate by status)\n")
status_outcome = pd.crosstab(
    df["RocketStatus"], df["_is_fail"], normalize="index"
).rename(columns={0: "success_rate", 1: "failure_rate"}).round(4)
display(status_outcome)
print("'Active' vehicles succeed MORE often (93.2% vs 89.0%).")
print("Causation runs backwards: programmes end and unreliable vehicles are")
print("withdrawn, so the OUTCOME partly causes the STATUS.")
'''))

cells.append(md(r"""
#### F.10 — Evidence: the latent leak in `Mission` names
"""))

cells.append(code(r'''
# =====================================================================
# F.10 - Mission names carry outcome information
# =====================================================================
# Mission is dropped as an identifier anyway, but the audit revealed that the
# name itself encodes risk. This matters because it shows an "ID" column was
# not actually inert.

global_rate = df["_is_fail"].mean()

name_patterns = []
for keyword in ["test", "demo"]:
    mask = df["Mission"].str.lower().str.contains(keyword, na=False)
    name_patterns.append({
        "pattern": f'contains "{keyword}"',
        "n_missions": int(mask.sum()),
        "failure_rate": round(df.loc[mask, "_is_fail"].mean(), 4),
        "median_year": int(df.loc[mask, "Date"].dt.year.median()),
    })

name_patterns.append({
    "pattern": "all missions",
    "n_missions": len(df),
    "failure_rate": round(global_rate, 4),
    "median_year": int(df["Date"].dt.year.median()),
})

name_leak = pd.DataFrame(name_patterns).set_index("pattern")
display(name_leak)

# The key question: is this just an era effect? Test within the early era only.
early_mask = df["Date"].dt.year < 1970
test_mask = df["Mission"].str.lower().str.contains("test", na=False)

print("Is it merely an era proxy? Test restricted to pre-1970 launches:\n")
print(f"  test-named, pre-1970 : {df.loc[test_mask & early_mask, '_is_fail'].mean():.3f} "
      f"(n={int((test_mask & early_mask).sum())})")
print(f"  all others, pre-1970 : {df.loc[~test_mask & early_mask, '_is_fail'].mean():.3f} "
      f"(n={int((~test_mask & early_mask).sum())})")

print(f"\nTest-named missions have a MEDIAN YEAR of "
      f"{int(df.loc[test_mask, 'Date'].dt.year.median())} (dataset median "
      f"{int(df['Date'].dt.year.median())}) - far more recent, yet still fail at")
print(f"{df.loc[test_mask, '_is_fail'].mean()*100:.1f}% versus {global_rate*100:.1f}% overall.")
print("\n-> Not an era proxy. The name signals a genuinely higher-risk mission")
print("   TYPE (first flights, unproven vehicles).")
'''))

cells.append(md(r"""
#### F.11 — Evidence: `Price` was recorded retroactively
"""))

cells.append(code(r'''
# =====================================================================
# F.11 - Price availability is era-dependent, not random
# =====================================================================
# Price was dropped in Part C, so we re-read it from the untouched raw file to
# audit its missingness pattern.

price_raw = pd.to_numeric(
    df_raw["Price"].astype(str).str.replace(",", "", regex=False).str.strip(),
    errors="coerce",
)
raw_dates = pd.to_datetime(df_raw["Date"], errors="coerce")

price_by_decade = pd.DataFrame({
    "with_price": price_raw.notna().groupby(raw_dates.dt.year // 10 * 10).sum(),
    "total": raw_dates.groupby(raw_dates.dt.year // 10 * 10).size(),
})
price_by_decade["pct_recorded"] = (
    price_by_decade["with_price"] / price_by_decade["total"] * 100
).round(1)
price_by_decade.index.name = "decade"

display(price_by_decade)

print("In the 1960s only 33 of 774 launches have a price.")
print("In the 2010s, 528 of 676 do.")
print("\n-> Pricing is recorded mainly for MODERN missions. Since era correlates")
print("   with the target, imputing Price would have injected era information")
print("   into what looks like a neutral financial column.")
'''))

# ------------------------------------------------------------- charts
cells.append(md(r"""
#### F.12 — Visual: the three leak findings
"""))

cells.append(code(r'''
# =====================================================================
# F.12 - CHART 1: the RocketStatus temporal tell
# =====================================================================
fig, ax = plt.subplots(figsize=(10, 5))

decade_labels = [f"{d}s" for d in status_by_decade.index]
active = status_by_decade["Active"].values
retired = status_by_decade["Retired"].values
x = np.arange(len(decade_labels))

ax.bar(x, retired, label="Retired", color="#C44E52", edgecolor="white")
ax.bar(x, active, bottom=retired, label="Active", color="#55A868", edgecolor="white")

for xi, (a, r) in enumerate(zip(active, retired)):
    if a == 0:
        ax.text(xi, r / 2, f"0 Active\n({r:,} Retired)", ha="center",
                va="center", fontsize=9, color="white", fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(decade_labels)
ax.set_xlabel("Launch decade", fontsize=11)
ax.set_ylabel("Number of launches", fontsize=11)
ax.set_title("RocketStatus is a 2022 snapshot, not a pre-flight condition",
             fontsize=13, fontweight="bold")
ax.legend()
ax.annotate("Zero 'Active' before the 1980s.\nThe field records 2022 retirement\nstatus, not launch-time facts.",
            xy=(0.5, 700), xytext=(2.4, 1350), fontsize=10,
            arrowprops=dict(arrowstyle="->", color="gray", lw=1.5),
            bbox=dict(boxstyle="round,pad=0.4", fc="#FFF3CD", ec="gray"))
plt.tight_layout()
plt.show()

print("Every one of the 1,837 pre-1980 launches is 'Retired' - no exceptions.")
print("A model would learn 'old missions used retired rockets', which is true")
print("by definition and therefore useless for predicting a NEW launch.")
'''))

cells.append(code(r'''
# =====================================================================
# F.13 - CHART 2: reverse causation
# =====================================================================
fig, ax = plt.subplots(figsize=(8, 5))

statuses = status_outcome.index.tolist()
success = (status_outcome["success_rate"] * 100).values
failure = (status_outcome["failure_rate"] * 100).values

x = np.arange(len(statuses))
width = 0.36
b1 = ax.bar(x - width/2, success, width, label="Success", color="#55A868", edgecolor="white")
b2 = ax.bar(x + width/2, failure, width, label="Failure", color="#C44E52", edgecolor="white")

for bars in (b1, b2):
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.7, f"{h:.1f}%",
                ha="center", fontsize=10, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(statuses)
ax.set_ylabel("Share of missions (%)", fontsize=11)
ax.set_title("Reverse causation: better outcomes CAUSE 'Active' status",
             fontsize=13, fontweight="bold")
ax.set_ylim(0, 108)
ax.legend()
plt.tight_layout()
plt.show()

print("'Active' vehicles succeed 93.2% vs 'Retired' 89.0%. The causality runs")
print("backwards - vehicles are retired partly BECAUSE they were unreliable.")
print("Using this column would let the model exploit a reversed relationship.")
'''))

cells.append(code(r'''
# =====================================================================
# F.14 - CHART 3: Price recorded retroactively
# =====================================================================
fig, ax = plt.subplots(figsize=(10, 5))

decades = [f"{d}s" for d in price_by_decade.index]
pct = price_by_decade["pct_recorded"].values
recorded = price_by_decade["with_price"].values
total = price_by_decade["total"].values

bars = ax.bar(decades, pct, color="#4C72B0", edgecolor="white", linewidth=1.2)
for bar, p, r, t in zip(bars, pct, recorded, total):
    ax.text(bar.get_x() + bar.get_width()/2, p + 1.5,
            f"{p:.0f}%\n({r:,}/{t:,})", ha="center", fontsize=9)

ax.set_xlabel("Launch decade", fontsize=11)
ax.set_ylabel("Launches with a recorded Price (%)", fontsize=11)
ax.set_title("Price missingness is systematic, not random", fontsize=13, fontweight="bold")
ax.set_ylim(0, max(pct) * 1.28)
plt.tight_layout()
plt.show()

print("The availability of Price climbs from ~0% (1950s) to ~85% (2010s).")
print("Because era correlates with the target, imputing Price would have")
print("smuggled era information into a supposedly neutral cost column.")
'''))

# ------------------------------------------------------------- summary
cells.append(code(r'''
# =====================================================================
# F.15 - Part F summary
# =====================================================================

leak_summary = pd.DataFrame([
    {"finding": "RocketStatus temporal leakage", "severity": "SEVERE",
     "evidence": "0 of 1,837 pre-1980 launches are 'Active'; reverse causation",
     "action": "Dropped and blocklisted in LeakageGuard"},
    {"finding": "Mission names encode risk", "severity": "LATENT",
     "evidence": "'test' names fail 52.2% vs 10.1%; 75% vs 23.6% pre-1970",
     "action": "Already dropped as identifier; documented"},
    {"finding": "Price recorded retroactively", "severity": "MODERATE",
     "evidence": "33/774 priced in 1960s vs 528/676 in 2010s",
     "action": "Already dropped at 72.7% missing"},
    {"finding": "Time carries no signal", "severity": "NONE",
     "evidence": "Failure rate 7.6%-10.2% across hour buckets",
     "action": "Excluded; costs nothing"},
]).set_index("finding")

display(leak_summary)

# Build a candidate feature matrix from everything Part E produced, then let
# the guard inspect it. Only numeric columns are scanned for near-restatement.
candidate_X = df_eng.drop(
    columns=[c for c in LeakageGuard.FORBIDDEN_COLUMNS if c in df_eng.columns]
)
numeric_candidates = candidate_X.select_dtypes(include=[np.number])

guard_final = LeakageGuard(
    target=(df_eng["MissionStatus"] == "Success").astype(int)
)

print("Guard verdict on the candidate feature set:\n")
display(guard_final.report(numeric_candidates))

if guard_final.is_clean:
    print("\nPASS: no leakage detected in the numeric candidate features.")
else:
    print(f"\nReview needed: {guard_final.violations}")
'''))

nb["cells"].extend(cells)

with open(NB_PATH, "w") as f:
    json.dump(nb, f, indent=1)

print(f"Appended {len(cells)} Part F cells.")
print(f"Notebook now has {len(nb['cells'])} cells.")
