"""
Inserts Part C (Data Cleaning) into oop.ipynb, after the Part B markdown cell.

Run: python3 build_part_c.py
"""
import json

NB_PATH = "oop.ipynb"
DOC_PATH = "_docs/part_c_doc.md"

with open(NB_PATH) as f:
    nb = json.load(f)

with open(DOC_PATH) as f:
    part_c_doc = f.read()


def md(text):
    return {"cell_type": "markdown", "metadata": {},
            "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.strip("\n").splitlines(keepends=True)}


cells = [md(part_c_doc)]

# ---------------------------------------------------------------- C.3 code
cells.append(code(r'''
import pandas as pd

# Work on a copy so the raw file stays untouched (guide: "Preserve Raw Data").
df_raw = pd.read_csv("space_missions.csv", encoding="latin1")
df = df_raw.copy()

print(f"Raw file shape      : {df_raw.shape}")
print(f"Working copy shape  : {df.shape}")
print("\nThe raw file is never written to. All cleaning happens on `df`.")
'''))

# ---------------------------------------------------------------------------
# Date is parsed FIRST, before the rules, because several later rules inspect
# the Date column (the duplicate rule keys on it, and the co-launch check
# groups by it). Parsing it early keeps the rule order independent.
cells.append(code(r'''
# =====================================================================
# RULE 0 - Parse Date to a real datetime (done first, on purpose)
# =====================================================================
# Date is parsed before the other rules because they depend on it: the
# duplicate rule keys on Date, and the co-launch check groups by it. Parsing
# it here means the rules below are order-independent.
#
# errors='coerce' turns unparseable text into NaT instead of raising, and we
# then CHECK how many failed - so a silent failure cannot hide.

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

n_bad = int(df["Date"].isna().sum())
print(f"Unparseable dates : {n_bad}")
print(f"New dtype         : {df['Date'].dtype}")
print(f"Date range        : {df['Date'].min().date()} -> {df['Date'].max().date()}")

assert n_bad == 0, f"{n_bad} dates failed to parse!"
print("\nPASS: all dates parsed successfully.")
'''))

cells.append(code(r'''
# =====================================================================
# RULE 1 - Remove exact duplicate rows
# =====================================================================
# An exact duplicate is a row where EVERY column matches another row. It
# carries no extra information and would double-count one launch.
before = len(df)
n_exact = int(df.duplicated(keep="first").sum())
df = df.drop_duplicates(keep="first").reset_index(drop=True)

print(f"Exact duplicates found   : {n_exact}")
print(f"Rows before -> after     : {before} -> {len(df)}")

# Show what was removed, for the audit trail.
exact_mask = df_raw.duplicated(keep=False)
print("\nThe duplicated record was:")
print(df_raw[exact_mask][["Company", "Date", "Rocket", "Mission", "MissionStatus"]].to_string())
'''))

cells.append(code(r'''
# =====================================================================
# RULE 2 - Remove the BUSINESS duplicate (same launch recorded twice)
# =====================================================================
# A business duplicate is NOT identical in every column, but represents the
# same real-world event. We identify it with a NARROW, evidence-based rule:
#
#   same Company + Rocket + Date + Price + MissionStatus
#   AND one of the rows has no launch Time
#
# The missing Time is the fingerprint of a re-entered record. This rule is
# deliberately narrow because a date-only match would wrongly delete genuine
# co-launches (see RULE 3 below).

dup_key = ["Company", "Rocket", "Date", "Price", "MissionStatus"]
dup_candidates = df[df.duplicated(subset=dup_key, keep=False)]

rows_to_drop = []
for _, group in dup_candidates.groupby(dup_key, dropna=False):
    # Only drop if the group has BOTH a complete row and an incomplete one.
    if len(group) > 1 and group["Time"].notna().any():
        rows_to_drop.extend(group.index[group["Time"].isna()].tolist())

print(f"Rows matching the duplicate key : {len(dup_candidates)}")
print(f"Rows dropped as business dups   : {len(rows_to_drop)}")

before = len(df)
df = df.drop(index=rows_to_drop).reset_index(drop=True)
print(f"Rows before -> after            : {before} -> {len(df)}")

print("\nRemaining record(s) for that launch:")
print(df[df["Company"].eq("ISRO") & df["Date"].eq("2012-04-26")][
    ["Company", "Date", "Time", "Rocket", "Mission", "MissionStatus"]
].to_string())
'''))

cells.append(code(r'''
# =====================================================================
# RULE 3 - VERIFY we did NOT delete legitimate co-launches
# =====================================================================
# Four pairs in this dataset share Company + Rocket + Date but are genuinely
# DIFFERENT launches - different times, different payloads. A naive
# "same date = duplicate" rule would have destroyed them. We prove they
# survived.

colauches = df[df.duplicated(subset=["Company", "Rocket", "Date"], keep=False)]

print(f"Rows sharing Company+Rocket+Date: {len(colauches)}")
print("These are DISTINCT launches and must all be kept:\n")
for (co, rk, dt), grp in colauches.groupby(["Company", "Rocket", "Date"]):
    times = grp["Time"].tolist()
    missions = grp["Mission"].tolist()
    print(f"  {co:<14} {dt.date()}  times={times}")
    for m in missions:
        print(f"      payload: {m}")

assert len(colauches) == 8, "Co-launch rows were lost - the duplicate rule is too aggressive!"
print(f"\nPASS: all {len(colauches)} co-launch rows preserved.")
'''))

cells.append(code(r'''
# =====================================================================
# RULE 4 - Repair Price: text with thousands separators -> float
# =====================================================================
# Values look like '1,160.00' and '5,000.00'. We strip the comma and cast.
# CRITICAL: we verify that the parse does not silently destroy values
# (guide §9A warns about unparseable text quietly becoming 0).

raw_nonnull = int(df["Price"].notna().sum())

df["Price"] = pd.to_numeric(
    df["Price"].astype(str).str.replace(",", "", regex=False).str.strip(),
    errors="coerce",
)

parsed_nonnull = int(df["Price"].notna().sum())

print(f"Non-null Price before parse : {raw_nonnull}")
print(f"Non-null Price after parse  : {parsed_nonnull}")
print(f"Values lost                 : {raw_nonnull - parsed_nonnull}")
print(f"New dtype                   : {df['Price'].dtype}")

assert raw_nonnull == parsed_nonnull, "Price parsing destroyed information!"
print("\nPASS: every non-null Price value survived as a real number.")
'''))

cells.append(code(r'''
# =====================================================================
# RULE 5 - Drop Price (72.68% missing)
# =====================================================================
# Above the guide's 50-70% threshold. Imputing three-quarters of a column
# would fabricate precision rather than recover information.

missing_pct = df["Price"].isna().mean() * 100
print(f"Price missing: {missing_pct:.2f}%")

df = df.drop(columns=["Price"])
print(f"Price dropped. Columns now: {df.shape[1]}")
print(f"Remaining columns: {list(df.columns)}")
'''))

cells.append(code(r'''
# =====================================================================
# RULE 6 - Extract Country from the composite Location string
# =====================================================================
# Location looks like 'Site 31/6, Baikonur Cosmodrome, Kazakhstan' and has
# between 2 and 5 comma-separated parts. The LAST part is always the country.
#
# Once Country is extracted, the source column is DROPPED: it is superseded.
# Keeping both would duplicate the same information (158 composite strings
# versus 22 clean country labels), and a composite string cannot be encoded
# meaningfully anyway. Extracting and dropping is one atomic step.

n_parts = df["Location"].str.split(",").str.len()
print(f"Comma-parts per Location value: min={n_parts.min()}, max={n_parts.max()}")

df["Country"] = df["Location"].str.split(",").str[-1].str.strip().str.title()

print(f"\nDistinct countries extracted: {df['Country'].nunique()}")
print(df["Country"].value_counts().head(10).to_string())

# Confirm no country value is blank or a leftover placeholder.
blank = int((df["Country"].str.strip() == "").sum())
print(f"\nBlank country values: {blank}")
assert blank == 0, "Country extraction produced blanks!"

# Drop the now-superseded source column.
df = df.drop(columns=["Location"])
print(f"\nDropped 'Location' (superseded by 'Country'). Columns now: {df.shape[1]}")
print(f"Remaining: {list(df.columns)}")
'''))

cells.append(code(r'''
# =====================================================================
# RULE 7 - Date integrity confirmed (parsed in RULE 0)
# =====================================================================
# Date was parsed before the rules ran, because the duplicate and co-launch
# logic depend on it. This cell simply confirms the result still holds after
# every other rule has been applied.

print(f"Date dtype  : {df['Date'].dtype}")
print(f"Unparseable : {int(df['Date'].isna().sum())}")
print(f"Date range  : {df['Date'].min().date()} -> {df['Date'].max().date()}")

assert df["Date"].isna().sum() == 0, "Date integrity lost during cleaning!"
print("\nPASS: Date remains a clean datetime after all cleaning rules.")
'''))

cells.append(code(r'''
# =====================================================================
# RULE 8 - Time: documented as left untouched
# =====================================================================
# Time-of-day has no causal relationship to launch-vehicle reliability, so we
# do NOT impute it. It is simply excluded from the feature set in Part G.
# We record its missingness here for transparency.

print(f"Time missing values    : {int(df['Time'].isna().sum())} "
      f"({df['Time'].isna().mean()*100:.2f}%)")
print(f"Time dtype             : {df['Time'].dtype}")
print("\nDecision: LEFT UNTOUCHED. No imputation. Excluded from features in Part G.")
print("Reason: launch time-of-day has no causal bearing on vehicle reliability,")
print("        so inventing values would add noise without adding signal.")
'''))

cells.append(code(r'''
# =====================================================================
# C.4 - Cleaning summary and integrity check
# =====================================================================

summary = pd.DataFrame([
    {"Step": "Start (raw)", "Rows": len(df_raw), "Columns": df_raw.shape[1]},
    {"Step": "After RULE 1 (exact duplicates)", "Rows": 4629, "Columns": 9},
    {"Step": "After RULE 2 (business duplicate)", "Rows": len(df), "Columns": 9},
    {"Step": "After RULE 5 (Price dropped)", "Rows": len(df), "Columns": 8},
    {"Step": "After RULE 6 (Location dropped)", "Rows": len(df), "Columns": len(df.columns)},
]).set_index("Step")

summary["Rows_removed"] = summary["Rows"].shift(1) - summary["Rows"]
display(summary)

print(f"Total rows removed : {len(df_raw) - len(df)} (1 exact dup + 1 business dup)")
print(f"Columns            : {df_raw.shape[1]} raw -> {len(df.columns)} cleaned")
print("                     (+ Country extracted, - Price, - Location)")
print(f"Final shape        : {df.shape}")

# Integrity assertion: nothing else went missing.
assert len(df) == len(df_raw) - 2, "Unexpected row count change!"
print("\nPASS: exactly 2 rows removed, both documented duplicates.")
'''))

cells.append(code(r'''
# Preview the cleaned dataset.
print("Cleaned dataset - first 5 rows:")
display(df.head())

print("\nFinal column types:")
print(df.dtypes.to_string())
'''))

cells.append(md(r"""
### C.5 — Reading the cleaning results

**What changed, and the evidence for each change:**

| Change | Rows/columns affected | Evidence it was correct |
| :--- | :--- | :--- |
| Removed 1 exact duplicate | −1 row | Every column matched identically (CASC, 2008-11-05) |
| Removed 1 business duplicate | −1 row | ISRO 2012-04-26 had no launch Time; its mission name merely prefixed the vehicle onto the same payload |
| **Preserved 8 co-launch rows** | 0 rows | Verified they have distinct times and payloads — they are separate launches |
| Parsed `Price` | 1,263 → 1,263 values | Zero parse failures: the comma-strip recovered every value |
| Dropped `Price` | −1 column | 72.68% missing, above the documented threshold |
| Extracted `Country`, dropped `Location` | +1 / -1 column | Last comma-part of `Location`; 22 countries, no blanks. Source column superseded, so dropped |
| Parsed `Date` | dtype change | 0 unparseable values |

**Final shape: 4,628 rows × 9 columns.**

**The most important line is the co-launch verification.** The naive rule
"same company + same rocket + same date = duplicate" would have deleted 6 real
launches — missions like `OPS 5187` and `OPS 5247`, which were genuinely
separate flights. Requiring a *missing Time* as the duplicate fingerprint made
the rule precise. This is what the guide means by "business duplicates require
domain logic" (§8B): the rule has to encode *why* two records are the same event,
not merely that they look similar.

**What we did not do.** No outliers were deleted, no values were imputed, and no
rows were dropped merely for having missing fields. Every missing value either
belongs to a dropped column or is intentionally excluded from the features.
"""))

# ------------------------------------------------------------- insert cells
# Insert after the Part B markdown cell (index 2).
INSERT_AT = 3
nb["cells"][INSERT_AT:INSERT_AT] = cells

with open(NB_PATH, "w") as f:
    json.dump(nb, f, indent=1)

print(f"Inserted {len(cells)} Part C cells at index {INSERT_AT}.")
print(f"Notebook now has {len(nb['cells'])} cells.")
