### Part C — Data Cleaning

Before touching any value, here is the full inventory of quality problems found in
Part A and what we decided to do about each one. Every decision is stated up
front so the code below is a record of *applying* rules, not a place where
decisions get made silently.

---

#### C.1 — Cleaning decision register

| # | Problem | Evidence | Decision | Reason |
| :-: | :--- | :--- | :--- | :--- |
| 1 | **Exact duplicate row** | 1 row (index 3530/3531: CASC, Long March 2D, 2008-11-05). Every column identical. | Drop one copy. | A fully identical record is redundant and would double-count one launch in the training signal. |
| 2 | **Business duplicate** | ISRO 2012-04-26, PSLV-XL: two rows share date, rocket, price (31) and outcome. The second row has **no launch time** and its mission name `"PSLV- C19 RISAT-1"` merely prefixes the vehicle onto the same payload `RISAT-1`. | Drop the time-less row. | The missing time is a recording gap in a re-entry of the same launch. Kept the complete record. |
| 3 | **Legitimate co-launches (NOT duplicates)** | 4 pairs share Company + Rocket + Date but have genuinely different times and payloads — e.g. ExPace 2019-12-07 flew at 02:55 and 08:52; US Air Force 1968-08-06 launched `OPS 5187` and `OPS 5247` separately. | **Keep all of them.** | These are distinct launches, sometimes two payloads on one date. Deleting them on a date-only match would destroy real observations. |
| 4 | **`Price` stored as text** | Values like `'1,160.00'`, `'5,000.00'` with thousands separators. | Strip non-numeric characters, cast to `float`. | Text cannot be used in arithmetic. Verified that all 1,265 non-null values parse successfully, so nothing is silently lost. |
| 5 | **`Price` missing 72.68%** | 3,365 of 4,630 rows blank. | Drop the column. | Above the guide's 50–70% threshold. Imputing three-quarters of a column would fabricate precision. |
| 6 | **`Location` is a composite string** | 158 unique strings with 2–5 comma-separated parts, e.g. `'Site 31/6, Baikonur Cosmodrome, Kazakhstan'`. | Split on `,`; take the last part as `Country`, then **drop `Location`**. | A composite string cannot express geography as a feature. The last part is always the country, so keeping the source would duplicate the same information across 158 raw strings versus 22 clean labels. |
| 7 | **All columns are `object`** | Every field loaded as text. | Cast `Date` to `datetime64`, `Price` to `float64`. | String dates cannot be sorted or differenced; string numbers cannot be averaged. |
| 8 | **`Time` missing 2.74%** | 127 rows blank. | Leave as-is; exclude from features. | Time-of-day has no causal relationship to launch-vehicle reliability, so no imputation is warranted. |
| 9 | **`MissionStatus` has 4 levels, two very rare** | `Prelaunch Failure` has only 4 rows. | Collapse to binary: `Success` vs `Not-Success`. | A 4-row class cannot be stratified or learned reliably. |

---

#### C.2 — What we are deliberately NOT doing

Just as important as the changes we make are the ones we refuse to make:

- **We are not deleting outliers.** Extreme values in this dataset are valid
  domain extremes, not errors. The guide's §8C rule of thumb is explicit: never
  delete a point merely because it complicates a distribution.
- **We are not overwriting the raw file.** All work happens on in-memory copies,
  so the original `space_missions.csv` stays byte-for-byte unchanged.
- **We are not dropping rows just because a field is missing.** Only the one
  genuinely duplicated re-entry is removed.
- **We are not imputing values anywhere.** Every missing value is either left
  alone or belongs to a dropped column. This avoids inventing data.

---

#### C.3 — Applying the rules

The cells below apply the register in order. Each step prints what it changed, so
the log in the output is the audit trail.

**One ordering note:** `Date` is parsed to a datetime *first*, before the other
rules. The duplicate rule keys on the date, and the co-launch verification groups
by it, so parsing it early keeps the rules order-independent. Rule 7 then simply
re-confirms that the date stayed clean after everything else ran.