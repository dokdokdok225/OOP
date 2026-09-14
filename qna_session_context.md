# Mangui-ob Notebook Q&A Session Context

## Purpose
This document records the context and beginner-friendly explanations used in the Q&A session for the revised Mangui-ob notebook. The refined notebook is `oop_refinement.ipynb`; the original `oop.ipynb` must remain untouched.

## Notebook scope
- Parts A and B are retained.
- Part C performs cleaning only: duplicates, missing values, types, invalids/outliers, date/price parsing, country extraction, and raw-data protection. It does not perform model preparation.
- Part D explains/plans feature preparation: `Launch_Year`, `Month_Sin`, `Month_Cos`, standardization versus normalization, one-hot encoding, and frequency encoding. Application occurs later.
- Part E creates causal company-history features using earlier missions only: `Company_Prior_Launches` and `Company_Prior_Failure_Rate`. It explains leakage and OOP simply and uses diagnostic comparisons only.
- Part F checks leakage involving `RocketStatus`, `Mission`, and `Price` with `LeakageGuard` and graphs.
- Part G separates inputs `X` from binary target `y`, with checks and graphs.
- Part H performs a stratified 80/20 split, fits preprocessing on `X_train` only, transforms `X_test`, validates the result, and saves `X_train.npy`, `X_test.npy`, `y_train.npy`, `y_test.npy`, and `feature_names.txt`.
- Part I is markdown-only summary tables. It states that no final predictive model has yet been trained or reported.

## Part A: dataset inspection
- Dataset: 4,630 rows and 9 initial columns.
- Initial columns are loaded as `object` text types.
- One exact duplicate was identified (indices 3530/3531: CASC, Long March 2D, 2008-11-05).
- `Price` has 3,365 missing values (72.68%).
- `Time` has 127 missing values (2.74%).
- `Mission` has 4,556 unique names across 4,630 rows (98.4% unique), so it behaves like an identifier and is not a useful general predictive feature.
- `Rocket` has 370 distinct rocket families; many occur only once or twice, so it has high cardinality.
- Candidate target: `MissionStatus`, later collapsed into `Success` (4,162; 89.89%) and `Not-Success` (468; 10.11%).
- Candidate inputs include company, country, launch year, rocket, and later company-history features.

### Cardinality
Cardinality means the number of unique values in a column. `Mission` has very high cardinality because almost every name is unique. `Rocket` also has high cardinality, but repeated rocket families can contain useful information.

### Identifiers and outliers
An identifier distinguishes records but does not generalize well. `Mission` is excluded because most names occur once and future missions will likely have unseen names. An outlier is an unusually high, low, or rare value; unusual does not mean incorrect. Valid historic launches, rare rockets, and extreme prices are retained unless confirmed as errors.

Examples of unusual values in the CSV include early launch dates such as 1957-10-04, high prices such as $5,000 million for Energiya/Buran and Energiya/Polyus, low prices around $2.5 million for Astra Rocket 3, rare rockets such as Juno II, and companies with very different launch counts (RVSN USSR appears about 1,777 times while some companies appear once).

## Part B: ML problem
Regression predicts a continuous number. `MissionStatus` is categorical, so this is classification, not regression. `Price` could be numeric but is too incomplete to be a reliable target.

The planned workflow has two stages:
1. **K-Means clustering:** unsupervised exploration that groups similar missions using numeric features such as launch era and company experience. Numeric inputs should be standardized because K-Means uses distances. Several values of K should be compared. Clustering does not see the target and is not the final predictor.
2. **Decision Tree classification:** supervised prediction of `Success` versus `Not-Success`. A tree learns readable if/else rules and can model interactions and non-linear relationships.

K-Means does not automatically turn into a decision tree. Clustering can reveal useful structure and motivate features; the decision tree then learns from known labels. The refined notebook prepares the workflow but does not report a final trained model.

### Overfitting
About 4,628 cleaned rows is reasonable for a small project, but overfitting can still occur due to high-cardinality inputs, class imbalance (89.89% success), and overly deep trees. Protections include excluding `Mission`, frequency encoding `Rocket`, stratified splitting, fitting preprocessing on training data only, leakage checks, and later limiting tree depth/minimum leaf sizes and using cross-validation. Accuracy alone is insufficient; use precision, recall, F1, and a confusion matrix. A much higher training score than test score indicates overfitting.

## Part C: cleaning and reasoning
- Work on a copy so the raw file remains unchanged.
- Remove the confirmed exact duplicate and the documented business duplicate (the same launch re-entered with incomplete information).
- Keep legitimate co-launches: same date/company/rocket can still represent separate payloads and launch times.
- Parse `Date` as datetime so it can be sorted and used to derive year/month.
- Parse `Price` text such as `1,160.00` into numbers, then drop it because 72.68% missingness would make imputation unreliable and fabricate precision.
- Split composite `Location` and take its final component as `Country`.
- Leave missing `Time` values alone rather than inventing times; `Time` is not used as an input.
- Keep original `MissionStatus` labels during cleaning; binary grouping belongs to modeling.
- Do not delete valid outliers or perform scaling/encoding in the cleaning stage.

Country gives a broad, easier-to-encode geographic feature and supports country-level comparisons, but it loses exact launch-site detail. The notebook extracts `Country` and drops composite `Location`; preserving a separate site feature would be an alternative if facility-level analysis were required.

## Part D: feature preparation
- `Launch_Year` captures technology era, infrastructure, and maturity.
- Month is cyclical: December and January are neighbours, although raw numbers 12 and 1 look far apart. `Month_Sin` and `Month_Cos` place months around a circle and preserve this relationship.
- **Standardization:** subtract the mean and divide by standard deviation; values are centered around 0 and generally comparable in spread. It is used with `StandardScaler` for numeric features such as launch year, prior launches, and failure rate, especially for distance-based methods.
- **Normalization:** commonly rescales values to a fixed range such as 0–1. It is different from standardization. The notebook chooses standardization.
- **One-hot encoding:** turns categories into separate 0/1 columns; e.g., `Company_NASA`, `Company_CASC`, and `Company_Arianespace`. This avoids incorrectly implying that category labels have numeric order. It is used for manageable `Company` and `Country` categories.
- **Frequency encoding:** replaces each `Rocket` name with its training-set frequency/count or proportion. It is not one-hot encoding. It avoids hundreds of rocket columns, though it represents how common a rocket is rather than its full identity. All preprocessing rules are fitted on training data only.

## Part E: company-history features
- `Company_Prior_Launches` counts the company’s launches before the current mission.
- `Company_Prior_Failure_Rate` is prior failures divided by prior launches. It measures historical reliability.
- “Prior” is essential: the current mission outcome must not be included in its own history, or the feature would leak the answer.
- OOP is explained as using earlier observations to describe a later/current observation: past launches can inform a current launch, but the current result cannot inform itself.
- `Company_Prior_Failures` was not retained because it was highly correlated with prior launches (about 0.952) and would add nearly duplicate information. Failure rate is more informative than a raw failure count.
- Part E creates and diagnostically compares these features; final model training is not reported there.

## Communication requirements
Use simple everyday language, define jargon once, keep explanations short enough for a beginner, point to notebook parts/cells, prefer tables/words over extra code for Phase 1, and do not claim that a final predictive model has been trained or reported.
