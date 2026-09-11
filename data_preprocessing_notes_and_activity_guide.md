# Comprehensive Guide to Data Preprocessing & Phase 1 Activity

**Course:** DSC 2135 – Object Oriented Programming / IS 4-513 – Intelligent Systems  
**Institution:** Ateneo de Davao University  
**Term / Date:** September 2026  
**Class Sections:** DataSci 2A (TTH 6:00–8:30 PM) | DataSci 4A (FRI 5:30–8:30 PM)  
**Motto:** *Fortes-in-Fide*

---

## Table of Contents
1. [Part I: Lecture Notes – Foundations of Data Preprocessing](#part-i-lecture-notes--foundations-of-data-preprocessing)
   - [1. Core Principle & The Golden Rule](#1-core-principle--the-golden-rule)
   - [2. Types of Data Analysis](#2-types-of-data-analysis)
   - [3. What Preprocessing Is: The Lifecycle](#3-what-preprocessing-is-the-lifecycle)
   - [4. Profiling: Know the Dataset First](#4-profiling-know-the-dataset-first)
   - [5. Data Types, Levels of Measurement, and Forms](#5-data-types-levels-of-measurement-and-forms)
   - [6. The Six Pillars of Data Quality Checks](#6-the-six-pillars-of-data-quality-checks)
   - [7. Data Cleaning and Treatment](#7-data-cleaning-and-treatment)
   - [8. Handling Missing Values, Duplicates, and Outliers](#8-handling-missing-values-duplicates-and-outliers)
   - [9. Data Transformation and Standardization](#9-data-transformation-and-standardization)
   - [10. Encoding Categorical Variables](#10-encoding-categorical-variables)
   - [11. Aggregation, Filtering, Reshaping, and Feature Preparation](#11-aggregation-filtering-reshaping-and-feature-preparation)
   - [12. Combining Datasets](#12-combining-datasets)
   - [13. Validation and Leakage Control](#13-validation-and-leakage-control)
   - [14. The Object-Oriented Programming (OOP) Perspective](#14-the-object-oriented-programming-oop-perspective)
   - [15. Common Pitfalls: What to Avoid](#15-common-pitfalls-what-to-avoid)
   - [16. The Master Preprocessing Checklist](#16-the-master-preprocessing-checklist)
2. [Part II: Activity Guide – Phase 1 Data Preprocessing (Data Scientist Track)](#part-ii-activity-guide--phase-1-data-preprocessing-data-scientist-track)
   - [1. Overview & General Instructions](#1-overview--general-instructions)
   - [2. Step-by-Step Activity Tasks (Parts A – I)](#2-step-by-step-activity-tasks-parts-a--i)
   - [3. Required Deliverables & Submission Checklist](#3-required-deliverables--submission-checklist)
   - [4. Final Reflection Prompts](#4-final-reflection-prompts)
   - [5. Evaluation Rubric & Submission Standards](#5-evaluation-rubric--submission-standards)

---

# Part I: Lecture Notes – Foundations of Data Preprocessing

---

### 1. Core Principle & The Golden Rule

> **The Golden Rule:**  
> *"Do not trust a clean-looking file until you know how it became clean."*

A machine learning model or analytical dashboard can only be as dependable as the raw data fed into it. A sophisticated model cannot compensate for erroneous, biased, or poorly prepared data.

---

### 2. Types of Data Analysis

The analytical goal directly dictates the nature and focus of data preparation:

| Type | Main Question | Preparation Focus |
| :--- | :--- | :--- |
| **Descriptive** | *What happened?* Counts, averages, historical trends. | Clear definitions, consistent aggregation units, and unified summaries. |
| **Diagnostic** | *Why did it happen?* Comparisons, segments, root-cause patterns. | Comparable categories, segment labels, and reliable groupings. |
| **Exploratory** | *What might be hidden?* Unsupervised relationships, correlations, anomalies. | Flexible inspection without aggressively deleting context or unusual points too early. |
| **Predictive** | *What is likely next?* Machine learning models, forecasting. | Clean target variable ($y$), disciplined train/validation/test split, strict data leakage prevention. |
| **Prescriptive** | *What should be done?* Optimization, decision options, trade-offs. | Business rules, boundary limits, constraints, and operational assumptions explicitly documented. |

---

### 3. What Preprocessing Is: The Lifecycle

**Data preprocessing** is the comprehensive preparation stage executed prior to analysis, reporting, predictive modeling, or long-term storage. It examines the structure, quality, format, and semantic meaning of data, applying justified, documented modifications so the data can be utilized accurately and reproducibly.

```
+-----------+       +-----------+       +---------------+       +------------+       +------------+
|  PROFILE  | ----> |   CLEAN   | ----> |   TRANSFORM   | ----> |  VALIDATE  | ----> |  DOCUMENT  |
| Understand|       |Fix Quality|       | Make Variables|       | Check the  |       |  Make it   |
| Structure |       |  Issues   |       |    Usable     |       |   Result   |       | Repeatable |
+-----------+       +-----------+       +---------------+       +------------+       +------------+
```

---

### 4. Profiling: Know the Dataset First

Never clean first and ask questions later. The contextual and domain meaning of each field controls every downstream preprocessing decision.

#### Questions to Ask First
- What does each row represent (the unit of analysis / observation)?
- What does each column measure?
- Which field serves as the unique primary identifier?
- What expected units, measurement ranges, and formats apply to each feature?

#### Dimensions to Inspect First
- **Shape:** Total count of rows and columns.
- **Data Types:** Numerical, textual string, date/timestamp, categorical, or boolean.
- **Missingness Profile:** Blank entries, `NaN`, `N/A`, `null`, or numeric placeholders (e.g., `999`, `-1`, or `0`).
- **Statistical Distributions:** Minimum, maximum, mean, median, quartiles, and frequent categories.

---

### 5. Data Types, Levels of Measurement, and Forms

#### A. Variable Types & Preparation Approaches
| Data Type | Concrete Examples | Common Preparation Workflow |
| :--- | :--- | :--- |
| **Numeric** | Age, salary, exam score, transaction quantity | Inspect domain range, test distributions, handle outliers, scale/normalize if required. |
| **Categorical** | Department, status, product type, region | Standardize spellings/case, group rare classes, apply encoding schemes. |
| **Datetime** | Birthdate, server log timestamp, checkout date | Parse string into standard ISO-8601 datetime object, extract intervals, durations, or calendar parts. |
| **Text** | Customer comments, descriptions, names | Strip whitespace, normalize casing, tokenize, remove uninformative tokens/noise carefully. |
| **Boolean** | Yes/No, True/False, Active/Inactive, 1/0 | Standardize into binary booleans; establish explicit logic for missing values. |

#### B. Levels of Measurement
- **Nominal:** Categories with no intrinsic order (e.g., Department: *HR*, *Finance*, *IT*). Requires one-hot encoding or frequency encoding.
- **Ordinal:** Categories with a definite ranking/order (e.g., Rating: *Low*, *Medium*, *High*). Supports ordinal encoding preserving relative magnitude.
- **Interval:** Numerical scales where intervals are equal, but zero is arbitrary (e.g., Temperature in Celsius).
- **Ratio:** Numerical scales featuring a meaningful, absolute zero point (e.g., Salary, Weight, Age). Supports full arithmetic operations.

#### C. Forms of Data
| Form | Examples | Core Preprocessing Concerns |
| :--- | :--- | :--- |
| **Structured** | Relational SQL tables, CSV files, Excel sheets | Data types, null values, exact/business duplicate records, relational constraints. |
| **Semi-structured** | JSON payloads, XML documents, system logs | Unnesting nested objects/arrays, resolving irregular schema keys, flattening structures. |
| **Unstructured** | Unformatted text comments, scanned images, audio files | Feature extraction, semantic labeling, vectorized representations, metadata extraction. |

---

### 6. The Six Pillars of Data Quality Checks

Conduct an initial audit evaluating the following six criteria:

1. **Completeness:** Are mandatory fields fully populated? What proportion of data is missing?
2. **Consistency:** Are labels, values, date formats, and spelling conventions uniform across records?
3. **Timeliness:** Does the dataset represent the appropriate, valid time window or reference period?
4. **Validity:** Do the entries adhere strictly to business syntax, domain ranges, and schema definitions?
5. **Uniqueness:** Are individual real-world entities recorded once, or are rows unintentionally duplicated?
6. **Accuracy:** Does the recorded data faithfully capture real-world operational truth?

---

### 7. Data Cleaning and Treatment

Data cleaning resolves errors, anomalies, and structural inconsistencies. Cleaning decisions must follow defensible rules rather than ad-hoc choices.

| Issue | Example | Remediation Strategy |
| :--- | :--- | :--- |
| **Typographical Inconsistency** | `"Davao"`, `"DVO"`, `"Davao City"` | Map variations to a standardized reference label using a controlled lookup dictionary. |
| **Impossible / Invalid Value** | $\text{Age} = -3$ or $\text{Score} = 150\%$ | Cross-check source system, replace with null, or flag with an explicit data quality marker. |
| **Wrong Data Type** | Numerical amount stored as text: `"$1,200.50"` | Strip non-numeric formatting characters and safely cast to `float` or `Decimal`. |
| **Duplicate Rows** | Exact same ID, timestamp, and amount repeated | Deduplicate based on defined primary key rules (e.g., retain most recent record). |

---

### 8. Handling Missing Values, Duplicates, and Outliers

#### A. Missing Value Treatments
Never substitute blanks with zeros unless zero carries an exact, valid real-world meaning.

| Remediation | When Appropriate | Inherent Risk / Trade-off |
| :--- | :--- | :--- |
| **Remove Rows** | Very small fraction of records missing; completely at random. | Reduces sample size; can introduce selection bias. |
| **Remove Columns** | Field is predominantly empty ($>50\text{--}70\%$) or uninformative. | Destroys potential predictive signals. |
| **Impute Values** | Missingness is low-to-moderate and predictably distributed (mean, median, mode, or k-NN). | Can distort feature variance, suppress covariance, or introduce false precision. |
| **Add Missing Flag** | Missingness itself indicates a meaningful pattern (e.g., loan co-signer omitted). | Expands dimensionality; increases model complexity. |
| **Consult Data Owner** | The business cause for blank entries is ambiguous. | Consumes project time, but prevents speculative assumptions. |

#### B. Duplicates: Exact vs. Business Duplicates
- **Exact Duplicates:** Every single column in the row is identical to another row. Remediated by dropping redundant rows.
- **Business Duplicates:** Critical operational identifiers match (e.g., Customer ID, Order Date, and SKU are identical, but timestamp differs by seconds due to retry logic). Requires domain logic: preserve the earliest record, the latest record, or the confirmed transaction.

#### C. Outlier Management
- **Detection Strategies:** Domain constraints, boxplot Interquartile Range ($\text{IQR} = Q_3 - Q_1$), or Z-score limits ($|Z| > 3$).
- **Verification Rule:** Distinguish between a genuine recording error (e.g., typing $500$ instead of $50$ for vehicle speed) and a valid extreme case (e.g., an executive salary or rare fraud event).
- **Rule of Thumb:** Never delete outliers simply because they complicate a distribution or degrade visual symmetry.

---

### 9. Data Transformation and Standardization

Transformation modifies data scales, representations, or schemas while preserving true information content.

$$\text{Convert} \longrightarrow \text{Standardize} \longrightarrow \text{Scale} \longrightarrow \text{Encode} \longrightarrow \text{Create (Feature Engineering)}$$

#### A. Data Type Conversion
- Text `"1,250.50"` $\longrightarrow$ Float `1250.50` (enables arithmetic operations and statistical summaries).
- Text `"2026/09/10"` $\longrightarrow$ Datetime object `2026-09-10` (enables chronological sorting and duration calculations).
- Mixed text `"TRUE"`, `"Yes"`, `"1"` $\longrightarrow$ Boolean `True` (prevents conditional logic bugs).
- Sentinel text `"N/A"`, `"null"`, `"-"` $\longrightarrow$ Proper `NaN` / `None` (prevents invalid calculation failures).
- *Caution:* Never execute silent coercion where unparseable characters secretly convert into `0`.

#### B. Data Standardization
- **Casing:** Convert strings to uniform case (`lower()`, `upper()`, or `title()`).
- **Whitespace:** Trim leading, trailing, and repeated intermediate white spaces.
- **Date Formats:** Harmonize diverse inputs (`9/10/26`, `Sept 10, 2026`) into ISO standard (`YYYY-MM-DD`).
- **Measurement Units:** Convert heterogeneous measurements into a uniform base metric (e.g., convert pounds to kilograms, or miles to kilometers).
- **Coding Systems:** Map inconsistent organizational codes (e.g., `HR`, `HRMDO`, `Human Resources`) via unified lookup tables.

#### C. Standardization vs. Normalization (Scaling)
- **Min-Max Normalization:** Rescales values into a fixed range (typically $[0, 1]$):
  $$X_{\text{norm}} = \frac{X - X_{\min}}{X_{\max} - X_{\min}}$$
  *Use case:* Neural networks, image pixel processing, or algorithms assuming bounded features. Sensitive to extreme outliers.
- **Standardization (Z-Score Scaling):** Centers data around a mean of $0$ with a standard deviation of $1$:
  $$Z = \frac{X - \mu}{\sigma}$$
  *Use case:* Distance-based algorithms (k-Means, k-NN, SVM) and gradient-based models (linear regression, logistic regression). Handles outliers better than min-max normalization.

---

### 10. Encoding Categorical Variables

Converting textual and categorical variables into numbers requires choosing an encoding method that matches the variable's measurement level:

| Encoding Method | Mechanism | Ideal Application | Hazard / Misuse |
| :--- | :--- | :--- | :--- |
| **Label Encoding** | Maps each unique category to an arbitrary integer ($0, 1, 2, \dots$). | Ordinal variables with clear rankings. | Applying to nominal data causes linear models to infer non-existent mathematical order ($2 > 1$). |
| **One-Hot Encoding** | Generates binary ($0/1$) indicator dummy columns for each level. | Nominal features with low cardinality. | Causes the "curse of dimensionality" when applied to high-cardinality features. |
| **Ordinal Encoding** | Maps ranked values to integers reflecting explicit hierarchy ($1, 2, 3$). | Explicit tiered attributes (e.g., *Low*, *Medium*, *High*). | Applying incorrect rank steps. |
| **Frequency / Count Encoding** | Replaces each category label with its observed sample frequency. | High-cardinality nominal variables (e.g., Zip codes, cities). | Different categories with identical counts will map to the exact same value. |

---

### 11. Aggregation, Filtering, Reshaping, and Feature Preparation

#### A. Structural Modifications
- **Filtering:** Subsetting observations matching explicit criteria. *Risk:* Silently hiding dropped records without documenting reasons.
- **Sorting:** Ordering records by indices or dates. *Risk:* Accidentally relying on row index order as an analytical feature.
- **Aggregation:** Condensing granular records into summary metrics (e.g., grouping daily purchases into monthly totals). *Risk:* Irretrievably discarding row-level variance.
- **Reshaping (Pivoting / Melting):** Transposing data between wide formats and long formats. *Risk:* Dropping identifiers or introducing unaligned multi-indexes.

#### B. Feature Engineering
Feature engineering derives new informative columns from raw inputs to capture domain logic:

| Raw Features | Derived Feature | Engineering Purpose |
| :--- | :--- | :--- |
| Date of Birth | $\text{Age} = \text{Current Date} - \text{DOB}$ | Replaces static timestamps with dynamic predictive indicators. |
| Transaction Timestamp | Hour of Day, Day of Week, Is Weekend | Captures behavioral periodicity, seasonality, and cyclic habits. |
| Start Timestamp & End Timestamp | $\text{Duration} = \text{End} - \text{Start}$ | Converts timestamp pairs into a continuous elapsed interval. |
| Quantity & Unit Price | $\text{Total Amount} = \text{Quantity} \times \text{Price}$ | Models financial scale and transaction magnitude directly. |
| Raw Text Feedback | Length, Sentiment Score, Keyword Indicators | Translates unstructured free text into structured tabular features. |

---

### 12. Combining Datasets

When joining or merging datasets, consider key integrity across the following dimensions:

1. **Key Uniqueness:** Verify whether join keys are strictly unique ($1:1$) or duplicate records will trigger row multiplication ($1:M$, $M:M$).
2. **Join Strategy:** Explicitly select Inner, Left, Right, or Full Outer joins based on analytical requirements.
3. **Unmatched Rows:** Analyze rows that fail to match to prevent systematic omissions.
4. **Column Collision:** Clarify column name overlaps to ensure authoritative source values are retained.
5. **Temporal Alignment:** Verify merged tables share the same timeframes and timezone baselines.

---

### 13. Validation and Leakage Control

#### A. Post-Transformation Validation Checklist
- Re-verify final row counts and column dimensions against initial profiling benchmarks.
- Compare statistical distributions before and after transformations (mean, standard deviation, quartiles).
- Confirm all placeholder values, nulls, and duplicate records have been resolved.
- Verify that numeric features lie within plausible real-world ranges.

#### B. Preventing Data Leakage
> **Data Leakage** occurs when information from outside the training dataset—specifically information unavailable during actual production runtime or the target itself—is introduced into the model training pipeline.

- **Split Early:** Perform train/test splits before computing imputation metrics, scalers, or encoders.
- **Fit on Train Only:** Always calculate parameters ($\mu$, $\sigma$, category frequencies, min/max values) using the training set only, then apply those fixed transformations to the validation and test sets.
- **Target Independence:** Never use target feature values ($y$) to clean, impute, or engineer independent features ($X$).
- **Temporal Integrity:** For time-series problems, never use future information to impute or predict historical records.

---

### 14. The Object-Oriented Programming (OOP) Perspective

Treat preprocessing as an engineered, modular, and reusable pipeline rather than an ad-hoc script of manual edits.

```
+-------------------------------------------------------------+
|                     PreprocessingPipeline                   |
|  + fit(X_train)                                             |
|  + transform(X)                                             |
+-------------------------------------------------------------+
       |                  |                    |
       v                  v                    v
+--------------+  +----------------+  +-------------------+
| CleaningRule |  |  Transformer   |  |     Validator     |
+--------------+  +----------------+  +-------------------+
```

#### Core OOP Abstractions:
- `DataSetProfile`: Encapsulates schema metadata, row/column counts, inferred types, and missingness metrics.
- `CleaningRule`: Encapsulates distinct detection logic and remediation methods for specific anomalies.
- `Transformer`: Implements standardized interfaces (e.g., `fit()`, `transform()`, `fit_transform()`) for conversions, encodings, and scalers.
- `Validator`: Evaluates transformed datasets against programmatic assertions and quality constraints.
- `PreprocessingPipeline`: Sequentially executes cleaning and transformation steps, logging transformations and parameters.

#### Pipeline Architecture:
$$\text{Raw Data (Read-Only)} + \text{Rules/Logic} \longrightarrow \text{Pipeline Processing} \longrightarrow \text{Clean Dataset} + \text{Audit Log}$$

---

### 15. Common Pitfalls: What to Avoid

1. **Cleaning Without Profiling:** Modifying records before understanding domain rules and schemas.
2. **Overwriting Raw Data:** Modifying original CSV or database sources directly.
3. **Blind Deletion:** Dropping missing values without evaluating whether they occur at random or follow systematic patterns.
4. **Treating Outliers as Bugs:** Deleting valid high-leverage data points simply because they look inconvenient.
5. **Inventing Categorical Order:** Using integer label encoding on nominal categories where no hierarchy exists.
6. **Scaling Before Splitting:** Fitting normalizers or scalers on the entire dataset prior to train/test partitioning.
7. **Subjective Cleaning:** Applying personal intuition rather than documented, rule-based standards.
8. **Silent Pipelines:** Modifying columns without logging changes, justifications, or counts of affected rows.

---

### 16. The Master Preprocessing Checklist

| Phase | Step | Focus Areas |
| :---: | :--- | :--- |
| **1** | **Understand** | Clarify the business objective, unit of analysis, column semantics, and target variables. |
| **2** | **Profile** | Inspect dataset dimensions, data types, missing value rates, and statistical summaries. |
| **3** | **Clean** | Resolve duplicate rows, correct invalid entries, and apply standardized labels. |
| **4** | **Transform** | Cast types, convert datetime features, encode categories, scale numeric ranges, and engineer derived features. |
| **5** | **Validate** | Verify integrity assertions, verify pre/post distributions, and confirm data leakage safeguards. |
| **6** | **Document** | Log all cleaning logic, operational assumptions, discarded rows, and dataset version markers. |

---
---

# Part II: Activity Guide – Phase 1 Data Preprocessing (Data Scientist Track)

**Course:** Intelligent Systems / Object Oriented Programming  
**Evaluation:** Midterm Practical Activity 1  
**Target Delivery Date:** September 10, 2026  

---

### 1. Overview & General Instructions

A machine-learning model can only learn from the data it receives. A sophisticated algorithm cannot compensate for poorly prepared data. Using your assigned dataset, prepare the data so that it can eventually support a machine-learning task. You are not yet required to build the final model.

#### Core Directives:
- **Individual Work:** Work individually using only your designated dataset.
- **No Manual Edits:** Do not modify the raw data file directly using Excel or spreadsheet software. All operations must be written in Python (Jupyter / Google Colab).
- **Preserve Raw Data:** Keep the original dataset unchanged in a dedicated raw directory.
- **Reproducible Code:** Provide reproducible code cells for every preprocessing operation.
- **Justify Every Decision:** Accompany each operation with a short explanation of why it was necessary.
- **Avoid Over-Processing:** Only apply transformations that improve the data. Unnecessary modifications can degrade downstream model performance.
- **Structure Your Notebook:** Organize your notebook using clean markdown headings, code comments, and concise interpretations.

---

### 2. Step-by-Step Activity Tasks (Parts A – I)

#### Part A – Dataset Inspection & Profiling
1. Determine the dataset dimensions (total rows and columns).
2. Enumerate all columns alongside their initial computational data types.
3. Classify variables into numerical and categorical categories.
4. Quantify missing values and identify duplicate records.
5. Identify potential outliers, anomalous entries, and invalid values.
6. Identify a candidate target variable ($y$), if present.
7. Identify candidate input features ($X$).
   - *Note:* If the dataset lacks an explicit target, specify the analytical or unsupervised learning problem it could support.

#### Part B – Define the Machine-Learning Problem
Identify the most suitable modeling framework for the dataset:
- **Classification:** Predicting discrete class labels.
- **Regression:** Predicting a continuous numeric quantity.
- **Clustering:** Unsupervised grouping of unlabelled records.
- **Other Analytical Task:** Anomaly detection, association mining, or time-series forecasting.
- *Justification:* Explain why the dataset attributes fit this problem type.

#### Part C – Data Cleaning
Inspect and handle data quality issues using explicit, rule-based logic:
- Missing values (impute, drop, or flag with justification).
- Duplicate observations (exact vs. business duplicates).
- Incorrect storage data types (e.g., numeric strings cast to proper numeric types).
- Inconsistent categorical entries (e.g., casing, typos, variant representations).
- Impossible or invalid domain entries.
- Outliers (evaluate whether points are recording errors or valid domain extremes).

#### Part D – Feature Transformation
Apply transformations that convert variables into representations suitable for statistical modeling:

| Transformation Technique | Functional Purpose | Appropriate Use Case |
| :--- | :--- | :--- |
| **Categorical Encoding** | Converts text categories into numeric matrices. | Algorithms requiring numeric tensors (One-Hot, Ordinal). |
| **Min-Max Normalization** | Bounds continuous values to a fixed range (typically $[0, 1]$). | Algorithms sensitive to feature scale without a normal distribution. |
| **Z-Score Standardization** | Centers continuous values to $\mu=0$ and $\sigma=1$. | Algorithms assuming zero-centered inputs or calculating Euclidean distances. |
| **Log / Power Transformation** | Mitigates heavy right-skewness and stabilizes variance. | Features spanning multiple orders of magnitude. |
| **Date/Time Extraction** | Derives cyclical and seasonal attributes from timestamps. | Problems where timing, seasonality, or duration matter. |
| **Feature Engineering** | Derives informative features from existing columns. | Creating indicators that capture domain relationships more directly. |

> **Required Discussion Question:**  
> Explicitly evaluate whether **normalization** or **standardization** is best suited for your numerical attributes and target problem, and explain your technical justification.

#### Part E – Feature Engineering
Derive at least **one** new feature that adds domain value.

*Formula Examples:*
- $\text{BMI} = \frac{\text{Weight (kg)}}{\left(\text{Height (m)}\right)^2}$
- $\text{Total Purchase} = \text{Quantity} \times \text{Unit Price}$
- $\text{Customer Age} = \text{Current Date} - \text{Birth Date}$
- $\text{Tenure} = \text{End Date} - \text{Start Date}$

*Explanation Requirement:* Detail why this derived feature provides stronger predictive or analytical signal than the raw attributes alone.

#### Part F – Check for Data Leakage
Evaluate whether any column contains information that would be unavailable during actual inference time:
- Does any feature directly or indirectly expose information about the target variable ($y$)?
- Could future information leak into past predictions?
- *Explanation:* Provide an evaluation of your dataset's leakage risks.

#### Part G – Feature and Target Separation
Separate your preprocessed dataset into feature matrices and target vectors:
```python
# Programmatic Separation Example
X = df.drop(columns=["target_variable_name"])
y = df["target_variable_name"]
```

#### Part H – Training and Testing Preparation
Prepare the dataset for model validation:
- Partition the data (e.g., 80% Training, 20% Testing; adjust and justify if dataset scale requires another ratio).
- **Leakage Control Requirement:** Compute all imputation, scaling, and encoding parameters using the **training set only**, then transform the test set using those pre-fit values.

#### Part I – Preprocessing Summary Table
Summarize all preprocessing interventions using the following table format:

| Issue / Need Identified | Specific Action Taken | Technical / Business Reason |
| :--- | :--- | :--- |
| *e.g., Inconsistent casing in City column* | *Applied .str.strip().str.title()* | *Standardized 14 unique variants into 3 valid municipal groups* |
| *e.g., Missing numerical entries in Age* | *Imputed using training set median* | *Preserved records while limiting skew from extreme values* |

---

### 3. Required Deliverables & Submission Checklist

- [ ] **Jupyter Notebook (`.ipynb`) or Google Colab link:** Fully executed with all cell outputs visible.
- [ ] **Raw Dataset Untouched:** Original file kept in its original form.
- [ ] **Cleaned Dataset Export:** Final preprocessed dataset saved to a `.csv` file.
- [ ] **Dataset Overview & Metadata:** Clear documentation of rows, columns, and variable types.
- [ ] **Machine Learning Problem Definition:** Proposed task (Classification, Regression, etc.) with reasoning.
- [ ] **Target ($y$) and Features ($X$) Defined:** Clear identification of input and output variables.
- [ ] **Data Quality Assessment:** Complete audit of nulls, duplicates, errors, and outliers.
- [ ] **Documented Cleaning Code:** Every cleaning step includes an explanation of the underlying decision.
- [ ] **Feature Transformations:** Documented encoding, scaling, and date transformations.
- [ ] **Engineered Feature:** At least one new derived feature with an explanation of its purpose.
- [ ] **Data Leakage Safeguards:** Audit confirming no target or future information leaks into features.
- [ ] **Train/Test Partitioning:** Training and testing split implemented without data leakage.
- [ ] **Preprocessing Summary Table:** Complete issue-action-reason summary table.
- [ ] **Final Reflection Responses:** Thorough answers to all four concluding reflection questions.

---

### 4. Final Reflection Prompts

1. **Impact Analysis:** Which specific preprocessing decision do you anticipate will have the greatest influence on the downstream performance of your model, and why?
2. **Technical Bottleneck:** Which variable presented the greatest preprocessing challenge, and what made resolving it difficult?
3. **Leakage Vulnerabilities:** What potential sources of data leakage did you identify in your dataset, and what steps did you take to prevent them?
4. **Residual Limitations:** What structural biases or limitations remain in the dataset that preprocessing could not resolve?

---

### 5. Evaluation Rubric & Submission Standards

#### Grading Rubric (100 Points Total)

| Evaluation Criterion | Maximum Points | Performance Expectations |
| :--- | :---: | :--- |
| **Dataset Understanding & Inspection** | **10** | Accurately identifies shape, columns, measurement levels, storage types, and business context. |
| **Data-Quality Identification** | **15** | Systematically identifies quality issues, anomalies, missing patterns, and structural errors. |
| **Appropriate Data Cleaning** | **20** | Applies suitable cleaning methods, preserves data integrity, and handles missing/duplicate data appropriately. |
| **Appropriate Transformation** | **15** | Selects appropriate scaling, encoding, and derived features while avoiding unnecessary transformations. |
| **Code Correctness & Quality** | **15** | Code runs end-to-end without errors, uses clean formatting, and is well-documented with comments. |
| **Track-Specific Preparation** | **10** | Sets up the data for machine learning (proper $X/y$ separation, train/test splitting, and leakage controls). |
| **Explanation & Justification** | **10** | Explains the reasoning behind every cleaning and transformation decision clearly and defensively. |
| **Organization & Completeness** | **5** | Follows all template specifications, includes required reflection answers, and meets submission guidelines. |
| **TOTAL SCORE** | **100** |  |

#### Pre-Submission Quality Checks
- [ ] **Restart and Run All:** Restart the notebook kernel and run all cells from top to bottom to verify execution without errors.
- [ ] **Visible Outputs:** Ensure all print statements, distribution plots, and table summaries render properly.
- [ ] **Confirm File Integrity:** Verify the raw data file remains completely unchanged.
- [ ] **Export Verification:** Open the exported preprocessed `.csv` file in a clean session to ensure it loads with valid columns and rows.
- [ ] **Review Explanations:** Confirm that all transformations and cleaning steps are supported by written technical justifications.