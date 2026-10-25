# OOP Midterm Project — Laptop Handoff

**Purpose:** Use this note to resume the OOP midterm work in a separate Pi session/device. This handoff covers the **DSC 2135 OOP Space Missions project only**. Ignore the unrelated MySQL and Information Management folders in the repository.

## Project at a glance

- **Project:** Space Missions launch-outcome classifier.
- **Course/track:** DSC 2135 — Object Oriented Programming; Data Scientist track.
- **Objective:** Prepare historical launch data, build and evaluate a binary classifier for `Success` vs `Not-Success`, and explain results and limitations.
- **Data:** `space_missions.csv` (4,630 initial records; 4,628 after duplicate handling). The target contains 4,160 Success and 468 Not-Success records.
- **Main project materials:** `oop.ipynb`; `uploads/activity_2_baseline_model.ipynb`; `uploads/activity_3_improve_compare_explain.ipynb`; `_drafts/midterm_project_paper.md`; `_docs/project_full_study_guide.md`; `_docs/context_handoff_prompt.md`.

## Project results to preserve

- Activity 1: leakage-aware preprocessing and a stratified 80/20 split (`random_state=42`), producing 3,702 training and 926 test rows with 46 features. Encoders/scalers are fitted on training data only; the test set is held out.
- Activity 2: `DummyClassifier` Not-Success F1 = 0.0000; class-weighted depth-limited decision tree Not-Success F1 = 0.3463.
- Activity 3: five candidate models compared by five-fold stratified CV on training data; tuned Random Forest (`n_estimators=500`, `max_depth=12`, `min_samples_leaf=2`, balanced class weights) scored 0.4215 Not-Success F1 on the held-out test set. False alarms fell from 183 to 112 while recall declined slightly.
- Limitations: random rather than chronological split, feature importance is not causation, and the source lacks important technical inputs. The model is a retrospective academic screener, not a launch-safety decision system.
- Do not invent or recalculate results casually. Consult the executed notebooks if any number is in question.

## OOP design material for the presentation

The presentation guidelines explicitly require an OOP design slide. The Activity 1 notebook contains real project classes and relationships:

- `FeatureBuilder` is an abstract base class with shared validation/state and an abstract derivation contract.
- `OperationalExperienceBuilder(FeatureBuilder)` specializes it to create leakage-safe, prior-launch-only operator history features.
- `LeakageGuard` checks feature matrices for forbidden/leaky fields and near-perfect target restatements.
- `FrequencyEncoder(BaseEstimator, TransformerMixin)` implements the scikit-learn transformer contract and is composed into the preprocessing `ColumnTransformer` / `Pipeline`.

Explain these actual responsibilities and connections rather than claiming the whole project is organized as one OO application. Useful concepts shown include abstraction, inheritance, encapsulation/properties, validation, and composition with the scikit-learn pipeline.

## Presentation guideline summary

The supplied PDF requests approximately **seven main slides** and a maximum of **15 minutes**: 10 minutes for the presentation plus 5 minutes for Q&A. A live program demo is not required. Required sequence:

1. Project title and overview
2. Data and project inputs
3. Program design and OOP application (classes, responsibilities, relationships, OOP concepts, simplified UML)
4. Methodology and workflow
5. Key implementation
6. Final output, results, and findings
7. Conclusion, limitation, and future improvement

The Data Scientist track is listed for Session 2. Keep slides concise and use evidence from the actual project (notebook snippets, diagrams, charts, and results).

## Latest report-formatting state

- The latest formatted report is **`Midterm_Project_Space_Missions_Ready.docx`** in the repository root on the originating PC.
- It uses explicit Calibri styles, redesigned table formatting, fixed unequal table widths, and a cover rebuilt with controlled spacing and an inline logo. The report body and table contents were checked as unchanged.
- An outside-session review found the earlier cover's blank-paragraph/floating-logo layout could shift. That issue was addressed in the `Ready` copy by removing the cover spacer paragraphs and making the logo inline. The cover section break and the page break after the Table of Contents were retained.
- The DOCX has **not** been visually rendered on the originating PC. On the laptop, open the `Ready` copy in Word and verify the cover stays on one page and that tables wrap/paginate cleanly before submission.
- **Important artifact transfer:** `Midterm_Project_Space_Missions_Ready.docx`, the original DOCX template, and `Midterm Project Presentation Guidelines.pdf` are local/untracked on the originating PC; this Git handoff commit does not include them. Transfer/copy those files separately if they are needed on the laptop. Older root DOCX copies may also exist; use the `_Ready.docx` copy as the latest one.
- The original template and guidelines were in `C:\Users\vonan\Downloads\` on the originating Windows installation.

## Cross-device workflow and guardrails

Pi intercom only connects sessions on the same machine. For PC-to-laptop continuity, use this handoff in Git: pull the repository, read this file, then inspect the listed notebooks/docs. Do not commit unrelated untracked coursework folders. Do not overwrite or push changes without reviewing the diff; commit only the files explicitly in scope. The user requested this handoff Markdown to be pushed; the DOCX artifacts are deliberately not added by this handoff commit.

## Recommended next task

On the laptop, transfer and visually review `Midterm_Project_Space_Missions_Ready.docx`. Then, if requested, create a concise seven-slide presentation following the guideline order above, using project evidence and a simple class-relationship diagram based on the actual classes in `oop.ipynb`.
