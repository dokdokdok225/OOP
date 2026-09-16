# DSC 2135 — Object Oriented Programming Project

Ateneo de Davao University · *Fortes-in-Fide*

Course project workspace: dataset inspection and preprocessing of
`space_missions.csv` (4,630 observations × 9 features), plus OOP lecture
references and activity guides.

## Contents

| Path | Description |
| --- | --- |
| `oop.ipynb` | **Main deliverable.** The activity parts. Check the notebook itself for which parts are complete — this table goes stale as work progresses. |
| `space_missions.csv` | Original raw Space Missions dataset. |
| `space_missions_engineered.csv` | Readable engineered table with selected source fields, derived features, and binary target. |
| `oop_rules.md` | Comprehensive OOP rules & architecture lecture reference. |
| `data_preprocessing_notes_and_activity_guide.md` | Data preprocessing lecture notes + Phase 1 activity guide. |
| `oop_preprocessing_notes.md` | Working notes on preprocessing decisions. |
| `execute_notebook.py` | Helper to run the notebook headlessly. |
| `activity_2_baseline_model.ipynb` | Activity 2 baseline classification notebook. |
| `_docs/project_data_cheat_sheet.md` | End-to-end data-flow cheat sheet: raw CSV → engineered features → `X`/`y` → saved arrays. |
| `_docs/model_fit_research.md` | Research and empirical comparison of suitable classification models. |
| `_drafts/` | Build scripts and notebook backups. These **are tracked in git** so no work is lost, but they are scratch files — not graded material. |
| `build_part_*.py` | Scratch scripts that generate notebook sections. Also scratch, not graded. |
| `main..py` | Empty file (0 bytes). The double dot in the name is a typo; harmless, left untouched. |

## Getting started

### Requirements

- Python 3.10+ (tested on 3.14)
- `pandas`, `numpy`, `jupyter`

```bash
python -m pip install pandas numpy jupyter
```

### Run the notebook

```bash
jupyter notebook oop.ipynb
```

The notebook reads `space_missions.csv` from the repository root using a
relative path, so it works from any clone location.

Headless execution:

```bash
python execute_notebook.py
```

## Working from multiple devices

This repository is the single source of truth. On a **new device**:

```bash
git clone https://github.com/dokdokdok225/OOP.git
cd OOP
```

Then, at the start and end of every work session:

```bash
git pull          # get changes made elsewhere
# ... edit ...
git add -A
git commit -m "Describe what changed"
git push
```

> **Important:** always `git pull` before you start editing and `git push`
> before you stop, or you will hit merge conflicts between devices.

## Branching

- `main` — stable, working state.
- Feature/experiment branches — optional; merge back into `main` when done.
