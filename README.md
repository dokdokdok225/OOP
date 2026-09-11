# DSC 2135 — Object Oriented Programming Project

Ateneo de Davao University · *Fortes-in-Fide*

Course project workspace: dataset inspection and preprocessing of
`space_missions.csv` (4,630 observations × 9 features), plus OOP lecture
references and activity guides.

## Contents

| Path | Description |
| --- | --- |
| `oop.ipynb` | **Main deliverable.** Part A (dataset inspection) and Part B (ML problem framing). |
| `space_missions.csv` | Dataset used by the notebook (committed so the notebook runs anywhere). |
| `oop_rules.md` | Comprehensive OOP rules & architecture lecture reference. |
| `data_preprocessing_notes_and_activity_guide.md` | Data preprocessing lecture notes + Phase 1 activity guide. |
| `oop_preprocessing_notes.md` | Working notes on preprocessing decisions. |
| `execute_notebook.py` | Helper to run the notebook headlessly. |
| `_drafts/` | Build scripts and notebook backups (historic; not part of the deliverable). |

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
