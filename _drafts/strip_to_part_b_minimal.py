"""
Reduces the notebook to Part A + the Part B PROBLEM DEFINITION ONLY.

Part B, per the activity guide, asks for three things and nothing more:
  1. Identify the most suitable modelling framework.
  2. Justify why the dataset attributes fit this problem type.
  3. (implicitly) state the target and features.

So we keep Part A and the user's own Part B write-up, and remove all the
clustering/ABC/analysis machinery that was added ahead of time.

Run: python3 strip_to_part_b_minimal.py
"""
import json

NB_PATH = "oop.ipynb"

with open(NB_PATH) as f:
    nb = json.load(f)

# Keep cells 0-2 only: Part A markdown, Part A code, Part B markdown.
# Cell 2 is the user's ORIGINAL Part B write-up, preserved verbatim.
nb["cells"] = nb["cells"][:3]

with open(NB_PATH, "w") as f:
    json.dump(nb, f, indent=1)

print(f"Notebook reduced to {len(nb['cells'])} cells.")
for i, c in enumerate(nb["cells"]):
    first = "".join(c["source"]).strip().split("\n")[0][:70]
    print(f"  {i:>3} {c['cell_type']:<8} {first}")
