"""
Strips the notebook back to Part A + Part B only, discarding Parts C-I so the
project can be rebuilt one part at a time.

Run: python3 strip_to_part_b.py
"""
import json

NB_PATH = "oop.ipynb"

with open(NB_PATH) as f:
    nb = json.load(f)

# Find where the Part C section begins and cut everything from there.
cut_at = None
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] == "markdown" and "# Part C — Data Cleaning" in "".join(c["source"]):
        cut_at = i
        break

if cut_at is None:
    raise SystemExit("Could not find the Part C header; nothing was changed.")

removed = len(nb["cells"]) - cut_at
nb["cells"] = nb["cells"][:cut_at]

# Drop any trailing separator-only markdown cells left behind.
while nb["cells"] and nb["cells"][-1]["cell_type"] == "markdown":
    text = "".join(nb["cells"][-1]["source"]).strip()
    if set(text) <= set("-\n \t") and text:
        nb["cells"].pop()
    else:
        break

with open(NB_PATH, "w") as f:
    json.dump(nb, f, indent=1)

print(f"Removed {removed} cells (Parts C-I).")
print(f"Notebook now has {len(nb['cells'])} cells: Part A + Part B only.")
for i, c in enumerate(nb["cells"]):
    first = "".join(c["source"]).strip().split("\n")[0][:70]
    print(f"  {i:>3} {c['cell_type']:<8} {first}")
