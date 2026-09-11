"""Executes oop.ipynb in place so every cell carries real outputs."""
import nbformat
from nbclient import NotebookClient

with open("oop.ipynb") as f:
    nb = nbformat.read(f, as_version=4)

client = NotebookClient(
    nb,
    timeout=600,
    kernel_name="python3",
    allow_errors=False,
    resources={"metadata": {"path": "."}},
)
client.execute()

with open("oop.ipynb", "w") as f:
    nbformat.write(nb, f)

n_err = sum(
    1
    for c in nb.cells
    if c.cell_type == "code"
    for o in c.get("outputs", [])
    if o.get("output_type") == "error"
)
print(f"Executed OK. Cells: {len(nb.cells)}. Error outputs: {n_err}")
