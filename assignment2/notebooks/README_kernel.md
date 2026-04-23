# Use `assignment2/.venv` in this notebook (Cursor / VS Code / Jupyter)

Notebooks run with the **kernel / Python interpreter you pick**, not “every venv on disk”. Installing into `.venv` only affects runs that actually use that interpreter.

## If `.venv` does not appear in “Select a Python Environment”

That list only shows environments the extension **finds easily**—often **`venv`** at the **workspace root**, not a nested **`assignment2/.venv`**.

Your project’s env is here: **`assignment2/.venv/bin/python`** (with a **dot**). That is **not** the same as a folder named **`venv`** at `/root/venv`.

**Fix A — Manual path (always works)**  
Command Palette → **Python: Select Interpreter** → **Enter interpreter path…** → paste:

`assignment2/.venv/bin/python`  
(if workspace is `/root`), or the full path, e.g. `/root/assignment2/.venv/bin/python`.

**Fix B — Open the assignment2 folder**  
**File → Open Folder…** → choose **`assignment2`** (not the parent). Then **`./.venv`** sits at the workspace root and usually appears in the list.

**Fix C — Workspace already at `/root`**  
`/root/.vscode/settings.json` can set  
`"python.defaultInterpreterPath": "${workspaceFolder}/assignment2/.venv/bin/python"`  
so the default is the assignment env (reload the window after saving).

## 1. Open the right folder (recommended)

Open **`assignment2`** as the workspace root (the folder that contains `.venv/` and `notebooks/`).  
Then `assignment2/.vscode/settings.json` can set the default interpreter to `./.venv/bin/python`.

## 2. Pick the interpreter (Cursor / VS Code)

1. `Ctrl+Shift+P` (mac: `Cmd+Shift+P`) → **Python: Select Interpreter**
2. Choose **Enter interpreter path…** → **Find…**
3. Select: `assignment2/.venv/bin/python` (Linux/mac) or `assignment2\.venv\Scripts\python.exe` (Windows)

Or click the Python version in the **status bar** and select the same path.

## 3. Pick the notebook kernel

With the `.ipynb` focused:

1. **Select Kernel** (top right of the notebook)
2. Choose **Python Environments…** → the same **`…/assignment2/.venv/bin/python`**

If you do not see it, run step 4 then reload the window (`Developer: Reload Window`).

## 4. Register Jupyter kernel (optional, helps Jupyter find `.venv`)

```bash
cd /path/to/assignment2
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install ipykernel jupyter matplotlib seaborn pandas numpy mysql-connector-python
python -m ipykernel install --user --name assignment2-venv --display-name "Assignment2 .venv"
```

Then **Select Kernel** → **assignment2-venv**.

## 5. Install notebook deps *into that* venv

```bash
source .venv/bin/activate
pip install matplotlib seaborn ...
```

Confirm in a notebook cell:

```python
import sys
print(sys.executable)
```

It should print a path under **`…/assignment2/.venv/`**.
