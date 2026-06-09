## Running adaptive MFD cell classification codes

Clone the repository
```bash
git clone https://github.com/chauj96/adaptive-mfd.git
cd adaptive-mfd
```

Install system dependencies (macOS) 
```bash
brew install suite-sparse swig
```

Create and activate a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

Install required Python packages
```bash
pip install .
```

Run the code
```bash
python main.py
```

## Optional: PETSc iterative solver
To use the PETSc-based solver, install PETSc and petsc4py:
```bash
brew install petsc
pip install petsc4py
```

Then set
```python
solver_type = "iterative"
```
in `main.py`.
