## Running the Python version (`src/adaptiveMFD_python`)

Install system dependencies (macOS)
```bash
brew install suite-sparse swig
```

Create and activate a virtual environment (from project root)
```bash
python3 -m venv venv
source venv/bin/activate
```

Navigate to the Python package directory
```bash
cd src/adaptiveMFD_python
```

Install required Python packages
```bash
pip install .
```

Run the code
```bash
python main.py
```
## Output

- VTU files for visualization (cell classification) in ParaView are written to `output_fault/`
- Solver error tables are printed in the terminal
- Relative error plots are displayed using matplotlib
