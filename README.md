## Running adaptive MFD cell classification codes

### Clone the repository
```bash
git clone https://github.com/chauj96/adaptive-mfd.git
cd adaptive-mfd
```

### Create and activate a Conda environment
```bash
conda create -n amfd python=3.12
conda activate amfd
```
### Install required packages
```bash
conda install -c conda-forge \
numpy scipy matplotlib pyvista pyyaml \
petsc petsc4py
```
### Run a simulation
#### Case 1: Manufactured-Solution Convergence Test
Run the convergence benchmark used to verify pressure and flux convergence rates:
```bash
python ex0_convergence.py
```
This script automatically performs both Global Adaptaion (GA) and Local Adaptation (LA) studies on a sequence of refined meshes and generates convergence plots.

Results are written to:
```text
output/ex0_convergence_GA/
output/ex0_convergence_LA/
```

#### Case 2: Benchmark Examples
Input files are provided in the `input/` directory. To run a case, pass the desired YAML configuration file as a command-line argument:
```bash
python main.py input/fullyPoly.yaml
```
Other available examples include:
```bash
python main.py input/twoFault_liso.yaml
python main.py input/twoFault_hetani.yaml
python main.py input/spe11b.yaml
```

Results are written automatically to:
```text
output/<case_name>/
```
where `<case_name>` is determined from the YAML filename.

#### Input Configuration
Simulation parameters are specified through YAML input files. These files define:

 - Mesh type (spe11b, two fault model, fully polyhedral model)
 - Analytical pressure field coefficients
 - Permeability tensor model
 - Boundary conditions
 - Tolerance values for adaptive classification
 - Adaptation strategy (`GA` or `LA`)
 - Pressure solver options (`direct`, `iterative`)
 - Flux error referecne
 - Saturation solver settings

This design allows different benchmark cases to be configured **without** modifying the source code.
