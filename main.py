import numpy as np
import yaml
import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
from mesh_loader import load_mesh
from physics import initPhysicalParams, projectAnalyticalField
from operators import createMmatrix, createBmatrix
from classification import classify_cells
from pressure_solver import solve_pressure
from saturation_solver import solve_saturation
from io_utils import print_flux_err, plot_flux_err, print_saturation_err, plot_saturation_err, print_sparsity_info, write_vtu

# ===== Step 0 =====
if len(sys.argv) != 2:
    raise ValueError("Usage: python main.py <input_file.yaml>")

input_file = sys.argv[1]
case_name = Path(input_file).stem
output_dir = os.path.join("output", case_name)
os.makedirs(output_dir, exist_ok=True)

with open(input_file, "r") as f:
    case = yaml.safe_load(f)

def energy_norm(v, M):
    return np.sqrt(v @ (M @ v))

# ===== Step 1: Load mesh and set up its geometric information =====
cell_struct, face_struct, vertices, Lx, Ly, Lz = load_mesh(case["mesh"])

# ===== Step 2: Physical/discrete operator setup =====
reference_flux = case["reference_flux"]

# Set analytical linear pressure field ( f(x,y,z) = ax + by + cz + d )
a = case["analytical_field"]["a"] / Lx
b = case["analytical_field"]["b"] / Ly
c = case["analytical_field"]["c"] / Lz
d = case["analytical_field"]["d"]

g_c = 0.0
dt_pressure = 1.0

# perm_tensor = "identity", "layered_isotropy", "het_anisotropy"
# bc_option = "linear", "corner2corner" 
cell_struct, face_struct, phys = initPhysicalParams(cell_struct, face_struct, Lx, Ly, Lz, perm_tensor=case["permeability"], bc_option=case["boundary_condition"])
cell_struct = createMmatrix(cell_struct, face_struct, ip_type="tpfa")
cell_struct = createBmatrix(cell_struct)
m_proj, p_proj = projectAnalyticalField(cell_struct, face_struct, phys, a, b, c, d)

n_cells = len(cell_struct)
solve_saturation_flag = case["saturation"]["enabled"]

# ===== Step 3: Classify cells and solve a pressure field =====
# Solver setup
# inner_product = "simple", "quasi_tpfa", "general_parametric", "bdvlm"
# solver_type = "direct", "iterative", "hypredrive"
tol_list = np.array(case["tol_list"])
inner_product = case["solver"]["inner_product"]
adaptation_level = case["adaptation_level"]
solver_type = case["solver"]["solver_type"]

# Compute full MFD 
flux_results = []
sat_results = []
sparsity_results = []
cellMarking_full = np.ones(n_cells, dtype=int)
m_full, p_full, M_full, nnz_full, memory_mb_full = solve_pressure(cell_struct, face_struct, cellMarking_full, inner_product, dt_pressure, g_c, solver_type, source_term=None)

write_vtu(os.path.join(output_dir, "mesh_full_MFD.vtu"), vertices, cell_struct, face_struct, cellMarking_full, "cellMarking", "cell_plot")
sparsity_results.append(["full MFD", 0.0, nnz_full, memory_mb_full, 0.0, 0.0])

if solve_saturation_flag:
    Sw0 = np.zeros(n_cells)
    Sw_inj = 1.0
    tEnd = case["saturation"]["tEnd"]
    dt_transport = case["saturation"]["dt"]
    Sw_hist_ref, time_hist_ref = solve_saturation(cell_struct, face_struct, m_full, Sw0, Sw_inj, tEnd=tEnd, dt=dt_transport, solver_type=solver_type)
    Sw_ref = Sw_hist_ref[:, -1]
    write_vtu(os.path.join(output_dir, "sat_full_MFD.vtu"), vertices, cell_struct, face_struct, Sw_ref, "saturation", "saturation_plot")

# Energy norm for flux
if reference_flux == "projection":
    e = m_full - m_proj
    flux_abs_err = energy_norm(e, M_full)
    flux_rel_err = (flux_abs_err / energy_norm(m_proj, M_full))
    # flux_rel_err = np.linalg.norm(m_full - m_proj) / np.linalg.norm(m_proj)
    # flux_abs_err = np.linalg.norm(m_full - m_proj)
    flux_results.append(["full MFD", flux_rel_err, flux_abs_err])

if solve_saturation_flag:
    sat_rel_err = np.linalg.norm(Sw_hist_ref[:, -1]  - Sw_ref) / np.linalg.norm(Sw_ref)
    sat_abs_err = np.linalg.norm(Sw_hist_ref[:, -1]  - Sw_ref)
    sat_results.append(["full MFD", sat_rel_err, sat_abs_err])


if reference_flux == "projection":
    flux_ref = m_proj
else:
    flux_ref = m_full

# Compute Adaptive MFD
for tol in tol_list:

    cellMarking = classify_cells(cell_struct, face_struct, m_proj, p_proj, vertices, a, b, c, d, tol, adaptation_level=adaptation_level, out_dir=output_dir)
    m_num, p_num, M_adapt, nnz_adapt, memory_mb_adapt = solve_pressure(cell_struct, face_struct, cellMarking, inner_product, dt_pressure, g_c, solver_type, source_term=None)
    n_tpfa_cells = n_cells - int(np.sum(cellMarking))
    sparsity_reduction = 100.0 * (1 - nnz_adapt / nnz_full)
    memory_reduction = 100.0 * (1 - memory_mb_adapt / memory_mb_full)
    sparsity_results.append([tol, n_tpfa_cells, nnz_adapt, memory_mb_adapt, sparsity_reduction,memory_reduction])

    if solve_saturation_flag:
        Sw_hist, time_hist = solve_saturation(cell_struct, face_struct, m_num, Sw0, Sw_inj, tEnd=tEnd, dt=dt_transport, solver_type=solver_type)
        Sw_final = Sw_hist[:,-1]
        write_vtu(os.path.join(output_dir, f"sat_tol_{tol:.1e}.vtu"), vertices, cell_struct, face_struct, Sw_final, "saturation", "saturation_plot")

        sat_rel_err = np.linalg.norm(Sw_final - Sw_ref) / np.linalg.norm(Sw_ref)
        sat_abs_err = np.linalg.norm(Sw_final - Sw_ref)
        sat_results.append([tol, sat_rel_err, sat_abs_err])

    e = m_num - flux_ref
    flux_abs_err = energy_norm(e, M_adapt)
    flux_rel_err = (flux_abs_err / energy_norm(flux_ref, M_adapt))
    # flux_rel_err = np.linalg.norm(m_num - flux_ref) / np.linalg.norm(flux_ref)
    # flux_abs_err = np.linalg.norm(m_num - flux_ref)
    flux_results.append([tol, flux_rel_err, flux_abs_err])

# Check flux relative/absolute error
print_sparsity_info(sparsity_results)
print_flux_err(flux_results)
plot_flux_err(flux_results, os.path.join(output_dir, "flux_error.png"))

if solve_saturation_flag:
    print_saturation_err(sat_results)
    plot_saturation_err(sat_results, os.path.join(output_dir, "saturation_error.png"))

plt.show()