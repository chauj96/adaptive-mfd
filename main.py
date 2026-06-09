import numpy as np
from mesh_loader import load_mesh
from physics import initPhysicalParams, projectAnalyticalField
from operators import createMmatrix, createBmatrix
from classification import classify_cells
from pressure_solver import solve_pressure
from saturation_solver import solve_saturation
from io_utils import print_pressure_err, plot_pressure_err, print_saturation_err, plot_saturation_err, write_vtu


# ===== Step 1: Load mesh =====
cell_struct, face_struct, vertices, Lx, Ly, Lz = load_mesh("twoFaults")

# Set analytical linear pressure field ( f(x,y,z) = ax + by + cz + d )
a = -1/Lx
b = -1/Ly
c = -1/Lz
d = 1.0

# ===== Step 2: Physical/discrete operator setup =====
g_c = 0.0
dt_pressure = 1.0

# perm_tensor = "identity", "layered_isotropy", "het_anisotropy"
# bc_option = "linear", "corner2corner" 
cell_struct, face_struct, phys = initPhysicalParams(cell_struct, face_struct, Lx, Ly, Lz, perm_tensor="het_anisotropy", bc_option="corner2corner")
cell_struct = createMmatrix(cell_struct, face_struct, ip_type="tpfa")
cell_struct = createBmatrix(cell_struct)
m_proj, p_proj = projectAnalyticalField(cell_struct, face_struct, phys, a, b, c, d)

n_cells = len(cell_struct)
Sw0 = np.zeros(n_cells)
Sw_inj = 1.0

# ===== Step 3: Classify cells and solve a pressure field =====
# Solver setup
# inner_produce = "simple", "quasi_tpfa", "general_parametric", "bdvlm"
# solver_type = "direct", "iterative"
tol_list = np.array([1, 1e-1, 1e-2, 1e-4, 1e-6, 1e-8])
n_tol = len(tol_list)
inner_product = "quasi_tpfa"
eps_solver = 1e-12
gmres_niter = 200
solver_type = "direct"

# Compute full MFD 
flux_results = []
sat_results = []
cellMarking_full = np.ones(n_cells, dtype=int)
m_full, p_full = solve_pressure(cell_struct, face_struct, cellMarking_full, inner_product, dt_pressure, g_c, eps_solver, gmres_niter, solver_type)
Sw_hist_ref, time_hist_ref = solve_saturation(cell_struct, face_struct, m_full, Sw0, Sw_inj, tEnd=0.25, dt=0.01)
Sw_ref = Sw_hist_ref[:, -1] 
# write_vtu("output_fault/sat_full.vtu", vertices, cell_struct, face_struct, Sw_ref, "saturation", "saturation_plot")

flux_rel_err = np.linalg.norm(m_full - m_proj) / np.linalg.norm(m_proj)
flux_abs_err = np.linalg.norm(m_full - m_proj)
flux_results.append(["full", flux_rel_err, flux_abs_err])

sat_rel_err = np.linalg.norm(Sw_hist_ref[:, -1]  - Sw_ref) / np.linalg.norm(Sw_ref)
sat_abs_err = np.linalg.norm(Sw_hist_ref[:, -1]  - Sw_ref)
sat_results.append(["full", sat_rel_err, sat_abs_err])

# Compute Adaptive MFD
for tol in tol_list:

    cellMarking = classify_cells(cell_struct, face_struct, m_proj, p_proj, vertices, a, b, c, d, tol)
    m_num, p_num = solve_pressure(cell_struct, face_struct, cellMarking, inner_product, dt_pressure, g_c, eps_solver, gmres_niter, solver_type)
    Sw_hist, time_hist = solve_saturation(cell_struct, face_struct, m_num, Sw0, Sw_inj, tEnd = 0.25, dt=0.01)
    Sw_final = Sw_hist[:,-1]
    # write_vtu(f"output_fault/sat_tol_{tol:.1e}.vtu", vertices, cell_struct, face_struct, Sw_final, "saturation", "saturation_plot")

    flux_rel_err = np.linalg.norm(m_num - m_full) / np.linalg.norm(m_proj)
    flux_abs_err = np.linalg.norm(m_num - m_full)
    flux_results.append([tol, flux_rel_err, flux_abs_err])

    sat_rel_err = np.linalg.norm(Sw_final - Sw_ref) / np.linalg.norm(Sw_ref)
    sat_abs_err = np.linalg.norm(Sw_final - Sw_ref)
    sat_results.append([tol, sat_rel_err, sat_abs_err])

# Check flux relative/absolute error
print_pressure_err(flux_results)
plot_pressure_err(flux_results)

print_saturation_err(sat_results)
plot_saturation_err(sat_results)