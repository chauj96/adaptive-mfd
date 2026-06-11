import numpy as np
from scipy.sparse import coo_matrix

from linear_solver import solve_linear_system


def solve_saturation(cell_struct, face_struct, m_num, Sw0, Sw_inj, tEnd, dt, solver_type="petsc"):

    n_cells = len(cell_struct)
    n_faces = len(face_struct)
    Sw = Sw0.copy()
    t = 0.0

    Sw_hist = [Sw.copy()]
    time_hist = [t]

    Vc = np.array([c["volume"] for c in cell_struct])
    phi = np.array([c["phi"] for c in cell_struct])
    acc = phi * Vc

    # ---- Pre-compute connectivity (constant across timesteps) ----

    # Flat cell-face pairs
    cell_face_lists = [np.asarray(c["faces"], dtype=int) for c in cell_struct]
    cell_sign_lists = [np.asarray(c["faces_orientation"], dtype=float).reshape(-1)
                       for c in cell_struct]
    face_counts = np.array([len(fl) for fl in cell_face_lists], dtype=int)

    cell_flat = np.repeat(np.arange(n_cells), face_counts)   # (total_cf,)
    face_flat = np.concatenate(cell_face_lists)               # (total_cf,)
    sign_flat = np.concatenate(cell_sign_lists)               # (total_cf,)

    # Face neighbor arrays
    face_cells_list = [np.asarray(f["cells"], dtype=int) for f in face_struct]
    face_ncells = np.array([len(fc) for fc in face_cells_list], dtype=int)
    face_cell0 = np.array([fc[0] for fc in face_cells_list], dtype=int)
    face_cell1 = np.array([fc[1] if len(fc) > 1 else -1
                           for fc in face_cells_list], dtype=int)

    # Signed flux per cell-face pair (constant)
    Fcf = sign_flat * m_num[face_flat]

    # Classify each cell-face pair
    ncells_cf = face_ncells[face_flat]
    is_interior = ncells_cf == 2
    is_boundary = ncells_cf == 1
    pos_flux = Fcf >= 0

    int_pos = is_interior & pos_flux    # interior, upwind from self
    int_neg = is_interior & ~pos_flux   # interior, upwind from neighbour
    bnd_pos = is_boundary & pos_flux    # boundary outflow
    bnd_neg = is_boundary & ~pos_flux   # boundary inflow

    # "Other" cell for interior faces
    c0_cf = face_cell0[face_flat]
    c1_cf = face_cell1[face_flat]
    other_cf = np.where(c1_cf == cell_flat, c0_cf, c1_cf)

    # ---- Pre-build constant COO entries (off-diagonal + flux-diagonal) ----

    # Interior positive: (c, c, Fcf)
    ip_rows, ip_cols, ip_vals = cell_flat[int_pos], cell_flat[int_pos], Fcf[int_pos]
    # Interior negative: (c, other, Fcf)
    in_rows, in_cols, in_vals = cell_flat[int_neg], other_cf[int_neg], Fcf[int_neg]
    # Boundary positive: (c, c, Fcf)
    bp_rows, bp_cols, bp_vals = cell_flat[bnd_pos], cell_flat[bnd_pos], Fcf[bnd_pos]

    offdiag_rows = np.concatenate([ip_rows, in_rows, bp_rows])
    offdiag_cols = np.concatenate([ip_cols, in_cols, bp_cols])
    offdiag_vals = np.concatenate([ip_vals, in_vals, bp_vals])

    # Boundary inflow RHS fix (constant)
    rhs_bnd_fix = np.zeros(n_cells)
    np.add.at(rhs_bnd_fix, cell_flat[bnd_neg], -Fcf[bnd_neg] * Sw_inj)

    # Pre-allocate full COO arrays (diagonal block + off-diagonal block)
    n_offdiag = len(offdiag_rows)
    n_total = n_cells + n_offdiag
    diag_idx = np.arange(n_cells)

    all_rows = np.empty(n_total, dtype=int)
    all_cols = np.empty(n_total, dtype=int)
    all_vals = np.empty(n_total)

    all_rows[:n_cells] = diag_idx
    all_cols[:n_cells] = diag_idx
    all_rows[n_cells:] = offdiag_rows
    all_cols[n_cells:] = offdiag_cols
    all_vals[n_cells:] = offdiag_vals         # constant portion

    # ---- Time loop (only updates diagonal values and RHS) ----

    while t < tEnd:

        dt_step = min(dt, tEnd - t)

        all_vals[:n_cells] = acc / dt_step    # update diagonal

        A = coo_matrix((all_vals, (all_rows, all_cols)),
                       shape=(n_cells, n_cells)).tocsr()

        rhs = (acc / dt_step) * Sw + rhs_bnd_fix

        Sw = solve_linear_system(A, rhs, solver_type=solver_type,
                                 label="Saturation")

        Sw = np.clip(Sw, 0, 1)

        t += dt_step

        Sw_hist.append(Sw.copy())
        time_hist.append(t)

    Sw_hist = np.array(Sw_hist).T
    time_hist = np.array(time_hist)

    return Sw_hist, time_hist