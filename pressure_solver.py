import numpy as np
from scipy.sparse import coo_matrix, bmat, diags
from inner_products import compute_inner_product_batch
from linear_solver import solve_linear_system
import time

def solve_pressure(cell_struct, face_struct, cellMarking, inner_product="simple", 
                   dt_pressure=1.0, g_c=0.0, solver_type="direct"):
    """
    Solve the mixed pressure system using adaptive TPFA/MFD operators.

    Parameters
    ----------
    cell_struct : list
        Cell geometry, physical properties, and local operator data.

    face_struct : list
        Face geometry, boundary-condition, and physical data.

    cellMarking : ndarray
        Cell classification array:
        0 = TPFA cell,
        1 = MFD cell.

    inner_product : str
        Inner-product type used for cells marked as MFD.

    dt_pressure : float
        Pressure time-step size used in the accumulation term.

    g_c : float
        Gravity coefficient.

    solver_type : str
        Linear solver backend, e.g. "direct", "petsc", or "iterative".

    Returns
    -------
    m : ndarray
        Numerical face fluxes.

    p : ndarray
        Numerical cell pressures.

    nnz_M : int
        Number of nonzeros in the global inner-product block M.
    """
    t_total = time.time()

    n_cells = len(cell_struct)
    n_faces = len(face_struct)
    dim = len(face_struct[0]["center"])

    # Precompute global face arrays
    face_centers = np.array([f["center"] for f in face_struct])
    face_normals = np.array([f["normal"] for f in face_struct])
    face_areas = np.array([f["area"] for f in face_struct])

    # Pre-extract cell data into contiguous arrays
    cell_centers_arr = np.array([c["center"] for c in cell_struct])
    cell_K_arr = np.array([c["K"] for c in cell_struct])
    cell_volumes_arr = np.array([c["volume"] for c in cell_struct])
    cell_face_lists = [np.asarray(c["faces"], dtype=int) for c in cell_struct]
    cell_sign_lists = [np.asarray(c["faces_orientation"], dtype=float).reshape(-1)
                       for c in cell_struct]

    face_counts = np.array([len(fl) for fl in cell_face_lists], dtype=int)
    unique_nf_vals = np.unique(face_counts)

    rows_list = []
    cols_list = []
    vals_list = []

    for nf in unique_nf_vals:
        cell_ids = np.where(face_counts == nf)[0]
        batch_size = len(cell_ids)

        # Stack per-cell arrays into batched 3D tensors
        face_ids_batch = np.array([cell_face_lists[c] for c in cell_ids])   # (batch, nf)
        signs_batch = np.array([cell_sign_lists[c] for c in cell_ids])      # (batch, nf)

        Cc_batch = cell_centers_arr[cell_ids]    # (batch, dim)
        K_batch = cell_K_arr[cell_ids]           # (batch, dim, dim)
        v_batch = cell_volumes_arr[cell_ids]     # (batch,)

        # Gather face geometry via advanced indexing
        Cf_batch = face_centers[face_ids_batch]  # (batch, nf, dim)
        Nf_batch = face_normals[face_ids_batch]  # (batch, nf, dim)
        Af_batch = face_areas[face_ids_batch]    # (batch, nf)

        # Vectorised geometry
        C_batch = Cf_batch - Cc_batch[:, None, :]                          # (batch, nf, dim)
        df_norms = np.linalg.norm(C_batch, axis=2)                         # (batch, nf)
        signf_batch = np.sign(
            np.sum((C_batch / df_norms[:, :, None]) * Nf_batch, axis=2))    # (batch, nf)
        N_batch = Af_batch[:, :, None] * signf_batch[:, :, None] * Nf_batch # (batch, nf, dim)

        # Compute invT — split by cellMarking for TPFA / MFD
        marking_batch = cellMarking[cell_ids]
        tpfa_mask = marking_batch == 0
        mfd_mask = ~tpfa_mask

        invT_batch = np.empty((batch_size, nf, nf))

        if np.any(tpfa_mask):
            invT_batch[tpfa_mask] = compute_inner_product_batch(
                "tpfa", C_batch[tpfa_mask], N_batch[tpfa_mask],
                K_batch[tpfa_mask], v_batch[tpfa_mask],
                Af_batch[tpfa_mask], dim)

        if np.any(mfd_mask):
            invT_batch[mfd_mask] = compute_inner_product_batch(
                inner_product, C_batch[mfd_mask], N_batch[mfd_mask],
                K_batch[mfd_mask], v_batch[mfd_mask],
                Af_batch[mfd_mask], dim,
                signf_vec=signf_batch[mfd_mask], Nf_mat=Nf_batch[mfd_mask])

        # Batched COO assembly
        sign_mat = signs_batch[:, :, None] * signs_batch[:, None, :]  # (batch, nf, nf)
        product = sign_mat * invT_batch
        # Column-major flatten per cell: transpose last two dims then row-major reshape
        vals_flat = product.transpose(0, 2, 1).reshape(batch_size, -1)  # (batch, nf*nf)

        ii, jj = np.meshgrid(np.arange(nf), np.arange(nf), indexing="ij")
        gi_local = ii.ravel('F')
        gj_local = jj.ravel('F')
        gi_all = face_ids_batch[:, gi_local]  # (batch, nf*nf)
        gj_all = face_ids_batch[:, gj_local]  # (batch, nf*nf)

        rows_list.append(gi_all.ravel())
        cols_list.append(gj_all.ravel())
        vals_list.append(vals_flat.ravel())

    rows = np.concatenate(rows_list)
    cols = np.concatenate(cols_list)
    vals = np.concatenate(vals_list)

    M = coo_matrix((vals, (rows, cols)), shape=(n_faces, n_faces)).tocsr()
    # Remove explicit zeros so that M.nnz reflects the actual sparsity pattern
    M.eliminate_zeros()
    B = buildBmatrix(cell_struct, face_struct)
    T = buildTmatrix(cell_struct)

    A_full = bmat([[M, -B.T], [B, diags(np.zeros(n_cells))]], format="csr")
    matrix = A_full.copy()

    rhs_Dirichlet = dirichletBoundary(cell_struct, face_struct)
    matrix, rhs_Neumann, _ = neumannBoundary(matrix, face_struct)

    p_n = np.zeros(n_cells)
    g_vec = np.array([0.0, 0.0, -g_c])
    face_rho = np.array([f["rho"] for f in face_struct])
    f_g = -face_rho * (face_normals @ g_vec) * face_areas

    BC_face_flux_ids = np.array(
        [i for i, s in enumerate(face_struct) if s.get("BC_flux") is not None], dtype=int
    )
    BC_face_flux_vals = np.array([face_struct[i]["BC_flux"] for i in BC_face_flux_ids])

    RHS = np.concatenate([f_g + rhs_Dirichlet, (1.0 / dt_pressure) * (T @ p_n)])
    matrix, RHS = enforcePrescribedDOFsStrong(BC_face_flux_ids, BC_face_flux_vals, matrix, RHS)

    sol3 = solve_linear_system(matrix, -RHS, solver_type=solver_type,
                               label="Pressure", refinement_iters=3)

    print(f"[Timer] TOTAL solve_pressure: {time.time() - t_total:.4f}s")

    return sol3[:n_faces], sol3[n_faces:], M.nnz


def buildBmatrix(cell_struct, face_struct):
    """
    Build the discrete divergence operator B.
    """
    n_cells = len(cell_struct)
    n_faces = len(face_struct)

    total_nnz = sum(len(cell["faces"]) for cell in cell_struct)

    rows = np.zeros(total_nnz, dtype=int)
    cols = np.zeros(total_nnz, dtype=int)
    vals = np.zeros(total_nnz, dtype=float)

    idx = 0
    for k, cell in enumerate(cell_struct):
        face_ids = np.array(cell["faces"], dtype=int)
        signs = np.array(cell["faces_orientation"], dtype=float)

        m = len(face_ids)
        rows[idx:idx+m] = k
        cols[idx:idx+m] = face_ids
        vals[idx:idx+m] = signs
        idx += m

    return coo_matrix((vals, (rows, cols)), shape=(n_cells, n_faces)).tocsr()


def buildTmatrix(cell_struct):
    """
    Build the accumulation matrix T = diag(phi * volume).
    """
    n_cells = len(cell_struct)
    
    phi_vals = np.array([cell["phi"] for cell in cell_struct])
    vol_vals = np.array([cell["volume"] for cell in cell_struct])
    t_vec = phi_vals * vol_vals

    return diags(t_vec, 0, shape=(n_cells, n_cells))


def dirichletBoundary(cell_struct, face_struct):
    """
    Assemble the RHS contribution from prescribed pressure boundaries.
    """
    # Apply Dirichlet boundary conditions
    n_faces = len(face_struct)
    rhs_Dirichlet = np.zeros(n_faces)

    # Build BC map once
    bc_map = {i: f["BC_pressure"] for i, f in enumerate(face_struct) 
              if "BC_pressure" in f and f["BC_pressure"] is not None}
    
    if not bc_map:
        return rhs_Dirichlet

    # Apply BCs
    bc_face_set = set(bc_map.keys()) 
    
    for cell in cell_struct:
        face_ids = cell["faces"]
        signs = cell["faces_orientation"]
        
        # Only check faces that might have BC
        for i, fid in enumerate(face_ids):
            if fid in bc_face_set: 
                rhs_Dirichlet[fid] = signs[i] * bc_map[fid]

    return rhs_Dirichlet

def neumannBoundary(A, face_struct):
    """
    Modify the system matrix to impose prescribed flux boundaries.
    """
    # Apply Neumann boundary conditions
    n_faces = len(face_struct)
    rhs_BC = np.zeros(n_faces)

    # BC face identification
    f_ids = np.array([i for i, x in enumerate(face_struct) if x.get("BC_flux") is not None], dtype=int)
    f_vals = np.array([face_struct[i]["BC_flux"] for i in f_ids])

    diag_mask = np.ones(A.shape[0])
    diag_mask[f_ids] = 0.0
    P = diags(diag_mask).tocsr()  # Projection matrix
    I_boundary = diags(1.0 - diag_mask).tocsr()
    A = P @ A @ P + I_boundary
    A.eliminate_zeros()

    rhs_BC[f_ids] = f_vals
    return A, rhs_BC, f_ids


def buildGravityRHS(face_struct, g):
    """
    Build the gravity contribution to the flux RHS.
    """
    # Build gravity RHS (fully vectorized)
    n_faces = len(face_struct)
    
    rho_vals = np.array([f["rho"] for f in face_struct])
    normals = np.array([f["normal"] for f in face_struct])
    areas = np.array([f["area"] for f in face_struct])
    
    g_vec = np.array([0.0, 0.0, -g])
    f_g = -rho_vals * (normals @ g_vec) * areas

    return f_g

def enforcePrescribedDOFsStrong(prescribedIdx, prescribedVal, A, b):
    """
    Enforce prescribed degrees of freedom by strong elimination.
    """
    nUnknowns = A.shape[0]

    # Convert scalar to array if needed
    if np.isscalar(prescribedVal):
        prescribedVal = np.full(len(prescribedIdx), prescribedVal)
    else:
        prescribedVal = np.array(prescribedVal).reshape(-1)

    isFree = np.ones(nUnknowns, dtype=bool)
    isFree[prescribedIdx] = False

    selectFree = diags(isFree.astype(float), 0, shape=(nUnknowns, nUnknowns))
    selectPrescribed = diags((~isFree).astype(float), 0, shape=(nUnknowns, nUnknowns))

    xPrescribed = np.zeros(nUnknowns)
    xPrescribed[prescribedIdx] = prescribedVal
    A_freeRows = selectFree @ A
    b = b - A_freeRows @ xPrescribed
    b[prescribedIdx] = prescribedVal

    A = A_freeRows @ selectFree + selectPrescribed
    
    return A.tocsr(), b


def block_prec(r, F_mm, A_pm, F_S, num_m_dofs):
    """
    Apply a block preconditioner for the mixed saddle-point system.
    """
    # Split residual into momentum and pressure parts
    r1 = r[:num_m_dofs]
    r2 = r[num_m_dofs:]

    # Solve flux block
    y1 = F_mm.solve(r1)
    
    # Solve Schur complement for pressure
    y2 = F_S.solve(r2 - A_pm @ y1)

    return np.concatenate([y1, y2])

