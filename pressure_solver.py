import numpy as np
from scipy.sparse import coo_matrix, bmat, diags

from inner_products import compute_inner_product
from linear_solver import solve_linear_system
import time

def solve_pressure(cell_struct, face_struct, cellMarking, inner_product="simple", 
                   dt_pressure=1.0, g_c=0.0, solver_type="direct"):
    t_total = time.time()

    n_cells = len(cell_struct)
    n_faces = len(face_struct)
    dim = len(face_struct[0]["center"])

    # Precompute global face arrays
    face_centers = np.array([f["center"] for f in face_struct])
    face_normals = np.array([f["normal"] for f in face_struct])
    face_areas = np.array([f["area"] for f in face_struct])

    face_counts = np.array([len(c["faces"]) for c in cell_struct], dtype=int)
    total_nnz = int(np.sum(face_counts * face_counts))

    rows = np.zeros(total_nnz, dtype=int)
    cols = np.zeros(total_nnz, dtype=int)
    vals = np.zeros(total_nnz, dtype=float)
    idx = 0

    # Cache for meshgrid indices
    gi_cache = {}
    gj_cache = {}
    for nf in np.unique(face_counts):
        ii, jj = np.meshgrid(np.arange(nf), np.arange(nf), indexing="ij")
        gi_cache[nf] = ii.ravel('F')
        gj_cache[nf] = jj.ravel('F')

    # Cell loop
    for cc in range(n_cells):
        face_ids = np.asarray(cell_struct[cc]["faces"], dtype=int)
        cell_nf = face_ids.size

        Cc = np.asarray(cell_struct[cc]["center"]).reshape(-1)
        K = np.asarray(cell_struct[cc]["K"])
        v = cell_struct[cc]["volume"]
        signs = np.asarray(cell_struct[cc]["faces_orientation"]).reshape(-1)
        
        Cf_mat = face_centers[face_ids]
        Nf_mat = face_normals[face_ids]
        Af_vec = face_areas[face_ids]

        C = Cf_mat - Cc
        df_norms = np.linalg.norm(C, axis=1)
        signf_vec = np.sign(np.sum((C / df_norms[:, None]) * Nf_mat, axis=1))
        N = Af_vec[:, None] * signf_vec[:, None] * Nf_mat

        ip_type = "tpfa" if cellMarking[cc] == 0 else inner_product
        invT = compute_inner_product(ip_type, C, N, K, v, Af_vec, dim,
                                     signf_vec=signf_vec, Nf_mat=Nf_mat)

        sign_mat = np.outer(signs, signs)

        gi = face_ids[gi_cache[cell_nf]]
        gj = face_ids[gj_cache[cell_nf]]
        n2 = cell_nf * cell_nf

        rows[idx:idx+n2] = gi
        cols[idx:idx+n2] = gj
        vals[idx:idx+n2] = (sign_mat * invT).ravel('F')
        idx += n2

    M = coo_matrix((vals, (rows, cols)), shape=(n_faces, n_faces)).tocsr()
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
    print("=" * 60)

    return sol3[:n_faces], sol3[n_faces:]


def buildBmatrix(cell_struct, face_struct):
    # Build divergence operator B (fully vectorized assembly)
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
    # Build time-stepping matrix T (fully vectorized)
    n_cells = len(cell_struct)
    
    phi_vals = np.array([cell["phi"] for cell in cell_struct])
    vol_vals = np.array([cell["volume"] for cell in cell_struct])
    t_vec = phi_vals * vol_vals

    return diags(t_vec, 0, shape=(n_cells, n_cells))


def dirichletBoundary(cell_struct, face_struct):
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
    # Apply Neumann boundary conditions
    n_faces = len(face_struct)
    rhs_BC = np.zeros(n_faces)

    # BC face identification
    f_ids = np.array([i for i, x in enumerate(face_struct) if x.get("BC_flux") is not None], dtype=int)
    f_vals = np.array([face_struct[i]["BC_flux"] for i in f_ids])

    # Modify matrix rows/cols for prescribed fluxes
    # A = A.tolil()
    # A[f_ids, :] = 0
    # A[:, f_ids] = 0
    # A[f_ids, f_ids] = 1.0
    # A = A.tocsr()

    # # Ensure A is already in CSR format
    # A = A.tocsr()

    # A[f_ids, :] = 0
    # A[:, f_ids] = 0
    # A[f_ids, f_ids] = 1.0
    # A.eliminate_zeros()

    diag_mask = np.ones(A.shape[0])
    diag_mask[f_ids] = 0.0
    P = diags(diag_mask).tocsr()  # Projection matrix
    I_boundary = diags(1.0 - diag_mask).tocsr()
    A = P @ A @ P + I_boundary
    A.eliminate_zeros()

    rhs_BC[f_ids] = f_vals
    return A, rhs_BC, f_ids


def buildGravityRHS(face_struct, g):
    # Build gravity RHS (fully vectorized)
    n_faces = len(face_struct)
    
    rho_vals = np.array([f["rho"] for f in face_struct])
    normals = np.array([f["normal"] for f in face_struct])
    areas = np.array([f["area"] for f in face_struct])
    
    g_vec = np.array([0.0, 0.0, -g])
    f_g = -rho_vals * (normals @ g_vec) * areas

    return f_g

def enforcePrescribedDOFsStrong(prescribedIdx, prescribedVal, A, b):
    
    # Enforce prescribed DOFs via strong elimination. Modifies matrix rows/cols to impose boundary conditions directly.

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
    # Block preconditioner application for saddle point systems.
    
    # Split residual into momentum and pressure parts
    r1 = r[:num_m_dofs]
    r2 = r[num_m_dofs:]

    # Solve flux block
    y1 = F_mm.solve(r1)
    
    # Solve Schur complement for pressure
    y2 = F_S.solve(r2 - A_pm @ y1)

    return np.concatenate([y1, y2])

