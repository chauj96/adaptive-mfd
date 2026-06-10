import numpy as np
from scipy.sparse import coo_matrix, bmat, diags
from scipy.sparse.linalg import spsolve

try:
    from petsc4py import PETSc
except ImportError:
    PETSc = None

from operators import orth
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

        if cellMarking[cc] == 0:
            td = np.sum(C * (N @ K), axis=1) / np.sum(C * C, axis=1)
            invT = np.diag(1.0 / np.abs(td))
        else:
            if inner_product == "simple":
                t_loc = 6.0 * np.trace(K) / dim
                Q = orth(N / Af_vec[:, None])
                U = np.eye(cell_nf) - Q @ Q.T
                di = np.diag(1.0 / Af_vec)
                invT_reg = (v / t_loc) * (di @ U @ di)
                invT = (C @ np.linalg.solve(K, C.T)) / v + invT_reg

            elif inner_product == "quasi_tpfa":
                W = N @ K @ N.T
                Qn = orth(N)
                P = np.eye(Qn.shape[0]) - Qn @ Qn.T
                diW = np.diag(1.0 / np.diag(W))
                invT_reg = (v / 2.0) * (P @ diW @ P)
                invT = (C @ np.linalg.solve(K, C.T)) / v + invT_reg

            elif inner_product == "general_parametric":
                W = N @ K @ N.T
                Qn = orth(N)
                P = np.eye(cell_nf) - Qn @ Qn.T
                diW = np.diag(1.0 / np.diag(W))
                invT_reg = (v / cell_nf) * (P @ diW @ P)
                invT = (C @ np.linalg.solve(K, C.T)) / v + invT_reg

            elif inner_product == "bdvlm":
                R = np.diag(Af_vec) @ C
                Nbd = (signf_vec[:, None] * Nf_mat) @ K
                M0 = R @ np.linalg.solve(R.T @ Nbd, R.T)
                NbdTNbd = Nbd.T @ Nbd
                PN = np.eye(cell_nf) - Nbd @ np.linalg.solve(NbdTNbd, Nbd.T)
                invT = np.diag(1.0 / Af_vec) @ (M0 + (1.0 / cell_nf) * PN) @ np.diag(1.0 / Af_vec)

            else:
                raise ValueError(f"Unknown inner_product: {inner_product}")

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

    # ======== NEED TO IMPROVE THEIR SPEED ===========
    t0 = time.time()
    if solver_type == "direct":
        sol3 = spsolve(matrix, -RHS)
        print(f"[Timer] Direct solve (initial): {time.time() - t0:.4f}s")
        
        t0 = time.time()
        for _ in range(3):
            r = matrix @ sol3 + RHS
            sol3 -= spsolve(matrix, r)
        print(f"[Timer] Iterative refinement (3 iters): {time.time() - t0:.4f}s")
    
    # ======== NEED TO IMPROVE ===============
    else:
        if PETSc is None:
            raise ImportError(
                "petsc4py is not installed. "
                "Install petsc/petsc4py or use solver_type='direct'."
            )
        
        # t_setup = time.time()
        # A_petsc = PETSc.Mat().createAIJ(
        #     size=matrix.shape, csr=(matrix.indptr, matrix.indices, matrix.data))
        # b_petsc = PETSc.Vec().createWithArray(-RHS)
        # x_petsc = A_petsc.createVecRight()
        #
        # ksp = PETSc.KSP().create()
        # ksp.setOperators(A_petsc)
        # ksp.setType('gmres')
        # ksp.setGMRESRestart(30)
        # ksp.setTolerances(rtol=1e-10, atol=1e-14, max_it=100)
        #
        # pc = ksp.getPC()
        # pc.setType('fieldsplit')
        # pc.setFieldSplitType(PETSc.PC.CompositeType.SCHUR)
        #
        # is_flux = PETSc.IS().createGeneral(range(n_faces))
        # is_pressure = PETSc.IS().createGeneral(range(n_faces, n_faces + n_cells))
        # pc.setFieldSplitIS(('flux', is_flux), ('pressure', is_pressure))
        #
        # pc.setFieldSplitSchurPreType(PETSc.PC.SchurPreType.SELF)
        # ksp.setFromOptions()
        # print(f"[Timer] PETSc setup: {time.time() - t_setup:.4f}s")
        #
        # t_solve = time.time()
        # ksp.solve(b_petsc, x_petsc)
        # sol3 = x_petsc.getArray().copy()
        # print(f"[Timer] PETSc initial solve: {time.time() - t_solve:.4f}s")
        # print(f"  - Iterations: {ksp.getIterationNumber()}, Residual: {ksp.getResidualNorm():.2e}")
        #
        # t_refine = time.time()
        # for i in range(1):
        #     r = matrix @ sol3 + RHS
        #     b_corr = PETSc.Vec().createWithArray(-r)
        #     x_corr = A_petsc.createVecRight()
        #     ksp.solve(b_corr, x_corr)
        #     sol3 += x_corr.getArray()
        #     b_corr.destroy()
        #     x_corr.destroy()
        #     print(f"  - Refinement {i+1}: {ksp.getIterationNumber()} iters")
        #
        # print(f"[Timer] PETSc refinement (1 iters): {time.time() - t_refine:.4f}s")
        # print(f"  - Final residual: {np.linalg.norm(matrix @ sol3 + RHS) / np.linalg.norm(RHS):.2e}")
        #
        # A_petsc.destroy()
        # b_petsc.destroy()
        # x_petsc.destroy()
        # ksp.destroy()

        # solving ls
        st = time.time()

        A_petsc = PETSc.Mat().createAIJ(
            size=matrix.shape, csr=(matrix.indptr, matrix.indices, matrix.data))
        ksp = PETSc.KSP().create()
        ksp.setOperators(A_petsc)
        b = A_petsc.createVecLeft()
        b.array[:] = -RHS
        x = A_petsc.createVecRight()


        ksp.setType("preonly")
        ksp.getPC().setType("lu")
        ksp.getPC().setFactorSolverType("mumps")
        ksp.setConvergenceHistory()
        ksp.solve(b, x)
        sol3 = x.array

        et = time.time()
        elapsed_time = et - st
        print("Linear solver time (Pressure):", elapsed_time, "seconds")

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

