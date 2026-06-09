import numpy as np
from scipy.sparse import diags, bmat, csr_matrix
from scipy.sparse.linalg import spsolve
from petsc4py import PETSc

def create_dummy_saddle_point(n_vel=100, n_pres=50):
    """
    Dummy saddle point system 
    
    [M   B^T] [u]   [f]
    [B    0 ] [p] = [g]
    
    M: velocity-velocity coupling (SPD)
    B: divergence operator
    """
    print("="*60)
    print("Creating dummy saddle point system")
    print("="*60)
    print(f"Velocity DOFs: {n_vel}")
    print(f"Pressure DOFs: {n_pres}")
    print(f"Total DOFs:    {n_vel + n_pres}")
    
    # M block: 1D Laplacian-like (SPD)
    # -u''(x) discretization
    main_diag = 2.0 * np.ones(n_vel)
    off_diag = -1.0 * np.ones(n_vel - 1)
    
    M = diags([off_diag, main_diag, off_diag], 
              [-1, 0, 1], 
              shape=(n_vel, n_vel), 
              format='csr')
    
    # B block: divergence/gradient operator
    # Simple finite difference: (u_{i+1} - u_i)
    B_rows = []
    B_cols = []
    B_vals = []
    
    for i in range(n_pres):
        # Each pressure couples to 2 velocities
        B_rows.append(i)
        B_cols.append(i)
        B_vals.append(1.0)
        
        if i < n_pres - 1:
            B_rows.append(i)
            B_cols.append(i + 1)
            B_vals.append(-1.0)
        
        # Extra coupling if n_vel > n_pres
        if i + n_pres < n_vel:
            B_rows.append(i)
            B_cols.append(i + n_pres)
            B_vals.append(0.5)
    
    B = csr_matrix((B_vals, (B_rows, B_cols)), 
                   shape=(n_pres, n_vel))
    
    # Full saddle point matrix
    # [M   B^T]
    # [B    0 ]
    A = bmat([
        [M,   B.T],
        [B,   None]
    ], format='csr')
    
    # RHS
    f = np.sin(np.linspace(0, 2*np.pi, n_vel))  # velocity RHS
    g = np.zeros(n_pres)  # pressure RHS (incompressibility)
    
    rhs = np.concatenate([f, g])
    
    print(f"Matrix size: {A.shape}")
    print(f"Matrix nnz:  {A.nnz}")
    print(f"RHS norm:    {np.linalg.norm(rhs):.2e}")
    print("="*60 + "\n")
    
    return A, rhs, n_vel, n_pres


def solve_with_direct(A, rhs):
    """Direct solver (baseline)"""
    print("="*60)
    print("DIRECT SOLVER (LU)")
    print("="*60)
    
    import time
    start = time.time()
    
    sol = spsolve(A, rhs)
    
    elapsed = time.time() - start
    
    residual = np.linalg.norm(A @ sol - rhs)
    rel_residual = residual / np.linalg.norm(rhs)
    
    print(f"Time:             {elapsed:.4f} seconds")
    print(f"Residual norm:    {residual:.2e}")
    print(f"Relative residual: {rel_residual:.2e}")
    print("="*60 + "\n")
    
    return sol


def solve_with_petsc_schur(A, rhs, n_vel, n_pres, verbose=True):
    """
    PETSc solver with Fieldsplit + Schur complement
    """
    print("="*60)
    print("PETSc SOLVER: GMRES + BLOCK SCHUR COMPLEMENT")
    print("="*60)
    
    import time
    start = time.time()
    
    # scipy sparse -> PETSc Mat
    A_csr = A.tocsr()
    petsc_mat = PETSc.Mat().createAIJ(
        size=A_csr.shape,
        csr=(A_csr.indptr, A_csr.indices, A_csr.data)
    )
    petsc_mat.assemblyBegin()
    petsc_mat.assemblyEnd()
    
    # Vectors
    petsc_rhs = PETSc.Vec().createSeq(len(rhs))
    petsc_rhs.setArray(rhs)
    
    petsc_sol = PETSc.Vec().createSeq(len(rhs))
    petsc_sol.set(0.0)
    
    # Index Sets for field splitting
    is_vel = PETSc.IS().createGeneral(list(range(n_vel)))
    is_pres = PETSc.IS().createGeneral(list(range(n_vel, n_vel + n_pres)))
    
    print(f"Velocity block: indices 0-{n_vel-1}")
    print(f"Pressure block: indices {n_vel}-{n_vel+n_pres-1}")
    
    # KSP solver
    ksp = PETSc.KSP().create()
    ksp.setOperators(petsc_mat)
    ksp.setType('fgmres')  # Flexible GMRES
    ksp.setTolerances(rtol=1e-10, atol=1e-15, max_it=200)
    
    # Monitor convergence
    if verbose:
        def monitor(ksp, its, rnorm):
            print(f"  Iteration {its:3d}: residual = {rnorm:.6e}")
        ksp.setMonitor(monitor)
    
    # Fieldsplit preconditioner
    pc = ksp.getPC()
    pc.setType('fieldsplit')
    pc.setFieldSplitType(PETSc.PC.CompositeType.SCHUR)
    
    # Schur factorization type:
    # DIAG:  [M  0 ] [I  M^{-1}B^T]
    #        [0  I ] [B     -S    ]
    # LOWER: [M  0 ] [I  M^{-1}B^T]
    #        [B  I ] [0     -S    ]
    # UPPER: [I  B^T] [M      0   ]
    #        [0   I ] [0     -S   ]
    # FULL:  [M  B^T] [I      0   ]
    #        [0  -S ] [0      I   ]
    
    pc.setFieldSplitSchurFactType(PETSc.PC.SchurFactType.LOWER)
    
    pc.setFieldSplitIS(('velocity', is_vel), ('pressure', is_pres))
    
    print(f"\nPreconditioner: Fieldsplit with Schur (LOWER)")
    
    # Setup sub-solvers for each block
    ksp.setUp()
    ksp_vel, ksp_pres = pc.getFieldSplitSubKSP()
    
    # Velocity block solver
    print(f"\nVelocity block solver:")
    ksp_vel.setType('preonly')
    pc_vel = ksp_vel.getPC()
    
    if n_vel < 5000:
        pc_vel.setType('lu')
        print(f"  Using direct LU")
    else:
        pc_vel.setType('hypre')
        pc_vel.setHYPREType('boomeramg')
        print(f"  Using Hypre BoomerAMG")
    
    # Pressure block (Schur complement) solver
    print(f"\nPressure/Schur block solver:")
    ksp_pres.setType('preonly')
    pc_pres = ksp_pres.getPC()
    pc_pres.setType('jacobi')
    print(f"  Using Jacobi preconditioner")
    
    # Alternative options for Schur complement:
    # pc_pres.setType('none')     # No preconditioning
    # pc_pres.setType('ilu')      # ILU
    # pc_pres.setType('hypre')    # AMG
    
    print("\n" + "-"*60)
    print("Starting solve...")
    print("-"*60 + "\n")
    
    # Solve
    ksp.solve(petsc_rhs, petsc_sol)
    
    elapsed = time.time() - start
    
    # Extract solution
    sol = petsc_sol.getArray().copy()
    
    # Convergence info
    converged_reason = ksp.getConvergedReason()
    iterations = ksp.getIterationNumber()
    residual_norm = ksp.getResidualNorm()
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Time:              {elapsed:.4f} seconds")
    print(f"Iterations:        {iterations}")
    print(f"Final residual:    {residual_norm:.6e}")
    print(f"Converged reason:  {converged_reason}", end="")
    
    if converged_reason > 0:
        print(" (✓ CONVERGED)")
    elif converged_reason == 0:
        print(" (still iterating)")
    else:
        print(" (✗ DIVERGED)")
    
    # Actual residual check
    actual_residual = np.linalg.norm(A @ sol - rhs)
    rel_residual = actual_residual / np.linalg.norm(rhs)
    
    print(f"Actual residual:   {actual_residual:.6e}")
    print(f"Relative residual: {rel_residual:.6e}")
    print("="*60 + "\n")
    
    # Cleanup
    petsc_mat.destroy()
    petsc_rhs.destroy()
    petsc_sol.destroy()
    is_vel.destroy()
    is_pres.destroy()
    ksp.destroy()
    
    return sol, iterations, elapsed


def compare_solutions(sol_direct, sol_petsc, n_vel):
    """
    두 solver 결과 비교
    """
    print("="*60)
    print("SOLUTION COMPARISON")
    print("="*60)
    
    # Split into velocity and pressure
    u_direct = sol_direct[:n_vel]
    p_direct = sol_direct[n_vel:]
    
    u_petsc = sol_petsc[:n_vel]
    p_petsc = sol_petsc[n_vel:]
    
    # Compute errors
    u_error = np.linalg.norm(u_direct - u_petsc)
    p_error = np.linalg.norm(p_direct - p_petsc)
    
    u_rel_error = u_error / (np.linalg.norm(u_direct) + 1e-15)
    p_rel_error = p_error / (np.linalg.norm(p_direct) + 1e-15)
    
    total_error = np.linalg.norm(sol_direct - sol_petsc)
    total_rel_error = total_error / np.linalg.norm(sol_direct)
    
    print(f"Velocity absolute error:     {u_error:.6e}")
    print(f"Velocity relative error:     {u_rel_error:.6e}")
    print(f"Pressure absolute error:     {p_error:.6e}")
    print(f"Pressure relative error:     {p_rel_error:.6e}")
    print(f"Total relative error:        {total_rel_error:.6e}")
    
    tolerance = 1e-6
    if total_rel_error < tolerance:
        print(f"\n✓ TEST PASSED (error < {tolerance})")
        status = True
    else:
        print(f"\n✗ TEST FAILED (error >= {tolerance})")
        status = False
    
    print("="*60 + "\n")
    
    return status


def main():
    """
    Main test function
    """
    print("\n" + "#"*60)
    print("# SADDLE POINT SYSTEM TEST")
    print("# PETSc with Block Schur Complement Preconditioner")
    print("#"*60 + "\n")
    
    # Create dummy system
    A, rhs, n_vel, n_pres = create_dummy_saddle_point(
        n_vel=200,
        n_pres=100
    )
    
    # Solve with direct solver
    sol_direct = solve_with_direct(A, rhs)
    
    # Solve with PETSc
    sol_petsc, iterations, time_petsc = solve_with_petsc_schur(
        A, rhs, n_vel, n_pres, 
        verbose=True
    )
    
    # Compare solutions
    test_passed = compare_solutions(sol_direct, sol_petsc, n_vel)
    
    # Summary
    print("="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Problem size:     {n_vel + n_pres} DOFs")
    print(f"PETSc iterations: {iterations}")
    print(f"Test status:      {'PASSED ✓' if test_passed else 'FAILED ✗'}")
    print("="*60)
    
    return test_passed


if __name__ == "__main__":
    success = main()
    
    # Exit with appropriate code
    import sys
    sys.exit(0 if success else 1)

# from petsc4py import PETSc

# print(f"PETSc version: {PETSc.Sys.getVersion()}")

# mat = PETSc.Mat().createAIJ([5, 5])
# mat.setUp()
# for i in range(5):
#     mat[i, i] = 2.0
#     if i > 0:
#         mat[i, i-1] = -1.0
#     if i < 4:
#         mat[i, i+1] = -1.0
# mat.assemblyBegin()
# mat.assemblyEnd()

# vec = PETSc.Vec().createSeq(5)
# vec.set(1.0)

# result = mat * vec
# print(f"Mat-vec result: {result.getArray()}")

# print("✓ PETSc basic test passed!")