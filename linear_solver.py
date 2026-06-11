import numpy as np
from scipy.sparse.linalg import spsolve

try:
    from petsc4py import PETSc
except ImportError:
    PETSc = None

import time

# Linear solver interface:
# - direct sparse solve via SciPy
# - optional iterative refinement
# - PETSc/MUMPS backend for large-scale problems

def solve_linear_system(A, b, solver_type="direct", label="", refinement_iters=0):
    """
    Solve A x = b using the specified back-end.

    Parameters
    ----------
    A : scipy.sparse matrix (CSR)
        System matrix.
    b : np.ndarray
        Right-hand side vector.
    solver_type : str
        "direct" — SciPy spsolve (with optional iterative refinement).
        "petsc"  — PETSc MUMPS LU factorization.
    label : str
        Name printed alongside timing output (e.g. "Pressure", "Saturation").
    refinement_iters : int
        Number of iterative-refinement steps applied after the initial solve.
        Only used when solver_type="direct".

    Returns
    -------
    x : np.ndarray
        Solution vector.
    """

    if solver_type == "direct":
        t0 = time.time()
        x = spsolve(A, b)
        print(f"[Timer] Direct solve ({label} initial): {time.time() - t0:.4f}s")

        if refinement_iters > 0:
            t0 = time.time()
            for _ in range(refinement_iters):
                r = A @ x - b
                x -= spsolve(A, r)
            print(f"[Timer] Iterative refinement ({label} {refinement_iters} iters): {time.time() - t0:.4f}s")

    elif solver_type in ("petsc", "iterative"):
        if PETSc is None:
            raise ImportError(
                "petsc4py is not installed. "
                "Install petsc/petsc4py or use solver_type='direct'."
            )

        st = time.time()

        A_petsc = PETSc.Mat().createAIJ(
            size=A.shape, csr=(A.indptr, A.indices, A.data))
        ksp = PETSc.KSP().create()
        ksp.setOperators(A_petsc)
        b_petsc = A_petsc.createVecLeft()
        b_petsc.array[:] = b
        x_petsc = A_petsc.createVecRight()

        ksp.setType("preonly")
        ksp.getPC().setType("lu")
        ksp.getPC().setFactorSolverType("mumps")
        ksp.setConvergenceHistory()
        ksp.solve(b_petsc, x_petsc)
        x = x_petsc.array.copy()

        A_petsc.destroy()
        b_petsc.destroy()
        x_petsc.destroy()
        ksp.destroy()

        elapsed_time = time.time() - st
        print(f"Linear solver time ({label}): {elapsed_time:.6f} seconds")

    else:
        raise ValueError(
            f"Unknown solver_type '{solver_type}'. Use 'direct' or 'petsc'."
        )

    return x
