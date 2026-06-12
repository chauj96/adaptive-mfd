import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

try:
    import hypredrive
    # Force-load the native extension now, before petsc4py: petsc's conda
    # libHYPRE shares the SONAME of hypredrive's (newer) bundled libHYPRE,
    # and the dynamic loader keeps whichever loads first for the whole
    # process. hypredrive needs its own.
    from hypredrive import driver as _  # noqa: F401
except ImportError:
    hypredrive = None

try:
    from petsc4py import PETSc
except ImportError:
    PETSc = None

import atexit
import time

# Linear solver interface:
# - direct sparse solve via SciPy
# - optional iterative refinement
# - PETSc/MUMPS backend for large-scale problems
# - hypredrive (hypre) backend: two-level MGR for the indefinite MFD
#   systems, BoomerAMG for definite ones

# Persistent hypredrive drivers, one per (configuration kind, object name),
# e.g. ("mgr", "pressure") and ("amg", "saturation"). Reusing a single
# HYPREDRV handle across solves avoids per-call create/parse/destroy
# overhead and defers the statistics summary to one table per driver —
# "STATISTICS SUMMARY for <name>:" — printed when the drivers are closed
# at interpreter exit.
_hypredrive_drivers = {}


def _hypredrive_driver(kind, name):
    key = (kind, name)
    drv = _hypredrive_drivers.get(key)
    if drv is None:
        if kind == "mgr":
            opts = hypredrive_mgr_options(name=name)
        else:
            opts = hypredrive_amg_options(name=name)
        drv = hypredrive.HypreDrive(options=opts)
        if not _hypredrive_drivers:
            # Registered after hypredrive's own session-finalize atexit hook,
            # so (LIFO) the drivers close — flushing the statistics summaries —
            # before the hypredrive runtime is torn down.
            atexit.register(_hypredrive_close_all)
        _hypredrive_drivers[key] = drv
    return drv


def _hypredrive_close_all():
    for drv in _hypredrive_drivers.values():
        drv.close()
    _hypredrive_drivers.clear()


def hypredrive_mgr_options(rel_tol=1e-10, max_iter=250, krylov_dim=100, print_level=0,
                           statistics=2, name="pressure"):
    """
    hypredrive options for the indefinite (saddle-point) MFD system:
    FGMRES (with dofmap-magnitude scaling) preconditioned by two-level MGR.

    At MGR level 0 the F-points are the face-flux DOFs (dofmap label 1),
    relaxed with one BoomerAMG V-cycle on the flux block; the resulting
    coarse (Galerkin RAP) operator on the cell-pressure DOFs (label 0)
    is an approximate Schur complement solved with one BoomerAMG V-cycle.
    """
    return {
        "general": {"name": name, "statistics": statistics, "exec_policy": "host"},
        "linear_system": {"init_guess_mode": "zeros"},
        "solver": {
            "fgmres": {
                "max_iter": max_iter,
                "krylov_dim": krylov_dim,
                "relative_tol": rel_tol,
                "absolute_tol": 0.0,
                "print_level": print_level,
            }
        },
        "preconditioner": {
            "mgr": {
                "tolerance": 0.0,
                "max_iter": 1,
                "print_level": 0,
                "coarse_th": 0.0,
                "level": {
                    "0": {
                        "f_dofs": [1],
                        "f_relaxation": {
                            "amg": {
                                "tolerance": 0.0,
                                "max_iter": 1,
                                "print_level": 0,
                            },
                        },
                        "g_relaxation": "none",
                        "restriction_type": "injection",
                        "prolongation_type": "jacobi",
                        "coarse_level_type": "rap",
                    },
                },
                "coarsest_level": {
                    "amg": {
                        "tolerance": 0.0,
                        "max_iter": 1,
                        "print_level": 0,
                    },
                },
            }
        },
    }


def hypredrive_amg_options(rel_tol=1e-10, max_iter=250, krylov_dim=60, print_level=0,
                           statistics=2, name="saturation"):
    """
    hypredrive options for definite systems (e.g. the saturation transport
    matrix): GMRES preconditioned by one BoomerAMG V-cycle.
    """
    return {
        "general": {"name": name, "statistics": statistics, "exec_policy": "host"},
        "linear_system": {"init_guess_mode": "zeros"},
        "solver": {
            "gmres": {
                "max_iter": max_iter,
                "krylov_dim": krylov_dim,
                "relative_tol": rel_tol,
                "absolute_tol": 0.0,
                "print_level": print_level,
            }
        },
        "preconditioner": {
            "reuse": "always",
            "amg": {
                "tolerance": 0.0,
                "max_iter": 1,
                "print_level": 0,
            }
        },
    }


def solve_linear_system(A, b, solver_type="direct", label="", refinement_iters=0,
                        dofmap=None):
    """
    Solve A x = b using the specified back-end.

    Parameters
    ----------
    A : scipy.sparse matrix (CSR)
        System matrix.
    b : np.ndarray
        Right-hand side vector.
    solver_type : str
        "direct"     — SciPy spsolve (with optional iterative refinement).
        "petsc"      — PETSc MUMPS LU factorization.
        "hypredrive" — hypre (via hypredrive) GMRES; uses two-level MGR
                       when a dofmap is given (indefinite MFD system),
                       BoomerAMG otherwise.
    label : str
        Name printed alongside timing output (e.g. "Pressure", "Saturation").
    refinement_iters : int
        Number of iterative-refinement steps applied after the initial solve.
        Only used when solver_type="direct".
    dofmap : np.ndarray, optional
        Per-row integer labels used by MGR to define the block structure:
        1 for face-flux DOFs (fine grid), 0 for cell-pressure DOFs (coarse
        grid). Only used when solver_type="hypredrive".

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

    elif solver_type == "hypredrive":
        if hypredrive is None:
            raise ImportError(
                "hypredrive is not installed. "
                "Install the hypredrive Python package "
                "(https://github.com/hypre-space/hypredrive) "
                "or use solver_type='direct'."
            )

        st = time.time()

        A = csr_matrix(A)
        n_rows = A.shape[0]
        kind = "mgr" if dofmap is not None else "amg"
        # The application-level label ("Pressure", "Saturation", ...) names
        # the hypredrive object, so each system class gets its own driver
        # and its own statistics summary table.
        name = label.lower() if label else kind
        drv = _hypredrive_driver(kind, name)
        try:
            drv.set_matrix_from_csr(A, row_start=0, row_end=n_rows - 1)
            drv.set_rhs(np.asarray(b, dtype=np.float64))
            if dofmap is not None:
                drv.set_dofmap(np.asarray(dofmap, dtype=np.intc))
            drv.solve()
            iterations = drv.last_iterations
            x = drv.get_solution().copy()
        except Exception:
            # Drop the (possibly inconsistent) handle so the next call
            # starts from a fresh driver.
            _hypredrive_drivers.pop((kind, name), None)
            drv.close()
            raise

        elapsed_time = time.time() - st
        print(f"Linear solver time ({label}, hypredrive, "
              f"{iterations} iters): {elapsed_time:.6f} seconds")

    else:
        raise ValueError(
            f"Unknown solver_type '{solver_type}'. "
            "Use 'direct', 'petsc', or 'hypredrive'."
        )

    return x
