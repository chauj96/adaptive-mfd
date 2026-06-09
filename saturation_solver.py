import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
try:
    from petsc4py import PETSc
except ImportError:
    PETSc = None
import time

def solve_saturation(cell_struct, face_struct, m_num, Sw0, Sw_inj, tEnd, dt):

    n_cells = len(cell_struct)
    Sw = Sw0.copy()
    t = 0.0

    Sw_hist = [Sw.copy()]
    time_hist = [t]

    Vc = np.array([c["volume"] for c in cell_struct])
    phi = np.array([c["phi"] for c in cell_struct])
    acc = phi * Vc

    while t < tEnd:

        dt_step = min(dt, tEnd - t)

        rows = []
        cols = []
        vals = []

        rhs = (acc / dt_step) * Sw

        for c in range(n_cells):

            rows.append(c)
            cols.append(c)
            vals.append(acc[c] / dt_step)

            faces = cell_struct[c]["faces"]
            sgns  = cell_struct[c]["faces_orientation"]

            for k in range(len(faces)):

                f = faces[k]
                Fcf = sgns[k] * m_num[f]

                neigh = np.asarray(face_struct[f]["cells"], dtype=int)

                if neigh.size == 2:
                    other = neigh[0] if neigh[1] == c else neigh[1]

                    if Fcf >= 0:
                        rows.append(c)
                        cols.append(c)
                        vals.append(Fcf)
                    else:
                        rows.append(c)
                        cols.append(other)
                        vals.append(Fcf)

                elif neigh.size == 1:
                    if Fcf >= 0:
                        rows.append(c)
                        cols.append(c)
                        vals.append(Fcf)
                    else:
                        rhs[c] -= Fcf * Sw_inj

        A = coo_matrix((vals, (rows, cols)), shape=(n_cells, n_cells)).tocsr()

        # Sw = spsolve(A, rhs)
        st = time.time()
        A_petsc = PETSc.Mat().createAIJ(
            size=A.shape, csr=(A.indptr, A.indices, A.data))
        ksp = PETSc.KSP().create()
        ksp.setOperators(A_petsc)
        b = A_petsc.createVecLeft()
        b.array[:] = -rhs
        x = A_petsc.createVecRight()

        ksp.setType("preonly")
        ksp.getPC().setType("lu")
        ksp.getPC().setFactorSolverType("mumps")
        ksp.setConvergenceHistory()
        ksp.solve(b, x)
        Sw = x.array

        et = time.time()
        elapsed_time = et - st
        print("Linear solver time (Saturation):", elapsed_time, "seconds")

        Sw = np.maximum(0, np.minimum(1, Sw))

        t += dt_step

        Sw_hist.append(Sw.copy())
        time_hist.append(t)

    Sw_hist = np.array(Sw_hist).T
    time_hist = np.array(time_hist)

    return Sw_hist, time_hist