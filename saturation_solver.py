import numpy as np
from scipy.sparse import coo_matrix

from linear_solver import solve_linear_system
import time

def solve_saturation(cell_struct, face_struct, m_num, Sw0, Sw_inj, tEnd, dt, solver_type="petsc"):

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

        Sw = solve_linear_system(A, rhs, solver_type=solver_type,
                                 label="Saturation")

        Sw = np.clip(Sw,0,1)

        t += dt_step

        Sw_hist.append(Sw.copy())
        time_hist.append(t)

    Sw_hist = np.array(Sw_hist).T
    time_hist = np.array(time_hist)

    return Sw_hist, time_hist