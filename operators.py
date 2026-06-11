import numpy as np
from inner_products import compute_inner_product, orth

# Construct local MFD operators (M, B) for each cell
# Supports TPFA, simple, and general parametric inner products

def createMmatrix(cell_struct, face_struct, ip_type="tpfa"):
    n_cells = len(cell_struct)
    dim = len(face_struct[0]["center"])

    for c in range(n_cells):

        face_ids = np.array(cell_struct[c]["faces"]).astype(int)
        cell_nf = len(face_ids)

        Cc = np.array(cell_struct[c]["center"]).reshape(-1)
        K = np.array(cell_struct[c]["K"])
        v = cell_struct[c]["volume"]
        signs = np.array(cell_struct[c]["faces_orientation"]).reshape(-1)

        # Build local C, N, a
        C = np.zeros((cell_nf, dim))
        N = np.zeros((cell_nf, dim))
        a = np.zeros(cell_nf)

        for k in range(cell_nf):
            f = face_ids[k]

            Cf = np.array(face_struct[f]["center"]).reshape(-1)
            Nf = np.array(face_struct[f]["normal"]).reshape(-1)
            Af = face_struct[f]["area"]

            df = Cf - Cc
            dot_val = np.dot(df, Nf) / np.linalg.norm(df)
            signf = np.sign(dot_val)

            if signf != signs[k]:
                raise ValueError(f"Orientation mismatch in cell {c}")

            C[k, :] = df
            N[k, :] = Af * signf * Nf
            a[k] = Af

        # Build M (=invT)
        invT = compute_inner_product(ip_type, C, N, K, v, a, dim)

        # Store
        cell_struct[c]["M"] = invT

    return cell_struct


def createBmatrix(cell_struct):

    for c in range(len(cell_struct)):
        cell_struct[c]["B"] = np.array(cell_struct[c]["faces_orientation"]).reshape(-1)

    return cell_struct