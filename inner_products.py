import numpy as np


def orth(A):
    """Orthonormal basis for the column space of A."""
    U, _, _ = np.linalg.svd(A, full_matrices=False)
    return U


def compute_inner_product(ip_type, C, N, K, v, Af_vec, dim,
                          signf_vec=None, Nf_mat=None):
    """
    Compute the inverse transmissibility matrix (invT) for a single cell.

    Parameters
    ----------
    ip_type : str
        Inner product type: "tpfa", "simple", "quasi_tpfa",
        "general_parametric", or "bdvlm".
    C : ndarray (nf, dim)
        Face-center minus cell-center vectors.
    N : ndarray (nf, dim)
        Weighted (area * sign * normal) face normal vectors.
    K : ndarray (dim, dim)
        Permeability tensor.
    v : float
        Cell volume.
    Af_vec : ndarray (nf,)
        Face areas.
    dim : int
        Spatial dimension.
    signf_vec : ndarray (nf,), optional
        Face orientation signs (required for "bdvlm").
    Nf_mat : ndarray (nf, dim), optional
        Unit face normals (required for "bdvlm").

    Returns
    -------
    invT : ndarray (nf, nf)
        Inverse transmissibility matrix.
    """
    cell_nf = C.shape[0]

    if ip_type == "tpfa":
        td = np.sum(C * (N @ K), axis=1) / np.sum(C * C, axis=1)
        invT = np.diag(1.0 / np.abs(td))

    elif ip_type == "simple":
        t_loc = 6.0 * np.trace(K) / dim
        Q = orth(N / Af_vec[:, None])
        U = np.eye(cell_nf) - Q @ Q.T
        di = np.diag(1.0 / Af_vec)
        invT_reg = (v / t_loc) * (di @ U @ di)
        invT = (C @ np.linalg.solve(K, C.T)) / v + invT_reg

    elif ip_type == "quasi_tpfa":
        W = N @ K @ N.T
        Qn = orth(N)
        P = np.eye(Qn.shape[0]) - Qn @ Qn.T
        diW = np.diag(1.0 / np.diag(W))
        invT_reg = (v / 2.0) * (P @ diW @ P)
        invT = (C @ np.linalg.solve(K, C.T)) / v + invT_reg

    elif ip_type == "general_parametric":
        W = N @ K @ N.T
        Qn = orth(N)
        P = np.eye(cell_nf) - Qn @ Qn.T
        diW = np.diag(1.0 / np.diag(W))
        invT_reg = (v / cell_nf) * (P @ diW @ P)
        invT = (C @ np.linalg.solve(K, C.T)) / v + invT_reg

    elif ip_type == "bdvlm":
        if signf_vec is None or Nf_mat is None:
            raise ValueError("bdvlm requires signf_vec and Nf_mat arguments.")
        R = np.diag(Af_vec) @ C
        Nbd = (signf_vec[:, None] * Nf_mat) @ K
        M0 = R @ np.linalg.solve(R.T @ Nbd, R.T)
        NbdTNbd = Nbd.T @ Nbd
        PN = np.eye(cell_nf) - Nbd @ np.linalg.solve(NbdTNbd, Nbd.T)
        invT = np.diag(1.0 / Af_vec) @ (M0 + (1.0 / cell_nf) * PN) @ np.diag(1.0 / Af_vec)

    else:
        raise ValueError(f"Unknown inner product type: {ip_type}")

    return invT
