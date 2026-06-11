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


def compute_inner_product_batch(ip_type, C, N, K, v, Af, dim,
                                signf_vec=None, Nf_mat=None):
    """
    Batched computation of inverse transmissibility matrices.

    All inputs carry a leading batch dimension.  Every per-cell matrix
    operation (SVD, solve, trace, …) is executed with NumPy's native
    batched routines — no Python-level loop over cells.

    Parameters
    ----------
    ip_type : str
        Inner product type: "tpfa", "simple", "quasi_tpfa",
        "general_parametric", or "bdvlm".
    C : ndarray (batch, nf, dim)
        Face-center minus cell-center vectors.
    N : ndarray (batch, nf, dim)
        Weighted face normal vectors (area * sign * normal).
    K : ndarray (batch, dim, dim)
        Permeability tensors.
    v : ndarray (batch,)
        Cell volumes.
    Af : ndarray (batch, nf)
        Face areas.
    dim : int
        Spatial dimension.
    signf_vec : ndarray (batch, nf), optional
        Face orientation signs (required for "bdvlm").
    Nf_mat : ndarray (batch, nf, dim), optional
        Unit face normals (required for "bdvlm").

    Returns
    -------
    invT : ndarray (batch, nf, nf)
        Inverse transmissibility matrices.
    """
    nf = C.shape[1]

    if ip_type == "tpfa":
        # td[b,i] = sum_j C[b,i,j] * (N @ K)[b,i,j]  /  sum_j C[b,i,j]^2
        NK = np.matmul(N, K)                              # (batch, nf, dim)
        td = np.sum(C * NK, axis=2) / np.sum(C * C, axis=2)  # (batch, nf)
        inv_td = 1.0 / np.abs(td)                         # (batch, nf)
        eye = np.eye(nf)
        invT = inv_td[:, :, None] * eye[None, :, :]       # broadcast diagonal

    elif ip_type == "simple":
        # Consistency term:  C @ K^{-1} @ C^T / v
        C_T = C.transpose(0, 2, 1)                        # (batch, dim, nf)
        KinvCT = np.linalg.solve(K, C_T)                  # (batch, dim, nf)
        consistency = np.matmul(C, KinvCT) / v[:, None, None]

        # Stabilisation: (v / t_loc) * diag(1/a) @ U @ diag(1/a)
        t_loc = 6.0 * np.trace(K, axis1=1, axis2=2) / dim # (batch,)
        N_over_A = N / Af[:, :, None]                      # (batch, nf, dim)
        Q, _, _ = np.linalg.svd(N_over_A, full_matrices=False)  # Q: (batch, nf, k)
        QQT = np.matmul(Q, Q.transpose(0, 2, 1))          # (batch, nf, nf)
        U_mat = np.eye(nf)[None, :, :] - QQT
        Af_outer = Af[:, :, None] * Af[:, None, :]         # (batch, nf, nf)
        invT_reg = (v / t_loc)[:, None, None] * (U_mat / Af_outer)

        invT = consistency + invT_reg

    elif ip_type == "quasi_tpfa":
        # Consistency
        C_T = C.transpose(0, 2, 1)
        KinvCT = np.linalg.solve(K, C_T)
        consistency = np.matmul(C, KinvCT) / v[:, None, None]

        # Stabilisation: (v / 2) * P @ diag(1/diag(W)) @ P
        W = np.matmul(np.matmul(N, K), N.transpose(0, 2, 1))
        Qn, _, _ = np.linalg.svd(N, full_matrices=False)
        P = np.eye(nf)[None, :, :] - np.matmul(Qn, Qn.transpose(0, 2, 1))
        W_diag_inv = 1.0 / np.diagonal(W, axis1=1, axis2=2)  # (batch, nf)
        P_diW_P = np.matmul(P, W_diag_inv[:, :, None] * P)
        invT_reg = (v / 2.0)[:, None, None] * P_diW_P

        invT = consistency + invT_reg

    elif ip_type == "general_parametric":
        # Consistency
        C_T = C.transpose(0, 2, 1)
        KinvCT = np.linalg.solve(K, C_T)
        consistency = np.matmul(C, KinvCT) / v[:, None, None]

        # Stabilisation: (v / nf) * P @ diag(1/diag(W)) @ P
        W = np.matmul(np.matmul(N, K), N.transpose(0, 2, 1))
        Qn, _, _ = np.linalg.svd(N, full_matrices=False)
        P = np.eye(nf)[None, :, :] - np.matmul(Qn, Qn.transpose(0, 2, 1))
        W_diag_inv = 1.0 / np.diagonal(W, axis1=1, axis2=2)
        P_diW_P = np.matmul(P, W_diag_inv[:, :, None] * P)
        invT_reg = (v / nf)[:, None, None] * P_diW_P

        invT = consistency + invT_reg

    elif ip_type == "bdvlm":
        if signf_vec is None or Nf_mat is None:
            raise ValueError("bdvlm requires signf_vec and Nf_mat arguments.")

        R = Af[:, :, None] * C                              # (batch, nf, dim)
        Nbd = np.matmul(signf_vec[:, :, None] * Nf_mat, K)  # (batch, nf, dim)

        RT = R.transpose(0, 2, 1)                           # (batch, dim, nf)
        RT_Nbd = np.matmul(RT, Nbd)                         # (batch, dim, dim)
        M0 = np.matmul(R, np.linalg.solve(RT_Nbd, RT))      # (batch, nf, nf)

        NbdT = Nbd.transpose(0, 2, 1)
        NbdTNbd = np.matmul(NbdT, Nbd)                      # (batch, dim, dim)
        PN = np.eye(nf)[None, :, :] - np.matmul(
            Nbd, np.linalg.solve(NbdTNbd, NbdT))            # (batch, nf, nf)

        inv_Af_outer = (1.0 / Af[:, :, None]) * (1.0 / Af[:, None, :])
        invT = inv_Af_outer * (M0 + (1.0 / nf) * PN)

    else:
        raise ValueError(f"Unknown inner product type: {ip_type}")

    return invT
