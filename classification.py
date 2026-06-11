import numpy as np
import os
from io_utils import write_vtu

# Compute residual-based indicator and classify cells as TPFA or MFD
# Also exports classification results for visualization

def classify_cells(cell_struct, face_struct, m_proj, p_proj, vertices, a, b, c, d, tol, out_dir="output"):
    """
    Classify cells as TPFA or MFD using the residual-based indicator.

    Parameters
    ----------
    cell_struct : list
        Cell geometry and operator data.

    face_struct : list
        Face geometry data.

    m_proj : ndarray
        Projected analytical flux field.

    p_proj : ndarray
        Projected analytical pressure field.

    tol : float
        Residual threshold used for TPFA/MFD classification.

    Returns
    -------
    cellMarking : ndarray
        Cell classification:
        0 = TPFA
        1 = MFD
    """

    n_cells = len(cell_struct)
    n_faces = len(face_struct)

    # Full MFD (will be used as a reference saturation solution)
    if isinstance(tol, str) and tol == "full":
        cellMarking = np.ones(n_cells, dtype=int)

        print(f"tol = full | TPFA cells = 0 / {n_cells}")

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        filename = os.path.join(out_dir, "mesh_full.vtu")
        write_vtu(filename, vertices, cell_struct, face_struct, cellMarking, "cellMarking", "cell_plot")

        return cellMarking

    face_centers = np.array([face_struct[f]["center"] for f in range(n_faces)])
    d_all = a * face_centers[:,0] + b * face_centers[:,1] + c * face_centers[:,2] + d

    # Output dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Create residual vector
    res_3D = np.zeros(n_faces)

    # Loop over all cells
    for cn in range(n_cells):

        face_ids = np.array(cell_struct[cn]["faces"]).astype(int)
        signs = np.array(cell_struct[cn]["faces_orientation"]).astype(float)

        M_K = signs * cell_struct[cn]["M"]
        B_K = cell_struct[cn]["B"]

        mK = signs * m_proj[face_ids]
        pK = p_proj[cn]
        d_K = signs * d_all[face_ids]

        DeltaP_K = -B_K * pK + d_K
        denom = np.linalg.norm(DeltaP_K)

        R_K = (M_K @ mK - B_K * pK + d_K) / denom

        res_3D[face_ids] += R_K

    # Face threshold
    face_exceeds = np.abs(res_3D) > tol

    # Cell aggregation
    cellMarking = np.zeros(n_cells, dtype=int)

    for c in range(n_cells):
        face_ids = np.array(cell_struct[c]["faces"]).astype(int)
        if np.any(face_exceeds[face_ids]):
            cellMarking[c] = 1   # MFD
        else:
            cellMarking[c] = 0   # TPFA

    tpfa_count = n_cells - np.sum(cellMarking)

    print("\n" + "=" * 50)
    if isinstance(tol, str):
        print(f"tol = {tol} | TPFA cells = {tpfa_count} / {n_cells}")
    else:
        print(f"tol = {tol:.1e} | TPFA cells = {tpfa_count} / {n_cells}")

    # Export VTU
    if isinstance(tol, str):
        filename = os.path.join(out_dir, f"mesh_{tol}.vtu")
    else:
        filename = os.path.join(out_dir, f"mesh_tol_{tol:.1e}.vtu")

    write_vtu(filename, vertices, cell_struct, face_struct, cellMarking, "cellMarking", "cell_plot")

    return cellMarking