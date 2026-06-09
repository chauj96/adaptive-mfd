import numpy as np

# Initialize physical parameters and boundary conditions
# Also provides analytical reference pressure and flux fields

def initPhysicalParams(cell_struct, face_struct, Lx, Ly, Lz, perm_tensor="identity", bc_option="linear"):

    # Physical constants 
    phi_vals = 0.3
    rho_vals = 1000.0
    g_val = 0.0
    gravity_dir = np.array([0.0, 0.0, -1.0])
    tol = 1e-6

    n_faces = len(face_struct)
    n_cells = len(cell_struct)

    for c in range(n_cells):

        z = cell_struct[c]["center"][2]

        if perm_tensor == "identity":

            K = np.eye(3)

        elif perm_tensor == "layered_isotropy":

            if z < 0.13:
                k_base = 1.0e-4
            elif z < 0.26:
                k_base = 1.0
            else:
                k_base = 1.0e3

            K = k_base * np.eye(3)

        elif perm_tensor == "het_anisotropy":

            if z < 0.13:
                k_base = 1.0e-4
            elif z < 0.26:
                k_base = 1.0
            else:
                k_base = 1.0e3

            kx = k_base
            ky = 100.0 * kx
            kz = 0.01 * kx

            K = np.diag([kx, ky, kz])

        else:
            raise ValueError(f"Unknown perm_tensor: {perm_tensor}")

        cell_struct[c]["K"] = K
        cell_struct[c]["phi"] = phi_vals
        cell_struct[c]["rho"] = rho_vals

    # Face centers
    f_centers = np.array([face_struct[f]["center"] for f in range(n_faces)])

    # Boundary detection 
    west_idx = np.where(np.abs(f_centers[:, 0] - 0.0) < tol)[0]
    east_idx = np.where(np.abs(f_centers[:, 0] - Lx) < tol)[0]

    south_idx = np.where(np.abs(f_centers[:, 1] - 0.0) < tol)[0]
    north_idx = np.where(np.abs(f_centers[:, 1] - Ly) < tol)[0]

    bottom_idx = np.where(np.abs(f_centers[:, 2] - 0.0) < tol)[0]
    top_idx = np.where(np.abs(f_centers[:, 2] - Lz) < tol)[0]

    BC_Dirichlet_map = {}
    BC_Neumann_map = {}

    if bc_option == "linear":

        grad_pref = np.array([-1.0 / Lx, 0.0, 0.0])

        # Assign Dirichlet boundary condition
        BC_Dirichlet_map = {**{int(f): 1.0 for f in west_idx}, **{int(f): 0.0 for f in east_idx}}

        # Assign Neumann boundary condition
        neumann_faces = np.concatenate([south_idx, north_idx, bottom_idx, top_idx])
        BC_Neumann_map = {int(f): 0.0 for f in neumann_faces}

    # TO DO: NEED TO DEBUG
    elif bc_option == "corner2corner":

        grad_pref = np.array([-1.0 / Lx, -1.0 / Ly, -1.0 / Lz])

        boundary_faces = np.unique(np.concatenate([west_idx, east_idx, south_idx, north_idx, bottom_idx, top_idx]))

        cell_centers = np.array([cell_struct[c]["center"] for c in range(n_cells)])

        inlet_target = np.array([0.0, 0.0, Lz])
        outlet_target = np.array([Lx, Ly, 0.0])

        inlet_cell = np.argmin(np.linalg.norm(cell_centers - inlet_target, axis=1))
        outlet_cell = np.argmin(np.linalg.norm(cell_centers - outlet_target, axis=1))

        inlet_faces_all = np.array(cell_struct[inlet_cell]["faces"])
        outlet_faces_all = np.array(cell_struct[outlet_cell]["faces"])

        inlet_faces = inlet_faces_all[np.isin(inlet_faces_all, boundary_faces)]
        outlet_faces = outlet_faces_all[np.isin(outlet_faces_all, boundary_faces)]

        dirichlet_faces = np.unique(np.concatenate([inlet_faces, outlet_faces]))

        BC_Dirichlet_map = {
            int(f): grad_pref @ f_centers[f] + 1.0 
            for f in dirichlet_faces
        }

        neumann_faces = np.setdiff1d(boundary_faces, dirichlet_faces)
        BC_Neumann_map = {int(f): 0.0 for f in neumann_faces}
    else:
        raise ValueError(f"Unknown bc_option: {bc_option}")

    # Assign face properties
    gravity_vec = g_val * gravity_dir
    for f in range(n_faces):
        face_struct[f]["gravity"] = gravity_vec
        face_struct[f]["rho"] = rho_vals
        face_struct[f]["BC_flux"] = BC_Neumann_map.get(f, None)
        face_struct[f]["BC_pressure"] = BC_Dirichlet_map.get(f, None)

    phys = {
        "grad_pref": grad_pref,
        "perm_tensor": perm_tensor,
    }

    return cell_struct, face_struct, phys


def projectAnalyticalField(cell_struct, face_struct, phys, a, b, c, d):

    nCells = len(cell_struct)
    nFaces = len(face_struct)

    gradp = np.array([a, b, c])

    # Pressure projection
    cell_centers = np.array([cell_struct[k]["center"] for k in range(nCells)])
    p_proj = cell_centers @ gradp + d

    # Flux projection (Harmonic average)
    m_proj = np.zeros(nFaces)

    for f in range(nFaces):

        n_f = np.asarray(face_struct[f]["normal"], dtype=float)
        n_f = n_f / np.linalg.norm(n_f)

        A_f = face_struct[f]["area"]

        neigh_cells = np.asarray(face_struct[f]["cells"], dtype=int)
        valid_cells = neigh_cells[neigh_cells >= 0]

        if len(valid_cells) == 1:

            Kf = cell_struct[valid_cells[0]]["K"]

        else:

            cL = valid_cells[0]
            cR = valid_cells[1]

            KL = cell_struct[cL]["K"]
            KR = cell_struct[cR]["K"]

            xL = np.asarray(cell_struct[cL]["center"], dtype=float)
            xR = np.asarray(cell_struct[cR]["center"], dtype=float)
            xf = np.asarray(face_struct[f]["center"], dtype=float)

            dL = np.linalg.norm(xf - xL)
            dR = np.linalg.norm(xf - xR)

            kL = np.diag(KL)
            kR = np.diag(KR)

            kf = (dL + dR) / (dL / kL + dR / kR)

            Kf = np.diag(kf)

        m_proj[f] = -A_f * np.dot(Kf @ gradp, n_f)

    return m_proj, p_proj