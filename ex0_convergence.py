import os
import numpy as np
import matplotlib.pyplot as plt
from mesh_loader import load_mesh
from operators import createMmatrix, createBmatrix
from classification import classify_cells
from pressure_solver import solve_pressure

"""
ex0 convergence benchmark.

This script reproduces the manufactured-solution convergence test used in
the adaptive MFD paper. For a sequence of uniformly refined meshes, it

1. constructs the adaptive TPFA/MFD classification (GA and LA),
2. solves the mixed pressure system with a manufactured source term,
3. computes pressure and flux errors against the analytical solution,
4. evaluates the TPFA cell fraction for different tolerances,
5. generates convergence plots and TPFA-fraction plots.

Outputs are written separately to

    output/ex0_convergence_GA/
    output/ex0_convergence_LA/

for direct comparison of Global Adaptation (GA) and Local Adaptation (LA).
"""

def exact_pressure(x, y):
    return np.sin(2*np.pi*x) * np.sin(2*np.pi*y) + x

def source_term(cell_struct):
    centers = np.array([c["center"] for c in cell_struct])
    volumes = np.array([c["volume"] for c in cell_struct])

    x = centers[:, 0]
    y = centers[:, 1]

    return (-8.0 * np.pi**2 * np.sin(2*np.pi*x) * np.sin(2*np.pi*y) * volumes)

def exact_flux(face_struct):
    m_exact = np.zeros(len(face_struct))

    for f, face in enumerate(face_struct):
        x, y, _ = face["center"]
        n = np.asarray(face["normal"], dtype=float)
        n = n / np.linalg.norm(n)
        u = -np.array([2*np.pi*np.cos(2*np.pi*x)*np.sin(2*np.pi*y) + 1.0, 2*np.pi*np.sin(2*np.pi*x)*np.cos(2*np.pi*y), 0.0])

        m_exact[f] = face["area"] * np.dot(u, n)

    return m_exact


def set_ex0_physics(cell_struct, face_struct):
    for c in cell_struct:
        c["K"] = np.eye(3)
        c["phi"] = 1.0
        c["rho"] = 1.0

    for f in face_struct:
        x, y, _ = f["center"]

        f["rho"] = 1.0
        f["gravity"] = np.array([0.0, 0.0, 0.0])

        if len(f["cells"]) == 1:
            f["BC_pressure"] = exact_pressure(x, y)
            f["BC_flux"] = None
        else:
            f["BC_pressure"] = None
            f["BC_flux"] = None

    return cell_struct, face_struct


def main():

    mesh_names = ["ex0_h8", "ex0_h16", "ex0_h32", "ex0_h64", "ex0_h128", "ex0_h256", "ex0_h512"]
    tau_list = np.array([1.0, 1e-1, 1e-2, 1e-3, 1e-4])
    adaptation_levels = ["GA", "LA"]
    inner_product = "simple"
    solver_type = "iterative"

    for adaptation_level in adaptation_levels:

        h_list = []

        rel_p_errors = np.zeros((len(mesh_names), len(tau_list)))
        rel_m_errors = np.zeros((len(mesh_names), len(tau_list)))
        tpfa_fracs = np.zeros((len(mesh_names), len(tau_list)))

        out_dir = f"output/ex0_convergence_{adaptation_level}"
        os.makedirs(out_dir, exist_ok=True)

        for i, mesh_name in enumerate(mesh_names):

            print("\n" + "=" * 60)
            print(f"Running {mesh_name} ({adaptation_level})")
            print("=" * 60)

            cell_struct, face_struct, vertices, Lx, Ly, Lz = load_mesh(mesh_name)

            n_cells = len(cell_struct)
            n_faces = len(face_struct)

            h_list.append(1.0 / int(mesh_name.replace("ex0_h", "")))

            cell_struct, face_struct = set_ex0_physics(cell_struct, face_struct)
            cell_struct = createMmatrix(cell_struct, face_struct, ip_type="tpfa")
            cell_struct = createBmatrix(cell_struct)

            centers = np.array([c["center"] for c in cell_struct])

            p_exact = exact_pressure(centers[:, 0], centers[:, 1])
            m_exact = exact_flux(face_struct)
            f_src = source_term(cell_struct)

            # Linear field used for classification
            a, b, c, d = 1.0, 1.0, 0.0, 1.0
            p_lin = centers[:, 0] + centers[:, 1] + 1.0
            grad_lin = np.array([a, b, c])
            m_lin = np.zeros(n_faces)

            for f, face in enumerate(face_struct):

                n = np.asarray(face["normal"], dtype=float)
                n = n / np.linalg.norm(n)
                m_lin[f] = (-face["area"] * np.dot(grad_lin, n))
   
            for j, tau in enumerate(tau_list):

                print(f"\n--- tau = {tau:.1e} ---")

                cell_marking = classify_cells(cell_struct, face_struct, m_lin, p_lin, vertices, a, b, c, d, tau, adaptation_level, out_dir=out_dir)
                n_tpfa = n_cells - int(np.sum(cell_marking))
                tpfa_fracs[i, j] = n_tpfa / n_cells

                m_num, p_num, _, _ = solve_pressure(cell_struct, face_struct, cell_marking, inner_product=inner_product, dt_pressure=1.0, g_c=0.0, solver_type=solver_type, source_term=f_src)
                rel_p_errors[i, j] = (np.linalg.norm(p_num - p_exact) / np.linalg.norm(p_exact))
                rel_m_errors[i, j] = (np.linalg.norm(m_num - m_exact) / np.linalg.norm(m_exact))

                print(f"TPFA cells = {n_tpfa} / {n_cells}")
                print(f"Relative pressure error = {rel_p_errors[i,j]:.6e}")
                print(f"Relative flux error     = {rel_m_errors[i,j]:.6e}")


        h_list = np.array(h_list)

        # pressure plots
        plt.figure()

        for j, tau in enumerate(tau_list):
            plt.loglog(h_list, rel_p_errors[:, j], "-o", label=rf"$\tau={tau:.0e}$")

        ref2 = ((h_list / h_list[0]) ** 2 * np.min(rel_p_errors[0, :]) * 0.15)

        plt.loglog(h_list, ref2, "--k")
        plt.text(h_list[2], ref2[2], r"$\mathcal{O}(h^2)$", fontsize=12, ha="center", va="center", bbox=dict(facecolor="white", edgecolor="none", pad=0.2))
        plt.xlabel("Cell size $h$", fontsize=14)
        plt.ylabel("Relative pressure error", fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True)
        plt.legend(loc="best", frameon=True, fontsize=12)
        plt.savefig(os.path.join(out_dir, "pressure_convergence.png"), dpi=600, bbox_inches="tight")
        plt.close()

        # flux plots
        plt.figure()

        for j, tau in enumerate(tau_list):
            plt.loglog(h_list, rel_m_errors[:, j], "-o", label=rf"$\tau={tau:.0e}$")

        ref1 = ((h_list / h_list[0]) * np.min(rel_m_errors[0, :]) * 0.15)

        plt.loglog(h_list, ref1, "--k")
        x_text = np.sqrt(h_list[1] * h_list[2])
        y_text = np.sqrt(ref1[1] * ref1[2])
        plt.text(x_text, y_text, r"$\mathcal{O}(h)$", fontsize=12, ha="center", va="center", bbox=dict(facecolor="white", edgecolor="none", pad=0.2))
        plt.xlabel("Cell size $h$", fontsize=14)
        plt.ylabel("Relative flux error", fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True)
        plt.legend(loc="best", frameon=True, fontsize=12)
        plt.savefig(os.path.join(out_dir, "flux_convergence.png"), dpi=600, bbox_inches="tight")
        plt.close()

        # tpfa fraction plots
        plt.figure()

        for k, mesh_name in enumerate(mesh_names):
            plt.semilogx(tau_list, 100 * tpfa_fracs[k, :], "-o", label=mesh_name)

        plt.gca().invert_xaxis()
        plt.xlabel("Tolerance", fontsize=14)
        plt.ylabel("TPFA fraction (%)", fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True)
        plt.legend(loc="best", frameon=True, fontsize=12)
        plt.savefig(os.path.join(out_dir, "tpfa_fraction.png"), dpi=600, bbox_inches="tight")
        plt.close()

if __name__ == "__main__":
    main()