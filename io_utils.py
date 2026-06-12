import numpy as np
import matplotlib.pyplot as plt

# Visualization and reporting utilities:
# - export cell data to VTU format for ParaView
# - print pressure, saturation, and sparsity statistics
# - generate error-versus-tolerance plots

def write_vtu(filename, V3, cell_struct, face_struct, cellData, cellDataName, flag):
    """
    Export cell-centered data to a VTU file for ParaView visualization.

    Parameters
    ----------
    filename : str
        Output VTU file.

    cellData : ndarray
        Cell-centered data to export.

    cellDataName : str
        Name of the exported variable.

    flag : str
        "cell_plot"       -> integer TPFA/MFD classification.
        "saturation_plot" -> floating-point saturation field.
    """

    nCells = len(cell_struct)

    nCells = len(cell_struct)
    nPts = V3.shape[0]

    VTK_POLYHEDRON = 42

    connectivity = []
    offsets = []
    types = [VTK_POLYHEDRON] * nCells

    faces_all = []
    faceoffsets = []

    off_conn = 0
    off_face = 0

    for c in range(nCells):

        fids = np.array(cell_struct[c]["faces"]).astype(int)

        vids = []
        for f in fids:
            vids.extend(face_struct[f]["verts"])

        vids = np.unique(vids)
        vids0 = vids

        connectivity.extend(vids0)
        off_conn += len(vids0)
        offsets.append(off_conn)

        rec = [len(fids)]

        for f in fids:
            v = np.array(face_struct[f]["verts"])
            rec.extend([len(v)])
            rec.extend(v.tolist())

        faces_all.extend(rec)
        off_face += len(rec)
        faceoffsets.append(off_face)

    with open(filename, "w") as fid:

        fid.write('<?xml version="1.0"?>\n')
        fid.write('<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">\n')
        fid.write('<UnstructuredGrid>\n')
        fid.write(f'<Piece NumberOfPoints="{nPts}" NumberOfCells="{nCells}">\n')

        # Points
        fid.write('<Points>\n')
        fid.write('<DataArray type="Float64" NumberOfComponents="3" format="ascii">\n')
        for p in V3:
            fid.write(f"{p[0]} {p[1]} {p[2]}\n")
        fid.write('</DataArray>\n</Points>\n')

        # Cells
        fid.write('<Cells>\n')

        fid.write('<DataArray type="Int32" Name="connectivity" format="ascii">\n')
        fid.write(" ".join(map(str, connectivity)))
        fid.write('\n</DataArray>\n')

        fid.write('<DataArray type="Int32" Name="offsets" format="ascii">\n')
        fid.write(" ".join(map(str, offsets)))
        fid.write('\n</DataArray>\n')

        fid.write('<DataArray type="UInt8" Name="types" format="ascii">\n')
        fid.write(" ".join(map(str, types)))
        fid.write('\n</DataArray>\n')

        fid.write('<DataArray type="Int32" Name="faces" format="ascii">\n')
        fid.write(" ".join(map(str, faces_all)))
        fid.write('\n</DataArray>\n')

        fid.write('<DataArray type="Int32" Name="faceoffsets" format="ascii">\n')
        fid.write(" ".join(map(str, faceoffsets)))
        fid.write('\n</DataArray>\n')

        fid.write('</Cells>\n')

        # Cell data
        fid.write(f'<CellData Scalars="{cellDataName}">\n')

        if flag == "cell_plot":
            fid.write(f'<DataArray type="Int32" Name="{cellDataName}" format="ascii">\n')
            fid.write(" ".join(map(str, cellData.astype(int))))
            
        elif flag == "saturation_plot":
            fid.write(f'<DataArray type="Float64" Name="{cellDataName}" format="ascii">\n')
            fid.write(" ".join(map(str, cellData.astype(float))))

        fid.write('\n</DataArray>\n</CellData>\n')

        fid.write('</Piece>\n</UnstructuredGrid>\n</VTKFile>\n')

    # print(f"Wrote {filename}")

def print_flux_err(results):

    print("\n=== Pressure/Flux Solver Results ===")
    print(f"{'tol':>10} | {'rel error':>12} | {'abs error':>12}")
    print("-"*40)

    for tol, rel_err, abs_err in results:
        if isinstance(tol, str):
            print(f"{tol:>10} | {rel_err:12.3e} | {abs_err:12.3e}")
        else:
            print(f"{tol:10.1e} | {rel_err:12.3e} | {abs_err:12.3e}")


def plot_flux_err(results, filename=None):

    numeric = [r for r in results if not isinstance(r[0], str)]

    tol = np.array([r[0] for r in numeric])

    rel_err = np.array([r[1] for r in numeric])
    abs_err = np.array([r[2] for r in numeric])

    plt.figure()

    plt.loglog(tol, rel_err, '-o', label="Relative Error")
    plt.loglog(tol, abs_err, '-s', label="Absolute Error")

    # reference slope
    plt.loglog(tol, tol, '--k')

    idx = max(len(tol) - 3, 0)

    plt.text(tol[idx], tol[idx], r"$\mathcal{O}(\tau)$", fontsize=12, ha='center', va='center', bbox=dict(facecolor='white', edgecolor='none', pad=0.2))

    plt.xlabel(r"Tolerance $\tau$", fontsize=14)
    plt.ylabel("Flux Error", fontsize=14)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.grid(True)

    plt.legend(loc="upper left", frameon=True, fontsize=12)

    if filename is not None:
        plt.savefig(filename, dpi=600, bbox_inches="tight")

def print_saturation_err(results):

    print("\n=== Saturation Solver Results ===")
    print(f"{'tol':>10} | {'rel error':>12} | {'abs error':>12}")
    print("-"*40)

    for tol, rel_err, abs_err in results:
        if isinstance(tol, str):
            print(f"{tol:>10} | {rel_err:12.3e} | {abs_err:12.3e}")
        else:
            print(f"{tol:10.1e} | {rel_err:12.3e} | {abs_err:12.3e}")

def plot_saturation_err(results, filename=None):

    numeric = [r for r in results if not isinstance(r[0], str)]

    tol = np.array([r[0] for r in numeric])
    rel_err = np.array([r[1] for r in numeric])
    abs_err = np.array([r[2] for r in numeric])

    plt.figure()

    plt.loglog(tol, rel_err, '-o', label="Relative Error")
    plt.loglog(tol, abs_err, '-s', label="Absolute Error")

    # reference slope
    plt.loglog(tol, tol, '--k')

    idx = max(len(tol) - 3, 0)

    plt.text(tol[idx], tol[idx], r"$\mathcal{O}(\tau)$", fontsize=12, ha='center', va='center', bbox=dict(facecolor='white', edgecolor='none', pad=0.2))

    plt.xlabel(r"Tolerance $\tau$", fontsize=14)
    plt.ylabel("Saturation Error", fontsize=14)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.grid(True)

    plt.legend(loc="upper left", frameon=True, fontsize=12)

    if filename is not None:
        plt.savefig(filename, dpi=600, bbox_inches="tight")

def print_sparsity_info(results):

    print("\n=== Sparsity Statistics ===")
    print(f"{'tol':>10} | {'TPFA cells':>10} | {'nnz(M)':>12} | {'Memory Mb (M)':>12} | {'sparsity red (%)':>18} | {'memory red (%)':>18}")
    print("-" * 101)

    for tol, n_tpfa_cells, nnz_M, memory_mb_M, sparsity_reduction, memory_reduction in results:
        if isinstance(tol, str):
            print(
                f"{tol:>10} | {int(n_tpfa_cells):10d} | {nnz_M:12d} | {memory_mb_M:18.2f} | {sparsity_reduction:18.2f} | {memory_reduction:18.2f}"
            )
        else:
            print(
                f"{tol:10.1e} | {int(n_tpfa_cells):10d} | {nnz_M:12d} | {memory_mb_M:18.2f} | {sparsity_reduction:18.2f} | {memory_reduction:18.2f}"
            )