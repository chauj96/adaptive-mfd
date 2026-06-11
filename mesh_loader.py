import os
import numpy as np
import zipfile
import xml.etree.ElementTree as ET
import pyvista as pv

# Mesh loading utilities:
# - loads benchmark meshes from VTU files
# - reconstructs cell and face connectivity
# - computes face orientations for each cell
# - returns geometric data structures used by the solvers
 
def load_mesh(mesh_name):
    """
    Load one of the benchmark meshes used in the adaptive MFD examples.
    """

    if mesh_name == "twoFaults":
        return load_vtu("meshes/twoFaults/fault_mesh.vtu")

    elif mesh_name == "spe11b":
        return load_vtu("meshes/spe11b/spe11b_mesh.vtu")

    elif mesh_name == "fullyPoly":
        return load_vtu("meshes/fullyPolyhedral/fullyPoly_mesh.vtu")

    elif mesh_name.startswith("ex0_"):

        ensure_ex0_meshes()

        if mesh_name == "ex0_h8":
            return load_vtu("meshes/ex0/ex0_h8.vtu")

        elif mesh_name == "ex0_h16":
            return load_vtu("meshes/ex0/ex0_h16.vtu")

        elif mesh_name == "ex0_h32":
            return load_vtu("meshes/ex0/ex0_h32.vtu")

        elif mesh_name == "ex0_h64":
            return load_vtu("meshes/ex0/ex0_h64.vtu")

        elif mesh_name == "ex0_h128":
            return load_vtu("meshes/ex0/ex0_h128.vtu")

        elif mesh_name == "ex0_h256":
            return load_vtu("meshes/ex0/ex0_h256.vtu")

        elif mesh_name == "ex0_h512":
            return load_vtu("meshes/ex0/ex0_h512.vtu")

    else:
        raise ValueError(f"Unknown mesh: {mesh_name}")

def load_vtu(filepath):
    """
    Load a VTU mesh and reconstruct cell and face data structures.

    Parameters
    ----------
    filepath : str
        Path to the VTU mesh file.

    Returns
    -------
    cell_struct : list
        Cell geometry and connectivity information.

    face_struct : list
        Face geometry and connectivity information.

    vertices : ndarray
        Vertex coordinates.

    Lx, Ly, Lz : float
        Domain dimensions computed from the mesh bounds.
    """

    mesh = pv.read(filepath)
    vertices = mesh.points
    n_cells = mesh.n_cells

    root = ET.parse(filepath).getroot()

    def read_array(name, dtype=float):
        for da in root.iter("DataArray"):
            if da.attrib.get("Name") == name:
                data = np.fromstring(da.text, sep=" ", dtype=dtype)
                ncomp = int(da.attrib.get("NumberOfComponents", "1"))
                if ncomp > 1:
                    data = data.reshape(-1, ncomp)
                return data
        raise KeyError(name)

    cell_centers = read_array("cellCenter", float)
    volumes = read_array("cellVolume", float)

    face_centers = read_array("faceCenter", float)
    face_normals = read_array("faceNormal", float)
    face_areas = read_array("faceArea", float)
    face_cells = read_array("faceCells", int)

    cell_faces_flat = read_array("cellFaces_flat", int)
    cell_face_offsets = read_array("cellFaceOffsets", int)

    face_verts_flat = read_array("faceVerts_flat", int)
    face_vert_offsets = read_array("faceVertOffsets", int)

    n_faces = len(face_centers)

    face_struct = []
    for f in range(n_faces):
        vstart = 0 if f == 0 else face_vert_offsets[f-1]
        vend = face_vert_offsets[f]
        verts = face_verts_flat[vstart:vend]

        cells = [int(c) for c in face_cells[f] if c >= 0]

        face_struct.append({
            "cells": cells,
            "verts": verts,
            "center": face_centers[f],
            "normal": face_normals[f],
            "area": face_areas[f],
        })

    cell_struct = []
    for c in range(n_cells):
        fstart = 0 if c == 0 else cell_face_offsets[c-1]
        fend = cell_face_offsets[c]
        faces = cell_faces_flat[fstart:fend]

        xc = cell_centers[c]
        signs = []
        normals = []

        for f in faces:
            xf = face_struct[f]["center"]
            nf = face_struct[f]["normal"]

            if np.dot(nf, xf - xc) < 0:
                signs.append(-1)
                normals.append(-nf)
            else:
                signs.append(1)
                normals.append(nf)

        cell_struct.append({
            "faces": faces,
            "center": xc,
            "volume": volumes[c],
            "faces_orientation": np.array(signs),
            "face_normals": np.array(normals),
        })

    bounds = mesh.bounds
    Lx = bounds[1] - bounds[0]
    Ly = bounds[3] - bounds[2]
    Lz = bounds[5] - bounds[4]

    return cell_struct, face_struct, vertices, Lx, Ly, Lz

def ensure_ex0_meshes():

    if os.path.exists("meshes/ex0"):
        return

    zip_path = "meshes/ex0.zip"

    if not os.path.exists(zip_path):
        raise FileNotFoundError(
            "Could not find meshes/ex0.zip"
        )

    print("Extracting ex0 meshes...")

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall("meshes")