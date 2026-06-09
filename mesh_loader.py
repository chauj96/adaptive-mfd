import os
import numpy as np
from scipy.io import loadmat
import xml.etree.ElementTree as ET
import pyvista as pv

# Mesh loader for different test cases (currently supports twoFaults)
# TO DO: add distorted unit cube case
 
def load_mesh(mesh_name):

    if mesh_name == "twoFaults":
        return load_vtu("meshes/twoFaults/fault_mesh.vtu")
    elif mesh_name == "spe11b":
        return load_vtu("meshes/spe11b/spe11b_mesh.vtu")
    elif mesh_name == "fullyPoly":
        return load_vtu("meshes/fullyPolyhedral/fullyPoly_mesh.vtu")
    else:
        raise ValueError(f"Unknown mesh: {mesh_name}")

# Directly loading data from .mat file (for debugging purpose)
def load_twofault():

    filepath = os.path.join("meshes", "twoFaults", "fault_mesh.mat")

    data = loadmat(filepath, simplify_cells=True)

    cell_struct = data["cell_struct"]
    face_struct = data["face_struct"]
    vertices = data["V3"]

    # Index conversion (from matlab to python)
    for c in cell_struct:
        c["faces"] = np.array(c["faces"]).astype(int) - 1

    # Size of the domain
    Lx = 1.0
    Ly = 1.0
    Lz = 0.4

    return cell_struct, face_struct, vertices, Lx, Ly, Lz

# Directly loading data from .mat file (for debugging purpose)
def load_spe11b():
    filepath = os.path.join("meshes", "spe11b", "spe11b_mesh.mat")

    data = loadmat(filepath, simplify_cells=True)

    cell_struct = data["cell_struct"]
    face_struct = data["face_struct"]
    vertices = data["V3"]

    for c in cell_struct:
        c["faces"] = np.array(c["faces"]).astype(int) - 1

    Lx = 8400
    Ly = 1200.02
    Lz = 100.0

    return cell_struct, face_struct, vertices, Lx, Ly, Lz


def load_vtu(filepath):

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