"""
Created on 28/07/2021, 09.04

@author: blauths
"""

import os
import subprocess

import fenics
import numpy as np

import cashocs
import cashocs._cli

dir_path = os.path.dirname(os.path.realpath(__file__))


def test_cli():
    cashocs._cli.convert([f"{dir_path}/mesh/mesh.msh", f"{dir_path}/mesh/mesh.xdmf"])
    mesh, subdomains, boundaries, dx, ds, dS = cashocs.import_mesh(
        dir_path + "/mesh/mesh.xdmf"
    )

    gmsh_coords = np.array(
        [
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 1],
            [0.499999999998694, 0],
            [1, 0.499999999998694],
            [0.5000000000020591, 1],
            [0, 0.5000000000020591],
            [0.2500000000010297, 0.7500000000010296],
            [0.3749999970924328, 0.3750000029075671],
            [0.7187499979760099, 0.2812500030636815],
            [0.6542968741702071, 0.6542968818888233],
        ]
    )

    assert abs(fenics.assemble(1 * dx) - 1) < 1e-14
    assert abs(fenics.assemble(1 * ds) - 4) < 1e-14

    assert abs(fenics.assemble(1 * ds(1)) - 1) < 1e-14
    assert abs(fenics.assemble(1 * ds(2)) - 1) < 1e-14
    assert abs(fenics.assemble(1 * ds(3)) - 1) < 1e-14
    assert abs(fenics.assemble(1 * ds(4)) - 1) < 1e-14

    assert np.allclose(mesh.coordinates(), gmsh_coords)

    assert os.path.isfile(f"{dir_path}/mesh/mesh.xdmf")
    assert os.path.isfile(f"{dir_path}/mesh/mesh.h5")
    assert os.path.isfile(f"{dir_path}/mesh/mesh_subdomains.xdmf")
    assert os.path.isfile(f"{dir_path}/mesh/mesh_subdomains.h5")
    assert os.path.isfile(f"{dir_path}/mesh/mesh_boundaries.xdmf")
    assert os.path.isfile(f"{dir_path}/mesh/mesh_boundaries.h5")

    subprocess.run(["rm", f"{dir_path}/mesh/mesh.xdmf"], check=True)
    subprocess.run(["rm", f"{dir_path}/mesh/mesh.h5"], check=True)
    subprocess.run(["rm", f"{dir_path}/mesh/mesh_subdomains.xdmf"], check=True)
    subprocess.run(["rm", f"{dir_path}/mesh/mesh_subdomains.h5"], check=True)
    subprocess.run(["rm", f"{dir_path}/mesh/mesh_boundaries.xdmf"], check=True)
    subprocess.run(["rm", f"{dir_path}/mesh/mesh_boundaries.h5"], check=True)
