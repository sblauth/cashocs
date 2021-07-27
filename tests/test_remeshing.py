"""
Created on 27/07/2021, 14.32

@author: blauths
"""

import cashocs
from fenics import *
import os
import shutil
import pytest


has_gmsh = False
query = shutil.which("gmsh")
if query is not None:
    has_gmsh = True
else:
    has_gmsh = False


dir_path = os.path.dirname(os.path.realpath(__file__))


def test_verification_remeshing():
    dir_path = os.path.dirname(os.path.realpath(__file__))

    config = cashocs.load_config(dir_path + "/config_remesh.ini")
    config.set("Mesh", "mesh_file", dir_path + "/mesh/remesh/mesh.xdmf")
    config.set("Mesh", "gmsh_file", dir_path + "/mesh/remesh/mesh.msh")
    config.set("Mesh", "geo_file", dir_path + "/mesh/remesh/mesh.geo")

    mesh, subdomains, boundaries, dx, ds, dS = cashocs.import_mesh(config)

    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V)
    p = Function(V)

    x = SpatialCoordinate(mesh)
    f = 2.5 * pow(x[0] + 0.4 - pow(x[1], 2), 2) + pow(x[0], 2) + pow(x[1], 2) - 1

    e = inner(grad(u), grad(p)) * dx - f * p * dx
    bcs = DirichletBC(V, Constant(0), boundaries, 1)

    J = u * dx

    sop = cashocs.ShapeOptimizationProblem(e, bcs, J, u, p, boundaries, config)
    assert cashocs.verification.shape_gradient_test(sop) > 1.9
    assert cashocs.verification.shape_gradient_test(sop) > 1.9
    assert cashocs.verification.shape_gradient_test(sop) > 1.9


@pytest.mark.skipif(not has_gmsh, reason="This test requires Gmsh")
def test_remeshing():
    os.system(f"python {dir_path}/remeshing_script.py")

    assert any(
        s.startswith("cashocs_remesh_") for s in os.listdir(f"{dir_path}/mesh/remesh")
    )
    assert any(
        s.startswith("._cashocs_remesh_temp_") for s in os.listdir(f"{dir_path}")
    )

    assert os.path.isdir(dir_path + "/temp")
    assert os.path.isdir(dir_path + "/temp/pvd")
    assert os.path.isfile(dir_path + "/temp/history.txt")
    assert os.path.isfile(dir_path + "/temp/history.json")
    assert os.path.isfile(dir_path + "/temp/optimized_mesh.msh")
    assert os.path.isfile(dir_path + "/temp/pvd/remesh_0_adjoint_0.pvd")
    assert os.path.isfile(dir_path + "/temp/pvd/remesh_1_adjoint_0.pvd")
    assert os.path.isfile(dir_path + "/temp/pvd/remesh_0_adjoint_0000003.vtu")
    assert os.path.isfile(dir_path + "/temp/pvd/remesh_0_state_0.pvd")
    assert os.path.isfile(dir_path + "/temp/pvd/remesh_1_state_0.pvd")
    assert os.path.isfile(dir_path + "/temp/pvd/remesh_0_state_0000003.vtu")
    assert os.path.isfile(dir_path + "/temp/pvd/remesh_0_shape_gradient.pvd")
    assert os.path.isfile(dir_path + "/temp/pvd/remesh_1_shape_gradient.pvd")
    assert os.path.isfile(dir_path + "/temp/pvd/remesh_0_shape_gradient000003.vtu")

    os.system(f"rm -r {dir_path}/mesh/remesh/cashocs_remesh_*")
    os.system(f"rm -r {dir_path}/._cashocs_remesh_temp_*")
    os.system("rm -r " + dir_path + "/temp")
