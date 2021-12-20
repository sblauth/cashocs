"""
Created on 27/07/2021, 14.43

@author: blauths
"""

import os

from fenics import *

import cashocs


dir_path = os.path.dirname(os.path.realpath(__file__))
config = cashocs.load_config(f"{dir_path}/config_remesh.ini")
config.set("Mesh", "mesh_file", dir_path + "/mesh/remesh/mesh.xdmf")
config.set("Mesh", "gmsh_file", dir_path + "/mesh/remesh/mesh.msh")
config.set("Mesh", "geo_file", dir_path + "/mesh/remesh/mesh.geo")
config.set("Output", "result_dir", dir_path + "/temp/")
config.add_section("Debug")
config.set("Debug", "remeshing", "True")

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
sop.solve(max_iter=4)
