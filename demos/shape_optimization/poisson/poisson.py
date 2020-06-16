"""
Created on 15/06/2020, 08.09

@author: blauths
"""

from fenics import *
from adpack.geometry import MeshGen
from adpack import ShapeOptimizationProblem
import numpy as np
import configparser



set_log_level(LogLevel.CRITICAL)

config = configparser.ConfigParser()
config.read('./config.ini')

# mesh, subdomains, boundaries, dx, ds, dS = MeshGen('./mesh/mesh.xdmf')
meshlevel = 25
degree = 1
dim = 2
mesh = UnitDiscMesh.create(MPI.comm_world, meshlevel, degree, dim)
dx = Measure('dx', mesh)
ds = Measure('ds', mesh)

boundary = CompiledSubDomain('on_boundary')
boundaries = MeshFunction('size_t', mesh, dim=1)
boundary.mark(boundaries, 1)

V = FunctionSpace(mesh, 'CG', 1)

bcs = DirichletBC(V, Constant(0), boundaries, 1)

x = SpatialCoordinate(mesh)
f = 2.5*pow(x[0] + 0.4 - pow(x[1], 2), 2) + pow(x[0], 2) + pow(x[1], 2) - 1

u = Function(V)
p = Function(V)

e = inner(grad(u), grad(p))*dx - f*p*dx

J = u*dx

optimization_problem = ShapeOptimizationProblem(e, bcs, J, u, p, boundaries, config)
optimization_problem.solve()
