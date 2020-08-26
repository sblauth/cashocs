"""
Created on 16/06/2020, 15.52

@author: blauths
"""

from fenics import *
import numpy as np
from cashocs import import_mesh



mesh, subdomains, boundaries, dx, ds, dS = import_mesh('./mesh/reference.xdmf')

sigma_out = 1.0
sigma_in = 10.0

V = FunctionSpace(mesh, 'CG', 1)

u = TrialFunction(V)
v = TestFunction(V)

a = sigma_out*inner(grad(u), grad(v))*dx(1) + sigma_in*inner(grad(u), grad(v))*dx(2)
L = Constant(-1)*v*ds(1)

bcs = DirichletBC(V, Constant(0), boundaries, 2)

reference = Function(V)
solve(a==L, reference, bcs)


mesh2, subdomains2, boundaries2, dx2, ds2, dS2 = import_mesh('./mesh/mesh.xdmf')
V2 = FunctionSpace(mesh2, 'CG', 1)

reference = interpolate(reference, V2)

np.save('reference.npy', reference.vector()[:])
