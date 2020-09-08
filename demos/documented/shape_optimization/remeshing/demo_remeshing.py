"""
Created on 08/09/2020, 09.31

@author: blauths
"""

from fenics import *
import cashocs



config = cashocs.create_config('./config.ini')

mesh, subdomains, boundaries, dx, ds, dS = cashocs.import_mesh(config)

V = FunctionSpace(mesh, 'CG', 1)
u = Function(V)
p = Function(V)

x = SpatialCoordinate(mesh)
f = 2.5*pow(x[0] + 0.4 - pow(x[1], 2), 2) + pow(x[0], 2) + pow(x[1], 2) - 1

e = inner(grad(u), grad(p))*dx - f*p*dx
bcs = DirichletBC(V, Constant(0), boundaries, 1)

J = u*dx

sop = cashocs.ShapeOptimizationProblem(e, bcs, J, u, p, boundaries, config)
sop.solve()


### Post Processing
#
# import matplotlib.pyplot as plt
# plt.figure(figsize=(10,5))
#
# ax_mesh = plt.subplot(1, 2, 1)
# fig_mesh = plot(mesh)
# plt.title('Discretization of the optimized geometry')
#
# ax_u = plt.subplot(1, 2, 2)
# ax_u.set_xlim(ax_mesh.get_xlim())
# ax_u.set_ylim(ax_mesh.get_ylim())
# fig_u = plot(u)
# plt.colorbar(fig_u, fraction=0.046, pad=0.04)
# plt.title('State variable u')
#
# plt.tight_layout()
# plt.savefig('./img_remeshing.png', dpi=150, bbox_inches='tight')
