"""
Created on 07/09/2020, 10.46

@author: blauths
"""

from fenics import *
import cashocs



config = cashocs.create_config('./config.ini')
mesh, subdomains, boundaries, dx, ds, dS = cashocs.import_mesh('./mesh/mesh.xdmf')

v_elem = VectorElement('CG', mesh.ufl_cell(), 2)
p_elem = FiniteElement('CG', mesh.ufl_cell(), 1)
V = FunctionSpace(mesh, MixedElement([v_elem, p_elem]))

v_in = Expression(('-1.0/4.0*(x[1] - 2.0)*(x[1] + 2.0)', '0.0'), degree=2)
bc_in = DirichletBC(V.sub(0), v_in, boundaries, 1)
bc_no_slip = cashocs.create_bcs_list(V.sub(0), Constant((0,0)), boundaries, [2,4])
bcs = [bc_in] + bc_no_slip

up = Function(V)
u, p = split(up)
vq = Function(V)
v, q = split(vq)

e = inner(grad(u), grad(v))*dx - p*div(v)*dx - q*div(u)*dx

J = inner(grad(u), grad(u))*dx

sop = cashocs.ShapeOptimizationProblem(e, bcs, J, up, vq, boundaries, config)
sop.solve()



### Post Processing
#
# import matplotlib.pyplot as plt
# plt.figure(figsize=(15,3))
#
# ax_mesh = plt.subplot(1, 3, 1)
# fig_mesh = plot(mesh)
# plt.title('Discretization of the optimized geometry')
#
# ax_u = plt.subplot(1, 3, 2)
# ax_u.set_xlim(ax_mesh.get_xlim())
# ax_u.set_ylim(ax_mesh.get_ylim())
# fig_u = plot(u)
# plt.colorbar(fig_u, fraction=0.046, pad=0.04)
# plt.title('State variable u')
#
#
# ax_p = plt.subplot(1, 3, 3)
# ax_p.set_xlim(ax_mesh.get_xlim())
# ax_p.set_ylim(ax_mesh.get_ylim())
# fig_p = plot(p)
# plt.colorbar(fig_p, fraction=0.046, pad=0.04)
# plt.title('State variable p')
#
# plt.tight_layout()
# plt.savefig('./img_shape_stokes.png', dpi=150, bbox_inches='tight')
