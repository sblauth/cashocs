"""
Created on 08/09/2020, 09.27

@author: blauths
"""

from fenics import *
import cashocs





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
# plt.savefig('./img_inverse_tomography.png', dpi=150, bbox_inches='tight')
