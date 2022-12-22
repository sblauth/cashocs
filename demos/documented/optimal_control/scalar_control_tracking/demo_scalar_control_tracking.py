# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.4
# ---

# ```{eval-rst}
# .. include:: ../../../global.rst
# ```
#
# (demo_scalar_control_tracking)=
# # Tracking of Scalar Functionals for Optimal Control Problems
#
# ## Problem Formulation
#
# In this demo we investigate cashocs functionality of tracking scalar functionals
# such as cost functional values and other quanitites, which typically
# arise after integration. For this, we investigate the problem
#
# $$
# \begin{align}
#     &\min\; J(y,u) = \frac{1}{2} \left( \int_{\Omega} y^2
#     \text{ d}x - C_{des} \right)^2 \\
#     &\text{ subject to } \qquad
#     \begin{alignedat}[t]{2}
#         -\Delta y &= u \quad &&\text{ in } \Omega,\\
#         y &= 0 \quad &&\text{ on } \Gamma.
#     \end{alignedat}
# \end{align}
# $$
#
# For this example, we do not consider control constraints,
# but search for an optimal control u in the entire space $L^2(\Omega)$,
# for the sake of simplicitiy. For the domain under consideration, we use the unit
# square $\Omega = (0, 1)^2$, since this is built into cashocs.
#
# In the following, we will describe how to solve this problem
# using cashocs. Moreover, we also detail alternative / equivalent FEniCS code which
# could be used to define the problem instead.
#
# ## Implementation
#
# The complete python code can be found in the file {download}`demo_scalar_control_tracking.py
# </../../demos/documented/optimal_control/scalar_control_tracking/demo_scalar_control_tracking.py>`
# and the corresponding config can be found in {download}`config.ini
# </../../demos/documented/optimal_control/scalar_control_tracking/config.ini>`.
#
# ### The state problem
#
# The difference to {ref}`demo_poisson` is that the cost functional does now track the
# value of the $L^2$ norm of $y$ against a desired value of $C_{des}$,
# and not the state $y$ itself. Other than that, the corresponding PDE constraint
# and its setup are completely analogous to {ref}`demo_poisson`

# +
from fenics import *

import cashocs

cashocs.set_log_level(cashocs.LogLevel.INFO)
config = cashocs.load_config("config.ini")
mesh, subdomains, boundaries, dx, ds, dS = cashocs.regular_mesh(25)
V = FunctionSpace(mesh, "CG", 1)

y = Function(V)
p = Function(V)
u = Function(V)
u.vector()[:] = 1.0

e = inner(grad(y), grad(p)) * dx - u * p * dx

bcs = cashocs.create_dirichlet_bcs(V, Constant(0), boundaries, [1, 2, 3, 4])
# -

# ### Definition of the scalar tracking type cost functional
#
# To define the desired tracking type functional, note that cashocs implements the
# functional for the following kind of cost functionals
#
# $$
# \begin{aligned}
#     J(y,u) &= \frac{1}{2} \vert \int_{\Sigma} f(y,u) \text{ d}m
#     - C_{des} \vert^2 \\
# \end{aligned}
# $$
#
# where $\Sigma$ is some part of the domain $\Omega$, e.g. the $\Omega$ itself, a
# subdomain, or its boundary $\Gamma$, and $\text{d}m$ is the corresponding integration
# measure.
#
# To define such a cost functional, we only need to define the integrands, i.e.,
#
# $$
# f(y,u) \text{ d}m
# $$
#
# which we will do in the UFL of FEniCS, as well as the goals of the tracking type
# functionals, i.e.,
#
# $$
# C_{des}.
# $$
#
# To do so, we use the {py:class}`cashocs.ScalarTrackingFunctional` class. We first
# define the integrand $f(y,u) \text{d}m = y^2 \text{d}x$ via

integrand = y * y * dx

# and define $C_{des}$ as

tracking_goal = 1.0

# With these definitions, we can define the cost functional as

J_tracking = cashocs.ScalarTrackingFunctional(integrand, tracking_goal)

# :::{note}
# The factor in front of the quadratic term can also be adapted, by using the keyword
# argument {python}`weight` of {py:class}`cashocs.ScalarTrackingFunctional`. Note that
# the default factor is {python}`0.5`, and that each weight will be multiplied by this
# value.
# :::
#
# Finally, we set up our optimization problem and solve it with the
# {py:meth}`solve <cashocs.OptimalControlProblem.solve>` method of the optimization
# problem

ocp = cashocs.OptimalControlProblem(e, bcs, J_tracking, y, u, p, config=config)
ocp.solve()

# To verify, that our approach is correct, we also print the value of the integral,
# which we want to track

print("L2-Norm of y squared: " + format(assemble(y * y * dx), ".3e"))


# Finally, we visualize the result with the lines

# +
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
fig = plot(u)
plt.colorbar(fig, fraction=0.046, pad=0.04)
plt.title("Control variable u")

plt.subplot(1, 2, 2)
fig = plot(y)
plt.colorbar(fig, fraction=0.046, pad=0.04)
plt.title("State variable y")

plt.tight_layout()
# plt.savefig('./img_scalar_control_tracking.png', dpi=150, bbox_inches='tight')
# -

# and the result looks as follows
# ![](/../../demos/documented/optimal_control/scalar_control_tracking/img_scalar_control_tracking.png)
