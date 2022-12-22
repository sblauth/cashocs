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
# (demo_control_solver)=
# # cashocs as Solver for Optimal Control Problems
#
# ## Problem Formulation
#
# As a model problem we again consider the one from {ref}`demo_poisson`, but now
# we do not use cashocs to compute the adjoint equation and derivative of the (reduced)
# cost functional, but supply these terms ourselves. The optimization problem is
# given by
#
# $$
# \begin{align}
#     &\min\; J(y,u) = \frac{1}{2} \int_{\Omega} \left( y - y_d \right)^2
#     \text{ d}x + \frac{\alpha}{2} \int_{\Omega} u^2 \text{ d}x \\
#     &\text{ subject to } \qquad
#     \begin{alignedat}[t]{2}
#         -\Delta y &= u \quad &&\text{ in } \Omega,\\
#         y &= 0 \quad &&\text{ on } \Gamma.
#     \end{alignedat}
# \end{align}
# $$
#
# (see, e.g., [Tröltzsch - Optimal Control of Partial Differential Equations](
# https://doi.org/10.1090/gsm/112) or [Hinze, Pinnau, Ulbrich, and Ulbrich -
# Optimization with PDE constraints](
# https://doi.org/10.1007/978-1-4020-8839-1)).
#
# For the numerical solution of this problem we consider exactly the same setting as
# in {ref}`demo_poisson`, and most of the code will be identical to this.
#
# ## Implementation
#
# The complete python code can be found in the file {download}`demo_control_solver.py
# </../../demos/documented/cashocs_as_solver/control_solver/demo_control_solver.py>`
# and the corresponding config can be found in {download}`config.ini
# </../../demos/documented/cashocs_as_solver/control_solver/config.ini>`.
#
# ### Recapitulation of {ref}`demo_poisson`
#
# For using cashocs exclusively as solver, the procedure is very similar to regularly
# using it, with a few additions after defining the optimization problem. In particular,
# up to the initialization of the optimization problem, our code is exactly the same as
# in {ref}`demo_poisson`, i.e., we use

# +
from fenics import *

import cashocs

config = cashocs.load_config("config.ini")
mesh, subdomains, boundaries, dx, ds, dS = cashocs.regular_mesh(25)
V = FunctionSpace(mesh, "CG", 1)

y = Function(V)
p = Function(V)
u = Function(V)

e = inner(grad(y), grad(p)) * dx - u * p * dx

bcs = cashocs.create_dirichlet_bcs(V, Constant(0), boundaries, [1, 2, 3, 4])

y_d = Expression("sin(2*pi*x[0])*sin(2*pi*x[1])", degree=1)
alpha = 1e-6
J = cashocs.IntegralFunctional(
    Constant(0.5) * (y - y_d) * (y - y_d) * dx + Constant(0.5 * alpha) * u * u * dx
)

ocp = cashocs.OptimalControlProblem(e, bcs, J, y, u, p, config=config)
# -

# ### Supplying Custom Adjoint Systems and Derivatives
#
# When using cashocs as a solver, the user can specify their custom weak forms of
# the adjoint system and of the derivative of the reduced cost functional. For our
# optimization problem, the adjoint equation is given by
#
# $$
# \begin{alignedat}{2}
#     - \Delta p &= y - y_d \quad &&\text{ in } \Omega, \\
#     p &= 0 \quad &&\text{ on } \Gamma,
# \end{alignedat}
# $$
#
# and the derivative of the cost functional is then given by
#
# $$
# dJ(u)[h] = \int_\Omega (\alpha u + p) h \text{ d}x.
# $$
#
# :::{note}
# For a detailed derivation and discussion of these objects we refer to
# [Tröltzsch - Optimal Control of Partial Differential Equations](
# https://doi.org/10.1090/gsm/112) or [Hinze, Pinnau, Ulbrich, and Ulbrich -
# Optimization with PDE constraints](https://doi.org/10.1007/978-1-4020-8839-1).
# :::
#
# To specify that cashocs should use these equations instead of the automatically
# computed ones, we have the following code. First, we specify the derivative
# of the reduced cost functional via

dJ = Constant(alpha) * u * TestFunction(V) * dx + p * TestFunction(V) * dx

# Afterwards, the weak form for the adjoint system is given by

adjoint_form = (
    inner(grad(p), grad(TestFunction(V))) * dx - (y - y_d) * TestFunction(V) * dx
)
adjoint_bcs = bcs

# where we can "recycle" the homogeneous Dirichlet boundary conditions used for the
# state problem.
#
# For both objects, one has to define them as a single UFL form for cashocs, as with the
# state system and cost functional. In particular, the adjoint weak form has to be in
# the form of a nonlinear variational problem, so that
# {python}`fenics.solve(adjoint_form == 0, p, adjoint_bcs)` could be used to solve it.
# In particular, both forms have to include {py:class}`fenics.TestFunction` objects from
# the control space and adjoint space, respectively, and must not contain
# {py:class}`fenics.TrialFunction` objects.
#
# These objects are then supplied to the
# {py:class}`OptimalControlProblem <cashocs.OptimalControlProblem>` via

ocp.supply_custom_forms(dJ, adjoint_form, adjoint_bcs)

# :::{note}
# One can also specify either the adjoint system or the derivative of the cost
# functional, using the methods {py:meth}`supply_adjoint_forms
# <cashocs.OptimalControlProblem.supply_adjoint_forms>` or {py:meth}`supply_derivatives
# <cashocs.OptimalControlProblem.supply_derivatives>`.
# However, this is potentially dangerous, due to the following. The adjoint system
# is a linear system, and there is no fixed convention for the sign of the adjoint
# state. Hence, supplying, e.g., only the adjoint system, might not be compatible with
# the derivative of the cost functional which cashocs computes. In effect, the sign
# is specified by the choice of adding or subtracting the PDE constraint from the
# cost functional for the definition of a Lagrangian function, which is used to
# determine the adjoint system and derivative. cashocs internally uses the convention
# that the PDE constraint is added, so that, internally, it computes not the adjoint
# state $p$ as defined by the equations given above, but $-p$ instead.
# Hence, it is recommended to either specify all respective quantities with the
# {py:meth}`supply_custom_forms <cashocs.OptimalControlProblem.supply_custom_forms>`
# method.
# :::
#
# Finally, we can use the {py:meth}`solve <cashocs.OptimalControlProblem.solve>` method
# to solve the problem with the line

ocp.solve()

# as in {ref}`demo_poisson`.
#
# We visualize the results with the lines

# +
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
fig = plot(u)
plt.colorbar(fig, fraction=0.046, pad=0.04)
plt.title("Control variable u")

plt.subplot(1, 3, 2)
fig = plot(y)
plt.colorbar(fig, fraction=0.046, pad=0.04)
plt.title("State variable y")

plt.subplot(1, 3, 3)
fig = plot(y_d, mesh=mesh)
plt.colorbar(fig, fraction=0.046, pad=0.04)
plt.title("Desired state y_d")

plt.tight_layout()
# plt.savefig('./img_poisson.png', dpi=150, bbox_inches='tight')
# -

# The results are, of course, identical to {ref}`demo_poisson` and look as follows
# ![](/../../demos/documented/cashocs_as_solver/control_solver/img_control_solver.png)

# :::{note}
# In case we have multiple state equations as in {ref}`demo_multiple_variables`,
# one has to supply ordered lists of adjoint equations and boundary conditions,
# analogously to the usual procedure for cashocs.
#
# In the case of multiple control variables, the derivatives of the reduced cost
# functional w.r.t. each of these have to be specified, again using an ordered list.
# :::
