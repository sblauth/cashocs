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
# (demo_poisson_clover)=
# # Topology Optimization with a Poisson Equation
#
# ## Some remarks about topology optimization
#
# As topology optimization problems are very involved from a theoretical point of view,
# it is, at the moment, not possible to automatically derive topological derivatives.
# Therefore, cashocs cannot be used as "black-box" solver for topology optimization
# problems in general.
#
# Moreover, our framework for topology optimization of using a level-set function is
# quite flexible, but requires a lot of theoretical understanding. In the following, we
# briefly go over some theoretical foundations required for using cashocs' topology
# optimization features. We refer the reader, e.g., to CITE for an exhaustive treatment
# of these topics.
#
# ## Solving topology optimization problems with a level-set method
#
# The general form of a topology optimization problem is given by
#
# $$
# \min_\Omega J(\Omega),
# $$
#
# i.e., we want to minimize some cost functional {math}`J` depending on a geometry or
# set {math}`\Omega` which is contained in some hold-all domain {math}`\mathrm{D}` by
# varying the topology and shape of {math}`\Omega`. In contrast to shape optimization
# problems we have discussed previously, topology optimization considers topological
# changes of {math}`\Omega`, i.e., the addition or removal of material.
#
# One can represent the domain {math}`\Omega` with a continuous level-set function {math}`\Psi
# \colon \mathrm{D} \to \mathbb{R}` as follows:
#
# $$
# \begin{align}
#     \Psi(x) < 0 \quad &\Leftrightarrow \quad x \in \Omega,\\
#     \Psi(x) > 0 \quad &\Leftrightarrow \quad x \in \mathcal{D} \setminus \bar{\Omega},\\
#     \Psi(x) = 0 \quad &\Leftrightarrow \quad x \in \partial \Omega \setminus \partial \mathrm{D}.
# \end{align}
# $$
#
# Assuming that the topological derivative of the problem exists, which we denote by
# {math}`DJ(\Omega)`, then the generalized topological derivative is given by
#
# $$
# \mathcal{D}J(\Omega)(x) =
#     \begin{cases}
#         -DJ(\Omega)(x) \quad &\text{ if } x \in \Omega,\\
#         DJ(\Omega)(x) \quad &\text{ if } x \in \mathcal{D} \setminus \bar{\Omega}.
#     \end{cases}
# $$
#
# This is motivated by the fact that if there exists some constant {math}`c > 0` such
# that
#
# $$
# \mathcal{D}J(\Omega)(x) = c\Psi(x) \quad \text{ for all } x \in \mathrm{D},
# $$
#
# then we have that {math}`DJ(\Omega) \geq 0` which is a necessary condition for
# {math}`\Omega` being a local minimizer of the problem.
# According to this, several optimization algorithms have been developed in the
# literature to solve topology optimization problems with level-set methods. We refer
# the reader to [Blauth and Sturm - Quasi-Newton methods for topology optimization
# using a level-set method](https://doi.org/10.1007/s00158-023-03653-2) for a detailed
# discussion of the algorithms implemented in cashocs.
#
#
# ## Problem Formulation
#
# In this demo, we investigate the basics of using cashocs for solving topology
# optimization problems with a level-set approach. This approachs goes back to
# [Amstutz and Andrä - A new algorithm for topology optimization using a level-set
# method](https://doi.org/10.1016/j.jcp.2005.12.015) and in cashocs we have implemented
# novel gradient-type and quasi-Newton methods for solving such problems from [Blauth
# and Sturm - Quasi-Newton methods for topology optimization using a level-set
# method](https://doi.org/10.1007/s00158-023-03653-2).
#
# In this demo, we consider the following model problem
#
# $$
# \begin{align}
#     &\min_{\Omega,u} J(\Omega, u) = \frac{1}{2} \int_\mathrm{D} \left( u - u_\mathrm{des} \right)^2 \text{ d}x\\
#     &\text{subject to} \qquad
#     \begin{alignedat}[t]{2}
#         - \Delta u + \alpha_\Omega u &= f_\Omega \quad &&\text{ in } \mathrm{D},\\
#         u &= 0 \quad &&\text{ on } \partial \mathrm{D}.
#     \end{alignedat}
# \end{align}
# $$
#
# Here, {math}`\alpha_\Omega` and {math}`f_\Omega` have jumps between {math}`\Omega`
# and {math}`\mathrm{D} \setminus \bar{\Omega}` and are given by
# {math}`\alpha_\Omega(x) = \chi_\Omega(x)\alpha_\mathrm{in} +
# \chi_{\Omega^c}(x) \alpha_\mathrm{out}` and {math}`f_\Omega(x) = \chi_\Omega(x)
# f_\mathrm{in} + \chi_{\Omega^c}(x) f_\mathrm{out}` with {math}`\alpha_\mathrm{in},
# \alpha_\mathrm{out} > 0` and {math}`f_\mathrm{in}, f_\mathrm{out} \in \mathbb{R}`, and
# {math}`\Omega^c = \mathrm{D} \setminus \bar{\Omega}` is the complement of
# {math}`\Omega` in {math}`\mathrm{D}`. Moreover, {math}`u_\mathrm{des}` is a desired
# state, which in our case comes from the solution of the state equation for a desired
# shape, which we want to recover by the above problem. The generalized topological
# derivative for this problem is given by
#
# $$
# \mathcal{D}J(\Omega)(x) = (\alpha_\mathrm{out} - \alpha_\mathrm{in}) u(x)p(x)
# - (f_\mathrm{out} - f_\mathrm{in})p(x),
# $$
# where {math}`u` solves the state equation and {math}`p` solves the adjoint equation.
#
# ## Implementation
#
# The complete python code can be found in the file {download}`demo_poisson_clover.py
# </../../demos/documented/topology_optimization/poisson_clover/demo_poisson_clover.py>`,
# and the corresponding config can be found in {download}`config.ini
# </../../demos/documented/topology_optimization/poisson_clover/config.ini>`.
#
# ### Initialization and setup
#
# As usual, we start by using a wildcard import for FEniCS and by importing cashocs

# +
from fenics import *

import cashocs

# -

# Next, as before, we can specify the verbosity of cashocs with the line

cashocs.set_log_level(cashocs.LogLevel.INFO)

# As with the other problem types, the solution algorithms of cashocs can be adapted
# with the help of configuration files, which is loaded with the {py:func}`load_config <
# cashocs.load_config>` function

cfg = cashocs.load_config("config.ini")

# In the next step, we define the mesh used for discretization of the hold-all domain
# {math}`\mathrm{D}`. To do so, we use the in-built {py:func}`cashocs.regular_box_mesh`
# function, which gives us a discretization of {math}`\mathrm{D} = (-2, 2)^2`.

mesh, subdomains, boundaries, dx, ds, dS = cashocs.regular_box_mesh(
    n=32, start_x=-2, start_y=-2, end_x=2, end_y=2, diagonal="crossed"
)

# Now, we define the function spaces required for solving the state system, i.e., a
# finite element space consisting of piecewise linear Lagrange elements, and we define
# a finite element space consisting of piecewise constant elements, which is used to
# discretize the jumping coefficients {math}`\alpha_\Omega` and {math}`f_\Omega`.

V = FunctionSpace(mesh, "CG", 1)
DG0 = FunctionSpace(mesh, "DG", 0)

# Now, we define the level-set function used to represent our domain {math}`\Omega` and
# initialize {math}`\Omega` to the empty set by setting {math}`\Psi = 1` with the lines

psi = Function(V)
psi.vector()[:] = 1.0

# We define the constants for the jumping coefficients as follows

# +
f_1 = 10.0
f_2 = 1.0
f_diff = f_1 - f_2

alpha_1 = 10.0
alpha_2 = 1.0
alpha_diff = alpha_1 - alpha_2

alpha = Function(DG0)
f = Function(DG0)
# -
#


def create_desired_state():
    a = 4.0 / 5.0
    b = 2

    f_expr = Expression(
        "0.1 * ( "
        + "(sqrt(pow(x[0] - a, 2) + b * pow(x[1], 2)) - 1)"
        + "* (sqrt(pow(x[0] + a, 2) + b * pow(x[1], 2)) - 1)"
        + "* (sqrt(b * pow(x[0], 2) + pow(x[1] - a, 2)) - 1)"
        + "* (sqrt(b * pow(x[0], 2) + pow(x[1] + a, 2)) - 1)"
        + "- 0.001)",
        degree=1,
        a=a,
        b=b,
    )
    psi_des = interpolate(f_expr, V)

    cashocs.interpolate_levelset_function_to_cells(psi_des, alpha_1, alpha_2, alpha)
    cashocs.interpolate_levelset_function_to_cells(psi_des, f_1, f_2, f)

    y_des = Function(V)
    v = TestFunction(V)

    F = dot(grad(y_des), grad(v)) * dx + alpha * y_des * v * dx - f * v * dx
    bcs = cashocs.create_dirichlet_bcs(V, Constant(0.0), boundaries, [1, 2, 3, 4])

    cashocs.newton_solve(F, y_des, bcs, is_linear=True)

    return y_des, psi_des


y_des, psi_des = create_desired_state()

y = Function(V)
p = Function(V)
F = dot(grad(y), grad(p)) * dx + alpha * y * p * dx - f * p * dx
bcs = cashocs.create_dirichlet_bcs(V, Constant(0.0), boundaries, [1, 2, 3, 4])
J = cashocs.IntegralFunctional(Constant(0.5) * pow(y - y_des, 2) * dx)

# Now, we have to define the topological derivative of the problem, which we do with

dJ_in = Constant(alpha_diff) * y * p - Constant(f_diff) * p
dJ_out = Constant(alpha_diff) * y * p - Constant(f_diff) * p

# ::::{note}
# We remark that the generalized topological derivative for this problem is identical
# in {math}`\Omega` and {math}`\Omega^c`, which is usually not the case. For this
# reason, the special structure of the problem can be exploited with the following
# lines in the configuration file
# ```{code-block} ini
# :caption: config.ini
# [TopologyOptimization]
# topological_derivative_is_identical = True
# ```
# ::::


def update_level_set():
    cashocs.interpolate_levelset_function_to_cells(psi, alpha_1, alpha_2, alpha)
    cashocs.interpolate_levelset_function_to_cells(psi, f_1, f_2, f)


top = cashocs.TopologyOptimizationProblem(
    F, bcs, J, y, p, psi, dJ_in, dJ_out, update_level_set, config=cfg
)
top.solve(algorithm="bfgs", rtol=0.0, atol=0.0, angle_tol=1.0, max_iter=100)

# We visualize the results with the following code

# +
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))

ax_mesh = plt.subplot(1, 2, 1)
top.plot_shape()
plt.title("Obtained shape")

ax_u = plt.subplot(1, 2, 2)
psi.vector()[:] = psi_des.vector()[:]
top.plot_shape()
plt.title("Desired shape")

plt.tight_layout()
# plt.savefig("./img_poisson_clover.png", dpi=150, bbox_inches="tight")
# -

# and the result looks like this
# ![](/../../demos/documented/topology_optimization/poisson_clover/img_poisson_clover.png)
#
# :::{note}
# The method {py:meth}`plot_shape <cashocs.TopologyOptimizationProblem.plot_shape>` can
# be used to visualize the current shape based on the level-set function `psi`. Note
# that the cells inside {math}`\Omega` are colored in yellow and the ones outside are
# colored blue.
