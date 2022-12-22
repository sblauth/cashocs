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
# (demo_heat_equation)=
# # Distributed Control for Time Dependent Problems
#
# ## Problem Formulation
#
# In this demo  we take a look at how time dependent problems can be treated with
# cashocs. To do so, we investigate a problem with a heat equation as PDE constraint,
# which was considered in [Blauth - Optimal Control and Asymptotic Analysis of the
# Cattaneo Model](https://nbn-resolving.org/urn:nbn:de:hbz:386-kluedo-53727). It reads
#
# $$
# \begin{align}
#     &\min\; J(y,u) = \frac{1}{2} \int_0^T \int_\Omega \left( y - y_d \right)^2
#     \text{ d}x \text{ d}t
#     + \frac{\alpha}{2} \int_0^T \int_\Omega u^2 \text{ d}x \text{ d}t \\
#     &\text{ subject to }\qquad
#     \begin{alignedat}[t]{2}
#         \partial_t y - \Delta y &= u \quad &&\text{ in } (0,T) \times \Omega,\\
#         y &= 0 \quad &&\text{ on } (0,T) \times \Gamma, \\
#         y(0, \cdot) &= y^{(0)} \quad &&\text{ in } \Omega.
#     \end{alignedat}
# \end{align}
# $$
#
# Since FEniCS does not have any direct built-in support for time dependent problems,
# we first have to perform a semi-discretization of the PDE system in the temporal
# component (e.g. via finite differences), and then solve the resulting sequence of
# PDEs.
#
# In particular, for the use with cashocs, we have to create not a single weak form and
# {py:class}`fenics.Function`, that can be re-used, like one would in classical FEniCS
# programs, but we have to create the corresponding objects a-priori for each time step.
#
# For the domain of this problem, we once again consider the space time cylinder given
# by $(0,T) \times \Omega = (0,1) \times (0,1)^2$. And for the initial condition we use
# $y^{(0)} = 0$.
#
# ## Temporal Discretization
#
# For the temporal discretization, we use the implicit Euler scheme as this is
# unconditionally stable for the parabolic heat equation. This means, we discretize the
# interval $[0,T]$ by a grid with nodes
# $t_k, k=0,\dots, n,\; \text{ with }\; t_0 := 0\; \text{ and }\; t_n := T$. Then, we
# approximate the time derivative $\partial_t y(t_{k+1})$ at some time $t_{k+1}$ by the
# backward difference
#
# $$
# \partial_t y(t_{k+1}) \approx \frac{y(t_{k+1}) - y(t_{k})}{\Delta t},
# $$
#
# where $\Delta t = t_{k+1} - t_{k}$, and thus get the sequence of PDEs
#
# $$
# \begin{alignedat}{2}
#     \frac{y_{k+1} - y_{k}}{\Delta t} - \Delta y_{k+1} &=
#     u_{k+1} \quad &&\text{ in } \Omega \quad \text{ for } k=0,\dots,n-1,\\
#     y_{k+1} &= 0 \quad &&\text{ on } \Gamma \quad \text{ for } k=0,\dots,n-1,
# \end{alignedat}
# $$
#
# Note, that $y_k \approx y(t_k), \text {and }\; u_k \approx u(t_k)$ are approximations
# of the continuous functions. The initial condition is included via
#
# $$
# y_0 = y^{(0)}.
# $$
#
# Moreover, for the cost functionals, we can discretize the temporal integrals using a
# rectangle rule. This means we approximate the cost functional via
#
# $$
# J(y, u) \approx \frac{1}{2} \sum_{k=0}^{n-1} \Delta t
# \left( \int_\Omega \left( y_{k+1} - (y_d)_{k+1} \right)^2 \text{ d}x
# + \alpha \int_\Omega u_{k+1}^2 \text{ d}x \right).
# $$
#
# Here, $(y_d)_k$ is an approximation of the desired state at time $t_k$.
#
# Let us now investigate how to solve this problem with cashocs.
#
# ## Implementation
#
# The complete python code can be found in the file {download}`demo_heat_equation.py
# </../../demos/documented/optimal_control/heat_equation/demo_heat_equation.py>` and
# the corresponding config can be found in {download}`config.ini
# </../../demos/documented/optimal_control/heat_equation/config.ini>`.
#
# ### Initialization
#
# This section is the same as for all previous problems and is done via

# +
from fenics import *
import numpy as np

import cashocs

config = cashocs.load_config("config.ini")
mesh, subdomains, boundaries, dx, ds, dS = cashocs.regular_mesh(20)
V = FunctionSpace(mesh, "CG", 1)
# -

# Next up, we specify the temporal discretization via

dt = 1 / 10
t_start = dt
t_end = 1.0
t_array = np.linspace(t_start, t_end, int(1 / dt))

# Here, {python}`t_array` is a numpy array containing all time steps. Note, that we do **not**
# include $t=0$ in the array. This is due to the fact, that the initial condition
# is prescribed and fixed.
#
# Due to the fact that we discretize the equation temporally, we do not only get a
# single {py:class}`fenics.Function` describing our state and control, but one
# {py:class}`fenics.Function` for each time step. Hence, we initialize these (together
# with the adjoint states) directly in lists

states = [Function(V) for i in range(len(t_array))]
controls = [Function(V) for i in range(len(t_array))]
adjoints = [Function(V) for i in range(len(t_array))]

# Note, that {python}`states[k]` corresponds to $y_{k+1}$ since indices start at
# {python}`0` in most programming languages (as it is the case in python).
#
# As the boundary conditions are not time dependent, we can initialize them now, and
# repeat them in a list, since they are the same for every state

bcs = cashocs.create_dirichlet_bcs(V, Constant(0), boundaries, [1, 2, 3, 4])
bcs_list = [bcs for i in range(len(t_array))]

# To define the sequence of PDEs, we will use a loop over all time steps. But before we
# can do that, we first initialize empty lists for the state equations, the
# approximations of the desired state, and the summands of the cost functional

y_d = []
e = []
J_list = []

# ### Definition of the optimization problem
#
# For the desired state, we define it with the help of a {py:class}`fenics.Expression`
# that is dependent on an additional parameter which models the time

alpha = 1e-5
y_d_expr = Expression(
    "exp(-20*(pow(x[0] - 0.5 - 0.25*cos(2*pi*t), 2) + pow(x[1] - 0.5 - 0.25*sin(2*pi*t), 2)))",
    degree=1,
    t=0.0,
)

# Next, we have the following for loop

for k in range(len(t_array)):
    t = t_array[k]
    y_d_expr.t = t

    y = states[k]

    if k == 0:
        y_prev = Function(V)
    else:
        y_prev = states[k - 1]

    p = adjoints[k]
    u = controls[k]

    state_eq = (
        Constant(1 / dt) * (y - y_prev) * p * dx
        + inner(grad(y), grad(p)) * dx
        - u * p * dx
    )

    e.append(state_eq)

    y_d.append(interpolate(y_d_expr, V))

    J_list.append(
        cashocs.IntegralFunctional(
            Constant(0.5 * dt) * (y - y_d[k]) * (y - y_d[k]) * dx
            + Constant(0.5 * dt * alpha) * u * u * dx
        )
    )

# ::::{admonition} Description of the for-loop
# At the beginning, the 'current' time t is determined from {python}`t_array`, and the
# expression for the desired state is updated to reflect the current time with the code
# :::python
# t = t_array[k]
# y_d_expr.t = t
# :::
#
# The following line sets the object {python}`y` to $y_{k+1}$
# :::python
# y = states[k]
# :::
#
# For the backward difference in the implicit Euler method, we also need $y_{k}$
# which we define as follows
# :::python
# if k == 0:
#     y_prev = Function(V)
# else:
#     y_prev = states[k - 1]
# :::
#
# which ensures that $y_0 = 0$, which corresponds to the initial condition
# $y^{(0)} = 0$. Hence, {python}`y_prev` indeed corresponds to $y_{k}$.
#
# Moreover, we get the current control and adjoint state via
# :::python
# p = adjoints[k]
# u = controls[k]
# :::
#
# This allows us to define the state equation at time t as
# :::python
# state_eq = (
#     Constant(1 / dt) * (y - y_prev) * p * dx
#     + inner(grad(y), grad(p)) * dx
#     - u * p * dx
# )
# :::
#
# This is then appended to the list of state constraints
# :::python
# e.append(state_eq)
# :::
#
# Further, we also put the current desired state into the respective list, i.e.,
# :::python
# y_d.append(interpolate(y_d_expr, V))
# :::
#
# Finally, we can define the k-th summand of the cost functional via
# :::python
# J_list.append(
#     cashocs.IntegralFunctional(
#         Constant(0.5 * dt) * (y - y_d[k]) * (y - y_d[k]) * dx
#         + Constant(0.5 * dt * alpha) * u * u * dx
#     )
# )
# :::
#
# and directly append this to the cost functional list.
# ::::

# Finally, we can define an optimal control problem as before, and solve it as in the
# previous demos (see, e.g., {ref}`demo_poisson`)

ocp = cashocs.OptimalControlProblem(
    e, bcs_list, J_list, states, controls, adjoints, config=config
)
ocp.solve()

# For a postprocessing, we perform the following steps

# +
u_file = File("./visualization/u.pvd")
y_file = File("./visualization/y.pvd")
temp_u = Function(V)
temp_y = Function(V)

for k in range(len(t_array)):
    t = t_array[k]

    temp_u.vector()[:] = controls[k].vector()[:]
    u_file << temp_u, t

    temp_y.vector()[:] = states[k].vector()[:]
    y_file << temp_y, t
# -

# which saves the result in the directory `./visualization/` as paraview .pvd files.
