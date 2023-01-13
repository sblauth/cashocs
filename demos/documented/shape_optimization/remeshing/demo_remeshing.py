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
# (demo_remeshing)=
# # Remeshing with cashocs
#
# ## Problem Formulation
#
# In this tutorial, we take a close look at how remeshing works in cashocs. To keep
# this discussion simple, we take a look at the model problem already investigated
# in {ref}`demo_shape_poisson`, i.e.,
#
# $$
# \begin{align}
#     &\min_\Omega J(u, \Omega) = \int_\Omega u \text{ d}x \\
#     &\text{subject to} \qquad
#     \begin{alignedat}[t]{2}
#         -\Delta u &= f \quad &&\text{ in } \Omega,\\
#         u &= 0 \quad &&\text{ on } \Gamma.
#     \end{alignedat}
# \end{align}
# $$
#
# As before, we use the unit disc
# $\Omega = \{ x \in \mathbb{R}^2 \,\mid\, \lvert\lvert x \rvert\rvert_2 < 1 \}$
# as initial geometry and the right-hand side $f$ is given by
#
# $$
# f(x) = 2.5 \left( x_1 + 0.4 - x_2^2 \right)^2 + x_1^2 + x_2^2 - 1.
# $$
#
# ## Implementation
#
# The complete python code can be found in the file {download}`demo_remeshing.py
# </../../demos/documented/shape_optimization/remeshing/demo_remeshing.py>`,
# and the corresponding config can be found in {download}`config.ini
# </../../demos/documented/shape_optimization/remeshing/config.ini>`.
# The corresponding mesh files are {download}`./mesh/mesh.geo
# </../../demos/documented/shape_optimization/remeshing/mesh/mesh.geo>` and
# {download}`./mesh/mesh.msh
# </../../demos/documented/shape_optimization/remeshing/mesh/mesh.msh>`.
#
# ### Pre-Processing with GMSH
#
# Before we can start with the actual cashocs implementation of remeshing, we have
# to take a closer look at how we can define a geometry with Gmsh. For this, .geo
# files are used.
#
# :::{hint}
# A detailed documentation and tutorials regarding the generation of geometries
# and meshes with Gmsh can be found [here](https://gmsh.info/doc/texinfo/gmsh.html).
# :::
#
# The file {download}`./mesh/mesh.geo
# </../../demos/documented/shape_optimization/remeshing/mesh/mesh.geo>`
# describes our geometry.
#
# :::{important}
# Any user defined variables that should be also kept for the remeshing, such
# as the characteristic lengths, must be lower-case, so that cashocs can distinguish
# them from the other GMSH commands. Any user defined variable starting with an upper
# case letter is not considered for the .geo file created for remeshing and will,
# thus, probably cause an error.
#
# In our case of the .geo file, the characteristic length is defined as {cpp}`lc`,
# and this is used to specify the (local) size of the discretization via so-called
# size fields. Note, that this variable is indeed taken into consideration for
# the remeshing as it starts with a lower case letter.
# :::
#
# The resulting mesh file was created over the command line
# with the command
#
# ```bash
# gmsh ./mesh/mesh.geo -o ./mesh/mesh.msh -2
# ```
#
# :::{note}
# For the purpose of this tutorial it is recommended to leave the `./mesh/mesh.msh`
# file as it is. In particular, carrying out the above command will overwrite
# the file and is, thus, not recommended. The command just highlights, how one
# would / could use GMSH to define their own geometries and meshes for cashocs
# or FEniCS.
# :::
#
# The resulting file is {download}`./mesh/mesh.msh
# </../../demos/documented/shape_optimization/remeshing/mesh/mesh.msh>`.
# This .msh file can be converted to the .xdmf format by using
# {py:func}`cashocs.convert` or alternatively, via the command line
#
# ```bash
# cashocs-convert ./mesh/mesh.msh ./mesh/mesh.xdmf
# ```
#
# To ensure that cashocs also finds these files, we have to specify them in the file
# {download}`config.ini
# </../../demos/documented/shape_optimization/remeshing/config.ini>`.
# For this, we have the following lines
#
# ```{code-block} ini
# :caption: config.ini
# [Mesh]
# mesh_file = ./mesh/mesh.xdmf
# gmsh_file = ./mesh/mesh.msh
# geo_file = ./mesh/mesh.geo
# remesh = True
# show_gmsh_output = True
# ```
#
# With this, we have specified the paths to the mesh files and also enabled the
# remeshing as well as the verbose output of GMSH to the terminal, as explained in
# {ref}`the corresponding documentation of the config files <config_shape_mesh>`.
#
# :::{note}
# Note, that the paths given in the config file can be either absolute or relative.
# In the latter case, they have to be relative to the location of the cashocs script
# which is used to solve the problem.
# :::
#
# With this, we can now focus on the implementation in python.
#
# ### Initialization
#
# The program starts as {ref}`demo_shape_poisson`, with importing FEniCS and cashocs

# +
from fenics import *

import cashocs

# -

# Afterwards, we specify the path to the mesh file

mesh_file = "./mesh/mesh.xdmf"

# In order to be able to use a remeshing, we have to parametrize the inputs for
# {py:class}`cashocs.ShapeOptimizationProblem` w.r.t. to the mesh file. We do so with
# the following function, which we explain more detailed later on


def parametrization(mesh_file: str):
    config = cashocs.load_config("./config.ini")

    mesh, subdomains, boundaries, dx, ds, dS = cashocs.import_mesh(mesh_file)

    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V)
    p = Function(V)

    x = SpatialCoordinate(mesh)
    f = 2.5 * pow(x[0] + 0.4 - pow(x[1], 2), 2) + pow(x[0], 2) + pow(x[1], 2) - 1

    e = inner(grad(u), grad(p)) * dx - f * p * dx
    bcs = DirichletBC(V, Constant(0), boundaries, 1)

    J = cashocs.IntegralFunctional(u * dx)

    args = (e, bcs, J, u, p, boundaries)
    kwargs = {"config": config}

    return args, kwargs


# ::::{admonition} Description of the {python}`parametrization` function
#
# The code inside the {python}`parametrization` function looks nearly identical to the
# setup of the problem considered in {ref}`demo_shape_poisson`. First, we load the
# config file and define the mesh with the commands
# :::python
# config = cashocs.load_config("./config.ini")
#
# mesh, subdomains, boundaries, dx, ds, dS = cashocs.import_mesh(mesh_file)
# :::
#
# Then, we define the {py:class}`fenics.FunctionSpace` and the
# {py:class}`fenics.Function` objects used for the state and adjoint variables
# :::python
# V = FunctionSpace(mesh, "CG", 1)
# u = Function(V)
# p = Function(V)
# :::
#
# In the following lines, we define the UFL form of the right-hand side
# :::python
# x = SpatialCoordinate(mesh)
# f = 2.5 * pow(x[0] + 0.4 - pow(x[1], 2), 2) + pow(x[0], 2) + pow(x[1], 2) - 1
# :::
#
# Next, we define the weak form of the PDE constraint and the corresponding boundary
# conditions
# :::python
# e = inner(grad(u), grad(p)) * dx - f * p * dx
# bcs = DirichletBC(V, Constant(0), boundaries, 1)
# :::
#
# and then the cost functional
# :::python
# J = cashocs.IntegralFunctional(u * dx)
# :::
#
# with this, we have defined all arguments that are required for the
# {py:class}`cashocs.ShapeOptimizationProblem`. In order to make them usable, we return
# them in two objects, the first being the tuple {python}`args`, which defines the
# positional parameters of the {py:class}`cashocs.ShapeOptimizationProblem`. The second
# return object is the dictionary {python}`kwargs` containing the keyword arguments.
# These should be usable analogously to {python}`*args` and {python}`**kwargs`, i.e.,
# the unpacking operators {python}`*` and {python}`**` should yield the respective
# arguments. In particular, the return values of the {python}`parametrization` function
# have to be valid inputs so that
# :::python
# mesh_file = ...
# args, kwargs = parametrization(mesh_file)
# sop = cashocs.ShapeOptimizationProblem(*args, **kwargs)
# :::
#
# is well-defined. Therefore, in our code, we write
# :::python
# args = (e, bcs, J, u, p, boundaries)
# kwargs = {"config": config}
#
# return args, kwargs
# :::
# ::::
#
# ### The shape optimization problem
#
# To define the shape optimization problem, we now have to pass the
# {python}`parametrization` function as well as the {python}`mesh_file` to its
# constructor. Solving the problem is now, again, as easy as calling its {py:meth}`solve
# <cashocs.ShapeOptimizationProblem.solve>` method.

sop = cashocs.ShapeOptimizationProblem(parametrization, mesh_file)
sop.solve()

# We visualize the result with the lines

# +
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))

ax_mesh = plt.subplot(1, 2, 1)
fig_mesh = plot(sop.mesh_handler.mesh)
plt.title("Discretization of the optimized geometry")

ax_u = plt.subplot(1, 2, 2)
ax_u.set_xlim(ax_mesh.get_xlim())
ax_u.set_ylim(ax_mesh.get_ylim())
fig_u = plot(sop.states[0])
plt.colorbar(fig_u, fraction=0.046, pad=0.04)
plt.title("State variable u")

plt.tight_layout()
# plt.savefig('./img_remeshing.png', dpi=150, bbox_inches='tight')
# -

# and get the following results
# ![](/../../demos/documented/shape_optimization/remeshing/img_remeshing.png)
#
# ::::{note}
# The example for remeshing is somewhat artificial, as the problem does not
# actually need remeshing. Therefore, the tolerances used in the config file, i.e.,
#
# ```{code-block} ini
# :caption: config.ini
# [MeshQuality]
# tol_lower = 0.1
# tol_upper = 0.25
# ```
#
# are comparatively large. However, this problem still shows all relevant
# aspects of remeshing in cashocs and can, thus, be transferred to "harder"
# problems that require remeshing.
# ::::
