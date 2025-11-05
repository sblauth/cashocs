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
# (demo_mpi_custom)=
# # Using a custom MPI communicator
#
# ## Topic
#
# In this demo, we explain how custom MPI communicators can be used with cashocs. Before
# going through this demo, it is recommended to look at {ref}`demo_mpi_comm_self`. Here,
# we show how we can split a MPI communicator to create new ones with the goal of
# solving similar problems on different MPI groups, where each group consists of
# multiple MPI processes. So this is similar to {ref}`demo_mpi_comm_self`, but now each
# sub-problem is also solved in parallel.
#
# ## Implementation
#
# The complete python code can be found in the file {download}`demo_mpi_custom.py
# </../../demos/documented/misc/mpi_custom/demo_mpi_custom.py>` and the
# corresponding config can be found in {download}`config.ini
# </../../demos/documented/misc/mpi_custom/config.ini>`.
#
# Let us, again, start with loading the required Python modules for the demo

# +
from fenics import *
from mpi4py import MPI
import numpy as np

import cashocs

# -

# Again, we load the {python}`MPI` submodule of the {python}`mpi4py` package, which is
# documented [here](https://mpi4py.readthedocs.io).
#
# ### MPI Initialization
#
# To create a new MPI communicator, we split the COMM_WORLD communicator into a new one,
# as described
# [here](https://mpitutorial.com/tutorials/introduction-to-groups-and-communicators/).

comm_world = MPI.COMM_WORLD
rank = comm_world.rank
color = np.mod(rank, 2)
comm = comm_world.Split(color, rank)
comm.Set_name("COMM_CUSTOM")

# This code splits COMM_WORLD into two groups, based on the color variable. MPI
# processes with even rank get the color {python}`0`, whereas processes with odd rank
# get the color {python}`1`.
#
# After having defined the new MPI communicator, we can proceed analogously to
# {ref}`demo_mpi_comm_self`. First, we define the result directories based on the color
# variable defined above. Each color becomes a new group of MPI processes, so each group
# has to write their output to a different result directory.

result_dir = f"./results_group_{color}"
config = cashocs.load_config("./config.ini")
config.set("Output", "result_dir", result_dir)

# :::{important}
# As discussed in {ref}`demo_mpi_comm_self` it is necessary that each group of MPI
# processes writes their files to a different result directory. If some groups write to
# the same directory, the produced files might be corrupt or errors can occur.
# :::
#
# As before, we supply the cashocs logging module with the used MPI communicator and
# attach different log files for the MPI groups

cashocs.log.set_comm(comm)
cashocs.log.add_logfile(f"./log_group_{color}.txt", level=cashocs.log.INFO)

# Finally, we load the computational mesh with the same communicator specified
# above

mesh, subdomains, boundaries, dx, ds, dS = cashocs.import_mesh(
    "./mesh/mesh.xdmf", comm=comm
)

# :::{important}
# It is required that the MPI communicators supplied to {py:func}`cashocs.import_mesh`
# and {py:meth}`cashocs.log.set_comm` are the same, otherwise problems can and will
# occur.
# :::
#
# ### Defining the PDE problem
#
# We again use the simple model example from {ref}`demo_shape_poisson` for this demo.
# As in {ref}`demo_mpi_comm_self`, we will create two MPI groups for this demo, so that
# we solve the problem with two different right-hand sides. But our approach with a
# custom MPI communicator will allow us to solve each problem in parallel. For this
# reason, we continue as in the {ref}`previous demo <demo_mpi_comm_self>`:

V = FunctionSpace(mesh, "CG", 1)
u = Function(V)
p = Function(V)
x = SpatialCoordinate(mesh)

# Now, we define the two different right-hand sides, one for each group, where we again
# use the color variable to distinguish between them

if color == 0:
    f = 2.5 * pow(x[0] + 0.4 - pow(x[1], 2), 2) + pow(x[0], 2) + pow(x[1], 2) - 1
else:
    f = 3.5 * pow(x[1] + 0.6 - pow(x[0], 2), 2) + pow(x[0], 2) + pow(x[1], 2) - 1

# The rest of the demo consists of setting up the state equation and the shape
# optimization problem, analogously to before

# +
e = inner(grad(u), grad(p)) * dx - f * p * dx
bcs = DirichletBC(V, Constant(0), boundaries, 1)

J = cashocs.IntegralFunctional(u * dx)

sop = cashocs.ShapeOptimizationProblem(e, bcs, J, u, p, boundaries, config=config)
sop.solve(algorithm="bfgs")

# -

# ::::{note}
# To run this demo (in parallel), we have use the command
#
# ```{code-block} bash
# mpirun -n 4 python demo_mpi_comm_self.py
# ```
#
# where the option {bash}`-n 4` specifies that we want to use four MPI tasks to run the
# problem. As described in the beginning, the 4 processes will be split into two groups
# of 2 MPI processes each, so that each shape optimization problem will be solved in
# parallel.
# ::::
#
# ### Results
#
# From the output we observe that we, indeed, solve two different problems with
# different right-hand sides, each of them in parallel. Additionally, we can see in the
# two produced log files that each group of MPI process did, in fact, do different
# things. As the meshes for each sub-problem are distributed between the two MPI
# processes (per group), we cannot easily visualize the results with matplotlib.
# However, you are encourage to run the demo and save the results as XDMF files and
# inspect them with Paraview.
