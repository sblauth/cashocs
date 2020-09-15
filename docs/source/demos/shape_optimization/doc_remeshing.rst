.. _demo_remeshing:

Remeshing with CASHOCS
======================

Problem Formulation
-------------------

In this tutorial, we take a close look at how remeshing works in CASHOCS. To keep
this discussion simple, we take a look at the model problem already investigated
in :ref:`demo_shape_poisson`, i.e.,

.. math::

    &\min_\Omega J(u, \Omega) = \int_\Omega u \text{ d}x \\
    &\text{subject to} \quad \left\lbrace \quad
    \begin{alignedat}{2}
    -\Delta u &= f \quad &&\text{ in } \Omega,\\
    u &= 0 \quad &&\text{ on } \Gamma.
    \end{alignedat} \right.

As before, we use the unit disc :math:`\Omega = \{ x \in \mathbb{R}^2 \,\mid\, \lvert\lvert x \rvert\rvert_2 < 1 \}`,
as initial geometry, and the right-hand side :math:`f` is given by

.. math:: f(x) = 2.5 \left( x_1 + 0.4 - x_2^2 \right)^2 + x_1^2 + x_2^2 - 1.



Implementation
--------------

The complete python code can be found in the file :download:`demo_remeshing.py </../../demos/documented/shape_optimization/remeshing/demo_remeshing.py>`,
and the corresponding config can be found in :download:`config.ini </../../demos/documented/shape_optimization/remeshing/config.ini>`.
The corresponding mesh files are :download:`./mesh/mesh.geo </../../demos/documented/shape_optimization/remeshing/mesh/mesh.geo>` and
:download:`./mesh/mesh.msh </../../demos/documented/shape_optimization/remeshing/mesh/mesh.msh>`.

Pre-Processing with GMSH
************************

Before we can start with the actual CASHOCS implementation of remeshing, we have
to take a closer look at how we can define a geometry with GMSH. For this, .geo
files are used.

.. hint::

    A detailed documentation and tutorials regarding the generation of geometries
    and meshes with GMSH can be found `here <https://gmsh.info/doc/texinfo/gmsh.html>`_.

The file :download:`./mesh/mesh.geo </../../demos/documented/shape_optimization/remeshing/mesh/mesh.geo>`
describes our geometry.

.. important::

    Any user defined variables that should be also kept for the remeshing, such
    as the characteristic lengths, must be lower case, so that CASHOCS can distinguish them
    from the other GMSH commands. Any user defined variable starting with an upper
    case letter is not considered for the .geo file created for remeshing and will,
    thus, probably cause an error.

    In our case of the .geo file, the characteristic length is defined as ``lc``,
    and this is used to specify the (local) size of the discretization via so-called
    size fields. Note, that this variable is indeed taken into consideration for
    the remeshing as it starts with a lower case letter.

The resulting mesh file was created over the command line
with the command ::

    gmsh ./mesh/mesh.geo -o ./mesh/mesh.msh -2

.. note::

    For the purpose of this tutorial it is recommended to leave the ``./mesh/mesh.msh``
    file as it is. In particular, carrying out the above command will overwrite
    the file and is, thus, not recommended. The command just highlights, how one
    would / could use GMSH to define their own geometries and meshes for CASHOCS
    or FEniCS.

The resulting file is :download:`./mesh/mesh.msh </../../demos/documented/shape_optimization/remeshing/mesh/mesh.msh>`.
This .msh file can be converted to the .xdmf format by using :ref:`cashocs-convert <cashocs_convert>`
as follows::

    cashocs-convert ./mesh/mesh.msh ./mesh/mesh.xdmf

from the command line.

.. hint::

    As the :ref:`cashocs-convert <cashocs_convert>` merely **converts** the .msh
    file to .xdmf, the user may very well use this command.

To ensure that CASHOCS also finds these files, we have to specify them in the file
:download:`config.ini </../../demos/documented/shape_optimization/remeshing/config.ini>`.
For this, we have the following lines ::

    [Mesh]
    mesh_file = ./mesh/mesh.xdmf
    gmsh_file = ./mesh/mesh.msh
    geo_file = ./mesh/mesh.geo
    remesh = True
    show_gmsh_output = True

With this, we have specified the paths to the mesh files and also enabled the remeshing
as well as the verbose output of GMSH to the terminal, as explained in :ref:`the
corresponding documentation of the config files <config_shape_mesh>`.

.. note::

    Note, that the paths given in the config file can be either absolute or relative.
    In the latter case, they have to be relative to the location of the CASHOCS script
    which is used to solve the problem.

With this, we can now focus on the implementation in python.

Initialization
**************

The program starts as :ref:`demo_shape_poisson`, with the following lines ::

    from fenics import *
    import cashocs


    config = cashocs.create_config('./config.ini')

with which we import FEniCS and CASHOCS, and read the config file. The mesh and
all other related objects are created with the command ::

    mesh, subdomains, boundaries, dx, ds, dS = cashocs.import_mesh(config)

Note, that in contrast to :ref:`demo_shape_poisson`, we cannot use a built-in mesh for this
tutorial since remeshing is only available for meshes generated by GMSH.

.. important::

    It is important to note that we have to pass the config as argument to
    :py:func:`import_mesh <cashocs.import_mesh>`. The alternative syntax ::

        mesh, subdomains, boundaries, dx, ds, dS = cashocs.import_mesh(./mesh/mesh.xdmf)

    is **NOT** equivalent for remeshing, even though the definition in the config
    file points to the same object, where the corresponding line reads ::

        mesh_file = ./mesh/mesh.xdmf

Definition of the state system
******************************

The definition of the state system is now completely analogous to the one in
:ref:`demo_shape_poisson`. Here, we just repeat the code for the sake of
completeness ::

    V = FunctionSpace(mesh, 'CG', 1)
    u = Function(V)
    p = Function(V)

    x = SpatialCoordinate(mesh)
    f = 2.5*pow(x[0] + 0.4 - pow(x[1], 2), 2) + pow(x[0], 2) + pow(x[1], 2) - 1

    e = inner(grad(u), grad(p))*dx - f*p*dx
    bcs = DirichletBC(V, Constant(0), boundaries, 1)


The shape optimization problem
******************************

The definition of the :py:class:`ShapeOptimizationProblem <cashocs.ShapeOptimizationProblem>`
as well as its solution is now also completely analogous to :ref:`demo_shape_poisson`,
and is done with the lines ::

    J = u*dx

    sop = cashocs.ShapeOptimizationProblem(e, bcs, J, u, p, boundaries, config)
    sop.solve()



The results should look like the one of :ref:`demo_shape_poisson`:

.. image:: /../../demos/documented/shape_optimization/remeshing/img_remeshing.png

.. note::

    The example for remeshing is somewhat artificial, as the problem does not
    actually need remeshing. Therefore, the tolerances used in the config file, i.e., ::

        tol_lower = 0.1
        tol_upper = 0.25

    are comparatively large. However, this problem still shows all relevant
    aspects of remeshing in CASHOCS and can, thus, be transferred to "harder"
    problems that require remeshing.
