.. _demo_custom_scalar_product:

Custom Scalar Products for Shape Gradient Computation
=====================================================

Problem Formulation
-------------------

In this demo, we show how to supply a custom bilinear form for the computation
of the shape gradient with cashocs. For the sake of simplicity, we again consider
our model problem from :ref:`demo_shape_poisson`, given by

.. math::

    &\min_\Omega J(u, \Omega) = \int_\Omega u \text{ d}x \\
    &\text{subject to} \quad \left\lbrace \quad
    \begin{alignedat}{2}
    -\Delta u &= f \quad &&\text{ in } \Omega,\\
    u &= 0 \quad &&\text{ on } \Gamma.
    \end{alignedat} \right.


For the initial domain, we use the unit disc :math:`\Omega = \{ x \in \mathbb{R}^2 \,\mid\, \lvert\lvert x \rvert\rvert_2 < 1 \}`, and the right-hand side :math:`f` is given by

.. math:: f(x) = 2.5 \left( x_1 + 0.4 - x_2^2 \right)^2 + x_1^2 + x_2^2 - 1.


Implementation
--------------

The complete python code can be found in the file :download:`demo_custom_scalar_product.py </../../demos/documented/shape_optimization/custom_scalar_product/demo_custom_scalar_product.py>`,
and the corresponding config can be found in :download:`config.ini </../../demos/documented/shape_optimization/custom_scalar_product/config.ini>`.


Initialization
**************

The demo program closely follows the one from :ref:`demo_shape_poisson`, so that up
to the definition of the :py:class:`ShapeOptimizationProblem <cashocs.ShapeOptimizationProblem>`,
the code is identical to the one in :ref:`demo_shape_poisson`, and given by ::

    from fenics import *

    import cashocs

    config = cashocs.load_config("./config.ini")

    meshlevel = 15
    degree = 1
    dim = 2
    mesh = UnitDiscMesh.create(MPI.comm_world, meshlevel, degree, dim)
    dx = Measure("dx", mesh)
    boundary = CompiledSubDomain("on_boundary")
    boundaries = MeshFunction("size_t", mesh, dim=1)
    boundary.mark(boundaries, 1)
    ds = Measure("ds", mesh, subdomain_data=boundaries)

    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V)
    p = Function(V)

    x = SpatialCoordinate(mesh)
    f = 2.5 * pow(x[0] + 0.4 - pow(x[1], 2), 2) + pow(x[0], 2) + pow(x[1], 2) - 1

    e = inner(grad(u), grad(p)) * dx - f * p * dx
    bcs = DirichletBC(V, Constant(0), boundaries, 1)

    J = cashocs.IntegralFunctional(u * dx)


Definition of the scalar product
********************************

To define the scalar product that shall be used for the shape optimization, we can
proceed analogously to :ref:`demo_neumann_control` and define the corresponding bilinear form
in FEniCS. However, note that one has to use a :py:class:`fenics.VectorFunctionSpace` with
piecewise linear Lagrange elements, i.e., one has to define the corresponding function space as ::

    VCG = VectorFunctionSpace(mesh, "CG", 1)

With this, we can now define the bilinear form as follows ::

    shape_scalar_product = (
        inner((grad(TrialFunction(VCG))), (grad(TestFunction(VCG)))) * dx
        + inner(TrialFunction(VCG), TestFunction(VCG)) * dx
    )

.. note::

    Note, that we cannot use the formulation ::

        shape_scalar_product = inner((grad(TrialFunction(VCG))), (grad(TestFunction(VCG))))*dx

    as this would not yield a coercive bilinear form for this problem. This is due to
    the fact that the entire boundary of :math:`\Omega` is variable. Hence, we actually
    need this second term.

Finally, we can set up the :py:class:`ShapeOptimizationProblem <cashocs.ShapeOptimizationProblem>`
and solve it with the lines ::

    sop = cashocs.ShapeOptimizationProblem(
        e, bcs, J, u, p, boundaries, config, shape_scalar_product=shape_scalar_product
    )
    sop.solve()


The result of the optimization looks like this

.. image:: /../../demos/documented/shape_optimization/custom_scalar_product/img_custom_scalar_product.png
