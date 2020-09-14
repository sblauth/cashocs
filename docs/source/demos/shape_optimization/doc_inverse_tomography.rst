.. _demo_inverse_tomography:

Inverse Problem in Electric Impedance Tomography
================================================

Problem Formulation
-------------------

For this demo, we investigate an inverse problem in the setting of electric
impedance tomography. It is based on the one used in the preprint
`Blauth, Nonlinear Conjugate Gradient Methods for PDE Constrained Shape Optimization
Based on Steklov-Poincare-Type Metrics <https://arxiv.org/abs/2007.12891>`_. The
problem reads

.. math::

    &\min_{\Omega} J(u, \Omega) = \sum_{i=1}{M} \frac{\nu_i}{2}
    \int_{\partial D} \left( u_i - m_i \right)^2 \text{ d}s \\
    &\text{subject to} \quad \left\lbrace \quad
    \begin{alignedat}{2}
        -\kappa^\text{in} \Delta u_i^\text{in} &= 0 \quad &&\text{ in } \Omega,\\
        -\kappa^\text{out} \Delta u_i^\text{out} &= 0 \quad &&\text{ in } D \setminus \Omega, \\
        \kappa^\text{out} \partial_n u^\text{out}_i &= f_i \quad &&\text{ on } \partial D, \\
        u^\text{out}_i &= u^\text{in}_i \quad &&\text{ on } \Gamma, \\
        \kappa^\text{out} \partial_n u^\text{out}_i &= \kappa^\text{in} \partial_n u^\text{in}_i \quad &&\text{ on } \Gamma, \\
        \int_{\partial D} u_i^\text{out} \text{ d}s &= 0.
    \end{alignedat}
    \right.

For a detailed description of this problem as well as its physical interpretation,
we refer to the preprint `Blauth, Nonlinear Conjugate Gradient Methods for PDE Constrained Shape Optimization
Based on Steklov-Poincare-Type Metrics <https://arxiv.org/abs/2007.12891>`_. To summarize the
setting briefly: The PDE constraints are models of electric impedance tomography.
There, some body :math:`\Omega` is inside another one (:math:`D \setminus \Omega`),
and both of them have different electric conductivities, namely :math:`\kappa^\text{in}`
for :math:`\Omega` and :math:`\kappa^\text{out}` for :math:`D \setminus \Omega`.
Moreover, :math:`f_i` denotes the electric current applied to the outer boundary
:math:`\partial D`. We assume that we have measurements :math:`m_i` of the electric
potential on the boundary given, and want to identify the shape of the interior
body from these measurements. This is modeled by the tracking-type cost functional,
which aims at minimizing the :math:`L^2(\Gamma)` distance between the simulated
and measured electric potential.

For our demo, we use as domain the unit square :math:`D = (0,1)^2`, and the initial
geometry of :math:`\Omega` is a square inside :math:`D`. We consider the case of
three measurements, so that :math:`M = 3`. For these, we consider the following cases

.. math::

    f_1 = 1 \quad \text{ on } \Gamma^l \cup \Gamma^r \qquad \text{ and } \qquad f_1 = -1 \quad \text{ on } \Gamma^t \cup \Gamma^b,\\
    f_2 = 1 \quad \text{ on } \Gamma^l \cup \Gamma^t \qquad \text{ and } \qquad f_2 = -1 \quad \text{ on } \Gamma^r \cup \Gamma^b,\\
    f_3 = 1 \quad \text{ on } \Gamma^l \cup \Gamma^b \qquad \text{ and } \qquad f_3 = -1 \quad \text{ on } \Gamma^r \cup \Gamma^t,

where :math:`\Gamma^l, \Gamma^r, \Gamma^t, \Gamma^b` are the left, right, top, and
bottom sides of the unit square.

Implementation
--------------

The complete python code can be found in the file :download:`demo_inverse_tomography.py </../../demos/documented/shape_optimization/inverse_tomography/demo_inverse_tomography.py>`,
and the corresponding config can be found in :download:`config.ini </../../demos/documented/shape_optimization/inverse_tomography/config.ini>`.


Initialization and generation of synthetic measurements
*******************************************************

We start our code by importing FEniCS and CASHOCS ::

    from fenics import *
    import cashocs

Next, we directly define the sample values of :math:`\kappa^\text{in}` and
:math:`\kappa^\text{out}` since these are needed later on ::

    kappa_out = 1e0
    kappa_in = 1e1

In the next part, we generate synthetic measurements, which correspond to :math:`m_i`.
To do this, we define a function :py:func:`generate_measurements()` as follows ::

    def generate_measurements():
    	mesh, subdomains, boundaries, dx, ds, dS = cashocs.import_mesh('./mesh/reference.xdmf')

    	cg_elem = FiniteElement('CG', mesh.ufl_cell(), 1)
    	r_elem = FiniteElement('R', mesh.ufl_cell(), 0)
    	V = FunctionSpace(mesh, MixedElement([cg_elem, r_elem]))

    	u, c = TrialFunctions(V)
    	v, d = TestFunctions(V)

        a = kappa_out*inner(grad(u), grad(v))*dx(1) + kappa_in*inner(grad(u), grad(v))*dx(2) + u*d*ds + v*c*ds
    	L1  = Constant(1)*v*(ds(3) + ds(4)) + Constant(-1)*v*(ds(1) + ds(2))
    	L2  = Constant(1)*v*(ds(3) + ds(2)) + Constant(-1)*v*(ds(1) + ds(4))
    	L3  = Constant(1)*v*(ds(3) + ds(1)) + Constant(-1)*v*(ds(2) + ds(4))

    	meas1 = Function(V)
    	meas2 = Function(V)
    	meas3 = Function(V)
    	solve(a==L1, meas1)
    	solve(a==L2, meas2)
    	solve(a==L3, meas3)

    	m1, _ = meas1.split(True)
    	m2, _ = meas2.split(True)
    	m3, _ = meas3.split(True)

    	return [m1, m2, m3]

.. note::

    The code executed in :py:func:`generate_measurements()` is used to solve the
    state problem on a reference domain, given by the mesh ``./mesh/reference.xdmf``.
    This mesh has the domain :math:`\Omega` as a circle in the center of the unit
    square. To distinguish between these two, we note that :math:`D \setminus \Omega`
    has the index / marker 1 and that :math:`\Omega` has the index / marker 2 in
    the corresponding GMSH file, which is then imported into ``subdomains``.

    Note, that we have to use a mixed finite element method to incorporate the
    integral constraint on the electric potential. The second component of the
    corresponding :py:class:`fenics.MixedFunctionSpace` ``V`` is just a scalar, one-dimensional,
    real element. The actual PDE constraint is then given by the part ::

        kappa_out*inner(grad(u), grad(v))*dx(1) + kappa_in*inner(grad(u), grad(v))*dx(2)

    and the integral constraint is realized with the saddle point formulation ::

        u*d*ds + v*c*ds

    The right hand sides ``L1``, ``L2``, and ``L3`` are just given by the Neumann
    boundary conditions as specified above.

    Finally, these PDEs are then solved via the :py:func:`fenics.solve` command,
    and then only the actual solution of the PDE (and not the Lagrange multiplier
    for the integral constraint) is returned.

As usual, we load the config into CASHOCS with the line ::

    config = cashocs.create_config('./config.ini')

Afterwards, we import the mesh into CASHOCS ::

    mesh, subdomains, boundaries, dx, ds, dS = cashocs.import_mesh('./mesh/mesh.xdmf')

Next, we define the :py:class:`fenics.FunctionSpace` object, which consists of
CG1 elements together with a scalar, real element, which acts as a Lagrange multiplier
for the integral constraint ::

    cg_elem = FiniteElement('CG', mesh.ufl_cell(), 1)
    r_elem = FiniteElement('R', mesh.ufl_cell(), 0)
    V = FunctionSpace(mesh, MixedElement([cg_elem, r_elem]))

Next, we compute the synthetic measurements via ::

    measurements = generate_measurements()


The PDE constraint
******************

Let us now investigate how the PDE constraint is defined. As we have a mixed
finite element problem, we proceed analogously to :ref:`demo_monolithic_problems`
and define the first state equation with the following lines ::

    uc1 = Function(V)
    u1, c1 = split(uc1)
    pd1 = Function(V)
    p1, d1 = split(pd1)
    e1 = kappa_out*inner(grad(u1), grad(p1))*dx(1) + kappa_in*inner(grad(u1), grad(p1))*dx(2) + u1*d1*ds + p1*c1*ds \
    	 - Constant(1)*p1*(ds(3) + ds(4)) - Constant(-1)*p1*(ds(1) + ds(2))

The remaining two experiments are defined completely analogously::

    uc2 = Function(V)
    u2, c2 = split(uc2)
    pd2 = Function(V)
    p2, d2 = split(pd2)
    e2 = kappa_out*inner(grad(u2), grad(p2))*dx(1) + kappa_in*inner(grad(u2), grad(p2))*dx(2) + u2*d2*ds + p2*c2*ds \
    	 - Constant(1)*p2*(ds(3) + ds(2)) - Constant(-1)*p2*(ds(1) + ds(4))

    uc3 = Function(V)
    u3, c3 = split(uc3)
    pd3 = Function(V)
    p3, d3 = split(pd3)
    e3 = kappa_out*inner(grad(u3), grad(p3))*dx(1) + kappa_in*inner(grad(u3), grad(p3))*dx(2) + u3*d3*ds + p3*c3*ds \
    	 - Constant(1)*p3*(ds(3) + ds(1)) - Constant(-1)*p3*(ds(2) + ds(4))

Finally, we group together the state equations as well as the state and adjoint variables
to (ordered) lists, as in :ref:`demo_multiple_variables` ::

    e = [e1, e2, e3]
    u = [uc1, uc2, uc3]
    p = [pd1, pd2, pd3]

Since the problem only has Neumann boundary conditions, we use ::

    bcs = None

the specify this.


The cost functional
*******************

The cost functional is then defined by first creating the individual summands,
and then summing them up::

    J1 = Constant(0.5)*pow(u1 - measurements[0], 2)*ds
    J2 = Constant(0.5)*pow(u2 - measurements[1], 2)*ds
    J3 = Constant(0.5)*pow(u3 - measurements[2], 2)*ds

    J = J1 + J2 + J3


The shape optimization problem can now be created as in :ref:`demo_shape_poisson`
and can be solved as easily, with the commands ::

    sop = cashocs.ShapeOptimizationProblem(e, bcs, J, u, p, boundaries, config)
    sop.solve()

The results should look like this

.. image:: /../../demos/documented/shape_optimization/inverse_tomography/img_inverse_tomography.png
