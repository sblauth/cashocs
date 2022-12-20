.. _tutorial_index:

cashocs Tutorial
================

Welcome to the cashocs tutorial. In the following, we present several example
programs that showcase how cashocs can be used to solve optimal control and
shape optimization problems.

.. include:: ../../../README.rst
    :start-after: readme_start_disclaimer
    :end-before: readme_end_disclaimer

However, we will also provide links to either the underlying theory of PDE
constrained optimization or to the relevant documentation of FEniCS in this
tutorial.

Note, that an overview over cashocs and its capabilities can be found in `Blauth, cashocs: A Computational, Adjoint-Based
Shape Optimization and Optimal Control Software <https://doi.org/10.1016/j.softx.2020.100646>`_.


.. toctree::
   :maxdepth: 2
   :caption: List of all demos:

   demos/optimal_control/optimal_control_index
   demos/shape_optimization/shape_optimization_index
   demos/cashocs_as_solver/solver_index


.. note::

    We recommend that you start with the introductory demos for
    optimal control problems, i.e., :ref:`demo_poisson` and :ref:`config_optimal_control`,
    as these demonstrate the basic ideas of cashocs. Additionally, they are a bit simpler than
    the introductory tutorials for shape optimization problems, i.e.,
    :ref:`demo_shape_poisson` and :ref:`config_shape_optimization`.

    Moreover, we note that some of cashocs functionality is explained only for optimal control, but not
    for shape optimization problems. This includes the contents of :ref:`demo_monolithic_problems`,
    :ref:`demo_picard_iteration`, :ref:`demo_heat_equation`, :ref:`demo_nonlinear_pdes`, 
    :ref:`demo_iterative_solvers`, :ref:`demo_state_constraints`, :ref:`demo_scalar_control_tracking`, and
    :ref:`demo_constraints`.
    However, the corresponding functionalities only deal with either the definition
    of the state system, its (numerical) solution, or the definition of suitable
    cost functionals. Therefore, they are straightforward to adapt to the case of
    shape optimization.

    On the contrary, the possibility to scale individual terms of a cost functional
    is only explained in :ref:`demo_scaling` for shape optimization problems, but
    works completely analogous for optimal control problems. 
    Moreover, the space mapping capabilities of cashocs are only documented for shape optimization 
    in :ref:`demo_space_mapping_semilinear_transmission` and 
    :ref:`demo_space_mapping_uniform_flow_distribution`. Again, space mapping works the same for 
    both optimal control and shape optimization.
