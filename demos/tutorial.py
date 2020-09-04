# Copyright (C) 2020 Sebastian Blauth
#
# This file is part of CASHOCS.
#
# CASHOCS is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# CASHOCS is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with CASHOCS.  If not, see <https://www.gnu.org/licenses/>.

"""Cashocs is a computational, adjoint based, shape optimization and optimal control software for python.
On this site, we are taking a deep look into its capabilities regarding these two fields of optimization,
which is done with various demo programs. Our intention is to give the user a detailed guide and overview
over what cashocs can do, and to also give an introduction of how to use it. As cashocs takes more of
a first optimize, then discretize approach, its features are manifold, but some familiarity with
PDE constrained optimization may be required to solve the problems successfully. In particular, cashocs
should **not** be treated as a black box solver for these kinds of problems.

For beginners, we recommend reading through the demos roughly in the same order as they are presented.
In particular, we also strongly recommend that you read the optimal control demos first, before you proceed
to the shape optimization demos, due to the following. First, optimal control is a much more established field
in PDE constrained optimization, and more well-known in general. Second, many concepts of the optimal control
demos are directly transferred to the shape optimization setting. Things such as the definition of (multiple)
PDE constraints, the different solution techniques for systems of PDEs (a monolithic approach vs a Picard iteration),
the treatment of time-dependent and nonlinear problems, as well as the choice of linear solvers are all applicable
to both kinds of problems. While we will still recall some of these concepts in the shape optimization demos,
they are explained in most detail in the demos for optimal control, so that one should look there first.

Without further ado, let us take a look at the demo programs.

# Optimal Control Demos

Here, we showcase the capabilities of cashocs for optimal control problems. As stated earlier,
newcomers should start reading here to get a grasp of how the problem definition in cashocs works,
as this is identical for optimal control and shape optimization problems.

.. include:: ./documented/optimal_control/01_poisson/doc_poisson.md

<br/>

.. include:: ./documented/optimal_control/01_poisson/doc_optimal_control_config.md

<br/>

.. include:: ./documented/optimal_control/02_box_constraints/doc_box_constraints.md

<br/>

.. include:: ./documented/optimal_control/03_neumann_control/doc_neumann_control.md

<br/>

.. include:: ./documented/optimal_control/04_multiple_variables/doc_multiple_variables.md

<br/>

.. include:: ./documented/optimal_control/05_monolithic_problems/doc_monolithic_problems.md

<br/>

.. include:: ./documented/optimal_control/06_picard_iteration/doc_picard_iteration.md

<br/>

.. include:: ./documented/optimal_control/07_stokes/doc_stokes.md

<br/>

.. include:: ./documented/optimal_control/08_heat_equation/doc_heat_equation.md

<br/>

.. include:: ./documented/optimal_control/09_nonlinear_pdes/doc_nonlinear_pdes.md

<br/>

.. include:: ./documented/optimal_control/10_dirichlet_control/doc_dirichlet_control.md

<br/>

.. include:: ./documented/optimal_control/11_iterative_solvers/doc_iterative_solvers.md

<br/>

.. include:: ./documented/optimal_control/12_state_constraints/doc_state_constraints.md

<br/>

.. include:: ./documented/optimal_control/13_sparse_control/doc_sparse_control.md



# Shape Optimization

In this second part, we take a look at shape optimization problems. It is assumed that
the reader is now somewhat familiar with the problem definition in cashocs, if not, please
refer to the optimal control demos.

<br/>

.. include:: ./documented/shape_optimization/01_shape_poisson/doc_shape_poisson.md

.. include:: ./documented/shape_optimization/01_shape_poisson/doc_shape_optimization_config.md

"""
