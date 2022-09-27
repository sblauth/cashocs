# Copyright (C) 2020-2022 Sebastian Blauth
#
# This file is part of cashocs.
#
# cashocs is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# cashocs is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with cashocs.  If not, see <https://www.gnu.org/licenses/>.

"""Implementation of a topology optimization problem."""

from __future__ import annotations

import copy
from typing import Callable, Literal, Union

import fenics
import ufl

from cashocs import _utils
from cashocs import io
from cashocs import types
from cashocs._optimization.topology_optimization import line_search_algorithm
from cashocs._optimization.topology_optimization import topology_optimization_algorithm


class TopologyOptimizationProblem:
    r"""A topology optimization problem.

    This class is used to define a topology optimization problem, and to solve
    it subsequently. For a detailed documentation, we refer to the
    :ref:`tutorial <tutorial_index>`. For easier input, when considering single (state
    or control) variables, these do not have to be wrapped into a list. Note, that in
    the case of multiple variables these have to be grouped into ordered lists, where
    ``state_forms``, ``bcs_list``, ``states``, ``adjoints`` have to have the same order
    (i.e. ``[y1, y2]`` and ``[p1, p2]``, where ``p1`` is the adjoint of ``y1`` and so
    on).
    """

    solver: Union[
        topology_optimization_algorithm.ConvexCombinationAlgorithm,
        topology_optimization_algorithm.SphereCombinationAlgorithm,
        line_search_algorithm.LineSearchTopologyAlgorithm,
    ]

    def __init__(  # pylint: disable=unused-argument
        self,
        state_forms: list[ufl.Form] | ufl.Form,
        bcs_list: list[list[fenics.DirichletBC]]
        | list[fenics.DirichletBC]
        | fenics.DirichletBC,
        cost_functional: list[types.CostFunctional],
        states: list[fenics.Function] | fenics.Function,
        adjoints: list[fenics.Function] | fenics.Function,
        levelset_function: fenics.Function,
        topological_derivative_neg: fenics.Function | ufl.Form,
        topological_derivative_pos: fenics.Function | ufl.Form,
        update_levelset: Callable,
        config: io.Config | None = None,
        topological_derivative_is_identical: bool = False,
        riesz_scalar_products: list[ufl.Form] | ufl.Form | None = None,
        re_normalize_levelset: bool = True,
        normalize_topological_derivative: bool = False,
        interpolation_scheme: Literal["angle", "volume"] = "angle",
        output_name: str | None = None,
        initial_guess: list[fenics.Function] | None = None,
        ksp_options: types.KspOptions | list[list[str | int | float]] | None = None,
        adjoint_ksp_options: types.KspOptions
        | list[list[str | int | float]]
        | None = None,
        desired_weights: list[float] | None = None,
    ) -> None:
        r"""Initializes the topology optimization problem.

        Args:
            state_forms: The weak form of the state equation (user implemented). Can be
                either a single UFL form, or a (ordered) list of UFL forms.
            bcs_list: The list of :py:class:`fenics.DirichletBC` objects describing
                Dirichlet (essential) boundary conditions. If this is ``None``, then no
                Dirichlet boundary conditions are imposed.
            cost_functional: UFL form of the cost functional. Can also be a list of
                summands of the cost functional
            states: The state variable(s), can either be a :py:class:`fenics.Function`,
                or a list of these.
            adjoints: The adjoint variable(s), can either be a
                :py:class:`fenics.Function`, or a (ordered) list of these.
            levelset_function: A :py:class:`fenics.Function` which represents the
                levelset function.
            topological_derivative_neg: The topological derivative inside the domain,
                where the levelset function is negative.
            topological_derivative_pos: The topological derivative inside the domain,
                where the levelset function is positive.
            update_levelset: A python function (without arguments) which is called to
                update the coefficients etc. when the levelset function is changed.
            config: The config file for the problem, generated via
                :py:func:`cashocs.create_config`. Alternatively, this can also be
                ``None``, in which case the default configurations are used, except for
                the optimization algorithm. This has then to be specified in the
                :py:meth:`solve <cashocs.OptimalControlProblem.solve>` method. The
                default is ``None``.
            topological_derivative_is_identical: A boolean flag, which indicates whether
                the topological derivatives inside and outside the domain coincide. As
                this is usually not the case, the default is ``False``.
            riesz_scalar_products: The scalar products of the control space. Can either
                be ``None`` or a single UFL form. If it is ``None``, the
                :math:`L^2(\Omega)` product is used (default is ``None``).
            re_normalize_levelset: A boolean flag, which indicates, whether the
                levelset function should be re-normalized after each iteration of the
                solution algorithm. The default is ``True``.
            normalize_topological_derivative: A boolean flag which is used to normalize
                the (generalized) topological derivative in each iteration of the
                solution algorithm. Default is ``False``.
            interpolation_scheme: One of ``angle`` or ``volume``. This determines
                whether the topological derivative is averaged by the volume of cells
                surrounding a vertex or by the angle.
            output_name: A path to a folder, where a .json file with the history of
                the optimization is saved.
            initial_guess: List of functions that act as initial guess for the state
                variables, should be valid input for :py:func:`fenics.assign`. Defaults
                to ``None``, which means a zero initial guess.
            ksp_options: A list of strings corresponding to command line options for
                PETSc, used to solve the state systems. If this is ``None``, then the
                direct solver mumps is used (default is ``None``).
            adjoint_ksp_options: A list of strings corresponding to command line options
                for PETSc, used to solve the adjoint systems. If this is ``None``, then
                the same options as for the state systems are used (default is
                ``None``).
            desired_weights: A list of values for scaling the cost functional terms. If
                this is supplied, the cost functional has to be given as list of
                summands. The individual terms are then scaled, so that term `i` has the
                magnitude of `desired_weights[i]` for the initial iteration. In case
                that `desired_weights` is `None`, no scaling is performed. Default is
                `None`.

        """
        self.state_forms = state_forms
        self.bcs_list = bcs_list
        self.cost_functional = cost_functional
        self.states = states
        self.adjoints = adjoints
        self.levelset_function = levelset_function
        self.topological_derivative_pos = topological_derivative_pos
        self.topological_derivative_neg = topological_derivative_neg
        self.update_levelset = update_levelset

        if config is None:
            self.config = io.Config()
        else:
            self.config = copy.deepcopy(config)

        self.topological_derivative_is_identical = topological_derivative_is_identical
        self.riesz_scalar_products = riesz_scalar_products
        self.re_normalize_levelset = re_normalize_levelset
        self.normalize_topological_derivative = normalize_topological_derivative
        self.interpolation_scheme = interpolation_scheme
        self.output_name = output_name

        self.mesh = self.levelset_function.function_space().mesh()
        self.dg0_space = fenics.FunctionSpace(self.mesh, "DG", 0)

    def solve(
        self,
        algorithm: str | None = None,
        tol: float = 1.0,
        max_iter: int = 100,
    ) -> None:
        """Solves the optimization problem.

        Args:
            algorithm: Selects the optimization algorithm. Valid choices are
                ``'gradient_descent'`` or ``'gd'`` for a gradient descent method,
                ``'conjugate_gradient'``, ``'nonlinear_cg'``, ``'ncg'`` or ``'cg'``
                for nonlinear conjugate gradient methods, ``'lbfgs'`` or ``'bfgs'``
                for limited memory BFGS methods, ``'sphere_combination'`` for Euler's
                method on the spehere, and ``'convex_combination'`` for a convex
                combination approach.
            tol: The relative tolerance used for the termination criterion (i.e. the
                angle between topological derivative and levelset function). Default is
                1 degree.
            max_iter: The maximum number of iterations the optimization algorithm
                can carry out before it is terminated. The default is 100.

        """
        if algorithm is None:
            self.algorithm = _utils.optimization_algorithm_configuration(
                self.config, algorithm
            )
        else:
            self.algorithm = algorithm

        self.tol = tol

        if self.algorithm.casefold() == "sphere_combination":
            self.solver = topology_optimization_algorithm.SphereCombinationAlgorithm(
                self
            )
        elif self.algorithm.casefold() == "convex_combination":
            self.solver = topology_optimization_algorithm.ConvexCombinationAlgorithm(
                self
            )
        else:
            self.solver = line_search_algorithm.LineSearchTopologyAlgorithm(
                self, self.algorithm
            )

        self.solver.run(tol, max_iter)
        self.solver.post_process()

    def plot_shape(self) -> None:
        """Visualize the current shape in a plot."""
        shape = fenics.Function(self.dg0_space)
        _utils.interpolate_levelset_function_to_cells(
            self.levelset_function, 1.0, 0.0, shape
        )
        fenics.plot(shape)
