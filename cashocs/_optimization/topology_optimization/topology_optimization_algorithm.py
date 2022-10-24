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

"""Solution algorithms for topology optimization."""

from __future__ import annotations

import abc
from typing import Callable, TYPE_CHECKING

import fenics
import numpy as np

from cashocs import _utils
from cashocs import nonlinear_solvers
from cashocs._optimization import optimization_algorithms
from cashocs._optimization.topology_optimization import topology_optimization_problem

if TYPE_CHECKING:
    from cashocs._optimization import optimal_control


class TopologyOptimizationAlgorithm(optimization_algorithms.OptimizationAlgorithm):
    """Parent class for solution algorithms for topology optimization."""

    def __init__(
        self,
        optimization_problem: topology_optimization_problem.TopologyOptimizationProblem,
    ) -> None:
        """Parent class for solvers for topology optimization problems.

        Args:
            optimization_problem: The corresponding optimization problem which shall be
                solved.

        """
        super().__init__(optimization_problem)

        self.levelset_function: fenics.Function = optimization_problem.levelset_function
        self.topological_derivative_neg = (
            optimization_problem.topological_derivative_neg
        )
        self.topological_derivative_pos = (
            optimization_problem.topological_derivative_pos
        )
        self.update_levelset: Callable = optimization_problem.update_levelset
        self.config = optimization_problem.config
        self.re_normalize_levelset = optimization_problem.re_normalize_levelset
        self.normalize_topological_derivative = (
            optimization_problem.normalize_topological_derivative
        )
        self.topological_derivative_is_identical = (
            optimization_problem.topological_derivative_is_identical
        )
        self.interpolation_scheme = optimization_problem.interpolation_scheme
        self.output_manager = optimization_problem.output_manager

        self._cashocs_problem: optimal_control.OptimalControlProblem = (
            optimization_problem._base_ocp
        )
        self._cashocs_problem.is_topology_problem = True

        self.mesh = optimization_problem.mesh
        self.cg1_space = fenics.FunctionSpace(self.mesh, "CG", 1)
        self.dg0_space = optimization_problem.dg0_space
        self.topological_derivative_vertex: fenics.Function = fenics.Function(
            self.cg1_space
        )
        self.levelset_function_prev = fenics.Function(self.cg1_space)

        self.cost_functional_values: list[float] = []
        self.angle_list: list[float] = []
        self.stepsize_list: list[float] = []

    @abc.abstractmethod
    def run(
        self,
        rtol: float | None = 0.0,
        atol: float | None = 1.0,
        max_iter: int | None = None,
    ) -> None:
        """Runs the optimization algorithm to solve the optimization problem.

        Args:
            rtol: Relative tolerance for the optimization algorithm.
            atol: Absolute tolerance for the optimization algorithm.
            max_iter: Maximum number of iterations for the optimization algorithm.

        """
        if rtol is None:
            self.rtol = 0.0
        else:
            self.rtol = rtol
        if atol is None:
            self.atol = 1.0
        else:
            self.atol = atol

        if max_iter is not None:
            self.config.set("OptimizationRoutine", "maximum_iterations", str(max_iter))

    def scalar_product(self, a: fenics.Function, b: fenics.Function) -> float:
        """Computes the scalar product between two functions.

        Args:
            a: The first function.
            b: The second function.

        Returns:
            The scalar product of a and b.

        """
        return self._cashocs_problem.form_handler.scalar_product([a], [b])

    def norm(self, a: fenics.Function) -> float:
        """Computes the norm of a function.

        Args:
            a: The function, whose norm is to be computed

        Returns:
            The norm of function a.

        """
        return float(np.sqrt(self.scalar_product(a, a)))

    def normalize(self, a: fenics.Function) -> None:
        """Normalizes a function.

        Args:
            a: The function which shall be normalized.

        """
        norm = self.norm(a)
        a.vector().vec().scale(1 / norm)
        a.vector().apply("")

    def compute_state_variables(self, cached: bool = True) -> None:
        """Solves the state system to compute the state variables.

        Args:
            cached: A boolean flag, which indicates, whether the system was solved
                already and only the cached solutions should be returned.

        """
        self.update_levelset()
        if not cached:
            self._cashocs_problem.state_problem.has_solution = False
        self._cashocs_problem.compute_state_variables()

    def average_topological_derivative(self) -> None:
        """Averages the topological derivative to make it piecewise continuous.

        The topological derivative is (typically) discontinuous and only defined for
        each cell of the mesh. This function averages the topological derivative  and
        computes a piecewise linear representation, which is used to update the levelset
        function.

        """
        # if self.interpolation_scheme == "average":
        #     _utils.interpolate_by_average(
        #         self.topological_derivative_pos,
        #         self.topological_derivative_neg,
        #         self.levelset_function,
        #         self.topological_derivative_vertex,
        #         1e-6,
        #     )
        if self.interpolation_scheme == "angle":
            _utils.interpolate_by_angle(
                self.topological_derivative_neg,
                self.topological_derivative_pos,
                self.levelset_function,
                self.topological_derivative_vertex,
            )
        else:
            _utils.interpolate_by_volume(
                self.topological_derivative_neg,
                self.topological_derivative_pos,
                self.levelset_function,
                self.topological_derivative_vertex,
            )

    def compute_gradient(self, cached: bool = True) -> None:
        """Computes the "topological gradient".

        This function solves the state and adjoint systems to compute the so-called
        topological gradient.

        Args:
            cached: A boolean flag, which indicates, whether the state and adjoint
                systems were solved already and only the cached solutions should be
                returned.

        """
        self.update_levelset()
        if not cached:
            self._cashocs_problem.state_problem.has_solution = False
            self._cashocs_problem.adjoint_problem.has_solution = False
        self._cashocs_problem.compute_adjoint_variables()

        if not self.topological_derivative_is_identical:
            self.average_topological_derivative()
        else:
            self.topological_derivative_vertex.vector()[:] = fenics.project(
                self.topological_derivative_pos, self.cg1_space
            ).vector()[:]

        if self.normalize_topological_derivative:
            norm = self.norm(self.topological_derivative_vertex)
            self.topological_derivative_vertex.vector().vec().scale(1.0 / norm)
            self.topological_derivative_vertex.vector().apply("")

        # self._smooth_topological_derivative()

    def _smooth_topological_derivative(self) -> None:
        cg1_space = self.topological_derivative_vertex.function_space()
        dx = fenics.Measure("dx", domain=cg1_space.mesh())
        self.comparison = fenics.Function(cg1_space)
        self.comparison.vector()[:] = self.topological_derivative_vertex.vector()[:]
        u = fenics.Function(cg1_space)
        v = fenics.TestFunction(cg1_space)

        res = (
            fenics.Constant(1e-3) * fenics.dot(fenics.grad(u), fenics.grad(v)) * dx
            + u * v * dx
            - self.topological_derivative_vertex * v * dx
        )
        nonlinear_solvers.newton_solve(res, u, bcs=[], is_linear=True)
        self.topological_derivative_vertex.vector()[:] = u.vector()[:]

    def compute_angle(self) -> float:
        """Computes the angle between topological gradient and levelset function.

        Returns:
            The angle between topological gradient and levelset function in degrees.

        """
        sp = (
            self.scalar_product(
                self.levelset_function, self.topological_derivative_vertex
            )
            / self.norm(self.levelset_function)
            / self.norm(self.topological_derivative_vertex)
        )
        sp = float(np.maximum(np.minimum(1.0, sp), -1.0))
        angle = np.arccos(sp)

        angle *= 360 / (2 * np.pi)

        return float(angle)

    def plot_shape(self) -> None:
        """Visualizes the current shape based on the levelset function."""
        shape = fenics.Function(self.dg0_space)
        _utils.interpolate_levelset_function_to_cells(
            self.levelset_function, 1.0, 0.0, shape
        )
        fenics.plot(shape)


class LevelSetTopologyAlgorithm(TopologyOptimizationAlgorithm):
    """Parent class for levelset-based solvers for topology optimization."""

    @abc.abstractmethod
    def move_levelset(self, stepsize: float) -> None:
        """Moves (updates) the levelset function based on the topological gradient.

        Args:
            stepsize: The stepsize which is used to update the levelset function.

        """
        pass

    def run(
        self,
        rtol: float | None = 0.0,
        atol: float | None = 1.0,
        max_iter: int | None = None,
    ) -> None:
        """Runs the optimization algorithm to solve the optimization problem.

        Args:
            rtol: Relative tolerance for the optimization algorithm.
            atol: Absolute tolerance for the optimization algorithm.
            max_iter: Maximum number of iterations for the optimization algorithm.

        """
        super().run(rtol, atol, max_iter)

        self.normalize(self.levelset_function)
        self.stepsize = 1.0
        self._cashocs_problem.state_problem.has_solution = False

        failed = False
        for k in range(self.config.getint("OptimizationRoutine", "maximum_iterations")):
            if failed:
                break

            self.iteration = k
            self.levelset_function_prev.vector().vec().aypx(
                0.0, self.levelset_function.vector().vec()
            )
            self.levelset_function_prev.vector().apply("")

            self._cashocs_problem.adjoint_problem.has_solution = False
            self.compute_gradient()

            self.objective_value = (
                self._cashocs_problem.reduced_cost_functional.evaluate()
            )
            self.cost_functional_values.append(self.objective_value)

            angle = self.compute_angle()
            self.gradient_norm = angle
            self.output()
            self.angle_list.append(angle)
            self.stepsize_list.append(self.stepsize)

            if self.convergence_test():
                print("\nOptimization successful!\n")
                break

            if k > 0:
                self.stepsize = float(np.minimum(2.0 * self.stepsize, 1.0))
            else:
                self.stepsize = float(
                    np.minimum(
                        self.config.getfloat("OptimizationRoutine", "initial_stepsize"),
                        1.0,
                    )
                )
            while True:
                self.move_levelset(self.stepsize)

                self._cashocs_problem.state_problem.has_solution = False
                self.compute_state_variables()
                cost_functional_new = (
                    self._cashocs_problem.reduced_cost_functional.evaluate()
                )
                if cost_functional_new <= self.objective_value:
                    break
                else:
                    self.stepsize *= 0.5

                if self.stepsize <= 1e-10:
                    if self.config.getboolean("OptimizationRoutine", "soft_exit"):
                        failed = True
                        break
                    else:
                        raise Exception("Stepsize computation failed.")


class ConvexCombinationAlgorithm(LevelSetTopologyAlgorithm):
    """An algorithm based on convex combination for topology optimization."""

    def move_levelset(self, stepsize: float) -> None:
        """Moves (updates) the levelset function based on the topological gradient.

        Args:
            stepsize: The stepsize which is used to update the levelset function.

        """
        self.levelset_function.vector().vec().aypx(
            0.0,
            (1.0 - stepsize) * self.levelset_function_prev.vector().vec()
            + stepsize
            * self.topological_derivative_vertex.vector().vec()
            / self.norm(self.topological_derivative_vertex),
        )
        self.levelset_function.vector().apply("")
        if self.re_normalize_levelset:
            self.normalize(self.levelset_function)


class SphereCombinationAlgorithm(LevelSetTopologyAlgorithm):
    """A solution algorithm which uses Euler's method on the sphere."""

    def move_levelset(self, stepsize: float) -> None:
        """Moves (updates) the levelset function based on the topological gradient.

        Args:
            stepsize: The stepsize which is used to update the levelset function.

        """
        angle = np.arccos(
            self.scalar_product(
                self.levelset_function_prev, self.topological_derivative_vertex
            )
            / self.norm(self.levelset_function_prev)
            / self.norm(self.topological_derivative_vertex)
        )

        a = (
            float(np.sin((1.0 - stepsize) * angle))
            * self.levelset_function_prev.vector().vec()
        )
        b = (
            float(
                np.sin(stepsize * angle) / self.norm(self.topological_derivative_vertex)
            )
            * self.topological_derivative_vertex.vector().vec()
        )
        self.levelset_function.vector().vec().aypx(
            0.0, float(1.0 / np.sin(angle)) * (a + b)
        )
        self.levelset_function.vector().apply("")
        if self.re_normalize_levelset:
            self.normalize(self.levelset_function)
