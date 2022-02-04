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

"""Parent class for optimization algorithms for PDE constrained optimization."""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import fenics

from cashocs import _exceptions
from cashocs import _loggers

if TYPE_CHECKING:
    from cashocs._optimization import optimization_problem as op


class OptimizationAlgorithm(abc.ABC):
    """Base class for optimization algorithms."""

    def __init__(self, optimization_problem: op.OptimizationProblem) -> None:
        """
        Args:
            optimization_problem: The corresponding optimization problem.
        """

        self.line_search_broken = False
        self.has_curvature_info = False

        self.form_handler = optimization_problem.form_handler
        self.state_problem = optimization_problem.state_problem
        self.config = self.state_problem.config
        self.adjoint_problem = optimization_problem.adjoint_problem

        self.gradient_problem = optimization_problem.gradient_problem
        self.cost_functional = optimization_problem.reduced_cost_functional
        self.gradient = optimization_problem.gradient
        self.search_direction = [
            fenics.Function(V) for V in self.form_handler.control_spaces
        ]

        self.optimization_variable_abstractions = (
            optimization_problem.optimization_variable_abstractions
        )

        self.gradient_norm = 1.0
        self.iteration = 0
        self.objective_value = 1.0
        self.gradient_norm_initial = 1.0
        self.relative_norm = 1.0
        self.stepsize = 1.0

        self.require_control_constraints = False

        self.requires_remeshing = False
        self.remeshing_its = False

        self.converged = False
        self.converged_reason = 0

        self.rtol = self.config.getfloat("OptimizationRoutine", "rtol", fallback=1e-3)
        self.atol = self.config.getfloat("OptimizationRoutine", "atol", fallback=0.0)
        self.maximum_iterations = self.config.getint(
            "OptimizationRoutine", "maximum_iterations", fallback=100
        )
        self.soft_exit = self.config.getboolean(
            "OptimizationRoutine", "soft_exit", fallback=False
        )

        if optimization_problem.is_shape_problem:
            self.temp_dict = optimization_problem.temp_dict
        else:
            self.temp_dict = None

        self.output_manager = optimization_problem.output_manager

    @abc.abstractmethod
    def run(self) -> None:
        """Solves the optimization problem."""

        pass

    def compute_gradient_norm(self) -> float:
        """Computes the norm of the gradient.

        Returns:
            The computed gradient norm.
        """

        return self.optimization_variable_abstractions.compute_gradient_norm()

    def output(self) -> None:
        """Writes the output (to console and files)."""

        self.output_manager.output(self)

    def output_summary(self) -> None:
        """Writes the summary of the optimization (to files and console)."""

        self.output_manager.output_summary(self)

    def nonconvergence(self) -> bool:
        """Checks for nonconvergence of the solution algorithm.

        Returns:
            A flag which is True, when the algorithm did not converge
        """

        if self.iteration >= self.maximum_iterations:
            self.converged_reason = -1
        if self.line_search_broken:
            self.converged_reason = -2
        if self.requires_remeshing:
            self.converged_reason = -3
        if self.remeshing_its:
            self.converged_reason = -4

        if self.converged_reason < 0:
            return True
        else:
            return False

    def _exit(self, message: str) -> None:

        if self.soft_exit:
            print(message)
        else:
            raise _exceptions.NotConvergedError("Optimization Algorithm", message)

    def post_processing(self) -> None:
        """Does a post processing after the optimization algorithm terminates."""

        if self.converged:
            self.output()
            self.output_summary()

        else:
            # maximum iterations reached
            if self.converged_reason == -1:
                self.output()
                self.output_summary()
                self._exit("Maximum number of iterations exceeded.")

            # Armijo line search failed
            elif self.converged_reason == -2:
                self.iteration -= 1
                self.output_summary()
                self._exit("Armijo rule failed.")

            # Mesh Quality is too low
            elif self.converged_reason == -3:
                self.iteration -= 1
                if self.optimization_variable_abstractions.mesh_handler.do_remesh:
                    _loggers.info(
                        "Mesh quality too low. Performing a remeshing operation.\n"
                    )
                    self.optimization_variable_abstractions.mesh_handler.remesh(self)
                else:
                    self.output_summary()
                    self._exit("Mesh quality is too low.")

            # Iteration for remeshing is the one exceeding the maximum number
            # of iterations
            elif self.converged_reason == -4:
                self.output_summary()
                self._exit("Maximum number of iterations exceeded.")

    def convergence_test(self) -> bool:
        """Checks, whether the algorithm converged successfully.

        Returns:
            A flag, which is True if the algorithm converged.
        """

        if self.iteration == 0:
            self.gradient_norm_initial = self.gradient_norm
        try:
            self.relative_norm = self.gradient_norm / self.gradient_norm_initial
        except ZeroDivisionError:
            self.relative_norm = 0.0
        if self.gradient_norm <= self.atol + self.rtol * self.gradient_norm_initial:
            if self.iteration == 0:
                self.objective_value = self.cost_functional.evaluate()
            self.converged = True
            return True

        return False

    def compute_gradient(self) -> None:
        """Computes the gradient of the reduced cost functional."""

        self.adjoint_problem.has_solution = False
        self.gradient_problem.has_solution = False
        self.gradient_problem.solve()

    def check_for_ascent(self) -> None:
        """Checks, whether the current search direction is an ascent direction.

        Reverts the direction to the negative gradient if an ascent direction is found.
        """

        directional_derivative = self.form_handler.scalar_product(
            self.gradient, self.search_direction
        )

        if directional_derivative >= 0:
            for i in range(len(self.gradient)):
                self.search_direction[i].vector().vec().aypx(
                    0.0, -self.gradient[i].vector().vec()
                )
            self.has_curvature_info = False

    def initialize_solver(self) -> None:
        """Initializes the solver."""

        self.converged = False

        try:
            self.iteration = self.temp_dict["OptimizationRoutine"].get(
                "iteration_counter", 0
            )
            self.gradient_norm_initial = self.temp_dict["OptimizationRoutine"].get(
                "gradient_norm_initial", 0.0
            )
        except (TypeError, AttributeError):
            self.iteration = 0
            self.gradient_norm_initial = 0.0

        self.relative_norm = 1.0
        self.state_problem.has_solution = False
