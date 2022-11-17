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

"""General optimization algorithm for PDE constrained optimization."""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import fenics
import numpy as np

from cashocs import _exceptions
from cashocs import _utils

if TYPE_CHECKING:
    from cashocs import _pde_problems
    from cashocs import _typing
    from cashocs._database import database
    from cashocs._optimization import line_search as ls
    from cashocs._optimization import optimization_variable_abstractions as ov


class OptimizationAlgorithm(abc.ABC):
    """Base class for optimization algorithms."""

    def __init__(
        self,
        db: database.Database,
        optimization_problem: _typing.OptimizationProblem,
        line_search: ls.LineSearch,
    ) -> None:
        """Initializes self.

        Args:
            db: The database of the problem.
            optimization_problem: The corresponding optimization problem.
            line_search: The corresponding line search.

        """
        self.db = db
        self.line_search = line_search

        self.config = self.db.config

        self.line_search_broken = False
        self.has_curvature_info = False

        self.optimization_problem = optimization_problem
        self.form_handler = optimization_problem.form_handler
        self.state_problem: _pde_problems.StateProblem = (
            optimization_problem.state_problem
        )
        self.adjoint_problem: _pde_problems.AdjointProblem = (
            optimization_problem.adjoint_problem
        )

        self.gradient_problem = optimization_problem.gradient_problem
        self.cost_functional = optimization_problem.reduced_cost_functional
        self.gradient = self.db.function_db.gradient
        self.search_direction = _utils.create_function_list(
            self.db.function_db.control_spaces
        )

        self.optimization_variable_abstractions: ov.OptimizationVariableAbstractions = (
            optimization_problem.optimization_variable_abstractions
        )

        self._gradient_norm: float = 1.0
        self._iteration: int = 0
        self._objective_value: float = 1.0
        self._gradient_norm_initial: float = 1.0
        self._relative_norm: float = 1.0

        self.requires_remeshing = False
        self.is_restarted: bool = False

        self._stepsize: float = 1.0
        self.converged = False
        self.converged_reason = 0

        self.rtol = self.config.getfloat("OptimizationRoutine", "rtol")
        self.atol = self.config.getfloat("OptimizationRoutine", "atol")
        self.maximum_iterations = self.config.getint(
            "OptimizationRoutine", "maximum_iterations"
        )
        self.soft_exit = self.config.getboolean("OptimizationRoutine", "soft_exit")

        self.output_manager = optimization_problem.output_manager
        self.initialize_solver()

    @property
    def iteration(self) -> int:
        """The number of iterations performed by the solver."""
        return self._iteration

    @iteration.setter
    def iteration(self, value: int) -> None:
        self.db.parameter_db.optimization_state["iteration"] = value
        self._iteration = value

    @property
    def objective_value(self) -> float:
        """The current objective value."""
        return self._objective_value

    @objective_value.setter
    def objective_value(self, value: float) -> None:
        self.db.parameter_db.optimization_state["objective_value"] = value
        self._objective_value = value

    @property
    def relative_norm(self) -> float:
        """The relative norm of the gradient."""
        return self._relative_norm

    @relative_norm.setter
    def relative_norm(self, value: float) -> None:
        self.db.parameter_db.optimization_state["relative_norm"] = value
        self._relative_norm = value

    @property
    def stepsize(self) -> float:
        """The accepted stepsize of the line search."""
        return self._stepsize

    @stepsize.setter
    def stepsize(self, value: float) -> None:
        self.db.parameter_db.optimization_state["stepsize"] = value
        self._stepsize = value

    @property
    def gradient_norm(self) -> float:
        """The current (absolute) gradient norm."""
        return self._gradient_norm

    @gradient_norm.setter
    def gradient_norm(self, value: float) -> None:
        self.db.parameter_db.optimization_state["gradient_norm"] = value
        self._gradient_norm = value

    @property
    def gradient_norm_initial(self) -> float:
        """The initial (absolute) norm of the gradient."""
        return self._gradient_norm_initial

    @gradient_norm_initial.setter
    def gradient_norm_initial(self, value: float) -> None:
        self.db.parameter_db.optimization_state["gradient_norm_initial"] = value
        self._gradient_norm_initial = value

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
        """Writes the output to console and files."""
        self.output_manager.output()

    def output_summary(self) -> None:
        """Writes the summary of the optimization (to files and console)."""
        self.output_manager.output_summary()

    def post_process(self) -> None:
        """Performs the non-console output related post-processing."""
        self.output_manager.post_process()

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

        return bool(self.converged_reason < 0)

    def _exit(self, message: str) -> None:
        """Exits the optimization algorithm and prints message.

        Args:
            message: The message that should be printed on exit.

        """
        if self.soft_exit:
            if fenics.MPI.rank(fenics.MPI.comm_world) == 0:
                print(message, flush=True)
            fenics.MPI.barrier(fenics.MPI.comm_world)
        else:
            raise _exceptions.NotConvergedError("Optimization Algorithm", message)

    def post_processing(self) -> None:
        """Does a post-processing after the optimization algorithm terminates."""
        if self.converged:
            self.output()
            self.post_process()
            self.output_summary()

        else:
            self.objective_value = self.cost_functional.evaluate()
            self.gradient_norm = np.NAN
            self.relative_norm = np.nan
            # maximum iterations reached
            if self.converged_reason == -1:
                self.output()
                self.post_process()
                self._exit("Maximum number of iterations exceeded.")

            # Armijo line search failed
            elif self.converged_reason == -2:
                self.iteration -= 1
                self.post_process()
                self._exit("Armijo rule failed.")

            # Mesh Quality is too low
            elif self.converged_reason == -3:
                self.iteration -= 1
                if not self.optimization_variable_abstractions.mesh_handler.do_remesh:
                    self.post_process()
                    self._exit("Mesh quality is too low.")

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
                self.search_direction[i].vector().apply("")
            self.has_curvature_info = False

    def initialize_solver(self) -> None:
        """Initializes the solver."""
        self.converged = False

        if self.db.parameter_db.temp_dict:
            try:
                self.iteration = self.db.parameter_db.temp_dict[
                    "OptimizationRoutine"
                ].get("iteration_counter", 0)
                self.gradient_norm_initial = self.db.parameter_db.temp_dict[
                    "OptimizationRoutine"
                ].get("gradient_norm_initial", 0.0)
            except TypeError:
                self.iteration = 0
                self.gradient_norm_initial = 0.0
        else:
            self.iteration = 0
            self.gradient_norm_initial = 0.0

        self.relative_norm = 1.0
        self.state_problem.has_solution = False
        self.adjoint_problem.has_solution = False
        self.gradient_problem.has_solution = False

    def evaluate_cost_functional(self) -> None:
        """Evaluates the cost functional and performs the output operation."""
        self.objective_value = self.cost_functional.evaluate()
        self.output()
