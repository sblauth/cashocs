# Copyright (C) 2020-2025 Fraunhofer ITWM and Sebastian Blauth
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

"""General line search."""

from __future__ import annotations

import abc
from typing import cast, TYPE_CHECKING

import fenics
import numpy as np
from petsc4py import PETSc

from cashocs import _utils
from cashocs import log

if TYPE_CHECKING:
    from scipy import sparse

    from cashocs import _typing
    from cashocs._database import database
    from cashocs._optimization import optimization_algorithms


class LineSearch(abc.ABC):
    """Abstract implementation of a line search."""

    def __init__(
        self, db: database.Database, optimization_problem: _typing.OptimizationProblem
    ) -> None:
        """Initializes self.

        Args:
            db: The database of the problem.
            optimization_problem: The corresponding optimization problem.

        """
        self.db = db
        self.problem_type = db.parameter_db.problem_type

        self.config = self.db.config
        self.form_handler = optimization_problem.form_handler
        self.state_problem = optimization_problem.state_problem
        self.optimization_variable_abstractions = (
            optimization_problem.optimization_variable_abstractions
        )
        self.cost_functional = optimization_problem.reduced_cost_functional

        if self.problem_type == "shape":
            if "deformation_function" in self.db.parameter_db.temp_dict.keys():
                self.deformation_function = self.db.parameter_db.temp_dict[
                    "deformation_function"
                ]
            else:
                self.deformation_function = fenics.Function(
                    self.db.function_db.control_spaces[0]
                )
            self.global_deformation_vector = self.deformation_function.vector().vec()

        self.stepsize = self.config.getfloat("LineSearch", "initial_stepsize")
        self.safeguard_stepsize = self.config.getboolean(
            "LineSearch", "safeguard_stepsize"
        )

        self.beta_armijo: float = self.config.getfloat("LineSearch", "beta_armijo")
        self.epsilon_armijo: float = self.config.getfloat(
            "LineSearch", "epsilon_armijo"
        )
        self.search_direction_inf = 1.0

        algorithm = _utils.optimization_algorithm_configuration(self.config)
        self.is_newton_like = algorithm.casefold() == "lbfgs"
        self.is_newton = algorithm.casefold() == "newton"
        self.is_steepest_descent = algorithm.casefold() == "gradient_descent"
        if self.is_newton:
            self.stepsize = 1.0

    def perform(
        self,
        solver: optimization_algorithms.OptimizationAlgorithm,
        search_direction: list[fenics.Function],
        has_curvature_info: bool,
        active_idx: np.ndarray | None = None,
        constraint_gradient: sparse.csr_matrix | None = None,
        dropped_idx: np.ndarray | None = None,
    ) -> None:
        """Performs a line search for the new iterate.

        Notes:
            This is the function that should be called in the optimization algorithm,
            it consists of a call to ``self.search`` and ``self.post_line_search``
            afterward.

        Args:
            solver: The optimization algorithm.
            search_direction: The current search direction.
            has_curvature_info: A flag, which indicates, whether the search direction
                is (presumably) scaled.
            active_idx: The list of active indices of the working set. Only needed
                for shape optimization with mesh quality constraints. Default is `None`.
            constraint_gradient: The gradient of the constraints for the mesh quality.
                Only needed for shape optimization with mesh quality constraints.
                Default is `None`.
            dropped_idx: The list of indicies for dropped constraints. Only needed
                for shape optimization with mesh quality constraints. Default is `None`.

        """
        deformation, is_remeshed = self.search(
            solver,
            search_direction,
            has_curvature_info,
            active_idx,
            constraint_gradient,
            dropped_idx,
        )
        if deformation is not None and self.config.getboolean(
            "ShapeGradient", "global_deformation"
        ):
            x = fenics.as_backend_type(deformation.vector()).vec()

            if not is_remeshed:
                transfer_matrix = solver.db.geometry_db.transfer_matrix
            else:
                transfer_matrix = solver.db.geometry_db.old_transfer_matrix

            transfer_matrix = cast(PETSc.Mat, transfer_matrix)

            transfer_matrix.multAdd(
                x, self.global_deformation_vector, self.global_deformation_vector
            )
            self.deformation_function.vector().apply("")

        self.post_line_search()

    def initialize_stepsize(
        self,
        solver: optimization_algorithms.OptimizationAlgorithm,
        search_direction: list[fenics.Function],
        has_curvature_info: bool,
    ) -> None:
        """Initializes the stepsize.

        Performs various ways for safeguarding (can be deactivated).

        Args:
            solver: The optimization algorithm.
            search_direction: The current search direction.
            has_curvature_info: A flag, which indicates whether the direction is
                (presumably) scaled.

        """
        self.search_direction_inf = np.max(
            [
                search_direction[i].vector().norm("linf")
                for i in range(len(search_direction))
            ]
        )

        if has_curvature_info:
            log.debug(
                "Stepsize computation has curvature information. "
                "Setting trial stepsize to 1.0."
            )
            self.stepsize = 1.0

        if solver.is_restarted:
            log.debug(
                "Solver has been restarted. "
                "Using initial_stepsize from config as trial stepsize."
            )
            self.stepsize = self.config.getfloat("LineSearch", "initial_stepsize")

        num_decreases = (
            self.optimization_variable_abstractions.compute_a_priori_decreases(
                search_direction, self.stepsize
            )
        )
        self.stepsize /= pow(self.beta_armijo, num_decreases)
        if num_decreases > 0:
            log.debug(
                "Stepsize is too large for the angle_change parameter in "
                "section MeshQuality. Making the step smaller to be feasible. "
                f"New step size: {self.stepsize:.3e}"
            )

        if self.safeguard_stepsize and solver.iteration == 0:
            search_direction_norm = np.sqrt(
                self.form_handler.scalar_product(search_direction, search_direction)
            )
            self.stepsize = float(
                np.minimum(self.stepsize, 100.0 / (1.0 + search_direction_norm))
            )
            log.debug(
                "Performed a safeguarding of the stepsize to avoid too large steps. "
                f"New step size after safeguarding: {self.stepsize:.3e}"
            )

    @abc.abstractmethod
    def search(
        self,
        solver: optimization_algorithms.OptimizationAlgorithm,
        search_direction: list[fenics.Function],
        has_curvature_info: bool,
        active_idx: np.ndarray | None = None,
        constraint_gradient: sparse.csr_matrix | None = None,
        dropped_idx: np.ndarray | None = None,
    ) -> tuple[fenics.Function | None, bool]:
        """Performs a line search.

        Args:
            solver: The optimization algorithm.
            search_direction: The current search direction.
            has_curvature_info: A flag, which indicates, whether the search direction
                is (presumably) scaled.
            active_idx: The list of active indices of the working set. Only needed
                for shape optimization with mesh quality constraints. Default is `None`.
            constraint_gradient: The gradient of the constraints for the mesh quality.
                Only needed for shape optimization with mesh quality constraints.
                Default is `None`.
            dropped_idx: The list of indicies for dropped constraints. Only needed
                for shape optimization with mesh quality constraints. Default is `None`.

        Returns:
            The accepted deformation or None, in case the deformation was not
            successful.

        """
        pass

    def post_line_search(self) -> None:
        """Performs tasks after the line search was successful."""
        self.form_handler.update_scalar_product()

    def _satisfies_armijo_condition(
        self,
        objective_step: float,
        current_function_value: float,
        decrease_measure: float,
    ) -> bool:
        """Checks whether the sufficient decrease condition is satisfied.

        Args:
            objective_step: The new objective value, after taking the step
            current_function_value: The old objective value
            decrease_measure: The directional derivative in direction of the search
                direction

        Returns:
            A boolean flag which is True in case the condition is satisfied.

        """
        if not self.problem_type == "topology":
            val = bool(
                objective_step
                < current_function_value + self.epsilon_armijo * decrease_measure
            )
        else:
            val = bool(objective_step <= current_function_value)
        return val
