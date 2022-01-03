# Copyright (C) 2020-2022 Sebastian Blauth
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

"""General blue print class for all optimization algorithms

This is the base class, on which ControlOptimizationAlgorithm and
ShapeOptimizationAlgorithm classes are based.

"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import fenics

from .._exceptions import NotConvergedError
from .._loggers import error, info


if TYPE_CHECKING:
    from .optimization_problem import OptimizationProblem


class OptimizationAlgorithm(abc.ABC):
    """Abstract class representing all kinds of optimization algorithms."""

    def __init__(self, optimization_problem: OptimizationProblem) -> None:
        """
        Parameters
        ----------
        optimization_problem : OptimizationProblem
            The corresponding optimization problem
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

        self.output_manager = optimization_problem.output_manager

    @abc.abstractmethod
    def run(self) -> None:
        pass

    def output(self) -> None:
        self.output_manager.output(self)

    def output_summary(self) -> None:
        self.output_manager.output_summary(self)

    def post_processing(self) -> None:
        """Does a post processing after the optimization algorithm terminates.

        Returns
        -------
        None
        """

        if self.converged:
            self.output()
            self.output_summary()

        else:
            # maximum iterations reached
            if self.converged_reason == -1:
                self.output()
                self.output_summary()
                if self.soft_exit:
                    print("Maximum number of iterations exceeded.")
                else:
                    raise NotConvergedError(
                        "Optimization Algorithm",
                        "Maximum number of iterations were exceeded.",
                    )

            # Armijo line search failed
            elif self.converged_reason == -2:
                self.iteration -= 1
                self.output_summary()
                if self.soft_exit:
                    print("Armijo rule failed.")
                else:
                    raise NotConvergedError(
                        "Armijo line search",
                        "Failed to compute a feasible Armijo step.",
                    )

            # Mesh Quality is too low
            elif self.converged_reason == -3:
                self.iteration -= 1
                if self.mesh_handler.do_remesh:
                    info("Mesh quality too low. Performing a remeshing operation.\n")
                    self.mesh_handler.remesh(self)
                else:
                    self.output_summary()
                    if self.soft_exit:
                        error("Mesh quality is too low.")
                    else:
                        raise NotConvergedError(
                            "Optimization Algorithm", "Mesh quality is too low."
                        )

            # Iteration for remeshing is the one exceeding the maximum number of iterations
            elif self.converged_reason == -4:
                self.output_summary()
                if self.soft_exit:
                    print("Maximum number of iterations exceeded.")
                else:
                    raise NotConvergedError(
                        "Optimization Algorithm",
                        "Maximum number of iterations were exceeded.",
                    )

    def nonconvergence(self) -> bool:
        """Checks for nonconvergence of the solution algorithm

        Returns
        -------
        bool
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
