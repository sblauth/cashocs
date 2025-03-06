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

"""Line search-based topology optimization algorithms."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cashocs import _exceptions
from cashocs import log
from cashocs._optimization.topology_optimization import topology_optimization_algorithm

if TYPE_CHECKING:
    from cashocs._database import database
    from cashocs._optimization import line_search as ls
    from cashocs._optimization.topology_optimization import bisection
    from cashocs._optimization.topology_optimization import (
        topology_optimization_problem,
    )


class DescentTopologyAlgorithm(
    topology_optimization_algorithm.TopologyOptimizationAlgorithm
):
    """A general solver class for topology optimization.

    This can use classical solution algorithms implemented in cashocs for topology
    optimization
    """

    projection: bisection.LevelSetVolumeProjector

    def __init__(
        self,
        db: database.Database,
        optimization_problem: topology_optimization_problem.TopologyOptimizationProblem,
        line_search: ls.LineSearch,
        algorithm: str | None,
    ) -> None:
        """This solver is used to invoke classical optimization algorithms.

        Args:
            db: The database of the problem.
            optimization_problem: The optimization problem which is to be solved.
            line_search: The line search for the problem.
            algorithm: The algorithm that is to be used.

        """
        super().__init__(db, optimization_problem, line_search)
        self.algorithm = algorithm

        self.iteration = 0
        self._cashocs_problem.config.set("Output", "verbose", "False")
        self._cashocs_problem.config.set("Output", "save_txt", "False")
        self._cashocs_problem.config.set("Output", "save_results", "False")
        self._cashocs_problem.config.set("Output", "save_state", "False")
        self._cashocs_problem.config.set("Output", "save_adjoint", "False")
        self._cashocs_problem.config.set("Output", "save_gradient", "False")
        self._cashocs_problem.config.set("OptimizationRoutine", "soft_exit", "True")
        self._cashocs_problem.config.set("OptimizationRoutine", "rtol", "0.0")
        self._cashocs_problem.config.set("OptimizationRoutine", "atol", "0.0")

        self._cashocs_problem._silent = True
        self._cashocs_problem.output_manager._silent = True

        self.successful = False
        self.loop_restart = False

        def pre_callback() -> None:
            self.projection.project()
            self.normalize(self.levelset_function)
            self.update_levelset()

        def post_callback() -> None:
            self.compute_gradient()
            self.db.parameter_db.optimization_state["no_adjoint_solves"] = (
                self._cashocs_problem.db.parameter_db.optimization_state[
                    "no_adjoint_solves"
                ]
            )
            self.db.parameter_db.optimization_state["no_state_solves"] = (
                self._cashocs_problem.db.parameter_db.optimization_state[
                    "no_state_solves"
                ]
            )
            self._cashocs_problem.db.function_db.gradient[0].vector().vec().aypx(
                0.0, -self.projected_gradient.vector().vec()
            )
            self._cashocs_problem.db.function_db.gradient[0].vector().apply("")

            self.angle = self.compute_angle()
            self.stepsize = self._cashocs_problem.solver.stepsize

            self.gradient_norm = self.compute_gradient_norm()
            self.objective_value = (
                self._cashocs_problem.reduced_cost_functional.evaluate()
            )

            if self.convergence_test():
                self.successful = True
                self.iteration -= 1
                self._cashocs_problem.db.function_db.gradient[0].vector().vec().set(0.0)
                self._cashocs_problem.db.function_db.gradient[0].vector().apply("")

            if not self.loop_restart and not self.successful:
                self.output()
            else:
                self.loop_restart = False

            self.iteration += 1

            if self.iteration >= self.max_iter:
                self._cashocs_problem.db.function_db.gradient[0].vector().vec().set(0.0)
                self._cashocs_problem.db.function_db.gradient[0].vector().apply("")

                exit_message = "Maximum number of iterations reached."
                if self.config.getboolean("OptimizationRoutine", "soft_exit"):
                    log.error(exit_message)
                else:
                    raise _exceptions.NotConvergedError(
                        "Topology Optimization Algorithm",
                        exit_message,
                    )

        self._cashocs_problem.inject_pre_callback(pre_callback)
        self._cashocs_problem.inject_post_callback(post_callback)

    def run(self) -> None:
        """Runs the optimization algorithm to solve the optimization problem."""
        self.iteration = 0

        stop_iter = -1

        while True:
            self._cashocs_problem.solve(
                algorithm=self.algorithm,
                rtol=self.rtol,
                atol=self.atol,
                max_iter=self.max_iter,
            )

            if self._cashocs_problem.solver.converged_reason < -1:
                self.iteration -= 1
                self.loop_restart = True

                if self.iteration == stop_iter:
                    raise _exceptions.NotConvergedError(
                        "Topology Optimization Solver", "The line search failed."
                    )

                stop_iter = self.iteration

            if self._cashocs_problem.solver.converged_reason >= 0:
                break
