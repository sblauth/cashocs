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

"""Line search-based topology optimization algorithms."""

from __future__ import annotations

from cashocs._optimization.topology_optimization import topology_optimization_algorithm
from cashocs._optimization.topology_optimization import topology_optimization_problem


class DescentTopologyAlgorithm(
    topology_optimization_algorithm.TopologyOptimizationAlgorithm
):
    """A general solver class for topology optimization.

    This can use classical solution algorithms implemented in cashocs for topology
    optimization
    """

    def __init__(
        self,
        optimization_problem: topology_optimization_problem.TopologyOptimizationProblem,
        algorithm: str | None,
    ) -> None:
        """This solver is used to invoke classical optimization algorithms.

        Args:
            optimization_problem: The optimization problem which is to be solved.
            algorithm: The algorithm that is to be used.

        """
        super().__init__(optimization_problem)
        self.algorithm = algorithm

        self.iteration = 0
        self._cashocs_problem.config.set("Output", "verbose", "False")
        self._cashocs_problem.config.set("Output", "save_txt", "False")
        self._cashocs_problem.config.set("Output", "save_results", "False")
        self._cashocs_problem.config.set("Output", "save_state", "False")
        self._cashocs_problem.config.set("Output", "save_adjoint", "False")
        self._cashocs_problem.config.set("Output", "save_gradient", "False")
        self._cashocs_problem.config.set("OptimizationRoutine", "soft_exit", "True")

        self.loop_restart = False

        def pre_hook() -> None:
            self.normalize(self.levelset_function)
            self.update_levelset()

        def post_hook() -> None:
            self.compute_gradient()
            self._cashocs_problem.gradient[0].vector().vec().aypx(
                0.0, -self.projected_gradient.vector().vec()
            )
            self._cashocs_problem.gradient[0].vector().apply("")

            self.angle = self.compute_angle()

            self.gradient_norm = self.compute_gradient_norm()
            self.objective_value = (
                self._cashocs_problem.reduced_cost_functional.evaluate()
            )

            if self.convergence_test():
                self.output()
                self._cashocs_problem.gradient[0].vector().vec().set(0.0)
                self._cashocs_problem.gradient[0].vector().apply("")
                print("\nOptimization successful!\n")

            if not self.loop_restart:
                self.output()
            else:
                self.loop_restart = False

            self.iteration += 1

            if self.iteration >= self.maximum_iterations:
                self._cashocs_problem.gradient[0].vector().vec().set(0.0)
                self._cashocs_problem.gradient[0].vector().apply("")
                print("Maximum number of iterations reached.")

        self._cashocs_problem.inject_pre_hook(pre_hook)
        self._cashocs_problem.inject_post_hook(post_hook)

    def run(self) -> None:
        """Runs the optimization algorithm to solve the optimization problem."""
        self.iteration = 0

        stop_iter = -1

        while True:
            self._cashocs_problem.solve(
                algorithm=self.algorithm,
                rtol=self.rtol,
                atol=self.atol,
                max_iter=self.maximum_iterations,
            )

            if self._cashocs_problem.solver.converged_reason < -1:
                self.iteration -= 1
                self.loop_restart = True

                if self.iteration == stop_iter:
                    print("The line search failed.")
                    break

                stop_iter = self.iteration

            if self._cashocs_problem.solver.converged_reason >= 0:
                break
