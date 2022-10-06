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
        self.max_iter = 0
        self.atol = 0.0
        self.rtol = 0.0
        self._cashocs_problem.config.set("Output", "verbose", "False")
        self._cashocs_problem.config.set("Output", "save_state", "False")
        self._cashocs_problem.config.set("Output", "save_adjoint", "False")
        self._cashocs_problem.config.set("Output", "save_gradient", "False")
        self._cashocs_problem.config.set("OptimizationRoutine", "soft_exit", "True")

        def pre_hook() -> None:
            self.normalize(self.levelset_function)
            self.update_levelset()

        def post_hook() -> None:
            self.compute_gradient()
            beta = self.scalar_product(
                self.topological_derivative_vertex, self.levelset_function
            )
            gamma = self.scalar_product(self.levelset_function, self.levelset_function)
            self._cashocs_problem.gradient[0].vector().vec().aypx(
                0.0,
                -(
                    self.topological_derivative_vertex.vector().vec()
                    - beta / gamma * self.levelset_function.vector().vec()
                ),
            )
            self._cashocs_problem.gradient[0].vector().apply("")

            angle = self.compute_angle()

            self.gradient_norm = angle
            self.objective_value = (
                self._cashocs_problem.reduced_cost_functional.evaluate()
            )
            self.stepsize = self._cashocs_problem.solver.stepsize
            self.cost_functional_values.append(self.objective_value)
            self.angle_list.append(angle)
            self.stepsize_list.append(self.stepsize)
            self.output()

            if self.convergence_test():
                self._cashocs_problem.gradient[0].vector().vec().set(0.0)
                self._cashocs_problem.gradient[0].vector().apply("")
                print("\nOptimization successful!\n")

            self.iteration += 1

            if self.iteration >= self.max_iter:
                self._cashocs_problem.gradient[0].vector().vec().set(0.0)
                self._cashocs_problem.gradient[0].vector().apply("")
                print("Maximum number of iterations reached.")

        self._cashocs_problem.inject_pre_hook(pre_hook)
        self._cashocs_problem.inject_post_hook(post_hook)

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

        self.iteration = 0
        self.max_iter = self.config.getint("OptimizationRoutine", "maximum_iterations")

        stop_iter = -1

        while True:
            self._cashocs_problem.solve(
                algorithm=self.algorithm, rtol=0.0, atol=0.0, max_iter=max_iter
            )

            if self._cashocs_problem.solver.converged_reason < -1:
                self.cost_functional_values.pop()
                self.angle_list.pop()
                self.stepsize_list.pop()
                self.iteration -= 1

                if self.iteration == stop_iter:
                    print("The line search failed.")
                    break

                stop_iter = self.iteration

            if self._cashocs_problem.solver.converged_reason >= 0:
                break
