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

"""Solution algorithms for topology optimization."""

from __future__ import annotations

import abc
from typing import Callable, cast, TYPE_CHECKING

import fenics
import numpy as np

try:
    from ufl_legacy import algorithms as ufl_algorithms
    import ufl_legacy as ufl
except ImportError:
    import ufl
    from ufl import algorithms as ufl_algorithms

from cashocs import _exceptions
from cashocs import _forms
from cashocs import _pde_problems
from cashocs import _utils
from cashocs._optimization import optimization_algorithms
from cashocs._optimization.topology_optimization import topology_optimization_problem

if TYPE_CHECKING:
    from cashocs._database import database
    from cashocs._optimization import line_search as ls
    from cashocs._optimization import optimal_control


class TopologyOptimizationAlgorithm(optimization_algorithms.OptimizationAlgorithm):
    """Parent class for solution algorithms for topology optimization."""

    form_handler: _forms.ControlFormHandler

    def __init__(
        self,
        db: database.Database,
        optimization_problem: topology_optimization_problem.TopologyOptimizationProblem,
        line_search: ls.LineSearch,
    ) -> None:
        """Parent class for solvers for topology optimization problems.

        Args:
            db: The database of the problem.
            optimization_problem: The corresponding optimization problem which shall be
                solved.
            line_search: The line search for the problem.

        """
        super().__init__(db, optimization_problem, line_search)

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
        self._cashocs_problem.db.parameter_db.problem_type = "topology"

        self.mesh = optimization_problem.mesh

        mpi_comm = self.mesh.mpi_comm()
        if self.interpolation_scheme == "angle" and mpi_comm.Get_size() > 1:
            raise _exceptions.InputError(
                "TopologyOptimizationProblem",
                "TopologyOptimization.interpolation_scheme",
                "The angle weighted interpolation option is not supported in parallel. "
                "Please use interpolation_scheme = volume in your config file.",
            )

        self.cg1_space = fenics.FunctionSpace(self.mesh, "CG", 1)
        self.dg0_space = optimization_problem.dg0_space
        self.topological_derivative_vertex: fenics.Function = fenics.Function(
            self.cg1_space
        )
        self.projected_gradient: fenics.Function = fenics.Function(self.cg1_space)
        self.levelset_function_prev = fenics.Function(self.cg1_space)
        self.setup_assembler()

        self.linear_solver = _utils.linalg.LinearSolver()

        self.projection = optimization_problem.projection

    def _generate_measure(self) -> ufl.Measure:
        """Generates the measure for projecting the topological derivative.

        Returns:
            The fenics measure which is used for projecting the topological derivative.

        """
        is_everywhere = False
        subdomain_id_list = []
        for integral in self.form_handler.riesz_scalar_products[0].integrals():
            integral_type = integral.integral_type()
            if integral_type not in ["cell", "dx"]:
                raise _exceptions.InputError(
                    "TopologyOptimizationProblem",
                    "riesz_scalar_products",
                    "The supplied scalar products have to be defined "
                    "over the volume only.",
                )
            subdomain_id = integral.subdomain_id()
            subdomain_id_list.append(subdomain_id)
            if subdomain_id == "everywhere":
                is_everywhere = True
                break

        mesh = (
            self.form_handler.riesz_scalar_products[0]
            .integrals()[0]
            .ufl_domain()
            .ufl_cargo()
        )
        subdomain_data = (
            self.form_handler.riesz_scalar_products[0].integrals()[0].subdomain_data()
        )
        if is_everywhere:
            measure = ufl.Measure("dx", mesh)
        else:
            measure = _utils.summation(
                [
                    ufl.Measure(
                        "dx", mesh, subdomain_data=subdomain_data, subdomain_id=id
                    )
                    for id in subdomain_id_list
                ]
            )

        return measure

    def setup_assembler(self) -> None:
        """Sets up the assembler for projecting the topological derivative."""
        self.form_handler = cast(_forms.ControlFormHandler, self.form_handler)
        modified_scalar_product = _utils.bilinear_boundary_form_modification(
            self.form_handler.riesz_scalar_products
        )
        test = modified_scalar_product[0].arguments()[0]
        dx = self._generate_measure()
        rhs = self.topological_derivative_vertex * test * dx
        try:
            self.assembler = fenics.SystemAssembler(
                modified_scalar_product[0], rhs, self.form_handler.control_bcs_list[0]
            )
        except (AssertionError, ValueError):
            estimated_degree = np.maximum(
                ufl_algorithms.estimate_total_polynomial_degree(
                    self.form_handler.riesz_scalar_products[0]
                ),
                ufl_algorithms.estimate_total_polynomial_degree(rhs),
            )
            self.assembler = fenics.SystemAssembler(
                modified_scalar_product[0],
                rhs,
                self.form_handler.control_bcs_list[0],
                form_compiler_parameters={"quadrature_degree": estimated_degree},
            )

        self.fenics_matrix = fenics.PETScMatrix(self.db.geometry_db.mpi_comm)
        self.assembler.assemble(self.fenics_matrix)
        self.fenics_matrix.ident_zeros()
        self.riesz_matrix = self.fenics_matrix.mat()
        self.b_tensor = fenics.PETScVector(self.db.geometry_db.mpi_comm)

    @abc.abstractmethod
    def run(self) -> None:
        """Runs the optimization algorithm to solve the optimization problem."""
        pass

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
        if self.interpolation_scheme == "angle":
            _utils.interpolate_by_angle(
                self.topological_derivative_neg,
                self.topological_derivative_pos,
                self.levelset_function,
                self.topological_derivative_vertex,
            )
        elif self.interpolation_scheme == "volume":
            _utils.interpolate_by_volume(
                self.topological_derivative_neg,
                self.topological_derivative_pos,
                self.levelset_function,
                self.topological_derivative_vertex,
            )

    def project_topological_derivative(self) -> None:
        """Projects the topological derivative to compute a topological gradient."""
        self.gradient_problem = cast(
            _pde_problems.ControlGradientProblem, self.gradient_problem
        )
        self.assembler.assemble(self.b_tensor)
        self.linear_solver.solve(
            self.topological_derivative_vertex,
            A=self.riesz_matrix,
            b=self.b_tensor.vec(),
            ksp_options=self.gradient_problem.riesz_ksp_options[0],
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
            self.topological_derivative_vertex.vector().vec().aypx(
                0.0,
                _utils.l2_projection(self.topological_derivative_pos, self.cg1_space)
                .vector()
                .vec(),
            )
            self.topological_derivative_vertex.vector().apply("")

        self.project_topological_derivative()

        if self.normalize_topological_derivative:
            norm = self.norm(self.topological_derivative_vertex)
            self.topological_derivative_vertex.vector().vec().scale(1.0 / norm)
            self.topological_derivative_vertex.vector().apply("")

        self.compute_projected_gradient()

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

    def compute_projected_gradient(self) -> None:
        """Computes the projected gradient."""
        beta = self.scalar_product(
            self.topological_derivative_vertex, self.levelset_function
        )
        gamma = self.scalar_product(self.levelset_function, self.levelset_function)
        self.projected_gradient.vector().vec().aypx(
            0.0, self.topological_derivative_vertex.vector().vec()
        )
        self.projected_gradient.vector().apply("")
        self.projected_gradient.vector().vec().axpby(
            -beta / gamma, 1.0, self.levelset_function.vector().vec()
        )
        self.projected_gradient.vector().apply("")

    def compute_gradient_norm(self) -> float:
        """Computes the norm of the projected gradient.

        Returns:
            The norm of the projected gradient.

        """
        self.compute_projected_gradient()
        norm: float = np.sqrt(
            self.scalar_product(self.projected_gradient, self.projected_gradient)
        )
        return norm


class LevelSetTopologyAlgorithm(TopologyOptimizationAlgorithm):
    """Parent class for levelset-based solvers for topology optimization."""

    @abc.abstractmethod
    def move_levelset(self, stepsize: float) -> None:
        """Moves (updates) the levelset function based on the topological gradient.

        Args:
            stepsize: The stepsize which is used to update the levelset function.

        """
        pass

    def run(self) -> None:
        """Runs the optimization algorithm to solve the optimization problem."""
        self.normalize(self.levelset_function)
        self.stepsize = 1.0
        self._cashocs_problem.state_problem.has_solution = False

        failed = False
        for k in range(self.config.getint("OptimizationRoutine", "max_iter")):
            if failed:
                break

            self.iteration = k
            self.levelset_function_prev.vector().vec().aypx(
                0.0, self.levelset_function.vector().vec()
            )
            self.levelset_function_prev.vector().apply("")

            self._cashocs_problem.adjoint_problem.has_solution = False
            self.compute_gradient()
            self.db.parameter_db.optimization_state["no_adjoint_solves"] = (
                self._cashocs_problem.db.parameter_db.optimization_state[
                    "no_adjoint_solves"
                ]
            )

            self.objective_value = (
                self._cashocs_problem.reduced_cost_functional.evaluate()
            )

            self.angle = self.compute_angle()
            self.gradient_norm = self.compute_gradient_norm()

            if self.convergence_test():
                break

            self.output()

            if k > 0:
                self.stepsize = float(np.minimum(1.5 * self.stepsize, 1.0))
            while True:
                self.move_levelset(self.stepsize)
                self.projection.project()
                self.normalize(self.levelset_function)
                self.update_levelset()

                self._cashocs_problem.state_problem.has_solution = False
                self.compute_state_variables()
                self.db.parameter_db.optimization_state["no_state_solves"] = (
                    self._cashocs_problem.db.parameter_db.optimization_state[
                        "no_state_solves"
                    ]
                )
                cost_functional_new = (
                    self._cashocs_problem.reduced_cost_functional.evaluate()
                )
                if cost_functional_new <= self.objective_value or (
                    self.projection.volume_restriction is not None and k == 0
                ):
                    break
                else:
                    self.stepsize *= 0.5

                if self.stepsize <= 1e-10:
                    if self.config.getboolean("OptimizationRoutine", "soft_exit"):
                        failed = True
                        break
                    else:
                        raise _exceptions.NotConvergedError(
                            "cashocs._optimization.topology_optimization."
                            "topology_optimization_algorithm",
                            "Stepsize computation failed.",
                        )

            self.iteration += 1
            if self.nonconvergence():
                break


class ConvexCombinationAlgorithm(LevelSetTopologyAlgorithm):
    """An algorithm based on convex combination for topology optimization."""

    def move_levelset(self, stepsize: float) -> None:
        """Moves (updates) the levelset function based on the topological gradient.

        Args:
            stepsize: The stepsize which is used to update the levelset function.

        """
        self.levelset_function.vector().vec().axpby(
            stepsize / self.norm(self.topological_derivative_vertex),
            0.0,
            self.topological_derivative_vertex.vector().vec(),
        )
        self.levelset_function.vector().apply("")
        self.levelset_function.vector().vec().axpy(
            1.0 - stepsize, self.levelset_function_prev.vector().vec()
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

        self.levelset_function.vector().vec().axpby(
            float(1.0 / np.sin(angle)) * float(np.sin((1.0 - stepsize) * angle)),
            0.0,
            self.levelset_function_prev.vector().vec(),
        )
        self.levelset_function.vector().vec().axpby(
            float(1.0 / np.sin(angle))
            * float(
                np.sin(stepsize * angle) / self.norm(self.topological_derivative_vertex)
            ),
            1.0,
            self.topological_derivative_vertex.vector().vec(),
        )
        self.levelset_function.vector().apply("")
        if self.re_normalize_levelset:
            self.normalize(self.levelset_function)
