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

"""Space mapping for shape optimization problems."""

from __future__ import annotations

import abc
import collections
import copy
import pathlib
from typing import Callable, TYPE_CHECKING

import fenics
import numpy as np
from typing_extensions import Literal

try:
    import ufl_legacy as ufl
except ImportError:
    import ufl

from cashocs import _exceptions
from cashocs import _utils
from cashocs import geometry
from cashocs import log
from cashocs._database import database
from cashocs._optimization.shape_optimization import shape_optimization_problem as sop
from cashocs.geometry import mesh_testing
from cashocs.io import output

if TYPE_CHECKING:
    from cashocs import _typing
    from cashocs import io


class FineModel(abc.ABC):
    """Base class for the fine model in space mapping shape optimization.

    Attributes:
        mesh: The FEM mesh for the fine model.
        cost_functional_value: The current cost functional value of the fine model.

    """

    cost_functional_value: float

    def __init__(self, mesh: fenics.Mesh) -> None:
        """Initializes self.

        Args:
            mesh: The finite element mesh of the coarse model, used for the space
                mapping with the fine model.

        """
        self.mesh = fenics.Mesh(mesh)

    @log.profile_execution_time("simulating the fine model")
    def simulate(self) -> None:
        """Simulates the fine model.

        This is done in this method so it can be decorated for logging purposes.
        """
        self.solve_and_evaluate()

    @abc.abstractmethod
    def solve_and_evaluate(self) -> None:
        """Solves and evaluates the fine model.

        This needs to be overwritten with a custom implementation.

        """
        pass


class CoarseModel:
    """Coarse Model for space mapping shape optimization."""

    def __init__(
        self,
        state_forms: ufl.Form | list[ufl.Form],
        bcs_list: (
            fenics.DirichletBC
            | list[fenics.DirichletBC]
            | list[list[fenics.DirichletBC]]
            | None
        ),
        cost_functional_form: list[_typing.CostFunctional] | _typing.CostFunctional,
        states: fenics.Function | list[fenics.Function],
        adjoints: fenics.Function | list[fenics.Function],
        boundaries: fenics.MeshFunction,
        config: io.Config | None = None,
        shape_scalar_product: ufl.Form | None = None,
        initial_guess: list[fenics.Function] | None = None,
        ksp_options: _typing.KspOption | list[_typing.KspOption] | None = None,
        adjoint_ksp_options: _typing.KspOption | list[_typing.KspOption] | None = None,
        gradient_ksp_options: _typing.KspOption | list[_typing.KspOption] | None = None,
        desired_weights: list[float] | None = None,
        preconditioner_forms: list[ufl.Form] | None = None,
        pre_callback: Callable | None = None,
        post_callback: Callable | None = None,
        linear_solver: _utils.linalg.LinearSolver | None = None,
        adjoint_linear_solver: _utils.linalg.LinearSolver | None = None,
        newton_linearizations: ufl.Form | list[ufl.Form] | None = None,
        excluded_from_time_derivative: list[int] | list[list[int]] | None = None,
    ) -> None:
        """Initializes self.

        Args:
            state_forms: The list of weak forms for the coarse state problem
            bcs_list: The list of boundary conditions for the coarse problem
            cost_functional_form: The cost functional for the coarse problem
            states: The state variables for the coarse problem
            adjoints: The adjoint variables for the coarse problem
            boundaries: A fenics MeshFunction which marks the boundaries.
            config: config: The config file for the problem, generated via
                :py:func:`cashocs.load_config`. Alternatively, this can also be
                ``None``, in which case the default configurations are used, except for
                the optimization algorithm. This has then to be specified in the
                :py:meth:`solve <cashocs.OptimalControlProblem.solve>` method. The
                default is ``None``.
            shape_scalar_product: The scalar product for the shape optimization problem
            initial_guess: The initial guess for solving a nonlinear state equation
            ksp_options: The list of PETSc options for the state equations.
            adjoint_ksp_options: The list of PETSc options for the adjoint equations.
            gradient_ksp_options: A list of dicts corresponding to command line options
                for PETSc, used to compute the (shape) gradient. If this is ``None``,
                either a direct or an iterative method is used (depending on the
                configuration, section OptimizationRoutine, key gradient_method).
            desired_weights: The desired weights for the cost functional
            preconditioner_forms: The list of forms for the preconditioner. The default
                is `None`, so that the preconditioner matrix is the same as the system
                matrix.
            pre_callback: A function (without arguments) that will be called before each
                solve of the state system
            post_callback: A function (without arguments) that will be called after the
                computation of the gradient.
            linear_solver: The linear solver (KSP) which is used to solve the linear
                systems arising from the discretized PDE.
            adjoint_linear_solver: The linear solver (KSP) which is used to solve the
                (linear) adjoint system.
            newton_linearizations: A (list of) UFL forms describing which (alternative)
                linearizations should be used for the (nonlinear) state equations when
                solving them (with Newton's method). The default is `None`, so that the
                Jacobian of the supplied state forms is used.
            excluded_from_time_derivative: For each state equation, a list of indices
                which are not part of the first order time derivative for pseudo time
                stepping. Example: Pressure for incompressible flow. Default is None.

        """
        self.state_forms = state_forms
        self.bcs_list = bcs_list
        self.cost_functional_form = cost_functional_form
        self.states = states
        self.adjoints = adjoints
        self.boundaries = boundaries

        if config is not None:
            self.config = config
        else:
            self.config = io.Config()
        self.config.validate_config()

        self.shape_scalar_product = shape_scalar_product
        self.initial_guess = initial_guess
        self.ksp_options = ksp_options
        self.adjoint_ksp_options = adjoint_ksp_options
        self.gradient_ksp_options = gradient_ksp_options
        self.desired_weights = desired_weights
        self.preconditioner_forms = preconditioner_forms
        self.pre_callback = pre_callback
        self.post_callback = post_callback
        self.linear_solver = linear_solver
        self.adjoint_linear_solver = adjoint_linear_solver
        self.newton_linearizations = newton_linearizations
        self.excluded_from_time_derivative = excluded_from_time_derivative

        self._pre_callback: Callable | None = None
        self._post_callback: Callable | None = None

        self.mesh = self.boundaries.mesh()
        self.coordinates_initial = self.mesh.coordinates().copy()

        config = copy.deepcopy(self.config)
        output_path = pathlib.Path(self.config.get("Output", "result_dir"))
        config.set("Output", "result_dir", str(output_path / "coarse_model"))

        self.shape_optimization_problem = sop.ShapeOptimizationProblem(
            self.state_forms,
            self.bcs_list,
            self.cost_functional_form,
            self.states,
            self.adjoints,
            self.boundaries,
            config=config,
            shape_scalar_product=self.shape_scalar_product,
            initial_guess=self.initial_guess,
            ksp_options=self.ksp_options,
            adjoint_ksp_options=self.adjoint_ksp_options,
            gradient_ksp_options=self.gradient_ksp_options,
            desired_weights=self.desired_weights,
            preconditioner_forms=self.preconditioner_forms,
            pre_callback=self.pre_callback,
            post_callback=self.post_callback,
            linear_solver=self.linear_solver,
            adjoint_linear_solver=self.adjoint_linear_solver,
            newton_linearizations=self.newton_linearizations,
            excluded_from_time_derivative=self.excluded_from_time_derivative,
        )

        self.coordinates_optimal = self.mesh.coordinates().copy()

    def optimize(self) -> None:
        """Solves the coarse model optimization problem."""
        self.shape_optimization_problem.inject_pre_post_callback(
            self._pre_callback, self._post_callback
        )
        self.shape_optimization_problem.solve()
        self.coordinates_optimal = self.mesh.coordinates().copy()


class ParameterExtraction:
    """Parameter extraction for space mapping shape optimization."""

    def __init__(
        self,
        coarse_model: CoarseModel,
        cost_functional_form: list[_typing.CostFunctional] | _typing.CostFunctional,
        states: fenics.Function | list[fenics.Function],
        config: io.Config | None = None,
        desired_weights: list[float] | None = None,
        mode: str = "initial",
    ) -> None:
        """Initializes self.

        Args:
            coarse_model: The coarse model optimization problem
            cost_functional_form: The cost functional for the parameter extraction
            states: The state variables for the parameter extraction
            config: config: The config file for the problem, generated via
                :py:func:`cashocs.load_config`. Alternatively, this can also be
                ``None``, in which case the default configurations are used, except for
                the optimization algorithm. This has then to be specified in the
                :py:meth:`solve <cashocs.OptimalControlProblem.solve>` method. The
                default is ``None``.
            desired_weights: The list of desired weights for the parameter extraction
            mode: The mode used for the initial guess of the parameter extraction. If
                this is coarse_optimum, the default, then the coarse model optimum is
                used as initial guess, if this is initial, then the initial guess for
                the optimization is used.

        """
        self.coarse_model = coarse_model
        self.mesh = coarse_model.mesh
        self.cost_functional_form = cost_functional_form
        self.mode = mode

        self.states = _utils.enlist(states)

        if config is not None:
            self.config = config
        else:
            self.config = io.Config()
        self.config.validate_config()

        self.desired_weights = desired_weights

        self._pre_callback: Callable | None = None
        self._post_callback: Callable | None = None

        self.adjoints = _utils.create_function_list(
            coarse_model.shape_optimization_problem.db.function_db.adjoint_spaces
        )

        dict_states = {
            coarse_model.shape_optimization_problem.states[i]: self.states[i]
            for i in range(len(self.states))
        }
        dict_adjoints = {
            coarse_model.shape_optimization_problem.adjoints[i]: self.adjoints[i]
            for i in range(len(self.adjoints))
        }
        mapping_dict = {}
        mapping_dict.update(dict_states)
        mapping_dict.update(dict_adjoints)

        self.state_forms = [
            ufl.replace(form, mapping_dict)
            for form in coarse_model.shape_optimization_problem.state_forms
        ]
        self.bcs_list = coarse_model.shape_optimization_problem.bcs_list
        self.shape_scalar_product = (
            coarse_model.shape_optimization_problem.shape_scalar_product
        )
        self.boundaries = coarse_model.shape_optimization_problem.boundaries
        self.initial_guess = coarse_model.shape_optimization_problem.initial_guess
        self.ksp_options = coarse_model.shape_optimization_problem.ksp_options
        self.adjoint_ksp_options = (
            coarse_model.shape_optimization_problem.adjoint_ksp_options
        )
        self.gradient_ksp_options = (
            coarse_model.shape_optimization_problem.gradient_ksp_options
        )
        self.preconditioner_forms = coarse_model.preconditioner_forms
        self.pre_callback = coarse_model.pre_callback
        self.post_callback = coarse_model.post_callback
        self.linear_solver = coarse_model.shape_optimization_problem.linear_solver
        self.adjoint_linear_solver = (
            coarse_model.shape_optimization_problem.adjoint_linear_solver
        )
        self.newton_linearizations = (
            coarse_model.shape_optimization_problem.newton_linearizations
        )
        self.excluded_from_time_derivative = (
            coarse_model.shape_optimization_problem.excluded_from_time_derivative
        )

        self.coordinates_initial = coarse_model.coordinates_initial

        a_priori_tester = mesh_testing.APrioriMeshTester(self.mesh)
        intersection_tester = mesh_testing.IntersectionTester(self.mesh)
        self.deformation_handler = geometry.DeformationHandler(
            self.mesh, a_priori_tester, intersection_tester
        )

        self.shape_optimization_problem: sop.ShapeOptimizationProblem | None = None

    def _solve(self, iteration: int = 0) -> None:
        """Solves the parameter extraction problem.

        Args:
            iteration: The current iteration of the space mapping method.
            initial_guess: The initial guesses for solving the problem.

        """
        if self.mode == "initial":
            self.deformation_handler.assign_coordinates(self.coordinates_initial)
        elif self.mode == "coarse_optimum":
            self.deformation_handler.assign_coordinates(
                self.coarse_model.coordinates_optimal
            )

        config = copy.deepcopy(self.config)
        output_path = pathlib.Path(self.config.get("Output", "result_dir"))
        config.set(
            "Output",
            "result_dir",
            str(output_path / f"parameter_extraction_{iteration}"),
        )

        self.shape_optimization_problem = sop.ShapeOptimizationProblem(
            self.state_forms,
            self.bcs_list,
            self.cost_functional_form,
            self.states,
            self.adjoints,
            self.boundaries,
            config=config,
            shape_scalar_product=self.shape_scalar_product,
            initial_guess=self.initial_guess,
            ksp_options=self.ksp_options,
            adjoint_ksp_options=self.adjoint_ksp_options,
            gradient_ksp_options=self.gradient_ksp_options,
            desired_weights=self.desired_weights,
            preconditioner_forms=self.preconditioner_forms,
            pre_callback=self.pre_callback,
            post_callback=self.post_callback,
            linear_solver=self.linear_solver,
            adjoint_linear_solver=self.adjoint_linear_solver,
            newton_linearizations=self.newton_linearizations,
            excluded_from_time_derivative=self.excluded_from_time_derivative,
        )
        if self.shape_optimization_problem is not None:
            self.shape_optimization_problem.inject_pre_post_callback(
                self._pre_callback, self._post_callback
            )

            self.shape_optimization_problem.solve()


class SpaceMappingProblem:
    """Space mapping method for shape optimization."""

    def __init__(
        self,
        fine_model: FineModel,
        coarse_model: CoarseModel,
        parameter_extraction: ParameterExtraction,
        method: Literal[
            "broyden", "bfgs", "lbfgs", "sd", "steepest_descent", "ncg"
        ] = "broyden",
        max_iter: int = 25,
        tol: float = 1e-2,
        use_backtracking_line_search: bool = False,
        broyden_type: Literal["good", "bad"] = "good",
        cg_type: Literal["FR", "PR", "HS", "DY", "HZ"] = "FR",
        memory_size: int = 10,
        verbose: bool = True,
        save_history: bool = False,
    ) -> None:
        """Initializes self.

        Args:
            fine_model: The fine model optimization problem
            coarse_model: The coarse model optimization problem
            parameter_extraction: The parameter extraction problem
            method: A string, which indicates which method is used to solve the space
                mapping. Can be one of "broyden", "bfgs", "lbfgs", "sd",
                "steepest descent", or "ncg". Default is "broyden".
            max_iter: Maximum number of space mapping iterations
            tol: The tolerance used for solving the space mapping iteration
            use_backtracking_line_search: A boolean flag, which indicates whether a
                backtracking line search should be used for the space mapping.
            broyden_type: A string, either "good" or "bad", determining the type of
                Broyden's method used. Default is "good"
            cg_type: A string, either "FR", "PR", "HS", "DY", "HZ", which indicates
                which NCG variant is used for solving the space mapping. Default is "FR"
            memory_size: The size of the memory for Broyden's method and the BFGS method
            verbose: A boolean flag which indicates, whether the output of the space
                mapping method should be verbose. Default is ``True``.
            save_history: A boolean flag which indicates, whether the history of the
                space mapping method should be saved to a .json file. Default is
                ``False``.

        """
        self.fine_model = fine_model
        self.coarse_model = coarse_model
        self.parameter_extraction = parameter_extraction
        self.method = method
        if self.method == "sd":
            self.method = "steepest_descent"
        elif self.method == "lbfgs":
            self.method = "bfgs"
        self.max_iter = max_iter
        self.tol = tol
        self.use_backtracking_line_search = use_backtracking_line_search
        self.broyden_type = broyden_type
        self.cg_type = cg_type
        self.memory_size = memory_size
        self.verbose = verbose
        self.save_history = save_history

        config = copy.deepcopy(self.coarse_model.config)
        config.set("Output", "save_state", "False")
        config.set("Output", "save_adjoint", "False")
        config.set("Output", "save_gradient", "False")

        self.db = database.Database(
            config,
            _utils.enlist(self.coarse_model.states),
            _utils.enlist(self.coarse_model.adjoints),
            self.coarse_model.ksp_options,  # type: ignore
            self.coarse_model.adjoint_ksp_options,  # type: ignore
            self.coarse_model.gradient_ksp_options,  # type: ignore
            self.coarse_model.cost_functional_form,  # type: ignore
            _utils.enlist(self.coarse_model.state_forms),
            _utils.check_and_enlist_bcs(self.coarse_model.bcs_list),
            self.coarse_model.preconditioner_forms,  # type: ignore
        )

        self.output_manager = output.OutputManager(self.db)

        self.coordinates_initial = self.coarse_model.coordinates_initial

        self.eps = 1.0
        self.converged = False
        self.iteration = 0

        self.x = self.fine_model.mesh

        self.deformation_space = fenics.VectorFunctionSpace(self.x, "CG", 1)
        a_priori_tester = mesh_testing.APrioriMeshTester(self.fine_model.mesh)
        intersection_tester = mesh_testing.IntersectionTester(self.fine_model.mesh)
        self.deformation_handler_fine = geometry.DeformationHandler(
            self.fine_model.mesh, a_priori_tester, intersection_tester
        )

        a_priori_tester = mesh_testing.APrioriMeshTester(self.coarse_model.mesh)
        intersection_tester = mesh_testing.IntersectionTester(self.coarse_model.mesh)
        self.deformation_handler_coarse = geometry.DeformationHandler(
            self.coarse_model.mesh, a_priori_tester, intersection_tester
        )

        self.z_star = [fenics.Function(self.deformation_space)]
        self.norm_z_star = 1.0
        self.p_current = [fenics.Function(self.deformation_space)]
        self.p_prev = [fenics.Function(self.deformation_space)]
        self.h = [fenics.Function(self.deformation_space)]
        self.v = [fenics.Function(self.deformation_space)]
        self.u = [fenics.Function(self.deformation_space)]
        self.transformation = fenics.Function(self.deformation_space)

        self.stepsize = 1.0
        self.x_save = None
        self.current_mesh_quality = 1.0

        self.diff = [fenics.Function(self.deformation_space)]
        self.temp = [fenics.Function(self.deformation_space)]
        self.dir_prev = [fenics.Function(self.deformation_space)]
        self.difference = [fenics.Function(self.deformation_space)]

        self.history_s: collections.deque = collections.deque()
        self.history_y: collections.deque = collections.deque()
        self.history_rho: collections.deque = collections.deque()
        self.history_alpha: collections.deque = collections.deque()

    def test_for_nonconvergence(self) -> None:
        """Tests, whether maximum number of iterations are exceeded."""
        if self.iteration >= self.max_iter:
            self.output_manager.post_process()
            raise _exceptions.NotConvergedError(
                "Space Mapping",
                "Maximum number of iterations exceeded.",
            )

    def smooth_deformation(self, a: list[fenics.Function]) -> list[fenics.Function]:
        """Smooths a deformation vector field with a Poincaré-Steklov operator.

        Args:
            a: The deformation vector field

        Returns:
            A smoothed deformation vector field, for the use in the scalar product.

        """
        shape_optimization_problem = self.coarse_model.shape_optimization_problem
        form_handler = shape_optimization_problem.form_handler

        lhs = form_handler.modified_scalar_product
        rhs = (
            ufl.dot(fenics.Constant((0.0, 0.0)), form_handler.test_vector_field)
            * shape_optimization_problem.form_handler.dx
        )
        bc_helper = fenics.Function(
            shape_optimization_problem.db.function_db.control_spaces[0]
        )
        bc_helper.vector().vec().aypx(0, a[0].vector().vec())
        bc_helper.vector().apply("")
        boundary = fenics.CompiledSubDomain("on_boundary")
        bcs = [
            fenics.DirichletBC(
                shape_optimization_problem.db.function_db.control_spaces[0],
                bc_helper,
                boundary,
            )
        ]

        result = [
            fenics.Function(shape_optimization_problem.db.function_db.control_spaces[0])
        ]
        fenics.solve(lhs == rhs, result[0], bcs)

        return result

    def _compute_initial_guess(self) -> None:
        """Compute initial guess for the space mapping by solving the coarse problem."""
        self.coarse_model.optimize()
        self.z_star = [
            self.deformation_handler_coarse.coordinate_to_dof(
                self.coarse_model.coordinates_optimal - self.coordinates_initial
            )
        ]
        self.deformation_handler_fine.assign_coordinates(
            self.coarse_model.coordinates_optimal
        )
        self.current_mesh_quality = geometry.compute_mesh_quality(self.fine_model.mesh)
        self.norm_z_star = np.sqrt(self._scalar_product(self.z_star, self.z_star))

    def output(self) -> None:
        """Uses the output manager to save / show the current results."""
        optimization_state = self.db.parameter_db.optimization_state
        optimization_state["iteration"] = self.iteration
        optimization_state["objective_value"] = self.fine_model.cost_functional_value
        optimization_state["gradient_norm"] = self.eps
        optimization_state["mesh_quality"] = self.current_mesh_quality
        optimization_state["stepsize"] = self.stepsize

        self.output_manager.output()

    def solve(self) -> None:
        """Solves the problem with the space mapping method."""
        log.begin("Solving the space mapping problem.")
        self._compute_initial_guess()

        self.fine_model.simulate()

        self.parameter_extraction._solve(  # pylint: disable=protected-access
            self.iteration
        )
        self.p_current[0].vector().vec().aypx(
            0.0,
            self.deformation_handler_coarse.coordinate_to_dof(
                self.parameter_extraction.mesh.coordinates()[:, :]
                - self.coordinates_initial
            )
            .vector()
            .vec(),
        )
        self.p_current[0].vector().apply("")
        self.eps = self._compute_eps()

        self.output()

        while not self.converged:
            self.dir_prev[0].vector().vec().aypx(
                0.0, -(self.p_prev[0].vector().vec() - self.z_star[0].vector().vec())
            )
            self.dir_prev[0].vector().apply("")
            self.temp[0].vector().vec().aypx(
                0.0, -(self.p_current[0].vector().vec() - self.z_star[0].vector().vec())
            )
            self.temp[0].vector().apply("")
            self._compute_search_direction(self.temp, self.h)

            self.stepsize = 1.0
            self.p_prev[0].vector().vec().aypx(0.0, self.p_current[0].vector().vec())
            self.p_prev[0].vector().apply("")
            self.iteration += 1
            self._update_iterates()

            self.current_mesh_quality = geometry.compute_mesh_quality(
                self.fine_model.mesh
            )
            self.output()

            if self.eps <= self.tol:
                self.converged = True
                break
            self.test_for_nonconvergence()

            self._update_broyden_approximation()
            self._update_bfgs_approximation()

        if self.converged:
            self.output_manager.output_summary()
            self.output_manager.post_process()
            log.end()

    def _update_broyden_approximation(self) -> None:
        """Updates the approximation of the mapping function with Broyden's method."""
        if self.method == "broyden":
            self.temp[0].vector().vec().aypx(
                0.0, self.p_current[0].vector().vec() - self.p_prev[0].vector().vec()
            )
            self.temp[0].vector().apply("")
            self._compute_broyden_application(self.temp, self.v)

            if self.memory_size > 0:
                if self.broyden_type == "good":
                    divisor = self._scalar_product(self.h, self.v)
                    self.u[0].vector().vec().axpby(
                        1.0 / divisor,
                        0.0,
                        self.h[0].vector().vec() - self.v[0].vector().vec(),
                    )
                    self.u[0].vector().apply("")

                    self.history_s.append([xx.copy(True) for xx in self.u])
                    self.history_y.append([xx.copy(True) for xx in self.h])

                elif self.broyden_type == "bad":
                    divisor = self._scalar_product(self.temp, self.temp)
                    self.u[0].vector().vec().axpby(
                        1.0 / divisor,
                        0.0,
                        self.h[0].vector().vec() - self.v[0].vector().vec(),
                    )
                    self.u[0].vector().apply("")

                    self.history_s.append([xx.copy(True) for xx in self.u])
                    self.history_y.append([xx.copy(True) for xx in self.temp])

                if len(self.history_s) > self.memory_size:
                    self.history_s.popleft()
                    self.history_y.popleft()

    def _update_bfgs_approximation(self) -> None:
        """Updates the approximation of the mapping function with the BFGS method."""
        if self.method == "bfgs":
            if self.memory_size > 0:
                self.temp[0].vector().vec().aypx(
                    0.0,
                    self.p_current[0].vector().vec() - self.p_prev[0].vector().vec(),
                )
                self.temp[0].vector().apply("")

                self.history_y.appendleft([xx.copy(True) for xx in self.temp])
                self.history_s.appendleft([xx.copy(True) for xx in self.h])
                curvature_condition = self._scalar_product(self.temp, self.h)

                if curvature_condition <= 0.0:
                    self.history_s.clear()
                    self.history_y.clear()
                    self.history_rho.clear()
                else:
                    rho = 1 / curvature_condition
                    self.history_rho.appendleft(rho)

                if len(self.history_s) > self.memory_size:
                    self.history_s.pop()
                    self.history_y.pop()
                    self.history_rho.pop()

    def _update_iterates(self) -> None:
        """Updates the iterates either directly or via a line search."""
        if not self.use_backtracking_line_search:
            success = self.deformation_handler_fine.move_mesh(self.h[0])
            if not success:
                raise _exceptions.CashocsException(
                    "The assignment of mesh coordinates was not "
                    "possible due to intersections"
                )

            self.fine_model.simulate()

            self.parameter_extraction._solve(  # pylint: disable=protected-access
                self.iteration
            )
            self.p_current[0].vector().vec().aypx(
                0.0,
                self.deformation_handler_coarse.coordinate_to_dof(
                    self.parameter_extraction.mesh.coordinates()[:, :]
                    - self.coordinates_initial
                )
                .vector()
                .vec(),
            )
            self.p_current[0].vector().apply("")
            self.eps = self._compute_eps()

        else:
            self.x_save = self.x.coordinates().copy()

            while True:
                if self.stepsize <= 1e-4:
                    raise _exceptions.NotConvergedError(
                        "Space Mapping Backtracking Line Search",
                        "The line search did not converge.",
                    )
                self.transformation.vector().vec().axpby(
                    self.stepsize, 0.0, self.h[0].vector().vec()
                )
                self.transformation.vector().apply("")
                success = self.deformation_handler_fine.move_mesh(self.transformation)
                if success:
                    self.fine_model.simulate()

                    # pylint: disable=protected-access
                    self.parameter_extraction._solve(self.iteration)
                    self.p_current[0].vector().vec().aypx(
                        0.0,
                        self.deformation_handler_coarse.coordinate_to_dof(
                            self.parameter_extraction.mesh.coordinates()[:, :]
                            - self.coordinates_initial
                        )
                        .vector()
                        .vec(),
                    )
                    self.p_current[0].vector().apply("")
                    eps_new = self._compute_eps()

                    if eps_new <= self.eps:
                        self.eps = eps_new
                        break
                    else:
                        self.deformation_handler_fine.revert_transformation()
                        self.stepsize /= 2

                else:
                    self.stepsize /= 2

    def _scalar_product(
        self, a: list[fenics.Function], b: list[fenics.Function]
    ) -> float:
        """Computes the scalar product between ``a`` and ``b``.

        Args:
            a: The first input for the scalar product
            b: The second input for the scalar product

        Returns:
            The scalar product between ``a`` and ``b``

        """
        return float(
            self.coarse_model.shape_optimization_problem.form_handler.scalar_product(
                a, b
            )
        )

    def _compute_search_direction(
        self, q: list[fenics.Function], out: list[fenics.Function]
    ) -> None:
        """Computes the search direction for a given rhs ``q``, saved to ``out``.

        Args:
            q: The rhs for computing the search direction
            out: The output list of functions, in which the search direction is stored.

        """
        if self.method == "steepest_descent":
            return self._compute_steepest_descent_application(q, out)
        elif self.method == "broyden":
            return self._compute_broyden_application(q, out)
        elif self.method == "bfgs":
            return self._compute_bfgs_application(q, out)
        elif self.method == "ncg":
            return self._compute_ncg_direction(q, out)
        else:
            raise _exceptions.InputError(
                "cashocs.space_mapping.shape_optimization.SpaceMapping",
                "method",
                "The method is not supported.",
            )

    @staticmethod
    def _compute_steepest_descent_application(
        q: list[fenics.Function], out: list[fenics.Function]
    ) -> None:
        """Computes the search direction for the steepest descent method.

        Args:
            q: The rhs for computing the search direction
            out: The output list of functions, in which the search direction is stored.

        """
        for i in range(len(out)):
            out[i].vector().vec().aypx(0.0, q[i].vector().vec())
            out[i].vector().apply("")

    def _compute_broyden_application(
        self, q: list[fenics.Function], out: list[fenics.Function]
    ) -> None:
        """Computes the search direction for Broyden's method.

        Args:
            q: The rhs for computing the search direction
            out: The output list of functions, in which the search direction is stored.

        """
        out[0].vector().vec().aypx(0.0, q[0].vector().vec())
        out[0].vector().apply("")

        for i in range(len(self.history_s)):
            if self.broyden_type == "good":
                alpha = self._scalar_product(self.history_y[i], out)
            elif self.broyden_type == "bad":
                alpha = self._scalar_product(self.history_y[i], q)
            else:
                raise _exceptions.CashocsException(
                    "Type of Broyden's method has to be either 'good' or 'bad'."
                )

            out[0].vector().vec().axpy(alpha, self.history_s[i][0].vector().vec())
            out[0].vector().apply("")

    def _compute_bfgs_application(
        self, q: list[fenics.Function], out: list[fenics.Function]
    ) -> None:
        """Computes the search direction for the LBFGS method.

        Args:
            q: The rhs for computing the search direction
            out: The output list of functions, in which the search direction is stored.

        """
        if self.memory_size > 0 and len(self.history_s) > 0:
            self.history_alpha.clear()
            out[0].vector().vec().aypx(0.0, q[0].vector().vec())
            out[0].vector().apply("")

            for i, _ in enumerate(self.history_s):
                alpha = self.history_rho[i] * self._scalar_product(
                    self.history_s[i], out
                )
                self.history_alpha.append(alpha)
                out[0].vector().vec().axpy(-alpha, self.history_y[i][0].vector().vec())
                out[0].vector().apply("")

            bfgs_factor = self._scalar_product(
                self.history_y[0], self.history_s[0]
            ) / self._scalar_product(self.history_y[0], self.history_y[0])
            out[0].vector().vec().scale(bfgs_factor)
            out[0].vector().apply("")

            for i, _ in enumerate(self.history_s):
                beta = self.history_rho[-1 - i] * self._scalar_product(
                    self.history_y[-1 - i], out
                )
                out[0].vector().vec().axpy(
                    self.history_alpha[-1 - i] - beta,
                    self.history_s[-1 - i][0].vector().vec(),
                )
                out[0].vector().apply("")

        else:
            out[0].vector().vec().aypx(0.0, q[0].vector().vec())
            out[0].vector().apply("")

    def _compute_ncg_direction(
        self, q: list[fenics.Function], out: list[fenics.Function]
    ) -> None:
        """Computes the search direction for the NCG methods.

        Args:
            q: The rhs for computing the search direction
            out: The output list of functions, in which the search direction is stored.

        """
        if self.iteration > 0:
            self.difference[0].vector().vec().aypx(
                0.0, q[0].vector().vec() - self.dir_prev[0].vector().vec()
            )
            self.difference[0].vector().apply("")

            if self.cg_type == "FR":
                beta_num = self._scalar_product(q, q)
                beta_denom = self._scalar_product(self.dir_prev, self.dir_prev)
                beta = beta_num / beta_denom
            elif self.cg_type == "PR":
                beta_num = self._scalar_product(q, self.difference)
                beta_denom = self._scalar_product(self.dir_prev, self.dir_prev)
                beta = beta_num / beta_denom
            elif self.cg_type == "HS":
                beta_num = self._scalar_product(q, self.difference)
                beta_denom = -self._scalar_product(out, self.difference)
                beta = beta_num / beta_denom
            elif self.cg_type == "DY":
                beta_num = self._scalar_product(q, q)
                beta_denom = -self._scalar_product(out, self.difference)
                beta = beta_num / beta_denom
            elif self.cg_type == "HZ":
                dy = -self._scalar_product(out, self.difference)
                y2 = self._scalar_product(self.difference, self.difference)

                self.difference[0].vector().vec().axpby(
                    -2 * y2 / dy, -1.0, out[0].vector().vec()
                )
                self.difference[0].vector().apply("")
                beta = -self._scalar_product(self.difference, q) / dy
            else:
                beta = 0.0
        else:
            beta = 0.0

        out[0].vector().vec().aypx(beta, q[0].vector().vec())
        out[0].vector().apply("")

    def _compute_eps(self) -> float:
        """Computes and returns the termination parameter epsilon."""
        self.diff[0].vector().vec().aypx(
            0.0, self.p_current[0].vector().vec() - self.z_star[0].vector().vec()
        )
        self.diff[0].vector().apply("")
        eps = float(
            np.sqrt(self._scalar_product(self.diff, self.diff)) / self.norm_z_star
        )

        return eps

    def inject_pre_callback(self, function: Callable | None) -> None:
        """Changes the a-priori callback of the OptimizationProblem.

        Args:
            function: A custom function without arguments, which will be called before
                each solve of the state system

        """
        self.coarse_model._pre_callback = function  # pylint: disable=protected-access
        # pylint: disable=protected-access
        self.parameter_extraction._pre_callback = function

    def inject_post_callback(self, function: Callable | None) -> None:
        """Changes the a-posteriori callback of the OptimizationProblem.

        Args:
            function: A custom function without arguments, which will be called after
                the computation of the gradient(s)

        """
        self.coarse_model._post_callback = function  # pylint: disable=protected-access
        # pylint: disable=protected-access
        self.parameter_extraction._post_callback = function

    def inject_pre_post_callback(
        self, pre_function: Callable | None, post_function: Callable | None
    ) -> None:
        """Changes the a-priori (pre) and a-posteriori (post) callbacks of the problem.

        Args:
            pre_function: A function without arguments, which is to be called before
                each solve of the state system
            post_function: A function without arguments, which is to be called after
                each computation of the (shape) gradient

        """
        self.inject_pre_callback(pre_function)
        self.inject_post_callback(post_function)
