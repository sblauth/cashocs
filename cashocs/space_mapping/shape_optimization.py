# Copyright (C) 2020-2021 Sebastian Blauth
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

"""Space mapping for shape optimization problems.

"""

import numpy as np
import fenics
from ufl import replace
from _collections import deque

import cashocs
from .._shape_optimization.shape_optimization_problem import ShapeOptimizationProblem
from .._exceptions import InputError, NotConvergedError, GeometryError
from ..utils import _check_and_enlist_functions
from ..geometry import DeformationHandler, compute_mesh_quality
from .._loggers import debug


class ParentFineModel:
    def __init__(self, mesh):

        self.mesh = fenics.Mesh(mesh)
        self.cost_functional_value = None

    def solve_and_evaluate(self):
        pass


class CoarseModel:
    def __init__(
        self,
        state_forms,
        bcs_list,
        cost_functional_form,
        states,
        adjoints,
        boundaries,
        config=None,
        shape_scalar_product=None,
        initial_guess=None,
        ksp_options=None,
        adjoint_ksp_options=None,
        desired_weights=None,
        scalar_tracking_forms=None,
    ):

        self.state_forms = state_forms
        self.bcs_list = bcs_list
        self.cost_functional_form = cost_functional_form
        self.states = states
        self.adjoints = adjoints
        self.boundaries = boundaries
        self.config = config
        self.shape_scalar_product = shape_scalar_product
        self.initial_guess = initial_guess
        self.ksp_options = ksp_options
        self.adjoint_ksp_options = adjoint_ksp_options
        self.desired_weights = desired_weights
        self.scalar_tracking_forms = scalar_tracking_forms

        self.mesh = self.boundaries.mesh()
        self.coordinates_initial = self.mesh.coordinates().copy()
        self.coordinates_optimal = None

        self.shape_optimization_problem = ShapeOptimizationProblem(
            self.state_forms,
            self.bcs_list,
            self.cost_functional_form,
            self.states,
            self.adjoints,
            self.boundaries,
            config=self.config,
            shape_scalar_product=self.shape_scalar_product,
            initial_guess=self.initial_guess,
            ksp_options=self.ksp_options,
            adjoint_ksp_options=self.adjoint_ksp_options,
            desired_weights=self.desired_weights,
            scalar_tracking_forms=self.scalar_tracking_forms,
        )

    def optimize(self):

        self.shape_optimization_problem.solve()
        self.coordinates_optimal = self.mesh.coordinates().copy()


class ParameterExtraction:
    def __init__(
        self,
        coarse_model,
        cost_functional_form,
        states,
        config=None,
        scalar_tracking_forms=None,
        desired_weights=None,
    ):

        """

        Parameters
        ----------
        coarse_model : CoarseModel
        cost_functional_form
        states
        config
        scalar_tracking_forms
        desired_weights
        """

        self.coarse_model = coarse_model
        self.mesh = coarse_model.mesh
        self.cost_functional_form = cost_functional_form

        ### states
        try:
            self.states = _check_and_enlist_functions(states)
        except InputError:
            raise InputError(
                "cashocs.space_mapping.shape_optimization.ParameterExtraction",
                "states",
                "Type of states is wring.",
            )

        self.config = config
        self.scalar_tracking_forms = scalar_tracking_forms
        self.desired_weights = desired_weights

        self.adjoints = [
            fenics.Function(V)
            for V in coarse_model.shape_optimization_problem.form_handler.adjoint_spaces
        ]

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
            replace(form, mapping_dict)
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

        self.coordinates_initial = coarse_model.coordinates_initial
        self.deformation_handler = DeformationHandler(self.mesh)

        self.shape_optimization_problem = None

    def _solve(self, initial_guess=None):
        if initial_guess is None:
            self.deformation_handler.assign_coordinates(self.coordinates_initial)
        else:
            self.deformation_handler.assign_coordinates(initial_guess)

        self.shape_optimization_problem = ShapeOptimizationProblem(
            self.state_forms,
            self.bcs_list,
            self.cost_functional_form,
            self.states,
            self.adjoints,
            self.boundaries,
            config=self.config,
            shape_scalar_product=self.shape_scalar_product,
            initial_guess=self.initial_guess,
            ksp_options=self.ksp_options,
            adjoint_ksp_options=self.adjoint_ksp_options,
            desired_weights=self.desired_weights,
            scalar_tracking_forms=self.scalar_tracking_forms,
        )

        self.shape_optimization_problem.solve()


class SpaceMapping:
    def __init__(
        self,
        fine_model,
        coarse_model,
        parameter_extraction,
        method="broyden",
        max_iter=25,
        tol=1e-2,
        use_backtracking_line_search=False,
        broyden_type="good",
        cg_type="FR",
        memory_size=10,
        verbose=True,
    ):
        """

        Parameters
        ----------
        fine_model : ParentFineModel
        coarse_model : CoarseModel
        parameter_extraction : ParameterExtraction
        method
        max_iter
        tol
        use_backtracking_line_search
        broyden_type
        cg_type
        memory_size
        verbose
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

        self.coordinates_initial = self.coarse_model.coordinates_initial

        self.eps = 1.0
        self.converged = False
        self.iteration = 0

        self.x = self.fine_model.mesh

        self.VCG = fenics.VectorFunctionSpace(self.x, "CG", 1)
        self.deformation_handler_fine = cashocs.DeformationHandler(self.fine_model.mesh)
        self.deformation_handler_coarse = cashocs.DeformationHandler(
            self.coarse_model.mesh
        )

        self.z_star = fenics.Function(self.VCG)
        self.p_current = fenics.Function(self.VCG)
        self.p_prev = fenics.Function(self.VCG)
        self.h = fenics.Function(self.VCG)
        self.v = fenics.Function(self.VCG)
        self.u = fenics.Function(self.VCG)

        self.x_save = None

        self.diff = fenics.Function(self.VCG)
        self.temp = fenics.Function(self.VCG)
        self.dir_prev = fenics.Function(self.VCG)
        self.difference = fenics.Function(self.VCG)

        self.history_s = deque()
        self.history_y = deque()
        self.history_rho = deque()
        self.history_alpha = deque()

    def solve(self):
        self.coarse_model.optimize()
        self.z_star = self.deformation_handler_coarse.coordinate_to_dof(
            self.coarse_model.coordinates_optimal - self.coordinates_initial
        )
        self.deformation_handler_fine.assign_coordinates(
            self.coarse_model.coordinates_optimal
        )
        self.current_mesh_quality = compute_mesh_quality(self.fine_model.mesh)
        self.norm_z_star = np.sqrt(self._scalar_product(self.z_star, self.z_star))

        self.fine_model.solve_and_evaluate()
        # self.parameter_extraction._solve()
        self.parameter_extraction._solve(
            initial_guess=self.coarse_model.coordinates_optimal
        )
        self.p_current.vector()[:] = self.deformation_handler_coarse.coordinate_to_dof(
            self.parameter_extraction.mesh.coordinates()[:, :]
            - self.coordinates_initial
        ).vector()[:]
        self.eps = self._compute_eps()

        if self.verbose:
            print(
                f"Space Mapping - Iteration {self.iteration:3d}:    Cost functional value = "
                f"{self.fine_model.cost_functional_value:.3e}    eps = {self.eps:.3e}"
                f"    Mesh Quality = {self.current_mesh_quality:1.2f}\n"
            )

        while not self.converged:
            self.dir_prev.vector()[:] = -(
                self.p_prev.vector()[:] - self.z_star.vector()[:]
            )
            self.temp.vector()[:] = -(
                self.p_current.vector()[:] - self.z_star.vector()[:]
            )
            self._compute_search_direction(self.temp, self.h)

            # if self._scalar_product(self.h, self.temp) <= 0.0:
            #     debug(
            #         "The computed search direction for space mapping did not yield a descent direction"
            #     )
            #     self.h.vector()[:] = self.temp.vector()[:]

            stepsize = 1.0
            self.p_prev.vector()[:] = self.p_current.vector()[:]
            if not self.use_backtracking_line_search:
                success = self.deformation_handler_fine.assign_coordinates(
                    self.x.coordinates()[:, :]
                    + stepsize
                    * self.deformation_handler_coarse.dof_to_coordinate(self.h)
                )
                if not success:
                    raise GeometryError(
                        "The assignment of mesh coordinates was not possible due to intersections"
                    )

                self.fine_model.solve_and_evaluate()
                # self.parameter_extraction._solve()
                self.parameter_extraction._solve(
                    initial_guess=self.coarse_model.coordinates_optimal
                )
                self.p_current.vector()[
                    :
                ] = self.deformation_handler_coarse.coordinate_to_dof(
                    self.parameter_extraction.mesh.coordinates()[:, :]
                    - self.coordinates_initial
                ).vector()[
                    :
                ]
                self.eps = self._compute_eps()

            else:
                self.x_save = self.x.coordinates().copy()

                while True:
                    if stepsize <= 1e-4:
                        raise NotConvergedError(
                            "Space Mapping Backtracking Line Search",
                            "The line search did not converge.",
                        )

                    success = self.deformation_handler_fine.assign_coordinates(
                        self.x_save
                        + stepsize
                        * self.deformation_handler_coarse.dof_to_coordinate(self.h)
                    )
                    if success:

                        self.fine_model.solve_and_evaluate()
                        # self.parameter_extraction._solve()
                        self.parameter_extraction._solve(
                            self.coarse_model.coordinates_optimal
                        )
                        self.p_current.vector()[
                            :
                        ] = self.deformation_handler_coarse.coordinate_to_dof(
                            self.parameter_extraction.mesh.coordinates()[:, :]
                            - self.coordinates_initial
                        ).vector()[
                            :
                        ]
                        eps_new = self._compute_eps()

                        if eps_new <= self.eps:
                            self.eps = eps_new
                            break
                        else:
                            stepsize /= 2

                    else:
                        stepsize /= 2

            self.iteration += 1
            self.current_mesh_quality = compute_mesh_quality(self.fine_model.mesh)
            if self.verbose:
                print(
                    f"Space Mapping - Iteration {self.iteration:3d}:    Cost functional value = "
                    f"{self.fine_model.cost_functional_value:.3e}    eps = {self.eps:.3e}"
                    f"    Mesh Quality = {self.current_mesh_quality:1.2f}    step size = {stepsize:.3e}"
                )

            if self.eps <= self.tol:
                self.converged = True
                break
            if self.iteration >= self.max_iter:
                break

            if self.method == "broyden":
                self.temp.vector()[:] = (
                    self.p_current.vector()[:] - self.p_prev.vector()[:]
                )
                self._compute_broyden_application(self.temp, self.v)

                if self.memory_size > 0:
                    if self.broyden_type == "good":
                        divisor = self._scalar_product(self.h, self.v)
                        self.u.vector()[:] = (
                            self.h.vector()[:] - self.v.vector()[:]
                        ) / divisor

                        self.history_s.append(self.u.copy(True))
                        self.history_y.append(self.h.copy(True))

                    elif self.broyden_type == "bad":
                        divisor = self._scalar_product(self.temp, self.temp)
                        self.u.vector()[:] = (
                            self.h.vector()[:] - self.v.vector()[:]
                        ) / divisor

                        self.history_s.append(self.u.copy(True))
                        self.history_y.append(self.temp.copy(True))

                    if len(self.history_s) > self.memory_size:
                        self.history_s.popleft()
                        self.history_y.popleft()

            elif self.method == "bfgs":
                if self.memory_size > 0:
                    self.temp.vector()[:] = (
                        self.p_current.vector()[:] - self.p_prev.vector()[:]
                    )

                    self.history_y.appendleft(self.temp.copy(True))
                    self.history_s.appendleft(self.h.copy(True))
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

        if self.converged:
            output = (
                f"\nStatistics --- Space mapping iterations: {self.iteration:4d}"
                + f" --- Final objective value: {self.fine_model.cost_functional_value:.3e}\n"
            )
            if self.verbose:
                print(output)

    def _scalar_product(self, a, b):
        return self.coarse_model.shape_optimization_problem.form_handler.scalar_product(
            a, b
        )

    def _compute_search_direction(self, q, out):
        if self.method == "steepest_descent":
            return self._compute_steepest_descent_application(q, out)
        elif self.method == "broyden":
            return self._compute_broyden_application(q, out)
        elif self.method == "bfgs":
            return self._compute_bfgs_application(q, out)
        elif self.method == "ncg":
            return self._compute_ncg_direction(q, out)
        else:
            raise InputError(
                "cashocs.space_mapping.shape_optimization.SpaceMapping",
                "method",
                "The method is not supported.",
            )

    @staticmethod
    def _compute_steepest_descent_application(q, out):
        out.vector()[:] = q.vector()[:]

    def _compute_broyden_application(self, q, out):
        out.vector()[:] = q.vector()[:]

        for i in range(len(self.history_s)):
            if self.broyden_type == "good":
                alpha = self._scalar_product(self.history_y[i], out)
            elif self.broyden_type == "bad":
                alpha = self._scalar_product(self.history_y[i], q)
            else:
                raise Exception(
                    "Type of Broyden's method has to be either 'good' or 'bad'."
                )

            out.vector()[:] += alpha * self.history_s[i].vector()[:]

    def _compute_bfgs_application(self, q, out):
        if self.memory_size > 0 and len(self.history_s) > 0:
            self.history_alpha.clear()
            out.vector()[:] = q.vector()[:]

            for i, _ in enumerate(self.history_s):
                alpha = self.history_rho[i] * self._scalar_product(
                    self.history_s[i], out
                )
                self.history_alpha.append(alpha)
                out.vector()[:] -= alpha * self.history_y[i].vector()[:]

            bfgs_factor = self._scalar_product(
                self.history_y[0], self.history_s[0]
            ) / self._scalar_product(self.history_y[0], self.history_y[0])
            out.vector()[:] *= bfgs_factor

            for i, _ in enumerate(self.history_s):
                beta = self.history_rho[-1 - i] * self._scalar_product(
                    self.history_y[-1 - i], out
                )
                out.vector()[:] += self.history_s[-1 - i].vector()[:] * (
                    self.history_alpha[-1 - i] - beta
                )

        else:
            out.vector()[:] = q.vector()[:]

    def _compute_ncg_direction(self, q, out):
        if self.iteration > 0:
            if self.cg_type == "FR":
                beta_num = self._scalar_product(q, q)
                beta_denom = self._scalar_product(self.dir_prev, self.dir_prev)
                self.beta = beta_num / beta_denom
            elif self.cg_type == "PR":
                self.difference.vector()[:] = q.vector()[:] - self.dir_prev.vector()[:]
                beta_num = self._scalar_product(q, self.difference)
                beta_denom = self._scalar_product(self.dir_prev, self.dir_prev)
                self.beta = beta_num / beta_denom
            elif self.cg_type == "HS":
                self.difference.vector()[:] = q.vector()[:] - self.dir_prev.vector()[:]
                beta_num = self._scalar_product(q, self.difference)
                beta_denom = -self._scalar_product(out, self.difference)
                self.beta = beta_num / beta_denom
            elif self.cg_type == "DY":
                self.difference.vector()[:] = q.vector()[:] - self.dir_prev.vector()[:]
                beta_num = self._scalar_product(q, q)
                beta_denom = -self._scalar_product(out, self.difference)
                self.beta = beta_num / beta_denom
            elif self.cg_type == "HZ":
                self.difference.vector()[:] = q.vector()[:] - self.dir_prev.vector()[:]
                dy = -self._scalar_product(out, self.difference)
                y2 = self._scalar_product(self.difference, self.difference)

                self.difference.vector()[:] = (
                    -self.difference.vector()[:] - 2 * y2 / dy * out.vector()[:]
                )
                self.beta = -self._scalar_product(self.difference, q) / dy
        else:
            self.beta = 0.0

        out.vector()[:] = q.vector()[:] + self.beta * out.vector()[:]

    def _compute_eps(self):
        self.diff.vector()[:] = self.p_current.vector()[:] - self.z_star.vector()[:]
        eps = np.sqrt(self._scalar_product(self.diff, self.diff)) / self.norm_z_star

        return eps
