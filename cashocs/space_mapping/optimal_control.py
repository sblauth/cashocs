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

"""Space mapping for optimal control problems

"""

import numpy as np
import fenics
from ufl import replace
from _collections import deque

from .._optimal_control.optimal_control_problem import OptimalControlProblem
from .._exceptions import InputError


class ParentFineModel:
    def __init__(self):
        self.control = None
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
        controls,
        adjoints,
        config=None,
        riesz_scalar_products=None,
        control_constraints=None,
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
        self.controls = controls
        self.adjoints = adjoints
        self.config = config
        self.riesz_scalar_products = riesz_scalar_products
        self.control_constraints = control_constraints
        self.initial_guess = initial_guess
        self.ksp_options = ksp_options
        self.adjoint_ksp_options = adjoint_ksp_options
        self.desired_weights = desired_weights
        self.scalar_tracking_forms = scalar_tracking_forms

        self.optimal_control_problem = OptimalControlProblem(
            self.state_forms,
            self.bcs_list,
            self.cost_functional_form,
            self.states,
            self.controls,
            self.adjoints,
            config=self.config,
            riesz_scalar_products=self.riesz_scalar_products,
            control_constraints=self.control_constraints,
            initial_guess=self.initial_guess,
            ksp_options=self.ksp_options,
            adjoint_ksp_options=self.adjoint_ksp_options,
            desired_weights=self.desired_weights,
            scalar_tracking_forms=self.scalar_tracking_forms,
        )

    def optimize(self):

        self.optimal_control_problem.solve()


class ParameterExtraction:
    def __init__(
        self,
        coarse_model,
        cost_functional_form,
        states,
        controls,
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
        controls
        config
        scalar_tracking_forms
        desired_weights

        Returns
        -------

        """

        self.coarse_model = coarse_model
        self.cost_functional_form = cost_functional_form

        ### states
        try:
            if type(states) == list and len(states) > 0:
                for i in range(len(states)):
                    if (
                        states[i].__module__ == "dolfin.function.function"
                        and type(states[i]).__name__ == "Function"
                    ):
                        pass
                    else:
                        raise InputError(
                            "cashocs.space_mapping.optimal_control.ParameterExtraction",
                            "states",
                            "states have to be fenics Functions.",
                        )

                self.states = states

            elif (
                states.__module__ == "dolfin.function.function"
                and type(states).__name__ == "Function"
            ):
                self.states = [states]
            else:
                raise InputError(
                    "cashocs.space_mapping.optimal_control.ParameterExtraction",
                    "states",
                    "Type of states is wrong.",
                )
        except:
            raise InputError(
                "cashocs.space_mapping.optimal_control.ParameterExtraction",
                "states",
                "Type of states is wrong.",
            )

        ### controls
        try:
            if type(controls) == list and len(controls) > 0:
                for i in range(len(controls)):
                    if (
                        controls[i].__module__ == "dolfin.function.function"
                        and type(controls[i]).__name__ == "Function"
                    ):
                        pass
                    else:
                        raise InputError(
                            "cashocs.space_mapping.optimal_control.ParameterExtraction",
                            "controls",
                            "controls have to be fenics Functions.",
                        )

                self.controls = controls

            elif (
                controls.__module__ == "dolfin.function.function"
                and type(controls).__name__ == "Function"
            ):
                self.controls = [controls]
            else:
                raise InputError(
                    "cashocs.space_mapping.optimal_control.ParameterExtraction",
                    "controls",
                    "Type of controls is wrong.",
                )
        except:
            raise InputError(
                "cashocs.space_mapping.optimal_control.ParameterExtraction",
                "controls",
                "Type of controls is wrong.",
            )

        self.config = config
        self.scalar_tracking_forms = scalar_tracking_forms
        self.desired_weights = desired_weights

        self.adjoints = [
            fenics.Function(V)
            for V in coarse_model.optimal_control_problem.form_handler.adjoint_spaces
        ]

        dict_states = {
            coarse_model.optimal_control_problem.states[i]: self.states[i]
            for i in range(len(self.states))
        }
        dict_adjoints = {
            coarse_model.optimal_control_problem.adjoints[i]: self.adjoints[i]
            for i in range(len(self.adjoints))
        }
        dict_controls = {
            coarse_model.optimal_control_problem.controls[i]: self.controls[i]
            for i in range(len(self.controls))
        }
        mapping_dict = {}
        mapping_dict.update(dict_states)
        mapping_dict.update(dict_adjoints)
        mapping_dict.update(dict_controls)
        self.state_forms = [
            replace(form, mapping_dict)
            for form in coarse_model.optimal_control_problem.state_forms
        ]
        self.bcs_list = coarse_model.bcs_list
        self.riesz_scalar_products = (
            coarse_model.optimal_control_problem.riesz_scalar_products
        )
        self.control_constraints = (
            coarse_model.optimal_control_problem.control_constraints
        )
        self.initial_guess = coarse_model.optimal_control_problem.initial_guess
        self.ksp_options = coarse_model.optimal_control_problem.ksp_options
        self.adjoint_ksp_options = (
            coarse_model.optimal_control_problem.adjoint_ksp_options
        )

        self.optimal_control_problem = None

    def _solve(self, initial_guesses=None):

        if initial_guesses is None:
            for i in range(len(self.controls)):
                self.controls[i].vector()[:] = 0.0
        else:
            for i in range(len(self.controls)):
                self.controls[i].vector()[:] = initial_guesses[i].vector()[:]

        self.optimal_control_problem = OptimalControlProblem(
            self.state_forms,
            self.bcs_list,
            self.cost_functional_form,
            self.states,
            self.controls,
            self.adjoints,
            config=self.config,
            riesz_scalar_products=self.riesz_scalar_products,
            control_constraints=self.control_constraints,
            initial_guess=self.initial_guess,
            ksp_options=self.ksp_options,
            adjoint_ksp_options=self.adjoint_ksp_options,
            desired_weights=self.desired_weights,
            scalar_tracking_forms=self.scalar_tracking_forms,
        )

        self.optimal_control_problem.solve()


class SpaceMapping:
    def __init__(self):
        pass

    def solve(self):
        pass

    def _inner_product(self):
        pass

    def _compute_search_direction(self):
        pass

    def _compute_steepest_descent_application(self):
        pass

    def _compute_broyden_application(self):
        pass

    def _compute_bfgs_application(self):
        pass

    def _compute_ncg_direction(self):
        pass

    def _compute_eps(self):
        pass
