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

"""Module for managing UFL forms for PDE constrained optimization."""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Optional, List, Union

from petsc4py import PETSc
import fenics
import ufl

from cashocs import utils

if TYPE_CHECKING:
    from cashocs import _optimization as op
    from cashocs._forms import shape_regularization


def _get_subdx(
    function_space: fenics.FunctionSpace, index: int, ls: List
) -> Union[None, List[int]]:
    """Computes the sub-indices for mixed function spaces based on the id of a subspace.

    Args:
        function_space: The function space, whose substructure is to be investigated.
        index: The id of the target function space.
        ls: A list of indices for the sub-spaces.

    Returns:
        The list of the sub-indices.
    """
    if function_space.id() == index:
        return ls
    if function_space.num_sub_spaces() > 1:
        for i in range(function_space.num_sub_spaces()):
            ans = _get_subdx(function_space.sub(i), index, ls + [i])
            if ans is not None:
                return ans

    return None


class FormHandler(abc.ABC):
    """Parent class for UFL form manipulation.

    This is subclassed by specific form handlers for either
    optimal control or shape optimization. The base class is
    used to determine common objects and to derive the UFL forms
    for the state and adjoint systems.
    """

    fe_shape_derivative_vector: fenics.PETScVector
    assembler: fenics.SystemAssembler
    fixed_indices: List[int]
    use_fixed_dimensions: bool = False
    test_vector_field: fenics.TestFunction
    gradient: List[fenics.Function]
    control_spaces: List[fenics.FunctionSpace]
    deformation_space: fenics.FunctionSpace
    controls: List[fenics.Function]
    control_dim: int
    riesz_projection_matrices: List[PETSc.Mat]
    uses_custom_scalar_product: bool = False

    def __init__(self, optimization_problem: op.OptimizationProblem) -> None:
        """Initializes self.

        Args:
            optimization_problem: The corresponding optimization problem
        """
        self.bcs_list = optimization_problem.bcs_list
        self.states = optimization_problem.states
        self.adjoints = optimization_problem.adjoints
        self.config = optimization_problem.config
        self.state_ksp_options = optimization_problem.ksp_options
        self.adjoint_ksp_options = optimization_problem.adjoint_ksp_options
        self.use_scalar_tracking = optimization_problem.use_scalar_tracking
        self.use_min_max_terms = optimization_problem.use_min_max_terms
        self.min_max_forms = optimization_problem.min_max_terms
        self.scalar_tracking_forms = optimization_problem.scalar_tracking_forms

        self.is_shape_problem = optimization_problem.is_shape_problem
        self.is_control_problem = optimization_problem.is_control_problem

        self.cost_functional_form = optimization_problem.cost_functional_form
        self.state_forms = optimization_problem.state_forms

        self.shape_regularization: Optional[
            shape_regularization.ShapeRegularization
        ] = None
        self.gradient_forms_rhs = None
        self.shape_derivative = None
        self.bcs_shape = None
        self.scalar_product_matrix = None
        self.mu_lame = None

        self.lagrangian_form = self.cost_functional_form + utils.summation(
            self.state_forms
        )
        self.cost_functional_shift = 0.0

        if self.use_scalar_tracking:
            self.scalar_cost_functional_integrands = [
                d["integrand"] for d in self.scalar_tracking_forms
            ]
            self.scalar_tracking_goals = [
                d["tracking_goal"] for d in self.scalar_tracking_forms
            ]

            dummy_meshes = [
                integrand.integrals()[0].ufl_domain().ufl_cargo()
                for integrand in self.scalar_cost_functional_integrands
            ]
            self.scalar_cost_functional_integrand_values = [
                fenics.Function(fenics.FunctionSpace(mesh, "R", 0))
                for mesh in dummy_meshes
            ]
            self.scalar_weights = [
                fenics.Function(fenics.FunctionSpace(mesh, "R", 0))
                for mesh in dummy_meshes
            ]

            self.no_scalar_tracking_terms = len(self.scalar_tracking_goals)
            try:
                for j in range(self.no_scalar_tracking_terms):
                    self.scalar_weights[j].vector().vec().set(
                        self.scalar_tracking_forms[j]["weight"]
                    )
            except KeyError:
                for j in range(self.no_scalar_tracking_terms):
                    self.scalar_weights[j].vector().vec().set(1.0)

        if self.use_min_max_terms:
            self.min_max_integrands = [d["integrand"] for d in self.min_max_forms]
            self.min_max_lower_bounds = [d["lower_bound"] for d in self.min_max_forms]
            self.min_max_upper_bounds = [d["upper_bound"] for d in self.min_max_forms]

            dummy_meshes = [
                integrand.integrals()[0].ufl_domain().ufl_cargo()
                for integrand in self.min_max_integrands
            ]
            self.min_max_integrand_values = [
                fenics.Function(fenics.FunctionSpace(mesh, "R", 0))
                for mesh in dummy_meshes
            ]

            self.no_min_max_terms = len(self.min_max_integrands)
            self.min_max_mu = []
            self.min_max_lambda = []
            for j in range(self.no_min_max_terms):
                self.min_max_mu.append(self.min_max_forms[j]["mu"])
                self.min_max_lambda.append(self.min_max_forms[j]["lambda"])

        self.state_dim = len(self.states)

        self.state_spaces = [x.function_space() for x in self.states]
        self.adjoint_spaces = [x.function_space() for x in self.adjoints]

        # Test if state_spaces coincide with adjoint_spaces
        if self.state_spaces == self.adjoint_spaces:
            self.state_adjoint_equal_spaces = True
        else:
            self.state_adjoint_equal_spaces = False

        self.mesh = self.state_spaces[0].mesh()
        self.dx = fenics.Measure("dx", self.mesh)

        self.trial_functions_state = [
            fenics.TrialFunction(V) for V in self.state_spaces
        ]
        self.test_functions_state = [fenics.TestFunction(V) for V in self.state_spaces]

        self.trial_functions_adjoint = [
            fenics.TrialFunction(V) for V in self.adjoint_spaces
        ]
        self.test_functions_adjoint = [
            fenics.TestFunction(V) for V in self.adjoint_spaces
        ]

        self.state_is_linear = self.config.getboolean("StateSystem", "is_linear")
        self.state_is_picard = self.config.getboolean("StateSystem", "picard_iteration")
        self.opt_algo = utils._optimization_algorithm_configuration(self.config)

        self._compute_state_equations()
        self._compute_adjoint_equations()

    def _compute_state_equations(self) -> None:
        """Calculates the weak form of the state equation for the use with fenics."""
        self.state_eq_forms = [
            fenics.derivative(
                self.state_forms[i], self.adjoints[i], self.test_functions_state[i]
            )
            for i in range(self.state_dim)
        ]

        if self.state_is_linear:
            self.linear_state_eq_forms = [
                ufl.replace(
                    self.state_eq_forms[i],
                    {self.states[i]: self.trial_functions_state[i]},
                )
                for i in range(self.state_dim)
            ]

            (
                self.state_eq_forms_lhs,
                self.state_eq_forms_rhs,
            ) = utils._split_linear_forms(self.linear_state_eq_forms)

    def _compute_adjoint_boundary_conditions(self) -> None:
        """Computes the boundary conditions for the adjoint systems."""
        if self.state_adjoint_equal_spaces:
            self.bcs_list_ad = [
                [fenics.DirichletBC(bc) for bc in self.bcs_list[i]]
                for i in range(self.state_dim)
            ]
            [
                [bc.homogenize() for bc in self.bcs_list_ad[i]]
                for i in range(self.state_dim)
            ]
        else:

            self.bcs_list_ad = [
                [1] * len(self.bcs_list[i]) for i in range(self.state_dim)
            ]

            for i in range(self.state_dim):
                for j, bc in enumerate(self.bcs_list[i]):
                    idx = bc.function_space().id()
                    subdx = _get_subdx(self.state_spaces[i], idx, ls=[])
                    adjoint_space = self.adjoint_spaces[i]
                    for num in subdx:
                        adjoint_space = adjoint_space.sub(num)
                    shape = adjoint_space.ufl_element().value_shape()
                    if shape == ():
                        bdry_value = fenics.Constant(0)
                    else:
                        bdry_value = fenics.Constant(
                            [0] * adjoint_space.ufl_element().value_size()
                        )

                    try:
                        self.bcs_list_ad[i][j] = fenics.DirichletBC(
                            adjoint_space,
                            bdry_value,
                            bc.domain_args[0],
                            bc.domain_args[1],
                        )
                    except AttributeError:
                        self.bcs_list_ad[i][j] = fenics.DirichletBC(
                            adjoint_space, bdry_value, bc.sub_domain
                        )

    def _compute_adjoint_scalar_tracking_forms(self) -> None:
        """Compute the part arising due to scalar_tracking_terms."""
        if self.use_scalar_tracking:
            for i in range(self.state_dim):
                for j in range(self.no_scalar_tracking_terms):
                    self.adjoint_eq_forms[i] += (
                        self.scalar_weights[j]
                        * (
                            self.scalar_cost_functional_integrand_values[j]
                            - fenics.Constant(self.scalar_tracking_goals[j])
                        )
                        * fenics.derivative(
                            self.scalar_cost_functional_integrands[j],
                            self.states[i],
                            self.test_functions_adjoint[i],
                        )
                    )

    def _compute_adjoint_min_max_forms(self) -> None:
        """Compute the part arising due to min_max_terms."""
        if self.use_min_max_terms:
            for i in range(self.state_dim):
                for j in range(self.no_min_max_terms):
                    if self.min_max_lower_bounds[j] is not None:
                        term_lower = self.min_max_lambda[j] + self.min_max_mu[j] * (
                            self.min_max_integrand_values[j]
                            - self.min_max_lower_bounds[j]
                        )
                        self.adjoint_eq_forms[i] += utils._min(
                            fenics.Constant(0.0), term_lower
                        ) * fenics.derivative(
                            self.min_max_integrands[j],
                            self.states[i],
                            self.test_functions_adjoint[i],
                        )

                    if self.min_max_upper_bounds[j] is not None:
                        term_upper = self.min_max_lambda[j] + self.min_max_mu[j] * (
                            self.min_max_integrand_values[j]
                            - self.min_max_upper_bounds[j]
                        )
                        self.adjoint_eq_forms[i] += utils._max(
                            fenics.Constant(0.0), term_upper
                        ) * fenics.derivative(
                            self.min_max_integrands[j],
                            self.states[i],
                            self.test_functions_adjoint[i],
                        )

    def _compute_adjoint_equations(self) -> None:
        """Calculates the weak form of the adjoint equation for use with fenics."""
        self.adjoint_eq_forms = [
            fenics.derivative(
                self.lagrangian_form,
                self.states[i],
                self.test_functions_adjoint[i],
            )
            for i in range(self.state_dim)
        ]

        self._compute_adjoint_scalar_tracking_forms()
        self._compute_adjoint_min_max_forms()

        self.linear_adjoint_eq_forms = [
            ufl.replace(
                self.adjoint_eq_forms[i],
                {self.adjoints[i]: self.trial_functions_adjoint[i]},
            )
            for i in range(self.state_dim)
        ]

        self.adjoint_eq_lhs, self.adjoint_eq_rhs = utils._split_linear_forms(
            self.linear_adjoint_eq_forms
        )

        self._compute_adjoint_boundary_conditions()

    def _pre_hook(self) -> None:
        pass

    def _post_hook(self) -> None:
        pass

    @abc.abstractmethod
    def scalar_product(
        self, a: List[fenics.Function], b: List[fenics.Function]
    ) -> float:
        """Computes the scalar product between a and b.

        Args:
            a: The first argument.
            b: The second argument.

        Returns:
            The scalar product of a and b.
        """
        pass

    def restrict_to_inactive_set(
        self, a: List[fenics.Function], b: List[fenics.Function]
    ) -> List[fenics.Function]:
        """Restricts a function to the inactive set.

        Note, that nothing will happen if the type of the optimization problem does not
        support box constraints.

        Args:
            a: The function, which shall be restricted to the inactive set
            b: The output function, which will contain the result and is overridden.

        Returns:
            The result of the restriction (overrides input b)
        """
        for j in range(len(self.gradient)):
            if not b[j].vector().vec().equal(a[j].vector().vec()):
                b[j].vector().vec().aypx(0.0, a[j].vector().vec())

        return b

    def restrict_to_active_set(
        self, a: List[fenics.Function], b: List[fenics.Function]
    ) -> List[fenics.Function]:
        """Restricts a function to the active set.

        Note, that nothing will happen if the type of the optimization problem does not
        support box constraints.

        Args:
            a: The function, which shall be restricted to the active set
            b: The output function, which will contain the result and is overridden.

        Returns:
            The result of the restriction (overrides input b)
        """
        for j in range(len(self.gradient)):
            b[j].vector().vec().set(0.0)

        return b

    def compute_active_sets(self) -> None:
        """Computes the active set for problems with box constraints."""
        pass

    def update_scalar_product(self) -> None:
        """Updates the scalar product."""
        pass

    def project_to_admissible_set(
        self, a: List[fenics.Function]
    ) -> List[fenics.Function]:
        """Projects a function ``a`` onto the admissible set."""
        pass
