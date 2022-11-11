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

"""Management for weak forms."""

from __future__ import annotations

import abc
from typing import Callable, List, TYPE_CHECKING, Union

import fenics
import ufl

from cashocs import _utils
from cashocs._database import database
from cashocs._optimization import cost_functional

if TYPE_CHECKING:
    from cashocs import _typing


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


def _hook() -> None:
    return None


class FormHandler(abc.ABC):
    """Parent class for UFL form manipulation.

    This is subclassed by specific form handlers for either
    optimal control or shape optimization. The base class is
    used to determine common objects and to derive the UFL forms
    for the state and adjoint systems.
    """

    def __init__(
        self, optimization_problem: _typing.OptimizationProblem, db: database.Database
    ) -> None:
        """Initializes self.

        Args:
            optimization_problem: The corresponding optimization problem.
            db: The database for the problem.

        """
        self.optimization_problem = optimization_problem
        self.db = db

        self.config = self.db.config
        self.bcs_list: List[List[fenics.DirichletBC]] = optimization_problem.bcs_list

        self.is_shape_problem: bool = optimization_problem.is_shape_problem
        self.is_control_problem = optimization_problem.is_control_problem

        self.control_dim: int = 1
        self.cost_functional_list: List[
            _typing.CostFunctional
        ] = optimization_problem.cost_functional_list
        self.state_forms = optimization_problem.state_forms

        self.lagrangian: cost_functional.Lagrangian = cost_functional.Lagrangian(
            self.cost_functional_list, self.state_forms
        )
        self.cost_functional_shift: float = 0.0

        self.dx: fenics.Measure = self.db.geometry_db.dx

        self.state_is_linear = self.config.getboolean("StateSystem", "is_linear")
        self.state_is_picard = self.config.getboolean("StateSystem", "picard_iteration")
        self.opt_algo: str = _utils.optimization_algorithm_configuration(self.config)

        self.gradient: List[fenics.Function] = []
        self.control_spaces: List[fenics.FunctionSpace] = []

        self.pre_hook: Callable[..., None] = _hook
        self.post_hook: Callable[..., None] = _hook

        self.state_eq_forms: List[ufl.Form] = []
        self.linear_state_eq_forms: List[ufl.Form] = []
        self.state_eq_forms_lhs: List[ufl.Form] = []
        self.state_eq_forms_rhs: List[ufl.Form] = []

        self.adjoint_eq_forms: List[ufl.Form] = []
        self.linear_adjoint_eq_forms: List[ufl.Form] = []
        self.adjoint_eq_lhs: List[ufl.Form] = []
        self.adjoint_eq_rhs: List[ufl.Form] = []

        self._compute_state_equations()
        self._compute_adjoint_equations()
        self.bcs_list_ad = self._compute_adjoint_boundary_conditions()

    def _compute_state_equations(self) -> None:
        """Calculates the weak form of the state equation for the use with fenics."""
        self.state_eq_forms = [
            fenics.derivative(
                self.state_forms[i],
                self.db.function_db.adjoints[i],
                self.db.function_db.test_functions_state[i],
            )
            for i in range(self.db.parameter_db.state_dim)
        ]

        if self.state_is_linear:
            self.linear_state_eq_forms = [
                ufl.replace(
                    self.state_eq_forms[i],
                    {
                        self.db.function_db.states[
                            i
                        ]: self.db.function_db.trial_functions_state[i]
                    },
                )
                for i in range(self.db.parameter_db.state_dim)
            ]

            (
                self.state_eq_forms_lhs,
                self.state_eq_forms_rhs,
            ) = _utils.split_linear_forms(self.linear_state_eq_forms)

    def _compute_adjoint_boundary_conditions(self) -> List[List[fenics.DirichletBC]]:
        """Computes the boundary conditions for the adjoint systems."""
        if self.db.parameter_db.state_adjoint_equal_spaces:
            bcs_list_ad = [
                [fenics.DirichletBC(bc) for bc in self.bcs_list[i]]
                for i in range(self.db.parameter_db.state_dim)
            ]
            for i in range(self.db.parameter_db.state_dim):
                for bc in bcs_list_ad[i]:
                    bc.homogenize()
        else:

            bcs_list_ad = [
                [1] * len(self.bcs_list[i])
                for i in range(self.db.parameter_db.state_dim)
            ]

            for i in range(self.db.parameter_db.state_dim):
                for j, bc in enumerate(self.bcs_list[i]):
                    idx = bc.function_space().id()
                    subdx: List[int] = []
                    _get_subdx(self.db.function_db.state_spaces[i], idx, ls=subdx)
                    adjoint_space = self.db.function_db.adjoint_spaces[i]
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
                        bcs_list_ad[i][j] = fenics.DirichletBC(
                            adjoint_space,
                            bdry_value,
                            bc.domain_args[0],
                            bc.domain_args[1],
                        )
                    except AttributeError:
                        bcs_list_ad[i][j] = fenics.DirichletBC(
                            adjoint_space, bdry_value, bc.sub_domain
                        )

        return bcs_list_ad

    def _compute_adjoint_equations(self) -> None:
        """Calculates the weak form of the adjoint equation for use with fenics."""
        self.adjoint_eq_forms = [
            self.lagrangian.derivative(
                self.db.function_db.states[i],
                self.db.function_db.test_functions_adjoint[i],
            )
            for i in range(self.db.parameter_db.state_dim)
        ]

        self.linear_adjoint_eq_forms = [
            ufl.replace(
                self.adjoint_eq_forms[i],
                {
                    self.db.function_db.adjoints[
                        i
                    ]: self.db.function_db.trial_functions_adjoint[i]
                },
            )
            for i in range(self.db.parameter_db.state_dim)
        ]

        self.adjoint_eq_lhs, self.adjoint_eq_rhs = _utils.split_linear_forms(
            self.linear_adjoint_eq_forms
        )

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
                b[j].vector().apply("")

        return b

    # pylint: disable=unused-argument
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
            b[j].vector().apply("")

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
