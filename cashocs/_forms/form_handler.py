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

"""Management for weak forms."""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import fenics

from cashocs import _utils
from cashocs._database import database

try:
    import ufl_legacy as ufl
except ImportError:
    import ufl

if TYPE_CHECKING:
    from cashocs import _typing
    from cashocs import io
    from cashocs._optimization import cost_functional as cf


def _get_subdx(
    function_space: fenics.FunctionSpace, index: int, ls: list
) -> None | list[int]:
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

        self.config: io.Config = self.db.config
        self.cost_functional_shift: float = 0.0
        self.lagrangian: cf.Lagrangian = self.db.form_db.lagrangian

        self.dx: ufl.Measure = self.db.geometry_db.dx

        self.opt_algo: str = _utils.optimization_algorithm_configuration(self.config)

    @abc.abstractmethod
    def scalar_product(
        self, a: list[fenics.Function], b: list[fenics.Function]
    ) -> float:
        """Computes the scalar product between a and b.

        Args:
            a: The first argument.
            b: The second argument.

        Returns:
            The scalar product of a and b.

        """
        pass

    @abc.abstractmethod
    def update_scalar_product(self) -> None:
        """Updates the scalar product."""
        pass

    @abc.abstractmethod
    def apply_shape_bcs(self, function: fenics.Function) -> None:
        """Applies the geometric boundary conditions / constraints to a function.

        Args:
            function: The function onto which the geometric constraints are imposed.
                Must be a vector CG1 function.

        """
        pass
