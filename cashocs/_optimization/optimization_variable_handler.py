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

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, List, Optional

import fenics


if TYPE_CHECKING:
    from .optimization_problem import OptimizationProblem


class OptimizationVariableHandler(abc.ABC):
    def __init__(self, optimization_problem: OptimizationProblem) -> None:

        self.gradient = optimization_problem.gradient
        self.form_handler = optimization_problem.form_handler

    @abc.abstractmethod
    def compute_decrease_measure(
        self, search_direction: Optional[List[fenics.Function]] = None
    ) -> float:

        pass

    @abc.abstractmethod
    def revert_variable_update(self) -> None:

        pass

    @abc.abstractmethod
    def update_optimization_variables(
        self, search_direction, stepsize: float, beta: float
    ) -> float:

        pass

    @abc.abstractmethod
    def compute_gradient_norm(self) -> float:

        pass

    @abc.abstractmethod
    def compute_a_priori_decreases(
        self, search_direction: List[fenics.Function], stepsize: float
    ) -> float:

        pass

    @abc.abstractmethod
    def requires_remeshing(self) -> bool:
        pass

    @abc.abstractmethod
    def project_ncg_search_direction(
        self, search_direction: List[fenics.Function]
    ) -> None:

        pass
