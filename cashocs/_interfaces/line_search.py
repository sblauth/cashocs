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

from __future__ import annotations

import abc
import weakref
from typing import TYPE_CHECKING, List, Union, Optional

import fenics

from ..utils import _optimization_algorithm_configuration


if TYPE_CHECKING:
    from .optimization_algorithm import OptimizationAlgorithm


class LineSearch(abc.ABC):
    def __init__(self, optimization_algorithm: OptimizationAlgorithm) -> None:
        """Initializes the line search object

        Parameters
        ----------
        optimization_algorithm : OptimizationAlgorithm
            The corresponding optimization algorihm
        """

        self.ref_algo = weakref.ref(optimization_algorithm)
        self.config = optimization_algorithm.config
        self.form_handler = optimization_algorithm.form_handler

        self.stepsize = self.config.getfloat(
            "OptimizationRoutine", "initial_stepsize", fallback=1.0
        )
        self.epsilon_armijo = self.config.getfloat(
            "OptimizationRoutine", "epsilon_armijo", fallback=1e-4
        )
        self.beta_armijo = self.config.getfloat(
            "OptimizationRoutine", "beta_armijo", fallback=2.0
        )
        self.armijo_stepsize_initial = self.stepsize

        self.cost_functional = optimization_algorithm.cost_functional

        algorithm = _optimization_algorithm_configuration(self.config)
        self.is_newton_like = algorithm == "lbfgs"
        self.is_newton = algorithm == "newton"
        self.is_steepest_descent = algorithm == "gradient_descent"
        if self.is_newton:
            self.stepsize = 1.0

    @abc.abstractmethod
    def decrease_measure(
        self, search_direction: Optional[fenics.Function] = None
    ) -> float:
        pass

    @abc.abstractmethod
    def search(
        self,
        search_direction: Union[fenics.Function, List[fenics.Function]],
        has_curvature_info: bool,
    ) -> None:
        pass
