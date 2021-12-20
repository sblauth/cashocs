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
from typing import TYPE_CHECKING, List, Union

import fenics


if TYPE_CHECKING:
    from .._forms import FormHandler


class PDEProblem(abc.ABC):
    def __init__(self, form_handler: FormHandler) -> None:
        """
        Parameters
        ----------
        form_handler: FormHandler
            The form handler for the problem
        """

        self.form_handler = form_handler
        self.config = form_handler.config

        self.has_solution = False

    @abc.abstractmethod
    def solve(self) -> Union[fenics.Function, List[fenics.Function]]:
        """

        Returns
        -------
        fenics.Function or list[fenics.Function]
            The solution of the PDE
        """

        pass
