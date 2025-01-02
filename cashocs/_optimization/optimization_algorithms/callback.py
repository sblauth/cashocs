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

"""Shape optimization problem."""

from __future__ import annotations

from typing import Callable, TYPE_CHECKING

from cashocs import _exceptions
from cashocs import _utils

if TYPE_CHECKING:
    from cashocs import _typing


class Callback:
    """Manages user-defined callbacks."""

    def __init__(self) -> None:
        """Initializes the callbacks."""
        self.pre_callback: (
            Callable[[], None] | Callable[[_typing.OptimizationProblem], None] | None
        ) = None
        self.post_callback: (
            Callable[[], None] | Callable[[_typing.OptimizationProblem], None] | None
        ) = None
        self.problem: _typing.OptimizationProblem | None = None

    def call_pre(self) -> None:
        """Calls the callback intended before solving the state system."""
        if self.pre_callback is not None:
            num_args = _utils.number_of_arguments(self.pre_callback)
            if num_args == 0:
                self.pre_callback()  # type: ignore
            elif num_args == 1:
                self.pre_callback(self.problem)  # type: ignore
            else:
                raise _exceptions.InputError(
                    "OptimizationProblem",
                    "pre_callback",
                    "The number of arguments for the pre_callback function can either "
                    "be one or zero.",
                )

    def call_post(self) -> None:
        """Calls the callback intended after computing the gradient."""
        if self.post_callback is not None:
            num_args = _utils.number_of_arguments(self.post_callback)
            if num_args == 0:
                self.post_callback()  # type: ignore
            elif num_args == 1:
                self.post_callback(self.problem)  # type: ignore
            else:
                raise _exceptions.InputError(
                    "OptimizationProblem",
                    "post_callback",
                    "The number of arguments for the pre_callback function can either "
                    "be one or zero.",
                )
