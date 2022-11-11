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

"""Shape optimization problem."""

from __future__ import annotations

from typing import Callable, Optional


class Callback:
    """Manages user-defined callbacks."""

    def __init__(self) -> None:
        """Initializes the callbacks."""
        self.pre_callback: Optional[Callable] = None
        self.post_callback: Optional[Callable] = None

    def call_pre(self) -> None:
        """Calls the callback intended before solving the state system."""
        if self.pre_callback is not None:
            self.pre_callback()  # pylint: disable=not-callable

    def call_post(self) -> None:
        """Calls the callback intended after computing the gradient."""
        if self.post_callback is not None:
            self.post_callback()  # pylint: disable=not-callable
