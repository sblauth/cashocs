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

"""Projection class for the level-set function."""

from __future__ import annotations

from typing import TYPE_CHECKING

import fenics
import scipy.optimize

from cashocs import _exceptions
from cashocs import _utils

try:
    import ufl_legacy as ufl
except ImportError:
    import ufl

if TYPE_CHECKING:
    from cashocs._database import database


class LevelSetVolumeProjector:
    """Class for a projection of the level-set function."""

    def __init__(
        self,
        levelset_function: fenics.Function,
        volume_restriction: float | tuple[float, float] | None = None,
        db: database.Database | None = None,
    ) -> None:
        """Initializes a class to project the levelset function.

        Args:
            levelset_function: A :py:class:`fenics.Function` which represents the
                levelset function.
            volume_restriction: A float or a tuple of floats that describes the
                volume restriction that the levelset function should fulfill.
                If this is ``None`` no projection is performed (default is ``None``).
                For a float we have an equality constraint for the volume and for a
                tuple an inequlity constraint for the volume.
            db: The database of the problem. The maximum number of the performed
                bisection iterations (default=100) and the tolerance for the bisection
                procedure (default=1e-4) are specified.

        """
        self.levelset_function = levelset_function
        self.levelset_function_temp = fenics.Function(
            self.levelset_function.function_space()
        )
        self.levelset_function_temp.vector().vec().aypx(
            0.0, self.levelset_function.vector().vec()
        )
        self.levelset_function_temp.vector().apply("")

        self.dx = ufl.Measure("dx", self.levelset_function.function_space().mesh())

        self.dg0_space = fenics.FunctionSpace(
            self.levelset_function.function_space().mesh(), "DG", 0
        )
        self.indicator_omega = fenics.Function(self.dg0_space)
        _utils.interpolate_levelset_function_to_cells(
            self.levelset_function, 1.0, 0.0, self.indicator_omega
        )
        self.vol = fenics.assemble(self.indicator_omega * self.dx)

        if volume_restriction is not None:
            if isinstance(volume_restriction, float):
                self.volume_restriction: tuple | None = tuple(
                    [volume_restriction, volume_restriction]
                )
            else:
                self.volume_restriction = volume_restriction
            if self.volume_restriction[1] < self.volume_restriction[0]:
                raise _exceptions.InputError(
                    "Bisection class",
                    "volume_restriction",
                    "The lower bound of the volume restriction"
                    " is bigger than the upper bound.",
                )
        else:
            self.volume_restriction = None

        if db is not None:
            self.max_iter_bisect = db.config.getint(
                "TopologyOptimization", "max_iter_bisection"
            )
            self.tol_bisect = db.config.getfloat(
                "TopologyOptimization", "tol_bisection"
            )

    def evaluate(self, iterate: float, target: float) -> float:
        """Computes the volume of a shape that is given by the level-set function.

        The function computes the volume of a shape that is represented by the level-
        set function that is shifted by iterate. The function returns the difference
        of the computed volume and target. This function is used for a bisection
        procedure that shifts the level-set function so that the volume of the shape
        reaches target.

        Args:
            iterate: A float that describes the shift parameter for the levelset
                function. It is the iterate in the bisection procedure. If the volume
                of the actual shape represented by the levelset function is desired it
                should be set to zero.
            target: The target / desired volume for the projection procedure. If the
                volume of the actual shape represented by the levelset function is
                desired it should be set to zero.

        Returns:
            The volume of the shape represented by the levelset function moved by
            iterate. It returns the volume difference to target.

        """
        self.levelset_function_temp.vector().vec().aypx(
            0.0, self.levelset_function.vector().vec() + iterate
        )
        self.levelset_function_temp.vector().apply("")
        _utils.interpolate_levelset_function_to_cells(
            self.levelset_function_temp, 1.0, 0.0, self.indicator_omega
        )
        vol = fenics.assemble(self.indicator_omega * self.dx)
        return float(vol - target)

    def project(self) -> None:
        """Projects the level-set function.

        This function shifts the level-set function by a constant such that the
        corresponding shape fulfills a volume constraint. This constant is computed
        by a bisection approach.

        """
        diff_levelset = abs(
            self.levelset_function.vector().max()
            - self.levelset_function.vector().min()
        )
        if diff_levelset <= self.tol_bisect or self.volume_restriction is None:
            return None

        _utils.interpolate_levelset_function_to_cells(
            self.levelset_function, 1.0, 0.0, self.indicator_omega
        )
        self.vol = fenics.assemble(self.indicator_omega * self.dx)

        if self.vol < self.volume_restriction[0]:
            interval_low = -abs(self.levelset_function.vector().max())
            interval_high = 0.0
            target = self.volume_restriction[0]
        elif self.vol > self.volume_restriction[1]:
            interval_low = 0.0
            interval_high = abs(self.levelset_function.vector().min())
            target = self.volume_restriction[1]
        else:
            return None

        xtol = self.tol_bisect
        while True:
            iterate = scipy.optimize.bisect(
                self.evaluate,
                interval_low,
                interval_high,
                xtol=xtol,
                maxiter=self.max_iter_bisect,
                args=target,
            )
            interval_low = iterate - self.tol_bisect
            interval_high = iterate + self.tol_bisect
            xtol *= 0.1
            if self.evaluate(iterate, target) < self.tol_bisect:
                break

        self.levelset_function.vector().vec().aypx(
            0.0, self.levelset_function.vector().vec() + iterate
        )
        self.levelset_function.vector().apply("")

        return None
