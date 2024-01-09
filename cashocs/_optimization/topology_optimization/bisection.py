from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING

import fenics
import scipy.optimize

from cashocs import _exceptions
from cashocs import _utils
from cashocs._optimization.topology_optimization import topology_optimization_problem

if TYPE_CHECKING:
    from cashocs._database import database


class projection_levelset:
    def __init__(
        self,
        levelset_function: fenics.Function,
        volume_restriction: Union[float, list[float]] | None = None,
    ) -> None:
        """Initializes a class to project the levelset function for topology
           optimization.

        Args:
            levelset_function: A :py:class:`fenics.Function` which represents the
                levelset function.
            volume_restriction: A float or a list of floats that describes the
                volume restriction that the levelset function should fulfill.
                If this is ``None`` no projection is performed (default is ``None``).

        """

        self.levelset_function = levelset_function
        self.levelset_function_temp = fenics.Function(
            self.levelset_function.function_space()
        )
        self.levelset_function_temp.vector().vec().aypx(
            0.0, self.levelset_function.vector().vec()
        )
        self.levelset_function_temp.vector().apply("")

        self.dx = fenics.Measure("dx", self.levelset_function.function_space().mesh())

        self.dg0_space = fenics.FunctionSpace(
            self.levelset_function.function_space().mesh(), "DG", 0
        )
        self.indicator_omega = fenics.Function(self.dg0_space)
        _utils.interpolate_levelset_function_to_cells(
            self.levelset_function, 1.0, 0.0, self.indicator_omega
        )
        self.vol = fenics.assemble(self.indicator_omega * self.dx)

        self.volume_restriction = volume_restriction
        if self.volume_restriction is not None:
            self.volume_restriction = _utils.enlist(self.volume_restriction)
            if len(self.volume_restriction) == 1:
                self.volume_restriction = [
                    self.volume_restriction[0],
                    self.volume_restriction[0],
                ]
            if self.volume_restriction[1] < self.volume_restriction[0]:
                raise _exceptions.InputError(
                    "Bisection class",
                    "volume_restriction",
                    "The lower bound of the volume restriction is bigger than the upper "
                    "bound.",
                )

        self.max_iter_bisect = 100
        self.tolerance_bisect = 1e-4

    def evaluate(self, iterate: float, target: float):
        """A function that computes the volume of the shape that is represented by
           the levelset funtion.

        Args:
            iterate: A float that describes the movement of the levelset function.
                It is the iterate in the bisection procedure. If the volume of the
                actual shape represented by the levelset function is desired it
                should be set to zero.
            target: The target for the projection procedure. If the volume of the
                actual shape represented by the levelset function is desired it
                should be set to zero.

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
        self.vol = fenics.assemble(self.indicator_omega * self.dx)
        return self.vol - target

    def project(self):
        """Projects the levelset function onto the admissible set by a bisection
        approach."""

        if (
            abs(
                self.levelset_function.vector().max()
                - self.levelset_function.vector().min()
            )
            <= self.tolerance_bisect
            or self.volume_restriction is None
        ):
            return

        _utils.interpolate_levelset_function_to_cells(
            self.levelset_function, 1.0, 0.0, self.indicator_omega
        )
        self.vol = fenics.assemble(self.indicator_omega * self.dx)

        if self.vol < self.volume_restriction[0]:
            lower = -abs(self.levelset_function.vector().max())
            upper = 0.0
            xtol = self.tolerance_bisect
            while True:
                iterate = scipy.optimize.bisect(
                    self.evaluate,
                    lower,
                    upper,
                    xtol=xtol,
                    maxiter=self.max_iter_bisect,
                    args=self.volume_restriction[0],
                )
                lower = iterate - self.tolerance_bisect
                upper = iterate + self.tolerance_bisect
                xtol *= 0.1
                if (
                    self.evaluate(iterate, self.volume_restriction[0])
                    < self.tolerance_bisect
                ):
                    break
            self.levelset_function.vector().vec().aypx(
                0.0, self.levelset_function.vector().vec() + iterate
            )
            self.levelset_function.vector().apply("")
        elif self.vol > self.volume_restriction[1]:
            lower = 0.0
            upper = abs(self.levelset_function.vector().min())
            xtol = self.tolerance_bisect
            while True:
                iterate = scipy.optimize.bisect(
                    self.evaluate,
                    lower,
                    upper,
                    xtol=xtol,
                    maxiter=self.max_iter_bisect,
                    args=self.volume_restriction[1],
                )
                lower = iterate - self.tolerance_bisect
                upper = iterate + self.tolerance_bisect
                xtol *= 0.1
                if (
                    self.evaluate(iterate, self.volume_restriction[1])
                    < self.tolerance_bisect
                ):
                    break
            self.levelset_function.vector().vec().aypx(
                0.0, self.levelset_function.vector().vec() + iterate
            )
            self.levelset_function.vector().apply("")
        else:
            return
