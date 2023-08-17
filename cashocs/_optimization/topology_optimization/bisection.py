from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING

import fenics
import scipy.optimize

from cashocs import _utils
from cashocs._optimization.topology_optimization import topology_optimization_algorithm

if TYPE_CHECKING:
    from cashocs._database import database

class projection_levelset:

    def __init__(
        self,
        algorithm: topology_optimization_algorithm.TopologyOptimizationAlgorithm,
        db: database.Database,
    ) -> None:
        """Initializes self.
        Args:
            algorithm: The corresponding optimization algorithm.
            db: The database of the problem.
        """

        self.levelset_function: fenics.Function = algorithm.levelset_function
        self.levelset_function_temp = fenics.Function(
            self.levelset_function.function_space()
        )
        self.levelset_function_temp.vector().vec().aypx(
            0.0, self.levelset_function.vector().vec()
        )
        self.levelset_function_temp.vector().apply("")

        self.dx = algorithm.dx
        self.algorithm = algorithm
        self.update_levelset = algorithm.update_levelset

        self.indicator_omega = fenics.Function(self.algorithm.dg0_space)
        _utils.interpolate_levelset_function_to_cells(self.levelset_function, 1.0, 0.0, self.indicator_omega)
        self.vol = fenics.assemble(self.indicator_omega * self.dx)

        self.volume_restriction = algorithm.volume_restriction
        if self.volume_restriction is not None:
            if len(self.volume_restriction) == 1:
                self.volume_restriction = [self.volume_restriction[0], self.volume_restriction[0]]

    def evaluate(self, iterate, target):
        self.levelset_function_temp.vector()[:] = self.levelset_function.vector()[:] + iterate
        self.levelset_function_temp.vector().apply("")
        _utils.interpolate_levelset_function_to_cells(self.levelset_function_temp, 1.0, 0.0, self.indicator_omega)
        self.vol = fenics.assemble(self.indicator_omega * self.dx)
        return (self.vol - target)


    def project(self):
        if self.volume_restriction is None or self.algorithm.iteration == 0:
            return

        _utils.interpolate_levelset_function_to_cells(self.levelset_function, 1.0, 0.0, self.indicator_omega)
        self.vol = fenics.assemble(self.indicator_omega * self.dx)

        if self.vol < self.volume_restriction[0]:
            max_levelset = abs(self.levelset_function.vector().max())
            iterate = scipy.optimize.bisect(self.evaluate, -max_levelset, 0., xtol=1e-4,
                                            args=self.volume_restriction[0])
            self.levelset_function.vector()[:] = self.levelset_function.vector()[:] + iterate
            self.update_levelset()
        elif self.vol > self.volume_restriction[1]:
            min_levelset = abs(self.levelset_function.vector().min())
            iterate = scipy.optimize.bisect(self.evaluate, 0., min_levelset, xtol=1e-4,
                                            args=self.volume_restriction[1])
            self.levelset_function.vector()[:] = self.levelset_function.vector()[:] + iterate
            self.update_levelset()
        else:
            return
