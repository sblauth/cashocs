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

"""Blueprints for shape optimization algorithms.

"""

from __future__ import annotations

import abc
import subprocess
from typing import TYPE_CHECKING

import fenics

from .._interfaces import OptimizationAlgorithm
from ..utils import write_out_mesh


if TYPE_CHECKING:
    from .shape_optimization_problem import ShapeOptimizationProblem


class ShapeOptimizationAlgorithm(OptimizationAlgorithm):
    """Blueprint for a solution algorithm for shape optimization problems"""

    def __init__(self, optimization_problem: ShapeOptimizationProblem) -> None:
        """Parent class for the optimization methods implemented in cashocs.optimization.methods

        Parameters
        ----------
        optimization_problem : ShapeOptimizationProblem
                the optimization problem
        """

        super().__init__(optimization_problem)

        self.line_search_broken = False
        self.requires_remeshing = False
        self.remeshing_its = False
        self.has_curvature_info = False

        self.mesh_handler = optimization_problem.mesh_handler

        self.shape_gradient_problem = optimization_problem.shape_gradient_problem
        self.gradient = self.shape_gradient_problem.gradient
        self.search_direction = fenics.Function(self.form_handler.deformation_space)

        self.temp_dict = optimization_problem.temp_dict
        if self.mesh_handler.do_remesh:
            if not self.config.getboolean("Debug", "remeshing", fallback=False):
                self.temp_dir = optimization_problem.temp_dir
        if self.config.getboolean("Mesh", "remesh", fallback=False):
            self.iteration = self.temp_dict["OptimizationRoutine"].get(
                "iteration_counter", 0
            )
        else:
            self.iteration = 0

        if self.mesh_handler.do_remesh:
            self.pvd_prefix = f"remesh_{self.temp_dict.get('remesh_counter', 0):d}_"

        if self.save_pvd:
            self.state_pvd_list = []
            for i in range(self.form_handler.state_dim):
                self.state_pvd_list.append(
                    self._generate_pvd_file(
                        self.form_handler.state_spaces[i],
                        f"state_{i:d}",
                        self.pvd_prefix,
                    )
                )

        if self.save_pvd_adjoint:
            self.adjoint_pvd_list = []
            for i in range(self.form_handler.state_dim):
                self.adjoint_pvd_list.append(
                    self._generate_pvd_file(
                        self.form_handler.state_spaces[i],
                        f"adjoint_{i:d}",
                        self.pvd_prefix,
                    )
                )

        if self.save_pvd_gradient:
            self.shape_gradient_pvd_file = self._generate_pvd_file(
                self.gradient.function_space(), "shape_gradient", self.pvd_prefix
            )

    @abc.abstractmethod
    def run(self) -> None:
        """Blueprint run method, overriden by the actual solution algorithms

        Returns
        -------
        None
        """

        pass

    def print_results(self) -> None:
        """Prints the current state of the optimization algorithm to the console.

        Returns
        -------
        None
        """

        super().print_results()

        if self.save_pvd_gradient:
            self.shape_gradient_pvd_file << self.gradient, float(self.iteration)

    def finalize(self) -> None:
        """Saves the history of the optimization algorithm

        Returns
        -------
        None
        """

        super().finalize()

        if self.mesh_handler.save_optimized_mesh:
            write_out_mesh(
                self.mesh_handler.mesh,
                self.mesh_handler.gmsh_file,
                f"{self.result_dir}/optimized_mesh.msh",
            )

        if self.mesh_handler.do_remesh:
            if not self.config.getboolean("Debug", "remeshing", fallback=False):
                subprocess.run(["rm", "-r", self.temp_dir], check=True)
                subprocess.run(
                    ["rm", "-r", self.mesh_handler.remesh_directory], check=True
                )
