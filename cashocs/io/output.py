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

"""Module managing the output of cashocs."""

from __future__ import annotations

from datetime import datetime as dt
import os
import pathlib
from typing import TYPE_CHECKING

from cashocs.io import managers

if TYPE_CHECKING:
    from cashocs._optimization import optimization_algorithms
    from cashocs._optimization import optimization_problem as op


class OutputManager:
    """Class handling all the output."""

    def __init__(self, optimization_problem: op.OptimizationProblem) -> None:
        """Initializes self.

        Args:
            optimization_problem: The corresponding optimization problem.

        """
        self.config = optimization_problem.config
        self.result_dir = self.config.get("Output", "result_dir")
        self.result_dir = self.result_dir.rstrip("/")

        self.time_suffix = self.config.getboolean("Output", "time_suffix")
        if self.time_suffix:
            dt_current_time = dt.now()
            self.suffix = (
                f"{dt_current_time.year}_{dt_current_time.month}_"
                f"{dt_current_time.day}_{dt_current_time.hour}_"
                f"{dt_current_time.minute}_{dt_current_time.second}"
            )
            self.result_dir = f"{self.result_dir}_{self.suffix}"

        save_txt = self.config.getboolean("Output", "save_txt")
        save_results = self.config.getboolean("Output", "save_results")
        save_pvd = self.config.getboolean("Output", "save_pvd")
        save_pvd_adjoint = self.config.getboolean("Output", "save_pvd_adjoint")
        save_pvd_gradient = self.config.getboolean("Output", "save_pvd_gradient")
        has_output = (
            save_txt
            or save_results
            or save_pvd
            or save_pvd_gradient
            or save_pvd_adjoint
        )

        if not os.path.isdir(self.result_dir):
            if has_output:
                pathlib.Path(self.result_dir).mkdir(parents=True, exist_ok=True)

        self.history_manager = managers.HistoryManager(
            optimization_problem, self.result_dir
        )
        self.pvd_file_manager = managers.PVDFileManager(
            optimization_problem, self.result_dir
        )
        self.result_manager = managers.ResultManager(
            optimization_problem, self.result_dir
        )
        self.mesh_manager = managers.MeshManager(optimization_problem, self.result_dir)
        self.temp_file_manager = managers.TempFileManager(optimization_problem)

    def output(self, solver: optimization_algorithms.OptimizationAlgorithm) -> None:
        """Writes the desired output to files and console.

        Args:
            solver: The optimization algorithm.

        """
        self.history_manager.print_to_console(solver)
        self.history_manager.print_to_file(solver)

        self.pvd_file_manager.save_to_file(solver)

        self.result_manager.save_to_dict(solver)

    def output_summary(
        self, solver: optimization_algorithms.OptimizationAlgorithm
    ) -> None:
        """Writes the summary to files and console.

        Args:
            solver: The optimization algorithm.

        """
        self.history_manager.print_console_summary(solver)
        self.history_manager.print_file_summary(solver)

    def post_process(
        self, solver: optimization_algorithms.OptimizationAlgorithm
    ) -> None:
        """Performs a post processing of the output.

        Args:
            solver: The optimization algorithm.

        """
        self.result_manager.save_to_json(solver)
        self.mesh_manager.save_optimized_mesh(solver)
        self.temp_file_manager.clear_temp_files(solver)

    def set_remesh(self, remesh_counter: int) -> None:
        """Sets the remesh prefix for pvd files.

        Args:
            remesh_counter: Number of times remeshing has been performed.

        """
        self.pvd_file_manager.set_remesh(remesh_counter)
