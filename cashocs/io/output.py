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

"""Management of cashocs output."""

from __future__ import annotations

from datetime import datetime as dt
import pathlib
from typing import TYPE_CHECKING

from cashocs.io import managers

if TYPE_CHECKING:
    from cashocs._database import database
    from cashocs._optimization import optimization_algorithms
    from cashocs._optimization import optimization_problem as op


class OutputManager:
    """Class handling all the output."""

    def __init__(
        self, optimization_problem: op.OptimizationProblem, db: database.Database
    ) -> None:
        """Initializes self.

        Args:
            optimization_problem: The corresponding optimization problem.
            db: The database of the problem.

        """
        self.db = db
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

        self.result_path = pathlib.Path(self.result_dir)

        save_txt = self.config.getboolean("Output", "save_txt")
        save_results = self.config.getboolean("Output", "save_results")
        save_state = self.config.getboolean("Output", "save_state")
        save_adjoint = self.config.getboolean("Output", "save_adjoint")
        save_gradient = self.config.getboolean("Output", "save_gradient")
        has_output = (
            save_txt or save_results or save_state or save_gradient or save_adjoint
        )

        if not self.result_path.is_dir():
            if has_output:
                self.result_path.mkdir(parents=True, exist_ok=True)

        self.history_manager = managers.HistoryManager(self.db, self.result_dir)
        self.xdmf_file_manager = managers.XDMFFileManager(
            optimization_problem, self.db, self.result_dir
        )

        self.result_manager = managers.ResultManager(
            self.db,
            self.result_dir,
            optimization_problem.has_cashocs_remesh_flag,
            optimization_problem.temp_dict,
        )
        self.mesh_manager = managers.MeshManager(self.db, self.result_dir)
        self.temp_file_manager = managers.TempFileManager(self.db)

    def output(self, solver: optimization_algorithms.OptimizationAlgorithm) -> None:
        """Writes the desired output to files and console.

        Args:
            solver: The optimization algorithm.

        """
        self.history_manager.print_to_console(solver)
        self.history_manager.print_to_file(solver)

        self.xdmf_file_manager.save_to_file(solver)

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
