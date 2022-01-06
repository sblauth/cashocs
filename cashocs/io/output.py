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

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from .managers import (
    PVDFileManager,
    TempFileManager,
    ResultManager,
    MeshManager,
    HistoryManager,
)


if TYPE_CHECKING:
    from .._optimization.optimization_problem import OptimizationProblem
    from .._optimization.optimization_algorithms import OptimizationAlgorithm


class OutputManager:
    def __init__(self, optimization_problem: OptimizationProblem) -> None:

        self.config = optimization_problem.config
        self.result_dir = self.config.get("Output", "result_dir", fallback="./results")

        self.time_suffix = self.config.getboolean(
            "Output", "time_suffix", fallback=False
        )
        if self.time_suffix:
            dt = datetime.now()
            self.suffix = (
                f"{dt.year}_{dt.month}_{dt.day}_{dt.hour}_{dt.minute}_{dt.second}"
            )
            if self.result_dir[-1] == "/":
                self.result_dir = f"{self.result_dir[:-1]}_{self.suffix}"
            else:
                self.result_dir = f"{self.result_dir}_{self.suffix}"

        save_txt = self.config.getboolean("Output", "save_txt", fallback=True)
        save_results = self.config.getboolean("Output", "save_results", fallback=True)
        save_pvd = self.config.getboolean("Output", "save_pvd", fallback=False)
        save_pvd_adjoint = self.config.getboolean(
            "Output", "save_pvd_adjoint", fallback=False
        )
        save_pvd_gradient = self.config.getboolean(
            "Output", "save_pvd_gradient", fallback=False
        )
        has_output = (
            save_txt
            or save_results
            or save_pvd
            or save_pvd_gradient
            or save_pvd_adjoint
        )

        if not os.path.isdir(self.result_dir):
            if has_output:
                Path(self.result_dir).mkdir(parents=True, exist_ok=True)

        self.history_manager = HistoryManager(optimization_problem, self.result_dir)
        self.pvd_file_manager = PVDFileManager(optimization_problem, self.result_dir)
        self.result_manager = ResultManager(optimization_problem, self.result_dir)
        self.mesh_manager = MeshManager(optimization_problem, self.result_dir)
        self.temp_file_manager = TempFileManager(optimization_problem)

    def output(self, solver: OptimizationAlgorithm) -> None:

        self.history_manager.print_to_console(solver)
        self.history_manager.print_to_file(solver)

        self.pvd_file_manager.save_to_file(solver)

        self.result_manager.save_to_dict(solver)

    def output_summary(self, solver: OptimizationAlgorithm) -> None:

        self.history_manager.print_console_summary(solver)
        self.history_manager.print_file_summary(solver)

        self.result_manager.save_to_json(solver)

        self.mesh_manager.save_optimized_mesh(solver)

        self.temp_file_manager.clear_temp_files(solver)

    def set_remesh(self, remesh_counter: int) -> None:
        self.pvd_file_manager.set_remesh(remesh_counter)
