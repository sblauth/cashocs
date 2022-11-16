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
from typing import Dict, List, Optional, TYPE_CHECKING

from cashocs.io import managers

if TYPE_CHECKING:
    from cashocs import _forms
    from cashocs._database import database
    from cashocs._optimization import optimization_algorithms


class OutputManager:
    """Class handling all the output."""

    def __init__(
        self,
        db: database.Database,
        form_handler: _forms.FormHandler,
        temp_dict: Optional[Dict] = None,
    ) -> None:
        """Initializes self.

        Args:
            db: The database of the problem.
            form_handler: The form handler of the problem.
            temp_dict: The dict which contains the remeshing initialization.

        """
        self.db = db
        self.config = self.db.config
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

        verbose = self.config.getboolean("Output", "verbose")
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

        self.managers: List[managers.IOManager] = []
        if verbose:
            self.managers.append(managers.ConsoleManager(self.db, self.result_dir))
        if save_txt:
            self.managers.append(managers.FileManager(self.db, self.result_dir))
        if save_state or save_adjoint or save_gradient:
            self.managers.append(
                managers.XDMFFileManager(form_handler, self.db, self.result_dir)
            )
        self.output_dict = {}
        if save_results:
            result_manager = managers.ResultManager(
                self.db,
                self.result_dir,
                temp_dict,
            )
            self.output_dict = result_manager.output_dict
            self.managers.append(result_manager)

        self.managers.append(managers.MeshManager(self.db, self.result_dir))
        self.managers.append(managers.TempFileManager(self.db, self.result_dir))

    def output(self, solver: optimization_algorithms.OptimizationAlgorithm) -> None:
        """Writes the desired output to files and console.

        Args:
            solver: The optimization algorithm.

        """
        for manager in self.managers:
            manager.output(solver)

    def output_summary(
        self, solver: optimization_algorithms.OptimizationAlgorithm
    ) -> None:
        """Writes the summary to files and console.

        Args:
            solver: The optimization algorithm.

        """
        for manager in self.managers:
            manager.output_summary(solver)

    def post_process(
        self, solver: optimization_algorithms.OptimizationAlgorithm
    ) -> None:
        """Performs a post processing of the output.

        Args:
            solver: The optimization algorithm.

        """
        for manager in self.managers:
            manager.post_process(solver)
