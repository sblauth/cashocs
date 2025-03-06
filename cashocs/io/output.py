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

"""Management of cashocs output."""

from __future__ import annotations

from datetime import datetime as dt
import pathlib
from typing import TYPE_CHECKING

from cashocs.io import managers

if TYPE_CHECKING:
    from cashocs._database import database


class OutputManager:
    """Class handling all the output."""

    def __init__(self, db: database.Database) -> None:
        """Initializes self.

        Args:
            db: The database of the problem.

        """
        self.db = db
        self.config = self.db.config
        self.result_dir = self.config.get("Output", "result_dir")
        self.result_dir = self.result_dir.rstrip("/")

        self._silent = False

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
        save_mesh = self.config.getboolean("Output", "save_mesh")
        save_checkpoints = save_state or save_gradient or save_adjoint or save_mesh
        has_output = save_txt or save_results or save_checkpoints

        if not self.result_path.is_dir():
            if has_output:
                self.result_path.mkdir(parents=True, exist_ok=True)

        checkpoints_path = pathlib.Path().joinpath(self.result_path, "checkpoints")
        if not checkpoints_path.is_dir() and save_checkpoints:
            checkpoints_path.mkdir(parents=True, exist_ok=True)

        self.managers: list[managers.IOManager] = []
        self.managers.append(
            managers.ConsoleManager(self.db, self.result_dir, verbose=verbose)
        )
        if save_txt:
            self.managers.append(managers.FileManager(self.db, self.result_dir))
        if save_state or save_adjoint or save_gradient:
            self.managers.append(managers.XDMFFileManager(self.db, self.result_dir))

        result_manager = managers.ResultManager(self.db, self.result_dir)
        self.output_dict = result_manager.output_dict
        self.managers.append(result_manager)

        if save_mesh:
            self.managers.append(managers.MeshManager(self.db, self.result_dir))
        self.managers.append(managers.TempFileManager(self.db, self.result_dir))

    def output(self) -> None:
        """Writes the desired output to files and console."""
        if not self._silent:
            for manager in self.managers:
                manager.output()

    def output_summary(self) -> None:
        """Writes the summary to files and console."""
        if not self._silent:
            for manager in self.managers:
                manager.output_summary()

    def post_process(self) -> None:
        """Performs a postprocessing of the output."""
        if not self._silent:
            for manager in self.managers:
                manager.post_process()
