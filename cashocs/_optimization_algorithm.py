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

"""General blue print class for all optimization algorithms

This is the base class, on which ControlOptimizationAlgorithm and
ShapeOptimizationAlgorithm classes are based.

"""

import os
from datetime import datetime
from pathlib import Path


class OptimizationAlgorithm:
    """Abstract class representing all kinds of optimization algorithms."""

    def __init__(self, optimization_problem):
        """

        Parameters
        ----------
        optimization_problem : cashocs.optimization_problem.OptimizationProblem
        """

        self.line_search_broken = False
        self.has_curvature_info = False

        self.form_handler = optimization_problem.form_handler
        self.state_problem = optimization_problem.state_problem
        self.config = self.state_problem.config
        self.adjoint_problem = optimization_problem.adjoint_problem

        self.cost_functional = optimization_problem.reduced_cost_functional

        self.objective_value = 1.0
        self.gradient_norm_initial = 1.0
        self.relative_norm = 1.0
        self.stepsize = 1.0

        self.converged = False
        self.converged_reason = 0

        self.verbose = self.config.getboolean("Output", "verbose", fallback=True)
        self.save_txt = self.config.getboolean("Output", "save_txt", fallback=True)
        self.save_results = self.config.getboolean(
            "Output", "save_results", fallback=True
        )
        self.rtol = self.config.getfloat("OptimizationRoutine", "rtol", fallback=1e-3)
        self.atol = self.config.getfloat("OptimizationRoutine", "atol", fallback=0.0)
        self.maximum_iterations = self.config.getint(
            "OptimizationRoutine", "maximum_iterations", fallback=100
        )
        self.soft_exit = self.config.getboolean(
            "OptimizationRoutine", "soft_exit", fallback=False
        )
        self.save_pvd = self.config.getboolean("Output", "save_pvd", fallback=False)
        self.save_pvd_adjoint = self.config.getboolean(
            "Output", "save_pvd_adjoint", fallback=False
        )
        self.save_pvd_gradient = self.config.getboolean(
            "Output", "save_pvd_gradient", fallback=False
        )

        self.has_output = (
            self.save_txt
            or self.save_pvd
            or self.save_pvd_gradient
            or self.save_pvd_adjoint
            or self.save_results
        )

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

        if not os.path.isdir(self.result_dir):
            if self.has_output:
                Path(self.result_dir).mkdir(parents=True, exist_ok=True)
