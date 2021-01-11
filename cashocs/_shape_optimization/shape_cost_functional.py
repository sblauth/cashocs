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

"""Implementation of a reduced shape cost functional

"""

import fenics



class ReducedShapeCostFunctional:
	"""Reduced cost functional for a shape optimization problem

	"""

	def __init__(self, form_handler, state_problem):
		"""Initializes the reduced cost functional

		Parameters
		----------
		form_handler : cashocs._forms.ShapeFormHandler
			the ControlFormHandler object for the optimization problem
		state_problem : cashocs._pde_problems.StateProblem
			the StateProblem object corresponding to the state system
		"""

		self.form_handler = form_handler
		self.state_problem = state_problem
		self.regularization = self.form_handler.regularization



	def evaluate(self):
		"""Evaluates the reduced cost functional

		Returns
		-------
		float
			the value of the reduced cost functional at the current control

		"""

		self.state_problem.solve()
		# self.regularization.update_geometric_quantities()

		return fenics.assemble(self.form_handler.cost_functional_form) + self.regularization.compute_objective()
