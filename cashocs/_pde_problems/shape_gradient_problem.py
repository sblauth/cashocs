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

"""Abstract implementation of a shape gradient problem.

This class uses the linear elasticity equations to project the
shape derivative to the shape gradient with a Riesz projection.
"""

import fenics
from petsc4py import PETSc

from ..utils import _setup_petsc_options, _solve_linear_problem



class ShapeGradientProblem:
	"""Riesz problem for the computation of the shape gradient.

	"""

	def __init__(self, form_handler, state_problem, adjoint_problem):
		"""Initialize the ShapeGradientProblem.

		Parameters
		----------
		form_handler : cashocs._forms.ShapeFormHandler
			The ShapeFormHandler object corresponding to the shape optimization problem.
		state_problem : cashocs._pde_problems.StateProblem
			The corresponding state problem.
		adjoint_problem : cashocs._pde_problems.AdjointProblem
			The corresponding adjoint problem.
		"""

		self.form_handler = form_handler
		self.state_problem = state_problem
		self.adjoint_problem = adjoint_problem

		self.gradient = fenics.Function(self.form_handler.deformation_space)
		self.gradient_norm_squared = 1.0

		self.config = self.form_handler.config

		# Generate the Krylov solver for the shape gradient problem
		self.ksp = PETSc.KSP().create()
		self.ksp_options = [
			['ksp_type', 'cg'],
			['pc_type', 'hypre'],
			['pc_hypre_type', 'boomeramg'],
			['pc_hypre_boomeramg_strong_threshold', 0.7],
			['ksp_rtol', 1e-20],
			['ksp_atol', 1e-50],
			['ksp_max_it', 1000]
		]
		_setup_petsc_options([self.ksp], [self.ksp_options])

		self.has_solution = False



	def solve(self):
		"""Solves the Riesz projection problem to obtain the shape gradient of the cost functional.

		Returns
		-------
		gradient : dolfin.function.function.Function
			The function representing the shape gradient of the (reduced) cost functional.
		"""

		self.state_problem.solve()
		self.adjoint_problem.solve()

		if not self.has_solution:

			self.form_handler.regularization.update_geometric_quantities()
			self.form_handler.assembler.assemble(self.form_handler.fe_shape_derivative_vector)
			b = fenics.as_backend_type(self.form_handler.fe_shape_derivative_vector).vec()
			_solve_linear_problem(self.ksp, self.form_handler.scalar_product_matrix, b, self.gradient.vector().vec(), self.ksp_options)
			self.gradient.vector().apply('')

			self.has_solution = True

			self.gradient_norm_squared = self.form_handler.scalar_product(self.gradient, self.gradient)

		return self.gradient
