"""
Created on 15/06/2020, 08.10

@author: blauths
"""

import fenics



class ShapeGradientProblem:
	"""A class that represents the shape gradient problem, used to compute the shape gradient

	Attributes
	----------
	shape_form_handler : adpack.forms.ShapeFormHandler
		the shape form handler, containing (amongst others) the UFL form of the shape derivative

	state_problem : adpack.pde_problems.state_problem.StateProblem
		the corresponding state problem

	adjoint_problem : adpack.pde_problems.adjoint_problem.AdjointProblem
		the corresponding adjoint problem

	gradient : dolfin.function.function.Function
		the shape gradient

	config : configparser.ConfigParser
		the config object of the problem

	has_solution : bool
		boolean flag, indicating whether the current shape gradient is up-to-date

	gradient_norm_squared : float
		norm of the shape gradient, squared
	"""

	def __init__(self, shape_form_handler, state_problem, adjoint_problem):
		"""Initialize the ShapeGradientProblem

		Parameters
		----------
		shape_form_handler : adpack.forms.ShapeFormHandler
			the ShapeFormHandler object corresponding to the shape optimization problem

		state_problem : adpack.pde_problems.state_problem.StateProblem
			the corresponding state problem

		adjoint_problem : adpack.pde_problems.adjoint_problem.AdjointProblem
			the corresponding adjoint problem
		"""

		self.shape_form_handler = shape_form_handler
		self.state_problem = state_problem
		self.adjoint_problem = adjoint_problem

		self.gradient = fenics.Function(self.shape_form_handler.deformation_space)
		self.gradient_norm_squared = 1.0

		self.config = self.shape_form_handler.config

		self.has_solution = False



	def solve(self):
		"""Solves the Riesz projection problem to obtain the gradient of the cost functional

		Returns
		-------
		gradient : dolfin.function.function.Function
			the function representing the gradient of the (reduced) cost functional

		"""

		self.state_problem.solve()
		self.adjoint_problem.solve()

		if not self.has_solution:

			self.shape_form_handler.regularization.update_geometric_quantities()

			self.shape_form_handler.ksp.setOperators(self.shape_form_handler.scalar_product_matrix)
			self.shape_form_handler.assembler.assemble(self.shape_form_handler.fe_shape_derivative_vector)
			b = fenics.as_backend_type(self.shape_form_handler.fe_shape_derivative_vector).vec()

			x = self.gradient.vector().vec()
			self.shape_form_handler.ksp.solve(b, x)

			if self.shape_form_handler.ksp.getConvergedReason() < 0:
				raise SystemExit('Krylov solver did not converge. Reason: ' + str(self.shape_form_handler.ksp.getConvergedReason()))

			self.has_solution = True

			self.gradient_norm_squared = self.shape_form_handler.scalar_product(self.gradient, self.gradient)

		return self.gradient



	def return_norm_squared(self):
		"""Returns the norm of the shape gradient, squared

		Returns
		-------
		gradient_norm_squared : float
			the norm of the shape gradient, squared
		"""

		self.solve()

		return self.gradient_norm_squared
