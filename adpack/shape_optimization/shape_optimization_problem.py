"""
Created on 15/06/2020, 08.00

@author: blauths
"""

import fenics
from ..forms import Lagrangian, ShapeFormHandler
from ..pde_problems.state_problem import StateProblem
from ..pde_problems.adjoint_problem import AdjointProblem
from ..pde_problems.shape_gradient_problem import ShapeGradientProblem
from .shape_cost_functional import ReducedCostFunctional
from .methods.gradient_descent import GradientDescent
from .methods.l_bfgs import LBFGS
from .methods.cg import CG



class ShapeOptimizationProblem:

	def __init__(self, state_forms, bcs_list, cost_functional_form, states, adjoints, boundaries, config, initial_guess=None):

		### Overloading, so that we do not have to use lists for single state single control
		### state_forms
		try:
			if type(state_forms) == list and len(state_forms) > 0:
				for i in range(len(state_forms)):
					if state_forms[i].__module__=='ufl.form' and type(state_forms[i]).__name__=='Form':
						pass
					else:
						raise SystemExit('state_forms have to be ufl forms')
				self.state_forms = state_forms
			elif state_forms.__module__ == 'ufl.form' and type(state_forms).__name__ == 'Form':
				self.state_forms = [state_forms]
			else:
				raise SystemExit('State forms have to be ufl forms')
		except:
			raise SystemExit('Type of state_forms is wrong.')
		self.state_dim = len(self.state_forms)

		### bcs_list
		try:
			if bcs_list == [] or bcs_list is None:
				self.bcs_list = [[]*self.state_dim]
			elif type(bcs_list) == list and len(bcs_list) > 0:
				if type(bcs_list[0]) == list:
					for i in range(len(bcs_list)):
						if type(bcs_list[i]) == list:
							pass
						else:
							raise SystemExit('bcs_list has inconsistent types.')
					self.bcs_list = bcs_list

				elif bcs_list[0].__module__ == 'dolfin.fem.dirichletbc' and type(bcs_list[0]).__name__ == 'DirichletBC':
					for i in range(len(bcs_list)):
						if bcs_list[i].__module__=='dolfin.fem.dirichletbc' and type(bcs_list[i]).__name__=='DirichletBC':
							pass
						else:
							raise SystemExit('bcs_list has inconsistent types.')
					self.bcs_list = [bcs_list]
			elif bcs_list.__module__ == 'dolfin.fem.dirichletbc' and type(bcs_list).__name__ == 'DirichletBC':
				self.bcs_list = [[bcs_list]]
			else:
				raise SystemExit('Type of bcs_list is wrong.')
		except:
			raise SystemExit('Type of bcs_list is wrong.')

		### cost_functional_form
		try:
			if cost_functional_form.__module__ == 'ufl.form' and type(cost_functional_form).__name__ == 'Form':
				self.cost_functional_form = cost_functional_form
			else:
				raise SystemExit('cost_functional_form has to be a ufl form')
		except:
			raise SystemExit('Type of cost_functional_form is wrong.')

		### states
		try:
			if type(states) == list and len(states) > 0:
				for i in range(len(states)):
					if states[i].__module__ == 'dolfin.function.function' and type(states[i]).__name__ == 'Function':
						pass
					else:
						raise SystemExit('states have to be fenics Functions.')

				self.states = states

			elif states.__module__ == 'dolfin.function.function' and type(states).__name__ == 'Function':
				self.states = [states]
			else:
				raise SystemExit('Type of states is wrong.')
		except:
			raise SystemExit('Type of states is wrong.')

		### adjoints
		try:
			if type(adjoints) == list and len(adjoints) > 0:
				for i in range(len(adjoints)):
					if adjoints[i].__module__ == 'dolfin.function.function' and type(adjoints[i]).__name__ == 'Function':
						pass
					else:
						raise SystemExit('adjoints have to fenics Functions.')

				self.adjoints = adjoints

			elif adjoints.__module__ == 'dolfin.function.function' and type(adjoints).__name__ == 'Function':
				self.adjoints = [adjoints]
			else:
				raise SystemExit('Type of adjoints is wrong.')
		except:
			raise SystemExit('Type of adjoints is wrong.')

		### config
		if config.__module__ == 'configparser' and type(config).__name__ == 'ConfigParser':
			self.config = config
		else:
			raise SystemExit('config has to be of configparser.ConfigParser type')

		self.boundaries = boundaries

		### initial guess
		if initial_guess is None:
			self.initial_guess = initial_guess
		else:
			try:
				if type(initial_guess) == list:
					self.initial_guess = initial_guess
				elif initial_guess.__module__ == 'dolfin.function.function' and type(initial_guess).__name__ == 'Function':
					self.initial_guess = [initial_guess]
			except:
				raise SystemExit('Initial guess has to be a list of functions')

		assert len(self.bcs_list) == self.state_dim, 'Length of states does not match'
		assert len(self.states) == self.state_dim, 'Length of states does not match'
		assert len(self.adjoints) == self.state_dim, 'Length of states does not match'
		if self.initial_guess is not None:
			assert len(self.initial_guess) == self.state_dim, 'Length of states does not match'
		### end overloading

		self.lagrangian = Lagrangian(self.state_forms, self.cost_functional_form)
		self.shape_form_handler = ShapeFormHandler(self.lagrangian, self.bcs_list, self.states, self.adjoints, self.boundaries, self.config)

		self.state_spaces = self.shape_form_handler.state_spaces
		self.adjoint_spaces = self.shape_form_handler.adjoint_spaces

		self.state_problem = StateProblem(self.shape_form_handler, self.initial_guess)
		self.adjoint_problem = AdjointProblem(self.shape_form_handler, self.state_problem)
		self.shape_gradient_problem = ShapeGradientProblem(self.shape_form_handler, self.state_problem, self.adjoint_problem)

		self.reduced_cost_functional = ReducedCostFunctional(self.shape_form_handler, self.state_problem)

		self.gradient = self.shape_gradient_problem.gradient
		self.objective_value = 1.0



	def norm_squared(self):
		"""Computes the norm (squared) of the shape gradient

		Returns
		-------
		 : float
			The square of the stationary measure

		"""

		return self.shape_gradient_problem.return_norm_squared()



	def solve(self):
		"""Solves the optimization problem by the method specified in the config file. See adpack.optimization.methds for details on the implemented solution methods

		Returns
		-------

			Updates self.state, self.control and self.adjoint according to the optimization method. The user inputs for generating the OptimalControlProblem class are actually manipulated there.

		"""

		self.algorithm = self.config.get('OptimizationRoutine', 'algorithm')

		if self.algorithm in ['gradient_descent', 'gd']:
			self.solver = GradientDescent(self)
		elif self.algorithm in ['lbfgs', 'bfgs']:
			self.solver = LBFGS(self)
		elif self.algorithm in ['cg', 'conjugate_gradient']:
			self.solver = CG(self)
		else:
			raise SystemExit('OptimizationRoutine.algorithm needs to be one of gd, lbfgs, or cg.')

		self.solver.run()
		self.solver.finalize()
