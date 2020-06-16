"""
Created on 26/02/2020, 11.13

@author: blauths
"""

import fenics
from ..forms import Lagrangian, FormHandler
from ..pde_problems.state_problem import StateProblem
from ..pde_problems.adjoint_problem import AdjointProblem
from ..pde_problems.gradient_problem import GradientProblem
from ..optimal_control.cost_functional import ReducedCostFunctional
from ..pde_problems.hessian_problem import HessianProblem
from ..pde_problems.semi_smooth_hessian import SemiSmoothHessianProblem
from ..pde_problems.unconstrained_hessian_problem import UnconstrainedHessianProblem
from .methods.gradient_descent import GradientDescent
from .methods.l_bfgs import LBFGS
from .methods.cg import CG
from .methods.newton import Newton
from .methods.semi_smooth_newton import SemiSmoothNewton
from .methods.primal_dual_active_set_method import PDAS
import time



class OptimalControlProblem:
	
	def __init__(self, state_forms, bcs_list, cost_functional_form, states, controls, adjoints, config,
				 riesz_scalar_products=None, control_constraints=None, initial_guess=None):
		"""The implementation of the optimization problem, used to generate all other classes and functionality. Also used to solve the problem.
		
		Parameters
		----------
		state_forms : ufl.form.Form or List[ufl.form.Form]
			the weak form of the state equation (user implemented)
		bcs_list : List[dolfin.fem.dirichletbc.DirichletBC] or List[List[dolfin.fem.dirichletbc.DirichletBC]]
			the list of DirichletBC objects describing essential boundary conditions
		cost_functional_form : ufl.form.Form
			the cost functional (user implemented)
		states : dolfin.function.function.Function or List[dolfin.function.function.Function]
			the state variable
		controls : dolfin.function.function.Function or List[dolfin.function.function.Function]
			the control variable
		adjoints : dolfin.function.function.Function or List[dolfin.function.function.Function]
			the adjoint variable
		config : configparser.ConfigParser
			the config file for the problem
		riesz_scalar_products :
		control_constraints : List[dolfin.function.function.Function] or List[float] or List[List]
			Box constraints posed on the control
		"""

		### Overloading, such that we do not have to use lists for single state single control
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

		### controls
		try:
			if type(controls) == list and len(controls) > 0:
				for i in range(len(controls)):
					if controls[i].__module__ == 'dolfin.function.function' and type(controls[i]).__name__ == 'Function':
						pass
					else:
						raise SystemExit('controls have to be fenics Functions.')

				self.controls = controls

			elif controls.__module__ == 'dolfin.function.function' and type(controls).__name__ == 'Function':
				self.controls = [controls]
			else:
				raise SystemExit('Type of controls is wrong.')
		except:
			raise SystemExit('Type of controls is wrong.')

		self.control_dim = len(self.controls)
		
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


		### riesz_scalar_products
		if riesz_scalar_products is None:
			dx = fenics.Measure('dx', self.controls[0].function_space().mesh())
			self.riesz_scalar_products = [fenics.inner(fenics.TrialFunction(self.controls[i].function_space()), fenics.TestFunction(self.controls[i].function_space())) * dx
										  for i in range(len(self.controls))]
		else:
			try:
				if type(riesz_scalar_products)==list and len(riesz_scalar_products) > 0:
					for i in range(len(riesz_scalar_products)):
						if riesz_scalar_products[i].__module__== 'ufl.form' and type(riesz_scalar_products[i]).__name__== 'Form':
							pass
						else:
							raise SystemExit('riesz_scalar_products have to be ufl forms')
					self.riesz_scalar_products = riesz_scalar_products
				elif riesz_scalar_products.__module__== 'ufl.form' and type(riesz_scalar_products).__name__== 'Form':
					self.riesz_scalar_products = [riesz_scalar_products]
				else:
					raise SystemExit('State forms have to be ufl forms')
			except:
				raise SystemExit('Type of riesz_scalar_prodcuts is wrong..')

		### control_constraints
		if control_constraints is None:
			self.control_constraints = []
			for control in self.controls:
				u_a = fenics.Function(control.function_space())
				u_a.vector()[:] = float('-inf')
				u_b = fenics.Function(control.function_space())
				u_b.vector()[:] = float('inf')
				self.control_constraints.append([u_a, u_b])
		else:
			try:
				if type(control_constraints) == list and len(control_constraints) > 0:
					if type(control_constraints[0]) == list:
						for i in range(len(control_constraints)):
							if type(control_constraints[i]) == list and len(control_constraints[i]) == 2:
								for j in range(2):
									if type(control_constraints[i][j]) in [float, int]:
										pass
									elif control_constraints[i][j].__module__ == 'dolfin.function.function' and type(control_constraints[i][j]).__name__ == 'Function':
										pass
									else:
										raise SystemExit('control_constraints has to be a list containing upper and lower bounds')
								pass
							else:
								raise SystemExit('control_constraints has to be a list containing upper and lower bounds')
						self.control_constraints = control_constraints
					elif (type(control_constraints[0]) in [float, int] or (control_constraints[0].__module__ == 'dolfin.function.function' and type(control_constraints[0]).__name__=='Function')) \
						and (type(control_constraints[1]) in [float, int] or (control_constraints[1].__module__ == 'dolfin.function.function' and type(control_constraints[1]).__name__=='Function')):

						self.control_constraints = [control_constraints]
					else:
						raise SystemExit('control_constraints has to be a list containing upper and lower bounds')

			except:
				raise SystemExit('control_constraints has to be a list containing upper and lower bounds')

		# recast floats into functions for compatibility
		if type(self.control_constraints[0][0]) in [float, int]:
			temp_constraints = self.control_constraints[:]
			self.control_constraints = []
			for i, control in enumerate(self.controls):
				u_a = fenics.Function(control.function_space())
				u_a.vector()[:] = temp_constraints[i][0]
				u_b = fenics.Function(control.function_space())
				u_b.vector()[:] = temp_constraints[i][1]
				self.control_constraints.append([u_a, u_b])

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
		assert len(self.riesz_scalar_products) == self.control_dim, 'Length of controls does not match'
		assert len(self.states) == self.state_dim, 'Length of states does not match'
		assert len(self.adjoints) == self.state_dim, 'Length of states does not match'
		assert len(self.control_constraints) == self.control_dim, 'Length of controls does not match'
		if self.initial_guess is not None:
			assert len(self.initial_guess) == self.state_dim, 'Length of states does not match'
		### end overloading

		self.lagrangian = Lagrangian(self.state_forms, self.cost_functional_form)
		self.form_handler = FormHandler(self.lagrangian, self.bcs_list, self.states, self.controls, self.adjoints, self.config, self.riesz_scalar_products, self.control_constraints)

		self.state_spaces = self.form_handler.state_spaces
		self.control_spaces = self.form_handler.control_spaces
		self.adjoint_spaces = self.form_handler.adjoint_spaces

		self.projected_difference = [fenics.Function(V) for V in self.control_spaces]

		self.state_problem = StateProblem(self.form_handler, self.initial_guess)
		self.adjoint_problem = AdjointProblem(self.form_handler, self.state_problem)
		self.gradient_problem = GradientProblem(self.form_handler, self.state_problem, self.adjoint_problem)
		
		if self.config.get('OptimizationRoutine', 'algorithm') == 'newton':
			self.hessian_problem = HessianProblem(self.form_handler, self.gradient_problem, self.control_constraints)
		if self.config.get('OptimizationRoutine', 'algorithm') == 'semi_smooth_newton':
			self.semi_smooth_hessian = SemiSmoothHessianProblem(self.form_handler, self.gradient_problem, self.control_constraints)
		if self.config.get('OptimizationRoutine', 'algorithm') == 'pdas':
			self.unconstrained_hessian = UnconstrainedHessianProblem(self.form_handler, self.gradient_problem)

		self.reduced_cost_functional = ReducedCostFunctional(self.form_handler, self.state_problem)

		self.gradients = self.gradient_problem.gradients
		self.objective_value = 1.0



	def stationary_measure_squared(self):
		"""Computes the stationary measure (squared) corresponding to box-constraints, in case they are present

		Returns
		-------
		 : float
			The square of the stationary measure

		"""

		for j in range(self.control_dim):
			self.projected_difference[j].vector()[:] = self.controls[j].vector()[:] - self.gradients[j].vector()[:]

		self.form_handler.project(self.projected_difference)

		for j in range(self.control_dim):
			self.projected_difference[j].vector()[:] = self.controls[j].vector()[:] - self.projected_difference[j].vector()[:]

		return self.form_handler.scalar_product(self.projected_difference, self.projected_difference)


		
	def solve(self):
		"""Solves the optimization problem by the method specified in the config file. See adpack.optimization.methds for details on the implemented solution methods
		
		Returns
		-------
		
			Updates self.state, self.control and self.adjoint according to the optimization method. The user inputs for generating the OptimalControlProblem class are actually manipulated there.

		"""
		
		self.algorithm = self.config.get('OptimizationRoutine', 'algorithm')
		
		if self.algorithm in ['gd', 'gradient_descent']:
			self.solver = GradientDescent(self)
		elif self.algorithm in ['lbfgs', 'bfgs']:
			self.solver = LBFGS(self)
		elif self.algorithm in ['cg', 'conjugate_gradient']:
			self.solver = CG(self)
		elif self.algorithm == 'newton':
			self.solver = Newton(self)
		elif self.algorithm in ['semi_smooth_newton', 'ss_newton']:
			self.solver = SemiSmoothNewton(self)
		elif self.algorithm in ['pdas', 'primal_dual_active_set']:
			self.solver = PDAS(self)
		else:
			raise SystemExit('OptimizationRoutine.algorithm needs to be one of gd, lbfgs, cg, newton, semi_smooth_newton or pdas.')
		
		self.solver.run()
		self.solver.finalize()
