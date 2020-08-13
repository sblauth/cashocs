"""
Created on 26/02/2020, 11.13

@author: blauths
"""

import fenics
from .._forms import Lagrangian, FormHandler
from .._pde_problems import (StateProblem, AdjointProblem, GradientProblem,
							 HessianProblem, SemiSmoothHessianProblem, UnconstrainedHessianProblem)
from .._optimal_control import ReducedCostFunctional
from .methods import GradientDescent, LBFGS, CG, Newton, PDAS, SemiSmoothNewton
import numpy as np



class OptimalControlProblem:
	"""Implements an optimal control problem

	This class is used to define an optimal control problem, and also to solve
	it subsequently. For a detailed documentation, see the examples in the "demos"
	folder. For easier input, when consider single (state or control) variables,
	these do not have to be wrapped into a list.
	Note, that in the case of multiple variables these have to be grouped into
	ordered lists, where state_forms, bcs_list, states, adjoints have to have
	the same order (i.e. [state1, state2, state3,...] and [adjoint1, adjoint2,
	adjoint3, ...], where adjoint1 is the adjoint of state1 and so on.
	"""

	def __init__(self, state_forms, bcs_list, cost_functional_form, states, controls, adjoints, config,
				 riesz_scalar_products=None, control_constraints=None, initial_guess=None):
		"""Initializes the optimal control problem.

		This is used to generate all classes and functionalities. First ensures
		consistent input as the __init__ function is overloaded. Afterwards, the
		solution algorithm is initialized.
		
		Parameters
		----------
		state_forms : ufl.form.Form or list[ufl.form.Form]
			the weak form of the state equation (user implemented). Can be either
			a single UFL form, or a (ordered) list of UFL forms
		bcs_list : list[dolfin.fem.dirichletbc.DirichletBC] or list[list[dolfin.fem.dirichletbc.DirichletBC]] or dolfin.fem.dirichletbc.DirichletBC or None
			the list of DirichletBC objects describing Dirichlet (essential) boundary conditions.
			If this is None, then no Dirichlet boundary conditions are imposed.
		cost_functional_form : ufl.form.Form
			UFL form of the cost functional
		states : dolfin.function.function.Function or list[dolfin.function.function.Function]
			the state variable(s), can either be a single fenics Function, or a (ordered) list of these
		controls : dolfin.function.function.Function or list[dolfin.function.function.Function]
			the control variable(s), can either be a single fenics Function, or a list of these
		adjoints : dolfin.function.function.Function or list[dolfin.function.function.Function]
			the adjoint variable(s), can either be a single fenics Function, or a (ordered) list of these
		config : configparser.ConfigParser
			the config file for the problem, generated via caospy.create_config(path_to_config)
		riesz_scalar_products : None or ufl.form.Form or list[ufl.form.Form], optional
			the scalar products of the control space. Can either be None, a single UFL form, or a
			(ordered) list of UFL forms. If None, the L^2(\Omega) product is used.
			(default is None)
		control_constraints : None or list[dolfin.function.function.Function] or list[float] or list[list[dolfin.function.function.Function]] or list[list[float]], optional
			Box constraints posed on the control, None means that there are none (default is None).
			The (inner) lists should contain two elements of the form [u_a, u_b], where u_a is the lower,
			and u_b the upper bound.
		initial_guess : list[dolfin.function.function.Function], optional
			list of functions that act as initial guess for the state variables, should be valid input for fenics.assign.
			(defaults to None (which means a zero initial guess))


		Examples
		--------
		The corresponding examples detailing the use of this class can be found
		in the "demos" folder.
		"""

		### Overloading, such that we do not have to use lists for a single state and a single control
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
				self.bcs_list = []
				for i in range(self.state_dim):
					self.bcs_list.append([])
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
		temp_constraints = self.control_constraints[:]
		self.control_constraints = []
		for idx, pair in enumerate(temp_constraints):
			if type(pair[0]) in [float, int]:
				lower_bound = fenics.Function(self.controls[idx].function_space())
				lower_bound.vector()[:] = pair[0]
			elif pair[0].__module__ == 'dolfin.function.function' and type(pair[0]).__name__ == 'Function':
				lower_bound = pair[0]
			else:
				raise SystemExit('Wrong type for the control constraints')

			if type(pair[1]) in [float, int]:
				upper_bound = fenics.Function(self.controls[idx].function_space())
				upper_bound.vector()[:] = pair[1]
			elif pair[1].__module__ == 'dolfin.function.function' and type(pair[1]).__name__ == 'Function':
				upper_bound = pair[1]
			else:
				raise SystemExit('Wrong type for the control constraints')

			self.control_constraints.append([lower_bound, upper_bound])


		if type(self.control_constraints[0][0]) in [float, int]:
			temp_constraints = self.control_constraints[:]
			self.control_constraints = []
			for i, control in enumerate(self.controls):
				u_a = fenics.Function(control.function_space())
				u_a.vector()[:] = temp_constraints[i][0]
				u_b = fenics.Function(control.function_space())
				u_b.vector()[:] = temp_constraints[i][1]
				self.control_constraints.append([u_a, u_b])

		### Check whether the control constraints are feasible
		for pair in self.control_constraints:
			assert np.alltrue(pair[0].vector()[:] < pair[1].vector()[:]), 'the lower bound must always be smaller than the upper bound'

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
			self.hessian_problem = HessianProblem(self.form_handler, self.gradient_problem)
		if self.config.get('OptimizationRoutine', 'algorithm') == 'semi_smooth_newton':
			self.semi_smooth_hessian = SemiSmoothHessianProblem(self.form_handler, self.gradient_problem, self.control_constraints)
		if self.config.get('OptimizationRoutine', 'algorithm') == 'pdas':
			self.unconstrained_hessian = UnconstrainedHessianProblem(self.form_handler, self.gradient_problem)

		self.reduced_cost_functional = ReducedCostFunctional(self.form_handler, self.state_problem)

		self.gradients = self.gradient_problem.gradients
		self.objective_value = 1.0



	def stationary_measure_squared(self):
		"""Computes the stationary measure (squared) corresponding to box-constraints

		In case there are no box constraints this reduces to the classical gradient
		norm.

		Returns
		-------
		 float
			The square of the stationary measure

		"""

		for j in range(self.control_dim):
			self.projected_difference[j].vector()[:] = self.controls[j].vector()[:] - self.gradients[j].vector()[:]

		self.form_handler.project_to_admissible_set(self.projected_difference)

		for j in range(self.control_dim):
			self.projected_difference[j].vector()[:] = self.controls[j].vector()[:] - self.projected_difference[j].vector()[:]

		return self.form_handler.scalar_product(self.projected_difference, self.projected_difference)


		
	def solve(self):
		"""Solves the optimization problem by the method specified in the config file.

		Updates / overwrites states, controls, and adjoints according
		to the optimization method, i.e., the user-input functions.

		Returns
		-------
		None
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
