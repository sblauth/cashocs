"""
Created on 24/02/2020, 08.45

@author: blauths
"""

import fenics
from ufl import replace
from .helpers import summ
import numpy as np
import time



class Lagrangian:
	
	def __init__(self, state_forms, cost_functional_form):
		"""
		A class that implements a Lagrangian, i.e., the sum of (reduced) cost functional and state constraint
		
		Parameters
		----------
		state_forms : List[ufl.form.Form]
			the weak forms of the state equation, as implemented by the user, either directly as one single form or a list of forms
		cost_functional_form : ufl.form.Form
			the cost functional, as implemented by the user
		"""
		
		self.state_forms = state_forms
		self.cost_functional_form = cost_functional_form

		self.lagrangian_form = self.cost_functional_form + summ(self.state_forms)
		
		
		
		
		
class FormHandler:

	def __init__(self, lagrangian, bcs_list, control_measures, states, controls, adjoints, config, control_constraints):
		"""The form handler implements all form manipulations needed in order to compute adjoint equations, sensitvities, etc.
		
		Parameters
		----------
		lagrangian : Lagrangian
			the lagrangian corresponding to the optimization problem
		bcs_list : List[List]
			the list of DirichletBCs for the state equation
		control_measures : List[ufl.measure.Measure]
			the measure corresponding to the domain of the control
		states : List[dolfin.function.function.Function]
			the function that acts as the state variable
		controls : List[dolfin.function.function.Function]
			the function that acts as the control variable
		adjoints : List[dolfin.function.function.Function]
			the function that acts as the adjoint variable
		config : configparser.ConfigParser
			the configparser object of the config file
		"""

		self.lagrangian = lagrangian
		self.bcs_list = bcs_list
		self.control_measures = control_measures
		self.states = states
		self.controls = controls
		self.adjoints = adjoints
		self.config = config
		self.control_constraints = control_constraints
		
		self.cost_functional_form = self.lagrangian.cost_functional_form
		self.state_forms = self.lagrangian.state_forms
		
		self.state_dim = len(self.states)
		self.control_dim = len(self.controls)
		
		self.state_spaces = [x.function_space() for x in self.states]
		self.control_spaces = [x.function_space() for x in self.controls]
		self.mesh = self.state_spaces[0].mesh()
		self.dx = fenics.Measure('dx', self.mesh)

		self.states_prime = [fenics.Function(V) for V in self.state_spaces]
		self.adjoints_prime = [fenics.Function(V) for V in self.state_spaces]
		
		self.hessian_actions = [fenics.Function(V) for V in self.control_spaces]
		
		self.arg_state1 = [fenics.Function(V) for V in self.state_spaces]
		self.arg_state2 = [fenics.Function(V) for V in self.state_spaces]
		
		self.arg_control1 = [fenics.Function(V) for V in self.control_spaces]
		self.arg_control2 = [fenics.Function(V) for V in self.control_spaces]
		
		self.test_directions = [fenics.Function(V) for V in self.control_spaces]
		
		self.trial_functions_state = [fenics.TrialFunction(V) for V in self.state_spaces]
		self.test_functions_state = [fenics.TestFunction(V) for V in self.state_spaces]
		
		self.trial_functions_control = [fenics.TrialFunction(V) for V in self.control_spaces]
		self.test_functions_control = [fenics.TestFunction(V) for V in self.control_spaces]

		self.temp = [fenics.Function(V) for V in self.control_spaces]

		self.compute_state_equations()
		self.compute_adjoint_equations()
		self.compute_gradient_equations()




		self.opt_algo = self.config.get('OptimizationRoutine', 'algorithm')

		if self.opt_algo == 'newton' or self.opt_algo == 'semi_smooth_newton' or (self.opt_algo == 'pdas' and self.config.get('OptimizationRoutine', 'inner_pdas') == 'newton'):
			self.compute_newton_forms()




	def scalar_product(self, a, b):
		"""Implements the scalar product needed for the algorithms

		Parameters
		----------
		a : List[dolfin.function.function.Function]
			The first input
		b : List[dolfin.function.function.Function]
			The second input

		Returns
		-------
		 : float
			The value of the scalar product

		"""

		return summ([fenics.assemble(fenics.inner(a[i], b[i])*self.control_measures[i]) for i in range(self.control_dim)])



	def compute_active_sets(self):

		self.idx_active_lower = [(self.controls[j].vector()[:] <= self.control_constraints[j][0]).nonzero()[0] for j in range(self.control_dim)]
		self.idx_active_upper = [(self.controls[j].vector()[:] >= self.control_constraints[j][1]).nonzero()[0] for j in range(self.control_dim)]

		self.idx_active = [np.concatenate((self.idx_active_lower[j], self.idx_active_upper[j])) for j in range(self.control_dim)]
		[self.idx_active[j].sort() for j in range(self.control_dim)]

		self.idx_inactive = [np.setdiff1d(np.arange(self.control_spaces[j].dim()), self.idx_active[j] ) for j in range(self.control_dim)]

		return None



	def project_active(self, a, b):

		for j in range(self.control_dim):
			self.temp[j].vector()[:] = 0.0
			self.temp[j].vector()[self.idx_active[j]] = a[j].vector()[self.idx_active[j]]
			b[j].vector()[:] = self.temp[j].vector()[:]

		return b



	def project_inactive(self, a, b):

		for j in range(self.control_dim):
			self.temp[j].vector()[:] = 0.0
			self.temp[j].vector()[self.idx_inactive[j]] = a[j].vector()[self.idx_inactive[j]]
			b[j].vector()[:] = self.temp[j].vector()[:]

		return b



	def project(self, a):

		for j in range(self.control_dim):
			a[j].vector()[:] = np.maximum(self.control_constraints[j][0], np.minimum(self.control_constraints[j][1], a[j].vector()[:]))

		return a
	


	def compute_state_equations(self):
		"""Compute the weak form of the state equation for the use with fenics
		
		Returns
		-------
		
			Creates self.state_eq_forms

		"""

		if self.config.get('StateEquation', 'picard_iteration'):
			self.state_picard_forms = [fenics.derivative(self.lagrangian.lagrangian_form, self.adjoints[i], self.test_functions_state[i]) for i in range(self.state_dim)]

		self.state_eq_forms = [fenics.derivative(self.lagrangian.lagrangian_form, self.adjoints[i], self.test_functions_state[i]) for i in range(self.state_dim)]

		if self.config.getboolean('StateEquation', 'is_linear'):
			for i in range(self.state_dim):
				self.state_eq_forms[i] = replace(self.state_picard_forms[i], {self.states[i] : self.trial_functions_state[i]})
	


	def compute_adjoint_equations(self):
		"""Computes the weak form of the adjoint equation for use with fenics
		
		Returns
		-------
		
			Creates self.adjoint_eq_form and self.bcs_ad, corresponding to homogenized BCs

		"""

		if self.config.get('StateEquation', 'picard_iteration'):
			self.adjoint_picard_forms = [fenics.derivative(self.lagrangian.lagrangian_form, self.states[i], self.test_functions_state[i]) for i in range(self.state_dim)]

		self.adjoint_eq_forms = [fenics.derivative(self.lagrangian.lagrangian_form, self.states[i], self.test_functions_state[i]) for i in range(self.state_dim)]
		self.adjoint_eq_lhs = []
		self.adjoint_eq_rhs = []

		for i in range(self.state_dim):
			self.adjoint_eq_forms[i] = replace(self.adjoint_eq_forms[i], {self.adjoints[i] : self.trial_functions_state[i]})
			a, L = fenics.system(self.adjoint_eq_forms[i])
			self.adjoint_eq_lhs.append(a)
			self.adjoint_eq_rhs.append(L)

		self.bcs_list_ad = [[fenics.DirichletBC(bc) for bc in self.bcs_list[i]] for i in range(self.state_dim)]
		[[bc.homogenize() for bc in self.bcs_list_ad[i]] for i in range(self.state_dim)]


	
	def compute_gradient_equations(self):
		"""Computes the variational form of the gradient equation, for the Riesz projection
		
		Returns
		-------
		
			Creates self.gradient_form_lhs and self.gradient_form_rhs

		"""
		
		self.gradient_forms_lhs = [fenics.inner(self.trial_functions_control[i], self.test_functions_control[i])*self.control_measures[i] for i in range(self.control_dim)]
		self.gradient_forms_rhs = [fenics.derivative(self.lagrangian.lagrangian_form, self.controls[i], self.test_functions_control[i]) for i in range(self.control_dim)]

	
	
	def compute_newton_forms(self):
		"""Computes the needed forms for a truncated Newton method
		
		Returns
		-------
		

		"""
		
		self.sensitivity_eqs_lhs = [fenics.derivative(self.state_forms[i], self.states[i], self.trial_functions_state[i]) for i in range(self.state_dim)]
		self.sensitivity_eqs_picard = [fenics.derivative(self.state_forms[i], self.states[i], self.states_prime[i]) for i in range(self.state_dim)]
		for i in range(self.state_dim):
			self.sensitivity_eqs_lhs[i] = replace(self.sensitivity_eqs_lhs[i], {self.adjoints[i] : self.test_functions_state[i]})
			self.sensitivity_eqs_picard[i] = replace(self.sensitivity_eqs_picard[i], {self.adjoints[i] : self.test_functions_state[i]})
		if self.state_dim > 1:
			self.sensitivity_eqs_rhs = [- summ([fenics.derivative(self.state_forms[i], self.states[j], self.states_prime[j]) for j in range(self.state_dim) if j != i])
										- summ([fenics.derivative(self.state_forms[i], self.controls[j], self.test_directions[j]) for j in range(self.control_dim)])
										for i in range(self.state_dim)]
		else:
			self.sensitivity_eqs_rhs = [- summ([fenics.derivative(self.state_forms[i], self.controls[j], self.test_directions[j]) for j in range(self.control_dim)])
										for i in range(self.state_dim)]
			
		for i in range(self.state_dim):
			self.sensitivity_eqs_rhs[i] = replace(self.sensitivity_eqs_rhs[i], {self.adjoints[i] : self.test_functions_state[i]})
			self.sensitivity_eqs_picard[i] -= self.sensitivity_eqs_rhs[i]


		self.L_y = [fenics.derivative(self.lagrangian.lagrangian_form, self.states[i], self.test_functions_state[i]) for i in range(self.state_dim)]
		self.L_u = [fenics.derivative(self.lagrangian.lagrangian_form, self.controls[i], self.test_functions_control[i]) for i in range(self.control_dim)]

		self.L_yy = [[fenics.derivative(self.L_y[i], self.states[j], self.arg_state2[j]) for j in range(self.state_dim)] for i in range(self.state_dim)]
		self.L_yu = [[fenics.derivative(self.L_u[i], self.states[j], self.arg_state2[j]) for j in range(self.state_dim)] for i in range(self.control_dim)]
		self.L_uy = [[fenics.derivative(self.L_y[i], self.controls[j], self.arg_control2[j]) for j in range(self.control_dim)] for i in range(self.state_dim)]
		self.L_uu = [[fenics.derivative(self.L_u[i], self.controls[j], self.arg_control2[j]) for j in range(self.control_dim)] for i in range(self.control_dim)]
		
		
		self.w_1 = [summ([replace(self.L_yy[i][j], {self.arg_state2[j] : self.states_prime[j]}) for j in range(self.state_dim)])
					+ summ([replace(self.L_uy[i][j] , {self.arg_control2[j] : self.test_directions[j]}) for j in range(self.control_dim)])
					for i in range(self.state_dim)]
		self.w_2 = [summ([replace(self.L_yu[i][j], {self.arg_state2[j] : self.states_prime[j]}) for j in range(self.state_dim)])
					+ summ([replace(self.L_uu[i][j], {self.arg_control2[j] : self.test_directions[j]}) for j in range(self.control_dim)])
					for i in range(self.control_dim)]
		
		self.adjoint_sensitivity_eqs_lhs = [fenics.derivative(self.state_forms[i], self.states[i], self.test_functions_state[i]) for i in range(self.state_dim)]
		self.adjoint_sensitivity_eqs_picard = [fenics.derivative(self.state_forms[i], self.states[i], self.test_functions_state[i]) for i in range(self.state_dim)]

		if self.state_dim > 1:
			for i in range(self.state_dim):
				self.adjoint_sensitivity_eqs_lhs[i] = replace(self.adjoint_sensitivity_eqs_lhs[i], {self.adjoints[i] : self.trial_functions_state[i]})
				self.adjoint_sensitivity_eqs_picard[i] = replace(self.adjoint_sensitivity_eqs_picard[i], {self.adjoints[i] : self.adjoints_prime[i]})

				self.w_1[i] -= summ([replace(fenics.derivative(self.state_forms[j], self.states[i], self.test_functions_state[i]),
											 {self.adjoints[j] : self.adjoints_prime[j]}) for j in range(self.state_dim) if j != i])
		else:
			self.adjoint_sensitivity_eqs_lhs[0] = replace(self.adjoint_sensitivity_eqs_lhs[0], {self.adjoints[0] : self.trial_functions_state[0]})
			self.adjoint_sensitivity_eqs_picard[0] = replace(self.adjoint_sensitivity_eqs_picard[0], {self.adjoints[0] : self.adjoints_prime[0]})

		for i in range(self.state_dim):
			self.adjoint_sensitivity_eqs_picard[i] -= self.w_1[i]
		
		
		self.adjoint_sensitivity_eqs_rhs = [summ([fenics.derivative(self.state_forms[j], self.controls[i], self.test_functions_control[i]) for j in range(self.state_dim)])
											for i in range(self.control_dim)]
		
		for i in range(self.control_dim):
			for j in range(self.state_dim):
				self.adjoint_sensitivity_eqs_rhs[i] = replace(self.adjoint_sensitivity_eqs_rhs[i], {self.adjoints[j] : self.adjoints_prime[j]})
		
		self.w_3 = [- self.adjoint_sensitivity_eqs_rhs[i] for i in range(self.control_dim)]
		
		self.hessian_lhs = [fenics.inner(self.trial_functions_control[i], self.test_functions_control[i])*self.control_measures[i] for i in range(self.control_dim)]
		self.hessian_rhs = [self.w_2[i] + self.w_3[i] for i in range(self.control_dim)]
