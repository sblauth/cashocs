"""
Created on 24/02/2020, 08.45

@author: blauths
"""

import fenics
from ufl import replace



class Lagrangian:
	
	def __init__(self, state_form, cost_functional_form):
		"""
		A class that implements a Lagrangian, i.e., the sum of (reduced) cost functional and state constraint
		
		Parameters
		----------
		state_form : ufl.form.Form
			the weak form of the state equation, as implemented by the user
		cost_functional_form : ufl.form.Form
			the cost functional, as implemented by the user
		"""
		
		self.state_form = state_form
		self.cost_functional_form =  cost_functional_form

		self.form = self.cost_functional_form + self.state_form
	




class FormHandler:

	def __init__(self, lagrangian, bcs, control_measure, state, control, adjoint, config):
		"""The form handler implements all form manipulations needed in order to compute adjoint equations, sensitvities, etc.
		
		Parameters
		----------
		lagrangian : Lagrangian
			the lagrangian corresponding to the optimization problem
		bcs : dolfin.fem.dirichletbc.DirichletBC
			the list of DirichletBCs for the state equation
		control_measure : ufl.measure.Measure
			the measure corresponding to the domain of the control
		state : dolfin.function.function.Function
			the function that acts as the state variable
		control : dolfin.function.function.Function
			the function that acts as the control variable
		adjoint : dolfin.function.function.Function
			the function that acts as the adjoint variable
		config : configparser.ConfigParser
			the configparser object of the config file
		"""
		
		self.lagrangian = lagrangian
		self.bcs = bcs
		self.control_measure = control_measure
		self.state = state
		self.control = control
		self.adjoint = adjoint
		self.config = config
		
		
		self.cost_functional_form = self.lagrangian.cost_functional_form
		self.state_form = self.lagrangian.state_form
		
		self.state_space = self.state.function_space()
		self.control_space = self.control.function_space()
		self.mesh = self.state_space.mesh()
		self.dx = fenics.Measure('dx', self.mesh)
		
		self.state_prime = fenics.Function(self.state_space)
		self.adjoint_prime = fenics.Function(self.state_space)
		
		self.hessian_action = fenics.Function(self.control_space)
		
		self.arg_state1 = fenics.Function(self.state_space)
		self.arg_state2 = fenics.Function(self.state_space)
		
		self.arg_control1 = fenics.Function(self.control_space)
		self.arg_control2 = fenics.Function(self.control_space)
		
		self.test_direction = fenics.Function(self.control_space)
		
		self.trial_function_state = fenics.TrialFunction(self.state_space)
		self.test_function_state = fenics.TestFunction(self.state_space)
		
		self.trial_function_control = fenics.TrialFunction(self.control_space)
		self.test_function_control = fenics.TestFunction(self.control_space)
		
		self.compute_state_equation()
		self.compute_adjoint_equation()
		self.compute_gradient_equation()
		self.compute_newton_forms()
		
	
	
	def compute_state_equation(self):
		"""Compute the weak form of the state equation for the use with fenics
		
		Returns
		-------
		None
			But creates self.state_eq_form

		"""
		
		self.state_eq_form = fenics.derivative(self.lagrangian.form, self.adjoint, self.test_function_state)
		
		if self.config.getboolean('StateEquation', 'is_linear'):
			self.state_eq_form = replace(self.state_eq_form, {self.state : self.trial_function_state})
			
	
	
	
	def compute_adjoint_equation(self):
		"""Computes the weak form of the adjoint equation for use with fenics
		
		Returns
		-------
		None
			But creates self.adjoint_eq_form and self.bcs_ad, corresponding to homogenized BCs

		"""
		
		self.adjoint_eq_form = fenics.derivative(self.lagrangian.form, self.state, self.trial_function_state)
		self.adjoint_eq_form = replace(self.adjoint_eq_form, {self.adjoint : self.test_function_state})
		self.adjoint_eq_form = fenics.adjoint(self.adjoint_eq_form)
		
		self.bcs_ad = [fenics.DirichletBC(bc) for bc in self.bcs]
		[bc.homogenize() for bc in self.bcs_ad]
	
	
	
	def compute_gradient_equation(self):
		"""Computes the variational form of the gradient equation, for the Riesz projection
		
		Returns
		-------
		None
			but creates self.gradient_form_lhs and self.gradient_form_rhs

		"""
		
		self.gradient_form_lhs = fenics.inner(self.trial_function_control, self.test_function_control)*self.control_measure
		self.gradient_form_rhs = fenics.derivative(self.lagrangian.form, self.control, self.test_function_control)
	
	
	
	def compute_newton_forms(self):
		"""Computes the needed forms for a truncated Newton method
		
		Returns
		-------
		None

		"""
		
		self.sensitivity_eq_lhs = fenics.derivative(self.state_form, self.state, self.trial_function_state)
		self.sensitivity_eq_lhs = replace(self.sensitivity_eq_lhs, {self.adjoint : self.test_function_state})
		
		self.sensitivity_eq_rhs = fenics.derivative(self.state_form, self.control, self.test_direction)
		self.sensitivity_eq_rhs = replace(self.sensitivity_eq_rhs, {self.adjoint : self.test_function_state})
		self.sensitivity_eq_rhs = -self.sensitivity_eq_rhs
		
		
		self.L_y = fenics.derivative(self.lagrangian.form, self.state, self.arg_state1)
		self.L_u = fenics.derivative(self.lagrangian.form, self.control, self.arg_control1)
		
		self.L_yy = fenics.derivative(self.L_y, self.state, self.arg_state2)
		self.L_yu = fenics.derivative(self.L_u, self.state, self.arg_state2)
		self.L_uy = fenics.derivative(self.L_y, self.control, self.arg_control2)
		self.L_uu = fenics.derivative(self.L_u, self.control, self.arg_control2)
		
		self.w_1 = replace(self.L_yy, {self.arg_state2 : self.state_prime, self.arg_state1 : self.test_function_state}) \
			  + replace(self.L_uy, {self.arg_control2 : self.test_direction, self.arg_state1 : self.test_function_state})
		self.w_2 = replace(self.L_yu, {self.arg_state2 : self.state_prime, self.arg_control1 : self.test_function_control}) \
				   + replace(self.L_uu, {self.arg_control2 : self.test_direction, self.arg_control1 : self.test_function_control})
		
		
		self.adjoint_sensitivity_lhs = fenics.adjoint(self.sensitivity_eq_lhs)
		
		self.adjoint_sensitivity_rhs = fenics.derivative(self.state_form, self.control, self.arg_control1)
		self.w_3 = replace(self.adjoint_sensitivity_rhs, {self.arg_control1 : self.test_function_control, self.adjoint : self.adjoint_prime})
		self.w_3 = -self.w_3
		
		
		self.hessian_lhs = fenics.inner(self.trial_function_control, self.test_function_control)*self.control_measure
		self.hessian_rhs = self.w_2 + self.w_3
