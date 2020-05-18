"""
Created on 24/02/2020, 08.45

@author: blauths
"""

import fenics
from ufl import replace
from .helpers import summ
import numpy as np
import time
from petsc4py import PETSc



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

	def __init__(self, lagrangian, bcs_list, states, controls, adjoints, config, control_scalar_products, control_constraints):
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
		self.states = states
		self.controls = controls
		self.adjoints = adjoints
		self.config = config
		self.control_scalar_products = control_scalar_products
		self.control_constraints = control_constraints
		
		self.cost_functional_form = self.lagrangian.cost_functional_form
		self.state_forms = self.lagrangian.state_forms
		
		self.state_dim = len(self.states)
		self.control_dim = len(self.controls)
		
		self.state_spaces = [x.function_space() for x in self.states]
		self.control_spaces = [x.function_space() for x in self.controls]
		self.adjoint_spaces = [x.function_space() for x in self.adjoints]
		# test if state_spaces == adjoint_spaces
		if self.state_spaces == self.adjoint_spaces:
			self.state_adjoint_equal_spaces = True
		else:
			self.state_adjoint_equal_spaces = False

		self.mesh = self.state_spaces[0].mesh()
		self.dx = fenics.Measure('dx', self.mesh)

		self.states_prime = [fenics.Function(V) for V in self.state_spaces]
		self.adjoints_prime = [fenics.Function(V) for V in self.adjoint_spaces]
		
		self.hessian_actions = [fenics.Function(V) for V in self.control_spaces]

		self.test_directions = [fenics.Function(V) for V in self.control_spaces]
		
		self.trial_functions_state = [fenics.TrialFunction(V) for V in self.state_spaces]
		self.test_functions_state = [fenics.TestFunction(V) for V in self.state_spaces]

		self.trial_functions_adjoint = [fenics.TrialFunction(V) for V in self.adjoint_spaces]
		self.test_functions_adjoint = [fenics.TestFunction(V) for V in self.adjoint_spaces]

		self.trial_functions_control = [fenics.TrialFunction(V) for V in self.control_spaces]
		self.test_functions_control = [fenics.TestFunction(V) for V in self.control_spaces]

		self.temp = [fenics.Function(V) for V in self.control_spaces]

		self.compute_state_equations()
		self.compute_adjoint_equations()
		self.compute_gradient_equations()

		self.opt_algo = self.config.get('OptimizationRoutine', 'algorithm')

		if self.opt_algo == 'newton' or self.opt_algo == 'semi_smooth_newton' or (self.opt_algo == 'pdas' and self.config.get('OptimizationRoutine', 'inner_pdas') == 'newton'):
			self.compute_newton_forms()

		# initialize the scalar products
		fe_scalar_product_matrices = [fenics.assemble(self.control_scalar_products[i], keep_diagonal=True) for i in range(self.control_dim)]
		self.scalar_product_matrices = [fenics.as_backend_type(fe_scalar_product_matrices[i]).mat() for i in range(self.control_dim)]
		[fe_scalar_product_matrices[i].ident_zeros() for i in range(self.control_dim)]
		self.riesz_projection_matrices = [fenics.as_backend_type(fe_scalar_product_matrices[i]).mat() for i in range(self.control_dim)]

		# test for symmetry
		for i in range(self.control_dim):
			if not self.scalar_product_matrices[i].isSymmetric():
				if not self.scalar_product_matrices[i].isSymmetric(1e-12):
					raise SystemExit('Error: Supplied scalar product form is not symmetric')

		opts = fenics.PETScOptions
		opts.clear()
		opts.set('ksp_type', 'cg')
		opts.set('pc_type', 'hypre')
		opts.set('pc_hypre_type', 'boomeramg')
		opts.set('pc_hypre_boomeramg_strong_threshold', 0.7)
		opts.set('ksp_rtol', 1e-16)
		opts.set('ksp_atol', 1e-50)
		opts.set('ksp_max_it', 100)
		# opts.set('ksp_monitor_true_residual')

		self.ksps = []
		for i in range(self.control_dim):
			ksp = PETSc.KSP().create()
			ksp.setFromOptions()
			ksp.setOperators(self.riesz_projection_matrices[i])
			self.ksps.append(ksp)



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

		result = 0.0

		for i in range(self.control_dim):
			x = fenics.as_backend_type(a[i].vector()).vec()
			y = fenics.as_backend_type(b[i].vector()).vec()

			temp, _ = self.scalar_product_matrices[i].getVecs()
			self.scalar_product_matrices[i].mult(x, temp)
			result += temp.dot(y)

		return result



	def compute_active_sets(self):

		self.idx_active_lower = [(self.controls[j].vector()[:] <= self.control_constraints[j][0].vector()[:]).nonzero()[0] for j in range(self.control_dim)]
		self.idx_active_upper = [(self.controls[j].vector()[:] >= self.control_constraints[j][1].vector()[:]).nonzero()[0] for j in range(self.control_dim)]

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
			a[j].vector()[:] = np.maximum(self.control_constraints[j][0].vector()[:], np.minimum(self.control_constraints[j][1].vector()[:], a[j].vector()[:]))

		return a
	


	def compute_state_equations(self):
		"""Compute the weak form of the state equation for the use with fenics
		
		Returns
		-------
		
			Creates self.state_eq_forms

		"""

		if self.config.getboolean('StateEquation', 'is_linear'):
			self.state_eq_forms = [replace(self.state_forms[i], {self.states[i] : self.trial_functions_state[i],
																 self.adjoints[i] : self.test_functions_state[i]}) for i in range(self.state_dim)]

		else:
			self.state_eq_forms = [fenics.derivative(self.state_forms[i], self.adjoints[i], self.test_functions_state[i]) for i in range(self.state_dim)]

		if self.config.get('StateEquation', 'picard_iteration'):
			self.state_picard_forms = [fenics.derivative(self.state_forms[i], self.adjoints[i], self.test_functions_state[i]) for i in range(self.state_dim)]

		if self.config.getboolean('StateEquation', 'is_linear'):
			self.state_eq_forms_lhs = []
			self.state_eq_forms_rhs = []
			for i in range(self.state_dim):
				a, L = fenics.system(self.state_eq_forms[i])
				self.state_eq_forms_lhs.append(a)
				self.state_eq_forms_rhs.append(L)


	def compute_adjoint_equations(self):
		"""Computes the weak form of the adjoint equation for use with fenics
		
		Returns
		-------
		
			Creates self.adjoint_eq_form and self.bcs_ad, corresponding to homogenized BCs

		"""

		self.lagrangian_temp_forms = [replace(self.lagrangian.lagrangian_form, {self.adjoints[i] : self.trial_functions_adjoint[i]}) for i in range(self.state_dim)]

		if self.config.get('StateEquation', 'picard_iteration'):
			self.adjoint_picard_forms = [fenics.derivative(self.lagrangian.lagrangian_form, self.states[i], self.test_functions_adjoint[i]) for i in range(self.state_dim)]

		self.adjoint_eq_forms = [fenics.derivative(self.lagrangian_temp_forms[i], self.states[i], self.test_functions_adjoint[i]) for i in range(self.state_dim)]
		self.adjoint_eq_lhs = []
		self.adjoint_eq_rhs = []

		for i in range(self.state_dim):
			a, L = fenics.system(self.adjoint_eq_forms[i])
			self.adjoint_eq_lhs.append(a)
			self.adjoint_eq_rhs.append(L)

		if self.state_adjoint_equal_spaces:
			self.bcs_list_ad = [[fenics.DirichletBC(bc) for bc in self.bcs_list[i]] for i in range(self.state_dim)]
			[[bc.homogenize() for bc in self.bcs_list_ad[i]] for i in range(self.state_dim)]

		else:
			def get_subdx(V, idx, ls):
				if V.id()==idx:
					return ls
				if V.num_sub_spaces() > 1:
					for i in range(V.num_sub_spaces()):
						ans = get_subdx(V.sub(i), idx, ls + [i])
						if ans is not None:
							return ans
				else:
					return None

			self.bcs_list_ad = [[None for bc in range(len(self.bcs_list[i]))] for i in range(self.state_dim)]

			for i in range(self.state_dim):
				for j, bc in enumerate(self.bcs_list[i]):
					idx = bc.function_space().id()
					subdx = get_subdx(self.state_spaces[i], idx, ls=[])
					W = self.adjoint_spaces[i]
					for num in subdx:
						W = W.sub(num)
					shape = W.ufl_element().value_shape()
					try:
						if shape == ():
							self.bcs_list_ad[i][j] = fenics.DirichletBC(W, fenics.Constant(0), bc.domain_args[0], bc.domain_args[1])
						else:
							self.bcs_list_ad[i][j] = fenics.DirichletBC(W, fenics.Constant([0]*W.ufl_element().value_size()), bc.domain_args[0], bc.domain_args[1])
					except AttributeError:
						if shape == ():
							self.bcs_list_ad[i][j] = fenics.DirichletBC(W, fenics.Constant(0), bc.sub_domain)
						else:
							self.bcs_list_ad[i][j] = fenics.DirichletBC(W, fenics.Constant([0]*W.ufl_element().value_size()), bc.sub_domain)


	
	def compute_gradient_equations(self):
		"""Computes the variational form of the gradient equation, for the Riesz projection
		
		Returns
		-------
		
			Creates self.gradient_form_lhs and self.gradient_form_rhs

		"""

		self.gradient_forms_rhs = [fenics.derivative(self.lagrangian.lagrangian_form, self.controls[i], self.test_functions_control[i]) for i in range(self.control_dim)]

	
	
	def compute_newton_forms(self):
		"""Computes the needed forms for a truncated Newton method

		Returns
		-------


		"""

		self.sensitivity_eqs_temp = [replace(self.state_forms[i], {self.adjoints[i] : self.test_functions_state[i]}) for i in range(self.state_dim)]

		self.sensitivity_eqs_lhs = [fenics.derivative(self.sensitivity_eqs_temp[i], self.states[i], self.trial_functions_state[i]) for i in range(self.state_dim)]
		if self.config.get('StateEquation', 'picard_iteration'):
			self.sensitivity_eqs_picard = [fenics.derivative(self.sensitivity_eqs_temp[i], self.states[i], self.states_prime[i]) for i in range(self.state_dim)]

		# need to distinguish due to empty sum in case state_dim = 1
		if self.state_dim > 1:
			self.sensitivity_eqs_rhs = [- summ([fenics.derivative(self.sensitivity_eqs_temp[i], self.states[j], self.states_prime[j]) for j in range(self.state_dim) if j != i])
										- summ([fenics.derivative(self.sensitivity_eqs_temp[i], self.controls[j], self.test_directions[j]) for j in range(self.control_dim)])
										for i in range(self.state_dim)]
		else:
			self.sensitivity_eqs_rhs = [- summ([fenics.derivative(self.sensitivity_eqs_temp[i], self.controls[j], self.test_directions[j]) for j in range(self.control_dim)])
										for i in range(self.state_dim)]

		# Add a "rhs" for the nonlinear picard iteration
		if self.config.get('StateEquation', 'picard_iteration'):
			for i in range(self.state_dim):
				self.sensitivity_eqs_picard[i] -= self.sensitivity_eqs_rhs[i]


		self.L_y = [fenics.derivative(self.lagrangian.lagrangian_form, self.states[i], self.test_functions_state[i]) for i in range(self.state_dim)]
		self.L_u = [fenics.derivative(self.lagrangian.lagrangian_form, self.controls[i], self.test_functions_control[i]) for i in range(self.control_dim)]

		self.L_yy = [[fenics.derivative(self.L_y[i], self.states[j], self.states_prime[j]) for j in range(self.state_dim)] for i in range(self.state_dim)]
		self.L_yu = [[fenics.derivative(self.L_u[i], self.states[j], self.states_prime[j]) for j in range(self.state_dim)] for i in range(self.control_dim)]
		self.L_uy = [[fenics.derivative(self.L_y[i], self.controls[j], self.test_directions[j]) for j in range(self.control_dim)] for i in range(self.state_dim)]
		self.L_uu = [[fenics.derivative(self.L_u[i], self.controls[j], self.test_directions[j]) for j in range(self.control_dim)] for i in range(self.control_dim)]


		self.w_1 = [summ([self.L_yy[i][j] for j in range(self.state_dim)])
					+ summ([self.L_uy[i][j] for j in range(self.control_dim)]) for i in range(self.state_dim)]
		self.w_2 = [summ([self.L_yu[i][j] for j in range(self.state_dim)])
					+ summ([self.L_uu[i][j] for j in range(self.control_dim)]) for i in range(self.control_dim)]

		self.adjoint_sensitivity_eqs_diag_temp = [replace(self.state_forms[i], {self.adjoints[i] : self.trial_functions_adjoint[i]}) for i in range(self.state_dim)]

		mapping_dict = {self.adjoints[j]: self.adjoints_prime[j] for j in range(self.state_dim)}
		self.adjoint_sensitivity_eqs_all_temp = [replace(self.state_forms[i], mapping_dict) for i in range(self.state_dim)]

		self.adjoint_sensitivity_eqs_lhs = [fenics.derivative(self.adjoint_sensitivity_eqs_diag_temp[i], self.states[i], self.test_functions_adjoint[i]) for i in range(self.state_dim)]
		if self.config.get('StateEquation', 'picard_iteration'):
			self.adjoint_sensitivity_eqs_picard = [fenics.derivative(self.adjoint_sensitivity_eqs_all_temp[i], self.states[i], self.test_functions_adjoint[i]) for i in range(self.state_dim)]

		if self.state_dim > 1:
			for i in range(self.state_dim):
				self.w_1[i] -= summ([fenics.derivative(self.adjoint_sensitivity_eqs_all_temp[j], self.states[i], self.test_functions_adjoint[i]) for j in range(self.state_dim) if j != i])
		else:
			pass

		# Add "rhs" for nonlinear picard iteration form
		for i in range(self.state_dim):
			self.adjoint_sensitivity_eqs_picard[i] -= self.w_1[i]

		self.adjoint_sensitivity_eqs_rhs = [summ([fenics.derivative(self.adjoint_sensitivity_eqs_all_temp[j], self.controls[i], self.test_functions_control[i]) for j in range(self.state_dim)])
											for i in range(self.control_dim)]

		self.w_3 = [- self.adjoint_sensitivity_eqs_rhs[i] for i in range(self.control_dim)]

		# self.hessian_lhs = self.control_scalar_products
		self.hessian_rhs = [self.w_2[i] + self.w_3[i] for i in range(self.control_dim)]
