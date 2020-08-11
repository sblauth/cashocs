"""
Created on 24/02/2020, 08.45

@author: blauths
"""

import fenics
from ufl import replace
from ufl.algorithms import expand_derivatives
from ufl.algorithms.estimate_degrees import estimate_total_polynomial_degree
from .helpers import summ
import numpy as np
from petsc4py import PETSc
from .shape_optimization.regularization import Regularization
import json
import os



class Lagrangian:
	"""
	A class that represents a Lagrangian function, used to derive the adjoint and gradient equations

	Attributes
	----------
	state_forms : List[ufl.form.Form]
		List of weak forms of the state equations, in the order that they should be solved

	cost_functional_form : ufl.form.Form
		a UFL form for the cost functional

	lagrangian_form : ufl.form.Form
		a UFL form representing the abstract Lagrangian as sum of cost functional and state equations
	"""
	
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
	"""
	A class that is used to manipulate the UFL forms (of the Lagrangian) to derivate adjoint and gradient equations, etc., for an optimal control problem.


	Attributes
	----------
	lagrangian : adpack.forms.Lagrangian
		the Lagrangian of the optimization problem

	bcs_list : list[list[dolfin.fem.dirichletbc.DirichletBC]]
		a list of Dirichlet boundary conditions

	states : list[dolfin.function.function.Function]
		a list of the state variables

	controls : list[dolfin.function.function.Function]
		a list of the controls

	adjoints : list[dolfin.function.function.Function]
		a list of the adjoint variables

	config : configparser.ConfigParser
		the configparser object storing the problems config

	riesz_scalar_products : list[ufl.form.Form]
		the scalar products used for the controls (determines the Hilbert space)

	control_constraints : list[list[dolfin.function.function.Function]]
		the control (box) constraints for the problem

	cost_functional_form : ufl.form.Form
		the UFL form of the cost functional

	state_forms : list[ufl.form.Form]
		the UFL forms of the state equation(s)

	state_dim : int
		number of state variables

	control_dim : int
		number of control variables

	state_spaces : list[dolfin.function.functionspace.FunctionSpace]
		a list of the finite element spaces of the state variables

	control_spaces : list[dolfin.function.functionspace.FunctionSpace]
		a list of the finite element spaces of the control variables

	adjoint_spaces : list[dolfin.function.functionspace.FunctionSpace]
		a list of the finite element spaces of the adjoint variables

	mesh : dolfin.cpp.mesh.Mesh
		the finite element mesh of the geometry

	dx : ufl.measure.Measure
		volume measure associated to mesh

	states_prime : list[dolfin.function.function.Function]
		a list of functions, used to compute the state sensitivites

	adjoints_prime : list[dolfin.function.function.Function]
		a list of functions, used to compute the adjoint sensitivities

	hessian_actions : list[dolfin.function.function.Function]
		a list of functions, used to store the action of the Hessian

	test_directions : list[dolfin.function.function.Function]
		a list of functions, used to store the directions onto which the Hessian is acting

	trial_functions_state : list[dolfin.function.argument.Argument]
		a list of TrialFunctions for the state variables

	test_functions_state : list[dolfin.function.argument.Argument]
		a list of TestFunctions for the state variables

	trial_functions_adjoint : list[dolfin.function.argument.Argument]
		a list of TrialFunctions for the adjoint variables

	test_functions_adjoint : list[dolfin.function.argument.Argument]
		a list of TestFunctions for the adjoint variables

	trial_functions_control : list[dolfin.function.argument.Argument]
		a list of TrialFunctions for the control variables

	test_functions_control : list[dolfin.function.argument.Argument]
		a list of TestFunctions for the control variables

	temp : list[dolfin.function.function.Function]
		a list of functions for temporary storage purposes

	opt_algo : str
		a string representing the chosen optimization algorithm

	scalar_product_matrices : list[petsc4py.PETSc.Mat]
		PETSc matrix for the control scalar product

	riesz_projection_matrices : list[petsc4py.PETSc.Mat]
		PETSc matrix for the identification of the gradient via the Riesz projection

	ksps : list[petsc4py.PETSc.KSP]
		PETSc Krylov solver object used for the solution of the Riesz problems

	idx_active_upper : list[list[int]]
		list indicating the indices where the control variables are at the upper box constraint

	idx_active_lower : list[list[int]]
		list indicating the indices where the control variables are at the lower box constraint

	idx_active : list[list[int]]
		list indicating the indices where the box constraints are active

	idx_inactive : list[list[int]]
		list indicating the indices where the box constraints are not active

	state_eq_forms : list[ufl.form.Form]
		Weak forms of the state equations (for treatment with fenics)

	state_picard_forms : list[ufl.form.Form]
		Weak forms of the state equations used if the state system shall be solved via a Picard iteration

	state_eq_forms_lhs : list[ufl.form.Form]
		left-hand-side of the state equations (in case they are linear)

	state_eq_forms_rhs : list[ufl.form.Form]
		right-hand-side of the state equations (in case they are linear)

	lagrangian_temp_forms : list[ufl.form.Form]
		temporary forms of the Lagrangian used for faster manipulation

	adjoint_picard_forms : list[ufl.form.Form]
		weak forms of the adjoint equations used if the state system shall be solved via a Picard iteration

	adjoint_eq_forms : list[ufl.form.Form]
		weak form of the adjoint equations

	adjoint_eq_lhs : list[ufl.form.Form]
		left-hand-side of the adjoint equations

	adjoint_eq_rhs : list[ufl.form.Form]
		right-hand-side of the adjoint equations

	bcs_list_ad : list[list[dolfin.fem.dirichletbc.DirichletBC]]
		a list of boundary conditions for the adjoint variables
		(it is assumed that for every DirichletBC of the state variables the corresponding adjoint DirichletBC is homogeneous)

	gradient_forms_rhs : list[ufl.form.Form]
		right-hand-side for the gradient Riesz problems

	sensitivity_eqs_temp : list[ufl.form.Form]
		temporary UFL forms for the sensitivity equations (for faster computation)

	sensitivity_eqs_lhs : list[ufl.form.Form]
		left-hand-side of the state (forward) sensitivity equations

	sensitivity_eqs_picard : list[ufl.form.Form]
		left-hand-side of the state sensitivity equations if they shall be solved via a Picard iteration

	sensitivity_eqs_rhs : list[ufl.form.Form]
		right-hand-side of the state sensitivity equations

	L_y : ufl.form.Form
		form that represents the partial derivative of the Lagrangian w.r.t. the state variables

	L_u : ufl.form.Form
		form that represents the partial derivative of the Lagrangian w.r.t. the control variables

	L_yy : ufl.form.Form
		form that represents the partial derivative of L_y w.r.t. the state variables

	L_yu : ufl.form.Form
		form that represents the partial derivative of L_y w.r.t. the control variables

	L_uy : ufl.form.Form
		form that represents the partial derivative of L_u w.r.t. the state variables

	L_uu : ufl.form.Form
		form that represents the partial derivative of L_u w.r.t. the control variables

	w_1 : ufl.form.Form
		temporary form used for the truncated Newton method

	w_2 : ufl.form.Form
		temporary form used for the truncated Newton method

	adjoint_sensitivity_eqs_diag_temp : list[ufl.form.Form]
		temporary forms used for the computation of the adjoint sensitivity equations

	adjoint_sensitivity_eqs_all_temp : list[ufl.form.Form]
		temporary forms used for the computation of the adjoint sensitvity equations

	adjoint_sensitivity_eqs_lhs : list[ufl.form.Form]
		left-hand-side of the adjoint sensitivity equations

	adjoint_sensitivity_eqs_picard : list[ufl.form.Form]
		weak form of the adjoint sensitivity equations if they shall be solved via a Picard iteration

	adjoint_sensitivity_eqs_rhs : list[ufl.form.Form]
		right-hand-side of the adjoint sensitivity equations

	w_3 : ufl.form.Form
		temporary form used for the truncated Newton method

	hessian_rhs : list[ufl.form.Form]
		right-hand-side for the Riesz problem used to identify the action of the Hessian onto test_directions
	"""

	def __init__(self, lagrangian, bcs_list, states, controls, adjoints, config, riesz_scalar_products, control_constraints):
		"""Initializes the FormHandler class

		Parameters
		----------
		lagrangian : adpack.forms.Lagrangian
			the lagrangian corresponding to the optimization problem

		bcs_list : list[list[dolfin.fem.dirichletbc.DirichletBC]]
			the list of DirichletBCs for the state equation

		states : list[dolfin.function.function.Function]
			the function that acts as the state variable

		controls : list[dolfin.function.function.Function]
			the function that acts as the control variable

		adjoints : list[dolfin.function.function.Function]
			the function that acts as the adjoint variable

		config : configparser.ConfigParser
			the configparser object of the config file

		riesz_scalar_products : list[ufl.form.Form]
			UFL forms of the scalar products for the control variables

		control_constraints : list[list[dolfin.function.function.Function]]
			the control constraints of the problem
		"""

		# Initialize the attributes from the arguments
		self.lagrangian = lagrangian
		self.bcs_list = bcs_list
		self.states = states
		self.controls = controls
		self.adjoints = adjoints
		self.config = config
		self.riesz_scalar_products = riesz_scalar_products
		self.control_constraints = control_constraints

		# Further initializations
		self.cost_functional_form = self.lagrangian.cost_functional_form
		self.state_forms = self.lagrangian.state_forms
		
		self.state_dim = len(self.states)
		self.control_dim = len(self.controls)
		
		self.state_spaces = [x.function_space() for x in self.states]
		self.control_spaces = [x.function_space() for x in self.controls]
		self.adjoint_spaces = [x.function_space() for x in self.adjoints]

		# Test if state_spaces coincide with adjoint_spaces
		if self.state_spaces == self.adjoint_spaces:
			self.state_adjoint_equal_spaces = True
		else:
			self.state_adjoint_equal_spaces = False

		self.mesh = self.state_spaces[0].mesh()
		self.dx = fenics.Measure('dx', self.mesh)

		# Define the necessary functions
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

		# Compute the necessary equations
		self.__compute_state_equations()
		self.__compute_adjoint_equations()
		self.__compute_gradient_equations()

		self.opt_algo = self.config.get('OptimizationRoutine', 'algorithm')

		if self.opt_algo == 'newton' or self.opt_algo == 'semi_smooth_newton' or (self.opt_algo == 'pdas' and self.config.get('OptimizationRoutine', 'inner_pdas') == 'newton'):
			self.__compute_newton_forms()

		# Initialize the scalar products
		fe_scalar_product_matrices = [fenics.assemble(self.riesz_scalar_products[i], keep_diagonal=True) for i in range(self.control_dim)]
		self.scalar_product_matrices = [fenics.as_backend_type(fe_scalar_product_matrices[i]).mat() for i in range(self.control_dim)]
		[fe_scalar_product_matrices[i].ident_zeros() for i in range(self.control_dim)]
		self.riesz_projection_matrices = [fenics.as_backend_type(fe_scalar_product_matrices[i]).mat() for i in range(self.control_dim)]

		# Test for symmetry of the scalar products
		for i in range(self.control_dim):
			if not self.scalar_product_matrices[i].isSymmetric():
				if not self.scalar_product_matrices[i].isSymmetric(1e-12):
					raise SystemExit('Error: Supplied scalar product form is not symmetric')

		# Initialize the PETSc Krylov solver for the Riesz projection problems
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
		"""Computes the scalar product between control type functions a and b

		Parameters
		----------
		a : list[dolfin.function.function.Function]
			The first argument
		b : list[dolfin.function.function.Function]
			The second argument

		Returns
		-------
		result : float
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
		"""Computes the indices corresponding to active and inactive sets.

		Returns
		-------
		None
		"""

		self.idx_active_lower = [(self.controls[j].vector()[:] <= self.control_constraints[j][0].vector()[:]).nonzero()[0] for j in range(self.control_dim)]
		self.idx_active_upper = [(self.controls[j].vector()[:] >= self.control_constraints[j][1].vector()[:]).nonzero()[0] for j in range(self.control_dim)]

		self.idx_active = [np.concatenate((self.idx_active_lower[j], self.idx_active_upper[j])) for j in range(self.control_dim)]
		[self.idx_active[j].sort() for j in range(self.control_dim)]

		self.idx_inactive = [np.setdiff1d(np.arange(self.control_spaces[j].dim()), self.idx_active[j] ) for j in range(self.control_dim)]

		return None



	def project_active(self, a, b):
		"""Projects a control type function a onto the active set, which is returned via the function b,  i.e., b is zero on the inactive set

		Parameters
		----------
		a : list[dolfin.function.function.Function]
			The first argument, to be projected onto the active set

		b : list[dolfin.function.function.Function]
			The second argument, which stores the result (is overwritten)

		Returns
		-------
		b : list[dolfin.function.function.Function]
			The result of the projection
		"""

		for j in range(self.control_dim):
			self.temp[j].vector()[:] = 0.0
			self.temp[j].vector()[self.idx_active[j]] = a[j].vector()[self.idx_active[j]]
			b[j].vector()[:] = self.temp[j].vector()[:]

		return b



	def project_inactive(self, a, b):
		"""Projects a control type function a onto the inactive set, which is returned via the function b, i.e., b is zero on the active set

		Parameters
		----------
		a : list[dolfin.function.function.Function]
			The control-type function that is to be projected onto the inactive set

		b : list[dolfin.function.function.Function]
			The storage for the result of the projection (is overwritten)

		Returns
		-------
		b : list[dolfin.function.function.Function]
			The result of the projection of a onto the inactive set
		"""

		for j in range(self.control_dim):
			self.temp[j].vector()[:] = 0.0
			self.temp[j].vector()[self.idx_inactive[j]] = a[j].vector()[self.idx_inactive[j]]
			b[j].vector()[:] = self.temp[j].vector()[:]

		return b



	def project(self, a):
		"""Projects a control type function a onto the set of admissible controls (given by the box constraints)

		Parameters
		----------
		a : list[dolfin.function.function.Function]
			The function which is to be projected onto the set of admissible controls (is overwritten)

		Returns
		-------
		a : list[dolfin.function.function.Function]
			The result of the projection
		"""

		for j in range(self.control_dim):
			a[j].vector()[:] = np.maximum(self.control_constraints[j][0].vector()[:], np.minimum(self.control_constraints[j][1].vector()[:], a[j].vector()[:]))

		return a
	


	def __compute_state_equations(self):
		"""Calculates the weak form of the state equation for the use with fenics
		
		Returns
		-------
		None
		"""

		if self.config.getboolean('StateEquation', 'is_linear', fallback=False):
			self.state_eq_forms = [replace(self.state_forms[i], {self.states[i] : self.trial_functions_state[i],
																 self.adjoints[i] : self.test_functions_state[i]}) for i in range(self.state_dim)]

		else:
			self.state_eq_forms = [fenics.derivative(self.state_forms[i], self.adjoints[i], self.test_functions_state[i]) for i in range(self.state_dim)]

		if self.config.get('StateEquation', 'picard_iteration', fallback=False):
			self.state_picard_forms = [fenics.derivative(self.state_forms[i], self.adjoints[i], self.test_functions_state[i]) for i in range(self.state_dim)]

		if self.config.getboolean('StateEquation', 'is_linear', fallback=False):
			self.state_eq_forms_lhs = []
			self.state_eq_forms_rhs = []
			for i in range(self.state_dim):
				a, L = fenics.system(self.state_eq_forms[i])
				self.state_eq_forms_lhs.append(a)
				self.state_eq_forms_rhs.append(L)



	def __compute_adjoint_equations(self):
		"""Calculates the weak form of the adjoint equation for use with fenics
		
		Returns
		-------
		None
		"""

		# Use replace -> derivative to speed up computations
		self.lagrangian_temp_forms = [replace(self.lagrangian.lagrangian_form, {self.adjoints[i] : self.trial_functions_adjoint[i]}) for i in range(self.state_dim)]

		if self.config.get('StateEquation', 'picard_iteration', fallback=False):
			self.adjoint_picard_forms = [fenics.derivative(self.lagrangian.lagrangian_form, self.states[i], self.test_functions_adjoint[i]) for i in range(self.state_dim)]

		self.adjoint_eq_forms = [fenics.derivative(self.lagrangian_temp_forms[i], self.states[i], self.test_functions_adjoint[i]) for i in range(self.state_dim)]
		self.adjoint_eq_lhs = []
		self.adjoint_eq_rhs = []

		for i in range(self.state_dim):
			a, L = fenics.system(self.adjoint_eq_forms[i])
			self.adjoint_eq_lhs.append(a)
			self.adjoint_eq_rhs.append(L)

		# Compute the  adjoint boundary conditions
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


	
	def __compute_gradient_equations(self):
		"""Calculates the variational form of the gradient equation, for the Riesz projection
		
		Returns
		-------
		None
		"""

		self.gradient_forms_rhs = [fenics.derivative(self.lagrangian.lagrangian_form, self.controls[i], self.test_functions_control[i]) for i in range(self.control_dim)]

	
	
	def __compute_newton_forms(self):
		"""Calculates the needed forms for the truncated Newton method

		Returns
		-------
		None
		"""

		# Use replace -> derivative to speed up the computations
		self.sensitivity_eqs_temp = [replace(self.state_forms[i], {self.adjoints[i] : self.test_functions_state[i]}) for i in range(self.state_dim)]

		self.sensitivity_eqs_lhs = [fenics.derivative(self.sensitivity_eqs_temp[i], self.states[i], self.trial_functions_state[i]) for i in range(self.state_dim)]
		if self.config.get('StateEquation', 'picard_iteration', fallback=False):
			self.sensitivity_eqs_picard = [fenics.derivative(self.sensitivity_eqs_temp[i], self.states[i], self.states_prime[i]) for i in range(self.state_dim)]

		# Need to distinguish cases due to empty sum in case state_dim = 1
		if self.state_dim > 1:
			self.sensitivity_eqs_rhs = [- summ([fenics.derivative(self.sensitivity_eqs_temp[i], self.states[j], self.states_prime[j]) for j in range(self.state_dim) if j != i])
										- summ([fenics.derivative(self.sensitivity_eqs_temp[i], self.controls[j], self.test_directions[j]) for j in range(self.control_dim)])
										for i in range(self.state_dim)]
		else:
			self.sensitivity_eqs_rhs = [- summ([fenics.derivative(self.sensitivity_eqs_temp[i], self.controls[j], self.test_directions[j]) for j in range(self.control_dim)])
										for i in range(self.state_dim)]

		# Add the right-hand-side for the picard iteration
		if self.config.get('StateEquation', 'picard_iteration', fallback=False):
			for i in range(self.state_dim):
				self.sensitivity_eqs_picard[i] -= self.sensitivity_eqs_rhs[i]

		# Compute forms for the truncated Newton method
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

		# Use replace -> derivative for faster computations
		self.adjoint_sensitivity_eqs_diag_temp = [replace(self.state_forms[i], {self.adjoints[i] : self.trial_functions_adjoint[i]}) for i in range(self.state_dim)]

		mapping_dict = {self.adjoints[j]: self.adjoints_prime[j] for j in range(self.state_dim)}
		self.adjoint_sensitivity_eqs_all_temp = [replace(self.state_forms[i], mapping_dict) for i in range(self.state_dim)]

		self.adjoint_sensitivity_eqs_lhs = [fenics.derivative(self.adjoint_sensitivity_eqs_diag_temp[i], self.states[i], self.test_functions_adjoint[i]) for i in range(self.state_dim)]
		if self.config.get('StateEquation', 'picard_iteration', fallback=False):
			self.adjoint_sensitivity_eqs_picard = [fenics.derivative(self.adjoint_sensitivity_eqs_all_temp[i], self.states[i], self.test_functions_adjoint[i]) for i in range(self.state_dim)]

		# Need cases distinction due to empty sum for state_dim == 1
		if self.state_dim > 1:
			for i in range(self.state_dim):
				self.w_1[i] -= summ([fenics.derivative(self.adjoint_sensitivity_eqs_all_temp[j], self.states[i], self.test_functions_adjoint[i]) for j in range(self.state_dim) if j != i])
		else:
			pass

		# Add right-hand-side for picard iteration
		for i in range(self.state_dim):
			self.adjoint_sensitivity_eqs_picard[i] -= self.w_1[i]

		self.adjoint_sensitivity_eqs_rhs = [summ([fenics.derivative(self.adjoint_sensitivity_eqs_all_temp[j], self.controls[i], self.test_functions_control[i]) for j in range(self.state_dim)])
											for i in range(self.control_dim)]

		self.w_3 = [- self.adjoint_sensitivity_eqs_rhs[i] for i in range(self.control_dim)]

		self.hessian_rhs = [self.w_2[i] + self.w_3[i] for i in range(self.control_dim)]





class ShapeFormHandler:
	"""A class that is used to manipulate the UFL forms (of the Lagrangian) to derivate adjoint equations
	and the shape derivative, etc., for a shape optimization problem.


	Attributes
	----------
	lagrangian : adpack.forms.Lagrangian
		the Lagrangian of the optimization problem

	bcs_list : list[list[dolfin.fem.dirichletbc.DirichletBC]]
		a list of Dirichlet boundary conditions

	states : list[dolfin.function.function.Function]
		a list of the state variables

	adjoints : list[dolfin.function.function.Function]
		a list of the adjoint variables

	boundaries : dolfin.cpp.mesh.MeshFunctionSizet
		a MeshFunction for the boundary markers

	config : configparser.ConfigParser
		the configparser object storing the problems config

	degree_estimation : bool
		boolean variable used to indicate whether there should be a custom degree estimation of the shape derivative (may be necessary)

	cost_functional_form : ufl.form.Form
		the UFL form of the cost functional

	state_forms : list[ufl.form.Form]
		the UFL forms of the state equation(s)

	state_dim : int
		number of state variables

	state_spaces : list[dolfin.function.functionspace.FunctionSpace]
		a list of the finite element spaces of the state variables

	adjoint_spaces : list[dolfin.function.functionspace.FunctionSpace]
		a list of the finite element spaces of the adjoint variables

	mesh : dolfin.cpp.mesh.Mesh
		the finite element mesh of the geometry

	dx : ufl.measure.Measure
		volume measure associated to mesh

	deformation_space : dolfin.function.functionspace.FunctionSpace
		Vector CG 1 FunctionSpace used for mesh deformations

	test_vector_field : dolfin.function.function.Function
		TestFunction of the deformation space, used for the shape derivative

	trial_functions_state : list[dolfin.function.argument.Argument]
		a list of TrialFunctions for the state variables

	test_functions_state : list[dolfin.function.argument.Argument]
		a list of TestFunctions for the state variables

	trial_functions_adjoint : list[dolfin.function.argument.Argument]
		a list of TrialFunctions for the adjoint variables

	test_functions_adjoint : list[dolfin.function.argument.Argument]
		a list of TestFunctions for the adjoint variables

	regularization : adpack.shape_optimization.regularization.Regularization
		class implementing regularizations

	estimated_degree : int
		estimated quadrature degree for the shape derivative

	assembler : dolfin.fem.assembling.SystemAssembler
		SystemAssembler object for the Riesz problem to identify the shape gradient

	fe_scalar_product_matrix : dolfin.cpp.la.PETScMatrix
		fenics representation of the Riesz scalar product matrix

	fe_shape_derivative_vector : dolfin.cpp.la.PETScVector
		fenics representation of the shape derivative assembled into a vector

	opt_algo : str
		a string representing the chosen optimization algorithm

	ksp : petsc4py.PETSc.KSP
		PETSc Krylov solver object for the shape gradient problem

	do_remesh : bool
		boolean flag for en- and disabling remeshing

	remesh_counter : int
		Counter for the number of remeshes performed

	gmsh_file : str
		path to the gmsh mesh file (used for remeshing)

	mesh_directory : str
		path to the folder containing the gmsh_file

	remesh_directory : str
		path to the remesh directory (given by "mesh_directory/remesh")

	config_save_file : str
		path to the save file of the original config file

	remesh_geo_file : str
		path to the .geo file for remeshing

	gmsh_file_init : str
		path to the initial gmsh file

	state_eq_forms : list[ufl.form.Form]
		Weak forms of the state equations (for treatment with fenics)

	state_picard_forms : list[ufl.form.Form]
		Weak forms of the state equations used if the state system shall be solved via a Picard iteration

	state_eq_forms_lhs : list[ufl.form.Form]
		left-hand-side of the state equations (in case they are linear)

	state_eq_forms_rhs : list[ufl.form.Form]
		right-hand-side of the state equations (in case they are linear)

	lagrangian_temp_forms : list[ufl.form.Form]
		temporary forms of the Lagrangian used for faster manipulation

	adjoint_picard_forms : list[ufl.form.Form]
		weak forms of the adjoint equations used if the state system shall be solved via a Picard iteration

	adjoint_eq_forms : list[ufl.form.Form]
		weak form of the adjoint equations

	adjoint_eq_lhs : list[ufl.form.Form]
		left-hand-side of the adjoint equations

	adjoint_eq_rhs : list[ufl.form.Form]
		right-hand-side of the adjoint equations

	bcs_list_ad : list[list[dolfin.fem.dirichletbc.DirichletBC]]
		a list of boundary conditions for the adjoint variables
		(it is assumed that for every DirichletBC of the state variables the corresponding adjoint DirichletBC is homogeneous)

	shape_derivative : ufl.form.Form
		UFL form of the shape derivative

	state_adjoint_ids : list[int]
		list of fenics id's for the state and adjoint variables (these do not need additional pull-backs)

	material_derivative_coeffs : list[ufl.coefficient.Coefficient]
		list of Coefficients that are eligible for the pull-back

	shape_bdry_def : list[int]
		list of boundary indices corresponding to the deformable boundary

	shape_bdry_fix : list[in]
		list of boundary indices corresponding to the fixed boundary

	CG1 : dolfin.function.functionspace.FunctionSpace
		scalar CG1 FunctionSpace used for computing the elasticity for the shape gradient problem

	DG0 : dolfin.function.functionspace.FunctionSpace
		scalar DG0 FunctionSpace for the computation of the shape gradient problem

	mu_lame : dolfin.function.function.Function
		CG1 Function representing the second lame parameter

	lambda_lame : float
		the first lame parameter

	damping_factor : float
		a damping factor for the linear elasticity equations (required to be positive in case the entire boundary is variable)

	volumes : ufl.coefficient.Coefficient
		DG0 Function representing the (weighted / scaled) cell volumes of the triangulation, or Constant(1), depending on whether inhomogeneous elasticity is enabled

	riesz_scalar_product : ufl.form.Form
		UFL form of the elasticity equations, used for projecting the shape derivative to the shape gradient

	bcs_shape : list[dolfin.fem.dirichletbc.DirichletBC]
		list of (homogeneous) Dirichlet boundary conditions for the shape gradient (for all boundaries in shape_bdry_fix)

	scalar_product_matrix : petsc4py.PETSc.Mat
		PETSc matrix corresponding to the elasticity equations
	"""

	def __init__(self, lagrangian, bcs_list, states, adjoints, boundaries, config):
		"""Initializes the ShapeFormHandler object

		Parameters
		----------
		lagrangian : adpack.forms.Lagrangian
			The Lagrangian corresponding to the shape optimization problem

		bcs_list : list[list[dolfin.fem.dirichletbc.DirichletBC]]
			list of boundary conditions for the state variables

		states : list[dolfin.function.function.Function]
			list of state variables

		adjoints : list[dolfin.function.function.Function]
			list of adjoint variables

		boundaries : dolfin.cpp.mesh.MeshFunctionSizet
			a MeshFunction for the boundary markers

		config : configparser.ConfigParser
			the configparser object storing the problems config
		"""

		self.lagrangian = lagrangian
		self.bcs_list = bcs_list
		self.states = states
		self.adjoints = adjoints
		self.boundaries = boundaries
		self.config = config

		self.degree_estimation = config.getboolean('ShapeGradient', 'degree_estimation', fallback=False)

		self.cost_functional_form = self.lagrangian.cost_functional_form
		self.state_forms = self.lagrangian.state_forms

		self.state_dim = len(self.states)

		self.state_spaces = [x.function_space() for x in self.states]
		self.adjoint_spaces = [x.function_space() for x in self.adjoints]

		# Test if state_spaces are identical to adjoint_spaces
		if self.state_spaces == self.adjoint_spaces:
			self.state_adjoint_equal_spaces = True
		else:
			self.state_adjoint_equal_spaces = False

		self.mesh = self.state_spaces[0].mesh()
		self.dx = fenics.Measure('dx', self.mesh)

		self.deformation_space = fenics.VectorFunctionSpace(self.mesh, 'CG', 1)
		self.test_vector_field = fenics.TestFunction(self.deformation_space)

		self.trial_functions_state = [fenics.TrialFunction(V) for V in self.state_spaces]
		self.test_functions_state = [fenics.TestFunction(V) for V in self.state_spaces]

		self.trial_functions_adjoint = [fenics.TrialFunction(V) for V in self.adjoint_spaces]
		self.test_functions_adjoint = [fenics.TestFunction(V) for V in self.adjoint_spaces]

		self.regularization = Regularization(self)

		# Calculate the necessary UFL forms
		self.__compute_state_equations()
		self.__compute_adjoint_equations()
		self.__compute_shape_derivative()
		self.__compute_shape_gradient_forms()

		if self.degree_estimation:
			self.estimated_degree = np.maximum(estimate_total_polynomial_degree(self.riesz_scalar_product), estimate_total_polynomial_degree(self.shape_derivative))
			self.assembler = fenics.SystemAssembler(self.riesz_scalar_product, self.shape_derivative, self.bcs_shape, form_compiler_parameters={'quadrature_degree' : self.estimated_degree})
		else:
			self.assembler = fenics.SystemAssembler(self.riesz_scalar_product, self.shape_derivative, self.bcs_shape)
		self.fe_scalar_product_matrix = fenics.PETScMatrix()
		self.fe_shape_derivative_vector = fenics.PETScVector()

		self.update_scalar_product()

		self.opt_algo = self.config.get('OptimizationRoutine', 'algorithm')

		if self.opt_algo == 'newton' or self.opt_algo == 'semi_smooth_newton' or (self.opt_algo == 'pdas' and self.config.get('OptimizationRoutine', 'inner_pdas') == 'newton'):
			raise SystemExit('Second order methods are not implemented yet')

		# Generate the Krylov solver for the shape gradient problem
		opts = fenics.PETScOptions
		opts.clear()
		# Options for a direct solver (debugging)
		# opts.set('ksp_type', 'preonly')
		# opts.set('pc_type', 'lu')
		# opts.set('pc_factor_mat_solver_type', 'mumps')
		# opts.set('mat_mumps_icntl_24', 1)

		opts.set('ksp_type', 'cg')
		opts.set('pc_type', 'hypre')
		opts.set('pc_hypre_type', 'boomeramg')
		opts.set('pc_hypre_boomeramg_strong_threshold', 0.7)
		opts.set('ksp_rtol', 1e-20)
		opts.set('ksp_atol', 1e-50)
		opts.set('ksp_max_it', 250)
		# opts.set('ksp_monitor_true_residual')

		self.ksp = PETSc.KSP().create()
		self.ksp.setFromOptions()

		# Remeshing initializations
		self.do_remesh = self.config.getboolean('Mesh', 'remesh', fallback=False)
		self.remesh_counter = self.config.getint('Mesh', 'remesh_counter', fallback=0)

		if self.do_remesh:
			self.gmsh_file = self.config.get('Mesh', 'gmsh_file')
			if self.remesh_counter == 0:
				self.config.set('Mesh', 'original_gmsh_file', self.gmsh_file)
			assert self.gmsh_file[-4:] == '.msh', 'Not a valid gmsh file'

			self.mesh_directory = os.path.split(self.config.get('Mesh', 'original_gmsh_file'))[0]
			self.remesh_directory = self.mesh_directory + '/remesh'
			if not os.path.exists(self.remesh_directory):
				os.mkdir(self.remesh_directory)
			self.config_save_file = self.remesh_directory + '/save_config.ini'
			self.remesh_geo_file = self.remesh_directory + '/remesh.geo'

		# create a copy of the initial config and mesh file
		if self.do_remesh and self.remesh_counter == 0:
			self.gmsh_file_init = self.remesh_directory + '/mesh_' + str(self.remesh_counter) + '.msh'
			copy_mesh = 'cp ' + self.gmsh_file + ' ' + self.gmsh_file_init
			os.system(copy_mesh)
			self.gmsh_file = self.gmsh_file_init

			with open(self.config_save_file, 'w') as file:
				self.config.write(file)



	def __compute_state_equations(self):
		"""Compute the weak form of the state equation for the use with fenics

		Returns
		-------
		None
		"""

		if self.config.getboolean('StateEquation', 'is_linear', fallback = False):
			self.state_eq_forms = [replace(self.state_forms[i], {self.states[i] : self.trial_functions_state[i],
																 self.adjoints[i] : self.test_functions_state[i]}) for i in range(self.state_dim)]

		else:
			self.state_eq_forms = [fenics.derivative(self.state_forms[i], self.adjoints[i], self.test_functions_state[i]) for i in range(self.state_dim)]

		if self.config.get('StateEquation', 'picard_iteration', fallback = False):
			self.state_picard_forms = [fenics.derivative(self.state_forms[i], self.adjoints[i], self.test_functions_state[i]) for i in range(self.state_dim)]

		if self.config.getboolean('StateEquation', 'is_linear', fallback = False):
			self.state_eq_forms_lhs = []
			self.state_eq_forms_rhs = []
			for i in range(self.state_dim):
				a, L = fenics.system(self.state_eq_forms[i])
				self.state_eq_forms_lhs.append(a)
				self.state_eq_forms_rhs.append(L)



	def __compute_adjoint_equations(self):
		"""Computes the weak form of the adjoint equation for use with fenics

		Returns
		-------
		None
		"""

		self.lagrangian_temp_forms = [replace(self.lagrangian.lagrangian_form, {self.adjoints[i] : self.trial_functions_adjoint[i]}) for i in range(self.state_dim)]

		if self.config.get('StateEquation', 'picard_iteration', fallback = False):
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



	def __compute_shape_derivative(self):
		"""Computes the shape derivative.
		Note: this only works properly if differential operators only act on state and adjoint variables, else the results are incorrect

		Returns
		-------
		None
		"""

		# Shape derivative of Lagrangian w/o regularization and pull-backs
		self.shape_derivative = fenics.derivative(self.lagrangian.lagrangian_form, fenics.SpatialCoordinate(self.mesh), self.test_vector_field)

		# Add pull-backs
		self.state_adjoint_ids = [coeff.id() for coeff in self.states] + [coeff.id() for coeff in self.adjoints]

		self.material_derivative_coeffs = []
		for coeff in self.lagrangian.lagrangian_form.coefficients():
			if coeff.id() in self.state_adjoint_ids:
				pass
			else:
				if not coeff.ufl_element().family() == 'Real':
					self.material_derivative_coeffs.append(coeff)

		for coeff in self.material_derivative_coeffs:
			# temp_space = fenics.FunctionSpace(self.mesh, coeff.ufl_element())
			# placeholder = fenics.Function(temp_space)
			# temp_form = fenics.derivative(self.lagrangian.lagrangian_form, coeff, placeholder)
			# material_derivative = replace(temp_form, {placeholder : fenics.dot(fenics.grad(coeff), self.test_vector_field)})

			material_derivative = fenics.derivative(self.lagrangian.lagrangian_form, coeff, fenics.dot(fenics.grad(coeff), self.test_vector_field))
			material_derivative = expand_derivatives(material_derivative)

			self.shape_derivative += material_derivative

		# Add regularization
		self.shape_derivative += self.regularization.compute_shape_derivative()



	def __compute_shape_gradient_forms(self):
		"""Calculates the necessary left-hand-sides for the shape gradient problem

		Returns
		-------
		None
		"""

		self.shape_bdry_def = json.loads(self.config.get('ShapeGradient', 'shape_bdry_def'))
		assert type(self.shape_bdry_def) == list, 'ShapeGradient.shape_bdry_def has to be a list'
		self.shape_bdry_fix = json.loads(self.config.get('ShapeGradient', 'shape_bdry_fix'))
		assert type(self.shape_bdry_fix) == list, 'ShapeGradient.shape_bdry_fix has to be a list'

		self.CG1 = fenics.FunctionSpace(self.mesh, 'CG', 1)
		self.DG0 = fenics.FunctionSpace(self.mesh, 'DG', 0)

		self.mu_lame = fenics.Function(self.CG1)
		self.lambda_lame = self.config.getfloat('ShapeGradient', 'lambda_lame')
		self.damping_factor = self.config.getfloat('ShapeGradient', 'damping_factor')

		if self.config.getboolean('ShapeGradient', 'inhomogeneous', fallback = False):
			self.volumes = fenics.project(fenics.CellVolume(self.mesh), self.DG0)
			vol_max = np.max(np.abs(self.volumes.vector()[:]))
			self.volumes.vector()[:] /= vol_max

		else:
			self.volumes = fenics.Constant(1.0)

		def eps(u):
			return fenics.Constant(0.5)*(fenics.grad(u) + fenics.grad(u).T)

		trial = fenics.TrialFunction(self.deformation_space)
		test = fenics.TestFunction(self.deformation_space)

		self.riesz_scalar_product = fenics.Constant(2)*self.mu_lame/self.volumes*fenics.inner(eps(trial), eps(test))*self.dx \
									+ fenics.Constant(self.lambda_lame)/self.volumes*fenics.div(trial)*fenics.div(test)*self.dx \
									+ fenics.Constant(self.damping_factor)/self.volumes*fenics.inner(trial, test)*self.dx

		self.bcs_shape = [fenics.DirichletBC(self.deformation_space, fenics.Constant([0]*self.deformation_space.ufl_element().value_size()), self.boundaries, i) for i in self.shape_bdry_fix]



	def __compute_mu_elas(self):
		"""Computes the second lame parameter mu_elas, based on Siebenborn et al.

		Returns
		-------
		None
		"""

		mu_def = self.config.getfloat('ShapeGradient', 'mu_def')
		mu_fix = self.config.getfloat('ShapeGradient', 'mu_fix')

		if np.abs(mu_def - mu_fix)/mu_fix > 1e-2:

			dx = self.dx
			opts = fenics.PETScOptions
			opts.clear()

			# opts.set('ksp_type', 'preonly')
			# opts.set('pc_type', 'lu')
			# opts.set('pc_factor_mat_solver_type', 'mumps')
			# opts.set('mat_mumps_icntl_24', 1)

			opts.set('ksp_type', 'cg')
			opts.set('pc_type', 'hypre')
			opts.set('pc_hypre_type', 'boomeramg')
			opts.set('ksp_rtol', 1e-16)
			opts.set('ksp_atol', 1e-50)
			opts.set('ksp_max_it', 100)

			phi = fenics.TrialFunction(self.CG1)
			psi = fenics.TestFunction(self.CG1)

			a = fenics.inner(fenics.grad(phi), fenics.grad(psi))*dx
			L = fenics.Constant(0.0)*psi*dx

			bcs = [fenics.DirichletBC(self.CG1, fenics.Constant(mu_fix), self.boundaries, i) for i in self.shape_bdry_fix]
			bcs += [fenics.DirichletBC(self.CG1, fenics.Constant(mu_def), self.boundaries, i) for i in self.shape_bdry_def]

			A, b = fenics.assemble_system(a, L, bcs)
			A = fenics.as_backend_type(A).mat()
			b = fenics.as_backend_type(b).vec()
			x, _ = A.getVecs()

			ksp = PETSc.KSP().create()
			ksp.setFromOptions()
			ksp.setOperators(A)
			ksp.setUp()
			ksp.solve(b, x)
			if ksp.getConvergedReason() < 0:
				raise SystemExit('Krylov solver did not converge. Reason: ' + str(ksp.getConvergedReason()))

			# TODO: Add possibility to switch this behavior (sqrt for 3D)
			self.mu_lame.vector()[:] = x[:]
			# self.mu_lame.vector()[:] = np.sqrt(x[:])

		else:
			self.mu_lame.vector()[:] = mu_fix



	def update_scalar_product(self):
		"""Updates the left-hand-side of the linear elasticity equations (needed when the geometry changes)

		Returns
		-------
		None
		"""

		self.__compute_mu_elas()

		self.assembler.assemble(self.fe_scalar_product_matrix)
		self.fe_scalar_product_matrix.ident_zeros()
		self.scalar_product_matrix = fenics.as_backend_type(self.fe_scalar_product_matrix).mat()



	def scalar_product(self, a, b):
		"""Computes the scalar product between deformation functions (needed for NCG and BFGS methods)

		Parameters
		----------
		a : dolfin.function.function.Function
			The first argument
		b : dolfin.function.function.Function
			The second argument

		Returns
		-------
		result : float
			The value of the scalar product
		"""

		x = fenics.as_backend_type(a.vector()).vec()
		y = fenics.as_backend_type(b.vector()).vec()

		temp, _ = self.scalar_product_matrix.getVecs()
		self.scalar_product_matrix.mult(x, temp)
		result = temp.dot(y)

		return result
