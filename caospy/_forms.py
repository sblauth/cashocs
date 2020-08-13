"""Private module forms of caospy.

This is used to carry out form manipulations
such as generating the UFL forms for the adjoint
system and for the Riesz gradient identificiation
problem.
"""

import fenics
from ufl import replace
from ufl.algorithms import expand_derivatives
from ufl.algorithms.estimate_degrees import estimate_total_polynomial_degree
from .utilities import summation
import numpy as np
from petsc4py import PETSc
from ._shape_optimization import Regularization
import json
import os
import warnings



class Lagrangian:
	"""Implementation of a Lagrangian

	This corresponds to the classical Lagrangian of a PDE constrained
	optimization problem, of the form

		L = J + e,

	where J is the cost functional and e the (weak) PDE constrained, tested by
	the adjoint variables. This is used to derive the adjoint and gradient
	equations needed for the optimization.

	See Also
	--------
	FormHandler : Derives the adjoint and gradient equations for optimal control problems
	ShapeFormHandler : Derives the adjoint equations and shape derivatives for
		shape optimization problems
	"""
	
	def __init__(self, state_forms, cost_functional_form):
		"""Initializes the Lagrangian
		
		Parameters
		----------
		state_forms : List[ufl.form.Form]
			the weak forms of the state equation, as implemented by the user,
			either directly as one single form or a list of forms
		cost_functional_form : ufl.form.Form
			the cost functional, as implemented by the user
		"""
		
		self.state_forms = state_forms
		self.cost_functional_form = cost_functional_form

		self.lagrangian_form = self.cost_functional_form + summation(self.state_forms)





class FormHandler:
	"""Class for UFL form manipulation

	This is used to symbolically (via UFL) derive the corresponding
	weak forms of the adjoint and gradient equations, that are later
	used in the solvers for the equations later on. These are needed
	as subroutines for the optimization (solution) algorithms.

	See Also
	--------
	ShapeFormHandler : Derives the adjoint equations and shape derivatives for
		shape optimization problems
	"""

	def __init__(self, lagrangian, bcs_list, states, controls, adjoints, config, riesz_scalar_products, control_constraints):
		"""Initializes the FormHandler class

		Parameters
		----------
		lagrangian : caospy._forms.Lagrangian
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

		if self.opt_algo == 'newton' or self.opt_algo == 'semi_smooth_newton' or \
				(self.opt_algo == 'pdas' and self.config.get('OptimizationRoutine', 'inner_pdas') == 'newton'):
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
		float
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



	def restrict_to_active_set(self, a, b):
		"""Restricts a function to the active set.

		Restricts a control type function a onto the active set,
		which is returned via the function b,  i.e., b is zero on the inactive set

		Parameters
		----------
		a : list[dolfin.function.function.Function]
			The first argument, to be projected onto the active set
		b : list[dolfin.function.function.Function]
			The second argument, which stores the result (is overwritten)

		Returns
		-------
		b : list[dolfin.function.function.Function]
			The result of the projection (overwrites input b)
		"""

		for j in range(self.control_dim):
			self.temp[j].vector()[:] = 0.0
			self.temp[j].vector()[self.idx_active[j]] = a[j].vector()[self.idx_active[j]]
			b[j].vector()[:] = self.temp[j].vector()[:]

		return b



	def restrict_to_inactive_set(self, a, b):
		"""Restricts a function to the inactive set

		Restricts a control type function a onto the inactive set,
		which is returned via the function b, i.e., b is zero on the active set

		Parameters
		----------
		a : list[dolfin.function.function.Function]
			The control-type function that is to be projected onto the inactive set
		b : list[dolfin.function.function.Function]
			The storage for the result of the projection (is overwritten)

		Returns
		-------
		b : list[dolfin.function.function.Function]
			The result of the projection of a onto the inactive set (overwrites input b)
		"""

		for j in range(self.control_dim):
			self.temp[j].vector()[:] = 0.0
			self.temp[j].vector()[self.idx_inactive[j]] = a[j].vector()[self.idx_inactive[j]]
			b[j].vector()[:] = self.temp[j].vector()[:]

		return b



	def project_to_admissible_set(self, a):
		"""Project a function to the set of admissible controls

		Projects a control type function a onto the set of admissible controls
		(given by the box constraints)

		Parameters
		----------
		a : list[dolfin.function.function.Function]
			The function which is to be projected onto the set of
			admissible controls (is overwritten)

		Returns
		-------
		a : list[dolfin.function.function.Function]
			The result of the projection (overwrites input a)
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
																 self.adjoints[i] : self.test_functions_state[i]})
								   for i in range(self.state_dim)]

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
		self.lagrangian_temp_forms = [replace(self.lagrangian.lagrangian_form,
											  {self.adjoints[i] : self.trial_functions_adjoint[i]})
									  for i in range(self.state_dim)]

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
		self.sensitivity_eqs_temp = [replace(self.state_forms[i],
											 {self.adjoints[i] : self.test_functions_state[i]})
									 for i in range(self.state_dim)]

		self.sensitivity_eqs_lhs = [fenics.derivative(self.sensitivity_eqs_temp[i], self.states[i], self.trial_functions_state[i]) for i in range(self.state_dim)]
		if self.config.get('StateEquation', 'picard_iteration', fallback=False):
			self.sensitivity_eqs_picard = [fenics.derivative(self.sensitivity_eqs_temp[i], self.states[i], self.states_prime[i]) for i in range(self.state_dim)]

		# Need to distinguish cases due to empty sum in case state_dim = 1
		if self.state_dim > 1:
			self.sensitivity_eqs_rhs = [- summation([fenics.derivative(self.sensitivity_eqs_temp[i], self.states[j], self.states_prime[j])
													 for j in range(self.state_dim) if j != i])
										- summation([fenics.derivative(self.sensitivity_eqs_temp[i], self.controls[j], self.test_directions[j])
													 for j in range(self.control_dim)])
										for i in range(self.state_dim)]
		else:
			self.sensitivity_eqs_rhs = [- summation([fenics.derivative(self.sensitivity_eqs_temp[i], self.controls[j], self.test_directions[j])
													 for j in range(self.control_dim)])
										for i in range(self.state_dim)]

		# Add the right-hand-side for the picard iteration
		if self.config.get('StateEquation', 'picard_iteration', fallback=False):
			for i in range(self.state_dim):
				self.sensitivity_eqs_picard[i] -= self.sensitivity_eqs_rhs[i]

		# Compute forms for the truncated Newton method
		self.L_y = [fenics.derivative(self.lagrangian.lagrangian_form, self.states[i], self.test_functions_state[i]) for i in range(self.state_dim)]
		self.L_u = [fenics.derivative(self.lagrangian.lagrangian_form, self.controls[i], self.test_functions_control[i]) for i in range(self.control_dim)]

		self.L_yy = [[fenics.derivative(self.L_y[i], self.states[j], self.states_prime[j])
					  for j in range(self.state_dim)] for i in range(self.state_dim)]
		self.L_yu = [[fenics.derivative(self.L_u[i], self.states[j], self.states_prime[j])
					  for j in range(self.state_dim)] for i in range(self.control_dim)]
		self.L_uy = [[fenics.derivative(self.L_y[i], self.controls[j], self.test_directions[j])
					  for j in range(self.control_dim)] for i in range(self.state_dim)]
		self.L_uu = [[fenics.derivative(self.L_u[i], self.controls[j], self.test_directions[j])
					  for j in range(self.control_dim)] for i in range(self.control_dim)]

		self.w_1 = [summation([self.L_yy[i][j] for j in range(self.state_dim)])
					+ summation([self.L_uy[i][j] for j in range(self.control_dim)]) for i in range(self.state_dim)]
		self.w_2 = [summation([self.L_yu[i][j] for j in range(self.state_dim)])
					+ summation([self.L_uu[i][j] for j in range(self.control_dim)]) for i in range(self.control_dim)]

		# Use replace -> derivative for faster computations
		self.adjoint_sensitivity_eqs_diag_temp = [replace(self.state_forms[i], {self.adjoints[i] : self.trial_functions_adjoint[i]})
												  for i in range(self.state_dim)]

		mapping_dict = {self.adjoints[j]: self.adjoints_prime[j] for j in range(self.state_dim)}
		self.adjoint_sensitivity_eqs_all_temp = [replace(self.state_forms[i], mapping_dict) for i in range(self.state_dim)]

		self.adjoint_sensitivity_eqs_lhs = [fenics.derivative(self.adjoint_sensitivity_eqs_diag_temp[i], self.states[i], self.test_functions_adjoint[i])
											for i in range(self.state_dim)]
		if self.config.get('StateEquation', 'picard_iteration', fallback=False):
			self.adjoint_sensitivity_eqs_picard = [fenics.derivative(self.adjoint_sensitivity_eqs_all_temp[i], self.states[i], self.test_functions_adjoint[i])
												   for i in range(self.state_dim)]

		# Need cases distinction due to empty sum for state_dim == 1
		if self.state_dim > 1:
			for i in range(self.state_dim):
				self.w_1[i] -= summation([fenics.derivative(self.adjoint_sensitivity_eqs_all_temp[j], self.states[i], self.test_functions_adjoint[i])
										  for j in range(self.state_dim) if j != i])
		else:
			pass

		# Add right-hand-side for picard iteration
		for i in range(self.state_dim):
			self.adjoint_sensitivity_eqs_picard[i] -= self.w_1[i]

		self.adjoint_sensitivity_eqs_rhs = [summation([fenics.derivative(self.adjoint_sensitivity_eqs_all_temp[j], self.controls[i], self.test_functions_control[i])
													   for j in range(self.state_dim)]) for i in range(self.control_dim)]

		self.w_3 = [- self.adjoint_sensitivity_eqs_rhs[i] for i in range(self.control_dim)]

		self.hessian_rhs = [self.w_2[i] + self.w_3[i] for i in range(self.control_dim)]





class ShapeFormHandler:
	"""Derives adjoint equations and shape derivatives

	This class is used analogously to the FormHandler class, but for
	shape optimization problems, where it is used to derive the adjoint equations
	and the shape derivatives.

	See Also
	--------
	FormHandler : Derives adjoint and gradient equations for optimal control problems
	"""

	def __init__(self, lagrangian, bcs_list, states, adjoints, boundaries, config):
		"""Initializes the ShapeFormHandler object

		Parameters
		----------
		lagrangian : caospy._forms.Lagrangian
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
			self.estimated_degree = np.maximum(estimate_total_polynomial_degree(self.riesz_scalar_product),
											   estimate_total_polynomial_degree(self.shape_derivative))
			self.assembler = fenics.SystemAssembler(self.riesz_scalar_product, self.shape_derivative, self.bcs_shape,
													form_compiler_parameters={'quadrature_degree' : self.estimated_degree})
		else:
			self.assembler = fenics.SystemAssembler(self.riesz_scalar_product, self.shape_derivative, self.bcs_shape)
		self.fe_scalar_product_matrix = fenics.PETScMatrix()
		self.fe_shape_derivative_vector = fenics.PETScVector()

		self.update_scalar_product()

		self.opt_algo = self.config.get('OptimizationRoutine', 'algorithm')

		if self.opt_algo == 'newton' or self.opt_algo == 'semi_smooth_newton' \
				or (self.opt_algo == 'pdas' and self.config.get('OptimizationRoutine', 'inner_pdas') == 'newton'):
			raise SystemExit('Second order methods are not implemented for shape optimization yet')

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
																 self.adjoints[i] : self.test_functions_state[i]})
								   for i in range(self.state_dim)]

		else:
			self.state_eq_forms = [fenics.derivative(self.state_forms[i], self.adjoints[i], self.test_functions_state[i])
								   for i in range(self.state_dim)]

		if self.config.get('StateEquation', 'picard_iteration', fallback = False):
			self.state_picard_forms = [fenics.derivative(self.state_forms[i], self.adjoints[i], self.test_functions_state[i])
									   for i in range(self.state_dim)]

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

		self.lagrangian_temp_forms = [replace(self.lagrangian.lagrangian_form,
											  {self.adjoints[i] : self.trial_functions_adjoint[i]})
									  for i in range(self.state_dim)]

		if self.config.get('StateEquation', 'picard_iteration', fallback = False):
			self.adjoint_picard_forms = [fenics.derivative(self.lagrangian.lagrangian_form, self.states[i], self.test_functions_adjoint[i])
										 for i in range(self.state_dim)]

		self.adjoint_eq_forms = [fenics.derivative(self.lagrangian_temp_forms[i], self.states[i], self.test_functions_adjoint[i])
								 for i in range(self.state_dim)]
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

		Note: this only works properly if differential operators only
		act on state and adjoint variables, else the results are incorrect

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

		if len(self.material_derivative_coeffs) > 0:
			warnings.warn('Shape derivative might be wrong, if differential operators act on variables other than states and adjoints.')

		for coeff in self.material_derivative_coeffs:
			# temp_space = fenics.FunctionSpace(self.mesh, coeff.ufl_element())
			# placeholder = fenics.Function(temp_space)
			# temp_form = fenics.derivative(self.lagrangian.lagrangian_form, coeff, placeholder)
			# material_derivative = replace(temp_form, {placeholder : fenics.dot(fenics.grad(coeff), self.test_vector_field)})

			material_derivative = fenics.derivative(self.lagrangian.lagrangian_form, coeff,
													fenics.dot(fenics.grad(coeff), self.test_vector_field))
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

		self.bcs_shape = [fenics.DirichletBC(self.deformation_space,
											 fenics.Constant([0]*self.deformation_space.ufl_element().value_size()), self.boundaries, i)
						  for i in self.shape_bdry_fix]



	def __compute_mu_elas(self):
		"""Computes the second lame parameter mu_elas, based on Siebenborn et al. [1]

		Returns
		-------
		None

		References
		----------
		[1] Schulz, V., Siebenborn, M. : Computational Comparison of Surface Metrics for
			PDE Constrained Shape Optimization, Computational Methods in Applied Mathematics,
			2016, Vol. 16, Iss. 3, https://doi.org/10.1515/cmam-2016-0009
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

			bcs = [fenics.DirichletBC(self.CG1, fenics.Constant(mu_fix), self.boundaries, i)
				   for i in self.shape_bdry_fix]
			bcs += [fenics.DirichletBC(self.CG1, fenics.Constant(mu_def), self.boundaries, i)
					for i in self.shape_bdry_def]

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
		"""Updates the linear elasticity equations to the current geometry

		Updates the left-hand-side of the linear elasticity equations
		(needed when the geometry changes)

		Returns
		-------
		None
		"""

		self.__compute_mu_elas()

		self.assembler.assemble(self.fe_scalar_product_matrix)
		self.fe_scalar_product_matrix.ident_zeros()
		self.scalar_product_matrix = fenics.as_backend_type(self.fe_scalar_product_matrix).mat()



	def scalar_product(self, a, b):
		"""Computes the scalar product between deformation functions

		Parameters
		----------
		a : dolfin.function.function.Function
			The first argument
		b : dolfin.function.function.Function
			The second argument

		Returns
		-------
		float
			The value of the scalar product
		"""

		x = fenics.as_backend_type(a.vector()).vec()
		y = fenics.as_backend_type(b.vector()).vec()

		temp, _ = self.scalar_product_matrix.getVecs()
		self.scalar_product_matrix.mult(x, temp)
		result = temp.dot(y)

		return result
