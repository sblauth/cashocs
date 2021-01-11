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

"""Private module forms of CASHOCS.

This is used to carry out form manipulations such as generating the UFL
 forms for the adjoint system and for the Riesz gradient identificiation
problems.
"""

import json

import fenics
import numpy as np
from petsc4py import PETSc
from ufl import replace
from ufl.algorithms import expand_derivatives
from ufl.algorithms.estimate_degrees import estimate_total_polynomial_degree
from ufl.log import UFLException

from ._exceptions import ConfigError, InputError, CashocsException
from ._loggers import warning
from ._shape_optimization import Regularization
from .utils import (_assemble_petsc_system, _optimization_algorithm_configuration, _setup_petsc_options, _solve_linear_problem, create_bcs_list, summation)



class Lagrangian:
	r"""Implementation of a Lagrangian.

	This corresponds to the classical Lagrangian of a PDE constrained
	optimization problem, of the form

	.. math:: L = J + e,

	where J is the cost functional and e the (weak) PDE constrained, tested by
	the adjoint variables. This is used to derive the adjoint and gradient
	equations needed for the optimization.

	See Also
	--------
	FormHandler : Derives necessary adjoint and gradient / shape derivative equations.
	"""

	def __init__(self, state_forms, cost_functional_form):
		"""Initializes the Lagrangian.

		Parameters
		----------
		state_forms : list[ufl.form.Form]
			The weak forms of the state equation, as implemented by the user.
		cost_functional_form : ufl.form.Form
			The cost functional, as implemented by the user.
		"""

		self.state_forms = state_forms
		self.cost_functional_form = cost_functional_form

		self.lagrangian_form = self.cost_functional_form + summation(self.state_forms)





class FormHandler:
	"""Parent class for UFL form manipulation.

	This is subclassed by specific form handlers for either
	optimal control or shape optimization. The base class is
	used to determine common objects and to derive the UFL forms
	for the state and adjoint systems.

	See Also
	--------
	ControlFormHandler : FormHandler for optimal control problems
	ShapeFormHandler : FormHandler for shape optimization problems
	"""

	def __init__(self, lagrangian, bcs_list, states, adjoints, config, ksp_options, adjoint_ksp_options):
		"""Initializes the form handler.

		Parameters
		----------
		lagrangian : cashocs._forms.Lagrangian
			The lagrangian of the optimization problem.
		bcs_list : list[list[dolfin.fem.dirichletbc.DirichletBC]]
			The list of DirichletBCs for the state equation.
		states : list[dolfin.function.function.Function]
			The function that acts as the state variable.
		adjoints : list[dolfin.function.function.Function]
			The function that acts as the adjoint variable.
		config : configparser.ConfigParser
			The configparser object of the config file.
		ksp_options : list[list[list[str]]]
			The list of command line options for the KSP for the
			state systems.
		adjoint_ksp_options : list[list[list[str]]]
			The list of command line options for the KSP for the
			adjoint systems.
		"""

		# Initialize the attributes from the arguments
		self.lagrangian = lagrangian
		self.bcs_list = bcs_list
		self.states = states
		self.adjoints = adjoints
		self.config = config
		self.state_ksp_options = ksp_options
		self.adjoint_ksp_options = adjoint_ksp_options

		# Further initializations
		self.cost_functional_form = self.lagrangian.cost_functional_form
		self.state_forms = self.lagrangian.state_forms

		self.state_dim = len(self.states)

		self.state_spaces = [x.function_space() for x in self.states]
		self.adjoint_spaces = [x.function_space() for x in self.adjoints]

		# Test if state_spaces coincide with adjoint_spaces
		if self.state_spaces == self.adjoint_spaces:
			self.state_adjoint_equal_spaces = True
		else:
			self.state_adjoint_equal_spaces = False

		self.mesh = self.state_spaces[0].mesh()
		self.dx = fenics.Measure('dx', self.mesh)

		self.trial_functions_state = [fenics.TrialFunction(V) for V in self.state_spaces]
		self.test_functions_state = [fenics.TestFunction(V) for V in self.state_spaces]

		self.trial_functions_adjoint = [fenics.TrialFunction(V) for V in self.adjoint_spaces]
		self.test_functions_adjoint = [fenics.TestFunction(V) for V in self.adjoint_spaces]

		self.state_is_linear = self.config.getboolean('StateSystem', 'is_linear', fallback = False)
		self.state_is_picard = self.config.getboolean('StateSystem', 'picard_iteration', fallback=False)
		self.opt_algo = _optimization_algorithm_configuration(config)

		if self.opt_algo == 'pdas':
			self.inner_pdas = self.config.get('AlgoPDAS', 'inner_pdas')

		self.__compute_state_equations()
		self.__compute_adjoint_equations()



	def __compute_state_equations(self):
		"""Calculates the weak form of the state equation for the use with fenics.

		Returns
		-------
		None
		"""

		if self.state_is_linear:
			self.state_eq_forms = [replace(self.state_forms[i], {self.states[i] : self.trial_functions_state[i],
																 self.adjoints[i] : self.test_functions_state[i]})
								   for i in range(self.state_dim)]

		else:
			self.state_eq_forms = [fenics.derivative(self.state_forms[i], self.adjoints[i], self.test_functions_state[i])
								   for i in range(self.state_dim)]

		if self.state_is_picard:
			self.state_picard_forms = [fenics.derivative(self.state_forms[i], self.adjoints[i], self.test_functions_state[i])
									   for i in range(self.state_dim)]

		if self.state_is_linear:
			self.state_eq_forms_lhs = []
			self.state_eq_forms_rhs = []
			for i in range(self.state_dim):
				try:
					a, L = fenics.system(self.state_eq_forms[i])
				except UFLException:
					raise CashocsException('The state system could not be transferred to a linear system.\n'
										   'Perhaps you specified that the system is linear, allthough it is not.\n'
										   'In your config, in the StateEquation section, try using is_linear = False.')
				self.state_eq_forms_lhs.append(a)
				if L.empty():
					zero_form = fenics.inner(fenics.Constant(np.zeros(self.test_functions_state[i].ufl_shape)), self.test_functions_state[i])*self.dx
					self.state_eq_forms_rhs.append(zero_form)
				else:
					self.state_eq_forms_rhs.append(L)



	def __compute_adjoint_equations(self):
		"""Calculates the weak form of the adjoint equation for use with fenics.

		Returns
		-------
		None
		"""

		# Use replace -> derivative to speed up computations
		self.lagrangian_temp_forms = [replace(self.lagrangian.lagrangian_form,
											  {self.adjoints[i] : self.trial_functions_adjoint[i]})
									  for i in range(self.state_dim)]

		if self.state_is_picard:
			self.adjoint_picard_forms = [fenics.derivative(self.lagrangian.lagrangian_form, self.states[i], self.test_functions_adjoint[i]) for i in range(self.state_dim)]

		self.adjoint_eq_forms = [fenics.derivative(self.lagrangian_temp_forms[i], self.states[i], self.test_functions_adjoint[i]) for i in range(self.state_dim)]
		self.adjoint_eq_lhs = []
		self.adjoint_eq_rhs = []

		for i in range(self.state_dim):
			a, L = fenics.system(self.adjoint_eq_forms[i])
			self.adjoint_eq_lhs.append(a)
			if L.empty():
				zero_form = fenics.inner(fenics.Constant(np.zeros(self.test_functions_adjoint[i].ufl_shape)), self.test_functions_adjoint[i])*self.dx
				self.adjoint_eq_rhs.append(zero_form)
			else:
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

			self.bcs_list_ad = [[1 for bc in range(len(self.bcs_list[i]))] for i in range(self.state_dim)]

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





class ControlFormHandler(FormHandler):
	"""Class for UFL form manipulation for optimal control problems.

	This is used to symbolically derive the corresponding weak forms of the
	adjoint and gradient equations (via UFL) , that are later used in the
	solvers for the equations later on. These are needed as subroutines for
	 the optimization (solution) algorithms.

	See Also
	--------
	ShapeFormHandler : Derives the adjoint equations and shape derivatives for shape optimization problems
	"""

	def __init__(self, lagrangian, bcs_list, states, controls, adjoints, config, riesz_scalar_products, control_constraints, ksp_options, adjoint_ksp_options, require_control_constraints):
		"""Initializes the ControlFormHandler class.

		Parameters
		----------
		lagrangian : cashocs._forms.Lagrangian
			The lagrangian corresponding to the optimization problem.
		bcs_list : list[list[dolfin.fem.dirichletbc.DirichletBC]]
			The list of DirichletBCs for the state equation.
		states : list[dolfin.function.function.Function]
			The function that acts as the state variable.
		controls : list[dolfin.function.function.Function]
			The function that acts as the control variable.
		adjoints : list[dolfin.function.function.Function]
			The function that acts as the adjoint variable.
		config : configparser.ConfigParser
			The configparser object of the config file.
		riesz_scalar_products : list[ufl.form.Form]
			The UFL forms of the scalar products for the control variables.
		control_constraints : list[list[dolfin.function.function.Function]]
			The control constraints of the problem.
		ksp_options : list[list[list[str]]]
			The list of command line options for the KSP for the
			state systems.
		adjoint_ksp_options : list[list[list[str]]]
			The list of command line options for the KSP for the
			adjoint systems.
		require_control_constraints : list[bool]
			A list of boolean flags that indicates, whether the i-th control
			has actual control constraints present.
		"""

		FormHandler.__init__(self, lagrangian, bcs_list, states, adjoints, config, ksp_options, adjoint_ksp_options)

		# Initialize the attributes from the arguments
		self.controls = controls
		self.riesz_scalar_products = riesz_scalar_products
		self.control_constraints = control_constraints
		self.require_control_constraints = require_control_constraints

		self.control_dim = len(self.controls)
		self.control_spaces = [x.function_space() for x in self.controls]

		# Define the necessary functions
		self.states_prime = [fenics.Function(V) for V in self.state_spaces]
		self.adjoints_prime = [fenics.Function(V) for V in self.adjoint_spaces]

		self.test_directions = [fenics.Function(V) for V in self.control_spaces]

		self.trial_functions_control = [fenics.TrialFunction(V) for V in self.control_spaces]
		self.test_functions_control = [fenics.TestFunction(V) for V in self.control_spaces]

		self.temp = [fenics.Function(V) for V in self.control_spaces]

		# Compute the necessary equations
		self.__compute_gradient_equations()

		if self.opt_algo == 'newton' or \
				(self.opt_algo == 'pdas' and self.inner_pdas == 'newton'):
			self.__compute_newton_forms()

		# Initialize the scalar products
		fenics_scalar_product_matrices = [fenics.assemble(self.riesz_scalar_products[i], keep_diagonal=True) for i in range(self.control_dim)]
		self.scalar_products_matrices = [fenics.as_backend_type(fenics_scalar_product_matrices[i]).mat() for i in range(self.control_dim)]

		copy_scalar_product_matrices = [fenics_scalar_product_matrices[i].copy() for i in range(self.control_dim)]
		[copy_scalar_product_matrices[i].ident_zeros() for i in range(self.control_dim)]
		self.riesz_projection_matrices = [fenics.as_backend_type(copy_scalar_product_matrices[i]).mat() for i in range(self.control_dim)]

		# Test for symmetry of the scalar products
		for i in range(self.control_dim):
			if not self.riesz_projection_matrices[i].isSymmetric():
				if not self.riesz_projection_matrices[i].isSymmetric(1e-15):
					if not (self.riesz_projection_matrices[i] - self.riesz_projection_matrices[i].copy().transpose()).norm() / self.riesz_projection_matrices[i].norm() < 1e-15:
						raise InputError('cashocs._forms.ControlFormHandler', 'riesz_scalar_products', 'Supplied scalar product form is not symmetric.')



	def scalar_product(self, a, b):
		"""Computes the scalar product between control type functions a and b.

		Parameters
		----------
		a : list[dolfin.function.function.Function]
			The first argument.
		b : list[dolfin.function.function.Function]
			The second argument.

		Returns
		-------
		float
			The value of the scalar product.
		"""

		result = 0.0

		for i in range(self.control_dim):
			x = fenics.as_backend_type(a[i].vector()).vec()
			y = fenics.as_backend_type(b[i].vector()).vec()

			temp, _ = self.scalar_products_matrices[i].getVecs()
			self.scalar_products_matrices[i].mult(x, temp)
			result += temp.dot(y)

		return result



	def compute_active_sets(self):
		"""Computes the indices corresponding to active and inactive sets.

		Returns
		-------
		None
		"""

		self.idx_active_lower = []
		self.idx_active_upper = []
		self.idx_active = []
		self.idx_inactive = []

		for j in range(self.control_dim):

			if self.require_control_constraints[j]:
				self.idx_active_lower.append((self.controls[j].vector()[:] <= self.control_constraints[j][0].vector()[:]).nonzero()[0])
				self.idx_active_upper.append((self.controls[j].vector()[:] >= self.control_constraints[j][1].vector()[:]).nonzero()[0])
			else:
				self.idx_active_lower.append([])
				self.idx_active_upper.append([])

			temp_active = np.concatenate((self.idx_active_lower[j], self.idx_active_upper[j]))
			temp_active.sort()
			self.idx_active.append(temp_active)
			self.idx_inactive.append(np.setdiff1d(np.arange(self.control_spaces[j].dim()), self.idx_active[j]))



	def restrict_to_active_set(self, a, b):
		"""Restricts a function to the active set.

		Restricts a control type function a onto the active set,
		which is returned via the function b,  i.e., b is zero on the inactive set.

		Parameters
		----------
		a : list[dolfin.function.function.Function]
			The first argument, to be projected onto the active set.
		b : list[dolfin.function.function.Function]
			The second argument, which stores the result (is overwritten).

		Returns
		-------
		b : list[dolfin.function.function.Function]
			The result of the projection (overwrites input b).
		"""

		for j in range(self.control_dim):
			if self.require_control_constraints[j]:
				self.temp[j].vector()[:] = 0.0
				self.temp[j].vector()[self.idx_active[j]] = a[j].vector()[self.idx_active[j]]
				b[j].vector()[:] = self.temp[j].vector()[:]

			else:
				b[j].vector()[:] = 0.0

		return b



	def restrict_to_lower_active_set(self, a, b):

		for j in range(self.control_dim):
			if self.require_control_constraints[j]:
				self.temp[j].vector()[:] = 0.0
				self.temp[j].vector()[self.idx_active_lower[j]] = a[j].vector()[self.idx_active_lower[j]]
				b[j].vector()[:] = self.temp[j].vector()[:]

			else:
				b[j].vector()[:] = 0.0

		return b



	def restrict_to_upper_active_set(self, a, b):

		for j in range(self.control_dim):
			if self.require_control_constraints[j]:
				self.temp[j].vector()[:] = 0.0
				self.temp[j].vector()[self.idx_active_upper[j]] = a[j].vector()[self.idx_active_upper[j]]
				b[j].vector()[:] = self.temp[j].vector()[:]

			else:
				b[j].vector()[:] = 0.0

		return b



	def restrict_to_inactive_set(self, a, b):
		"""Restricts a function to the inactive set.

		Restricts a control type function a onto the inactive set,
		which is returned via the function b, i.e., b is zero on the active set.

		Parameters
		----------
		a : list[dolfin.function.function.Function]
			The control-type function that is to be projected onto the inactive set.
		b : list[dolfin.function.function.Function]
			The storage for the result of the projection (is overwritten).

		Returns
		-------
		b : list[dolfin.function.function.Function]
			The result of the projection of a onto the inactive set (overwrites input b).
		"""

		for j in range(self.control_dim):
			if self.require_control_constraints[j]:
				self.temp[j].vector()[:] = 0.0
				self.temp[j].vector()[self.idx_inactive[j]] = a[j].vector()[self.idx_inactive[j]]
				b[j].vector()[:] = self.temp[j].vector()[:]

			else:
				b[j].vector()[:] = a[j].vector()[:]

		return b



	def project_to_admissible_set(self, a):
		"""Project a function to the set of admissible controls.

		Projects a control type function a onto the set of admissible controls
		(given by the box constraints).

		Parameters
		----------
		a : list[dolfin.function.function.Function]
			The function which is to be projected onto the set of admissible
			controls (is overwritten)

		Returns
		-------
		a : list[dolfin.function.function.Function]
			The result of the projection (overwrites input a)
		"""

		for j in range(self.control_dim):
			if self.require_control_constraints[j]:
				a[j].vector()[:] = np.maximum(self.control_constraints[j][0].vector()[:], np.minimum(self.control_constraints[j][1].vector()[:], a[j].vector()[:]))

		return a



	def __compute_gradient_equations(self):
		"""Calculates the variational form of the gradient equation, for the Riesz projection.

		Returns
		-------
		None
		"""

		self.gradient_forms_rhs = [fenics.derivative(self.lagrangian.lagrangian_form, self.controls[i], self.test_functions_control[i]) for i in range(self.control_dim)]



	def __compute_newton_forms(self):
		"""Calculates the needed forms for the truncated Newton method.

		Returns
		-------
		None
		"""

		# Use replace -> derivative to speed up the computations
		self.sensitivity_eqs_temp = [replace(self.state_forms[i],
											 {self.adjoints[i] : self.test_functions_state[i]})
									 for i in range(self.state_dim)]

		self.sensitivity_eqs_lhs = [fenics.derivative(self.sensitivity_eqs_temp[i], self.states[i], self.trial_functions_state[i]) for i in range(self.state_dim)]
		if self.state_is_picard:
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
		if self.state_is_picard:
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
		if self.state_is_picard:
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
		if self.state_is_picard:
			for i in range(self.state_dim):
				self.adjoint_sensitivity_eqs_picard[i] -= self.w_1[i]

		self.adjoint_sensitivity_eqs_rhs = [summation([fenics.derivative(self.adjoint_sensitivity_eqs_all_temp[j], self.controls[i], self.test_functions_control[i])
													   for j in range(self.state_dim)]) for i in range(self.control_dim)]

		self.w_3 = [- self.adjoint_sensitivity_eqs_rhs[i] for i in range(self.control_dim)]

		self.hessian_rhs = [self.w_2[i] + self.w_3[i] for i in range(self.control_dim)]





class ShapeFormHandler(FormHandler):
	"""Derives adjoint equations and shape derivatives.

	This class is used analogously to the ControlFormHandler class, but for
	shape optimization problems, where it is used to derive the adjoint equations
	and the shape derivatives.

	See Also
	--------
	ControlFormHandler : Derives adjoint and gradient equations for optimal control problems
	"""

	def __init__(self, lagrangian, bcs_list, states, adjoints, boundaries, config, ksp_options, adjoint_ksp_options, shape_scalar_product=None, deformation_space=None):
		"""Initializes the ShapeFormHandler object.

		Parameters
		----------
		lagrangian : cashocs._forms.Lagrangian
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
		ksp_options : list[list[list[str]]]
			The list of command line options for the KSP for the
			state systems.
		adjoint_ksp_options : list[list[list[str]]]
			The list of command line options for the KSP for the
			adjoint systems.
		shape_scalar_product : ufl.form.Form
			The weak form of the scalar product used to determine the
			shape gradient.
		"""

		FormHandler.__init__(self, lagrangian, bcs_list, states, adjoints, config, ksp_options, adjoint_ksp_options)

		self.boundaries = boundaries
		self.shape_scalar_product = shape_scalar_product

		self.degree_estimation = self.config.getboolean('ShapeGradient', 'degree_estimation', fallback=False)
		self.use_pull_back = self.config.getboolean('ShapeGradient', 'use_pull_back', fallback=True)
		
		if deformation_space is None:
			self.deformation_space = fenics.VectorFunctionSpace(self.mesh, 'CG', 1)
		else:
			self.deformation_space = deformation_space
			
		self.test_vector_field = fenics.TestFunction(self.deformation_space)

		self.regularization = Regularization(self)

		# Calculate the necessary UFL forms
		self.inhomogeneous_mu = False
		self.__compute_shape_derivative()
		self.__compute_shape_gradient_forms()
		self.__setup_mu_computation()
		
		if self.degree_estimation:
			self.estimated_degree = np.maximum(estimate_total_polynomial_degree(self.riesz_scalar_product),
											   estimate_total_polynomial_degree(self.shape_derivative))
			self.assembler = fenics.SystemAssembler(self.riesz_scalar_product, self.shape_derivative, self.bcs_shape,
													form_compiler_parameters={'quadrature_degree' : self.estimated_degree})
		else:
			try:
				self.assembler = fenics.SystemAssembler(self.riesz_scalar_product, self.shape_derivative, self.bcs_shape)
			except (AssertionError, ValueError):
				self.estimated_degree = np.maximum(estimate_total_polynomial_degree(self.riesz_scalar_product),
											   estimate_total_polynomial_degree(self.shape_derivative))
				self.assembler = fenics.SystemAssembler(self.riesz_scalar_product, self.shape_derivative, self.bcs_shape,
													form_compiler_parameters={'quadrature_degree' : self.estimated_degree})

		self.assembler.keep_diagonal = True
		self.fe_scalar_product_matrix = fenics.PETScMatrix()
		self.fe_shape_derivative_vector = fenics.PETScVector()

		self.update_scalar_product()
		
		# test for symmetry
		if not self.scalar_product_matrix.isSymmetric():
			if not self.scalar_product_matrix.isSymmetric(1e-15):
				if not (self.scalar_product_matrix - self.scalar_product_matrix.copy().transpose()).norm() / self.scalar_product_matrix.norm() < 1e-15:
					raise InputError('cashocs._forms.ShapeFormHandler', 'shape_scalar_product', 'Supplied scalar product form is not symmetric.')

		if self.opt_algo == 'newton' \
				or (self.opt_algo == 'pdas' and self.inner_pdas == 'newton'):
			raise NotImplementedError('Second order methods are not implemented for shape optimization yet')



	def __compute_shape_derivative(self):
		"""Computes the shape derivative.

		Returns
		-------
		None

		Notes
		-----
		This only works properly if differential operators only
		act on state and adjoint variables, else the results are incorrect.
		A corresponding warning whenever this could be the case is issued.
		"""

		# Shape derivative of Lagrangian w/o regularization and pull-backs
		self.shape_derivative = fenics.derivative(self.lagrangian.lagrangian_form, fenics.SpatialCoordinate(self.mesh), self.test_vector_field)

		# Add pull-backs
		if self.use_pull_back:
			self.state_adjoint_ids = [coeff.id() for coeff in self.states] + [coeff.id() for coeff in self.adjoints]
	
			self.material_derivative_coeffs = []
			for coeff in self.lagrangian.lagrangian_form.coefficients():
				if coeff.id() in self.state_adjoint_ids:
					pass
				else:
					if not (coeff.ufl_element().family() == 'Real'):
						self.material_derivative_coeffs.append(coeff)
	
			if len(self.material_derivative_coeffs) > 0:
				warning('Shape derivative might be wrong, if differential operators act on variables other than states and adjoints. \n'
							  'You can check for correctness of the shape derivative with cashocs.verification.shape_gradient_test\n')
	
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
		"""Calculates the necessary left-hand-sides for the shape gradient problem.

		Returns
		-------
		None
		"""

		self.shape_bdry_def = json.loads(self.config.get('ShapeGradient', 'shape_bdry_def', fallback='[]'))
		if not type(self.shape_bdry_def) == list:
			raise ConfigError('ShapeGradient', 'shape_bdry_def', 'The input has to be a list.')
		# if not len(self.shape_bdry_def) > 0:
		# 	raise ConfigError('ShapeGradient', 'shape_bdry_def','The input must not be empty.')

		self.shape_bdry_fix = json.loads(self.config.get('ShapeGradient', 'shape_bdry_fix', fallback='[]'))
		if not type(self.shape_bdry_def) == list:
			raise ConfigError('ShapeGradient', 'shape_bdry_fix', 'The input has to be a list.')

		self.shape_bdry_fix_x = json.loads(self.config.get('ShapeGradient', 'shape_bdry_fix_x', fallback='[]'))
		if not type(self.shape_bdry_fix_x) == list:
			raise ConfigError('ShapeGradient', 'shape_bdry_fix_x', 'The input has to be a list.')

		self.shape_bdry_fix_y = json.loads(self.config.get('ShapeGradient', 'shape_bdry_fix_y', fallback='[]'))
		if not type(self.shape_bdry_fix_y) == list:
			raise ConfigError('ShapeGradient', 'shape_bdry_fix_y', 'The input has to be a list.')

		self.shape_bdry_fix_z = json.loads(self.config.get('ShapeGradient', 'shape_bdry_fix_z', fallback='[]'))
		if not type(self.shape_bdry_fix_z) == list:
			raise ConfigError('ShapeGradient', 'shape_bdry_fix_z', 'The input has to be a list.')
		
		self.bcs_shape = create_bcs_list(self.deformation_space, fenics.Constant([0]*self.deformation_space.ufl_element().value_size()), self.boundaries, self.shape_bdry_fix)
		self.bcs_shape += create_bcs_list(self.deformation_space.sub(0), fenics.Constant(0.0), self.boundaries, self.shape_bdry_fix_x)
		self.bcs_shape += create_bcs_list(self.deformation_space.sub(1), fenics.Constant(0.0), self.boundaries, self.shape_bdry_fix_y)
		if self.deformation_space.num_sub_spaces() == 3:
			self.bcs_shape += create_bcs_list(self.deformation_space.sub(2), fenics.Constant(0.0), self.boundaries, self.shape_bdry_fix_z)
		
		
		self.CG1 = fenics.FunctionSpace(self.mesh, 'CG', 1)
		self.DG0 = fenics.FunctionSpace(self.mesh, 'DG', 0)
		
		if self.shape_scalar_product is None:
			# Use the default linear elasticity approach
	
			self.mu_lame = fenics.Function(self.CG1)
			self.lambda_lame = self.config.getfloat('ShapeGradient', 'lambda_lame', fallback=0.0)
			self.damping_factor = self.config.getfloat('ShapeGradient', 'damping_factor', fallback=0.0)
	
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
			
		else:
			# Use the scalar product supplied by the user
			
			self.riesz_scalar_product = self.shape_scalar_product
		



	def __setup_mu_computation(self):
		self.mu_def = self.config.getfloat('ShapeGradient', 'mu_def', fallback=1.0)
		self.mu_fix = self.config.getfloat('ShapeGradient', 'mu_fix', fallback=1.0)

		if np.abs(self.mu_def - self.mu_fix)/self.mu_fix > 1e-2:

			self.inhomogeneous_mu = True

			# dx = self.dx

			self.options_mu = [
				['ksp_type', 'cg'],
				['pc_type', 'hypre'],
				['pc_hypre_type', 'boomeramg'],
				['ksp_rtol', 1e-16],
				['ksp_atol', 1e-50],
				['ksp_max_it', 100]
			]
			self.ksp_mu = PETSc.KSP().create()
			_setup_petsc_options([self.ksp_mu], [self.options_mu])

			phi = fenics.TrialFunction(self.CG1)
			psi = fenics.TestFunction(self.CG1)

			self.a_mu = fenics.inner(fenics.grad(phi), fenics.grad(psi))*self.dx
			self.L_mu = fenics.Constant(0.0)*psi*self.dx

			self.bcs_mu = [fenics.DirichletBC(self.CG1, fenics.Constant(self.mu_fix), self.boundaries, i)
				   for i in self.shape_bdry_fix]
			self.bcs_mu += [fenics.DirichletBC(self.CG1, fenics.Constant(self.mu_def), self.boundaries, i)
					for i in self.shape_bdry_def]



	def __compute_mu_elas(self):
		"""Computes the second lame parameter mu_elas, based on `Schulz and
		Siebenborn, Computational Comparison of Surface Metrics for
		PDE Constrained Shape Optimization
		<https://doi.org/10.1515/cmam-2016-0009>`_

		Returns
		-------
		None
		"""
		
		if self.shape_scalar_product is None:
			if self.inhomogeneous_mu:
	
				A, b = _assemble_petsc_system(self.a_mu, self.L_mu, self.bcs_mu)
				x = _solve_linear_problem(self.ksp_mu, A, b, ksp_options=self.options_mu)
	
				if self.config.getboolean('ShapeGradient', 'use_sqrt_mu', fallback=False):
					self.mu_lame.vector()[:] = np.sqrt(x[:])
				else:
					self.mu_lame.vector()[:] = x[:]
	
			else:
				self.mu_lame.vector()[:] = self.mu_fix
			
			# for mpi compatibility
			self.mu_lame.vector().apply('')



	def update_scalar_product(self):
		"""Updates the linear elasticity equations to the current geometry.

		Updates the left-hand-side of the linear elasticity equations
		(needed when the geometry changes).

		Returns
		-------
		None
		"""
		
		self.__compute_mu_elas()

		self.assembler.assemble(self.fe_scalar_product_matrix)
		self.fe_scalar_product_matrix.ident_zeros()
		self.scalar_product_matrix = fenics.as_backend_type(self.fe_scalar_product_matrix).mat()



	def scalar_product(self, a, b):
		"""Computes the scalar product between two deformation functions.

		Parameters
		----------
		a : dolfin.function.function.Function
			The first argument.
		b : dolfin.function.function.Function
			The second argument.

		Returns
		-------
		float
			The value of the scalar product.
		"""

		x = fenics.as_backend_type(a.vector()).vec()
		y = fenics.as_backend_type(b.vector()).vec()

		temp, _ = self.scalar_product_matrix.getVecs()
		self.scalar_product_matrix.mult(x, temp)
		result = temp.dot(y)

		return result
