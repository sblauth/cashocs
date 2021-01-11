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

"""Class representing an optimal control problem.

"""

import fenics
import numpy as np

from .methods import CG, GradientDescent, LBFGS, Newton, PDAS
from .._exceptions import ConfigError, InputError
from .._forms import ControlFormHandler, Lagrangian
from .._optimal_control import ReducedCostFunctional
from .._pde_problems import (AdjointProblem, GradientProblem, HessianProblem, StateProblem, UnconstrainedHessianProblem)
from ..optimization_problem import OptimizationProblem
from ..utils import _optimization_algorithm_configuration



class OptimalControlProblem(OptimizationProblem):
	"""Implements an optimal control problem.

	This class is used to define an optimal control problem, and also to solve
	it subsequently. For a detailed documentation, see the examples in the :ref:`tutorial <tutorial_index>`.
	For easier input, when considering single (state or control) variables,
	these do not have to be wrapped into a list.
	Note, that in the case of multiple variables these have to be grouped into
	ordered lists, where state_forms, bcs_list, states, adjoints have to have
	the same order (i.e. ``[y1, y2]`` and ``[p1, p2]``, where ``p1`` is the adjoint of ``y1``
	and so on.
	"""

	def __init__(self, state_forms, bcs_list, cost_functional_form, states, controls, adjoints, config=None,
				 riesz_scalar_products=None, control_constraints=None, initial_guess=None, ksp_options=None,
				 adjoint_ksp_options=None, desired_weights=None):
		r"""This is used to generate all classes and functionalities. First ensures
		consistent input, afterwards, the solution algorithm is initialized.

		Parameters
		----------
		state_forms : ufl.form.Form or list[ufl.form.Form]
			The weak form of the state equation (user implemented). Can be either
			a single UFL form, or a (ordered) list of UFL forms.
		bcs_list : list[dolfin.fem.dirichletbc.DirichletBC] or list[list[dolfin.fem.dirichletbc.DirichletBC]] or dolfin.fem.dirichletbc.DirichletBC or None
			The list of DirichletBC objects describing Dirichlet (essential) boundary conditions.
			If this is ``None``, then no Dirichlet boundary conditions are imposed.
		cost_functional_form : ufl.form.Form or list[ufl.form.Form]
			UFL form of the cost functional.
		states : dolfin.function.function.Function or list[dolfin.function.function.Function]
			The state variable(s), can either be a :py:class:`fenics.Function`, or a list of these.
		controls : dolfin.function.function.Function or list[dolfin.function.function.Function]
			The control variable(s), can either be a :py:class:`fenics.Function`, or a list of these.
		adjoints : dolfin.function.function.Function or list[dolfin.function.function.Function]
			The adjoint variable(s), can either be a :py:class:`fenics.Function`, or a (ordered) list of these.
		config : configparser.ConfigParser or None
			The config file for the problem, generated via :py:func:`cashocs.create_config`.
			Alternatively, this can also be ``None``, in which case the default configurations
			are used, except for the optimization algorithm. This has then to be specified
			in the :py:meth:`solve <cashocs.OptimalControlProblem.solve>` method. The
			default is ``None``.
		riesz_scalar_products : None or ufl.form.Form or list[ufl.form.Form], optional
			The scalar products of the control space. Can either be None, a single UFL form, or a
			(ordered) list of UFL forms. If ``None``, the :math:`L^2(\Omega)` product is used.
			(default is ``None``).
		control_constraints : None or list[dolfin.function.function.Function] or list[float] or list[list[dolfin.function.function.Function]] or list[list[float]], optional
			Box constraints posed on the control, ``None`` means that there are none (default is ``None``).
			The (inner) lists should contain two elements of the form ``[u_a, u_b]``, where ``u_a`` is the lower,
			and ``u_b`` the upper bound.
		initial_guess : list[dolfin.function.function.Function], optional
			List of functions that act as initial guess for the state variables, should be valid input for :py:func:`fenics.assign`.
			Defaults to ``None``, which means a zero initial guess.
		ksp_options : list[list[str]] or list[list[list[str]]] or None, optional
			A list of strings corresponding to command line options for PETSc,
			used to solve the state systems. If this is ``None``, then the direct solver
			mumps is used (default is ``None``).
		adjoint_ksp_options : list[list[str]] or list[list[list[str]]] or None
			A list of strings corresponding to command line options for PETSc,
			used to solve the adjoint systems. If this is ``None``, then the same options
			as for the state systems are used (default is ``None``).

		Examples
		--------
		Examples how to use this class can be found in the :ref:`tutorial <tutorial_index>`.
		"""

		OptimizationProblem.__init__(self, state_forms, bcs_list, cost_functional_form, states, adjoints, config, initial_guess, ksp_options, adjoint_ksp_options, desired_weights)
		### Overloading, such that we do not have to use lists for a single state and a single control
		### controls
		try:
			if type(controls) == list and len(controls) > 0:
				for i in range(len(controls)):
					if controls[i].__module__ == 'dolfin.function.function' and type(controls[i]).__name__ == 'Function':
						pass
					else:
						raise InputError('cashocs._optimal_control.optimal_control_problem.OptimalControlProblem', 'controls', 'controls have to be fenics Functions.')

				self.controls = controls

			elif controls.__module__ == 'dolfin.function.function' and type(controls).__name__ == 'Function':
				self.controls = [controls]
			else:
				raise InputError('cashocs._optimal_control.optimal_control_problem.OptimalControlProblem', 'controls', 'Type of controls is wrong.')
		except:
			raise InputError('cashocs._optimal_control.optimal_control_problem.OptimalControlProblem', 'controls', 'Type of controls is wrong.')

		self.control_dim = len(self.controls)

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
							raise InputError('cashocs._optimal_control.optimal_control_problem.OptimalControlProblem', 'riesz_scalar_products', 'riesz_scalar_products have to be ufl forms')
					self.riesz_scalar_products = riesz_scalar_products
				elif riesz_scalar_products.__module__== 'ufl.form' and type(riesz_scalar_products).__name__== 'Form':
					self.riesz_scalar_products = [riesz_scalar_products]
				else:
					raise InputError('cashocs._optimal_control.optimal_control_problem.OptimalControlProblem', 'riesz_scalar_products', 'riesz_scalar_products have to be ufl forms')
			except:
				raise InputError('cashocs._optimal_control.optimal_control_problem.OptimalControlProblem', 'riesz_scalar_products', 'riesz_scalar_products have to be ufl forms')

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
										raise InputError('cashocs._optimal_control.optimal_control_problem.OptimalControlProblem', 'control_constraints',
														 'control_constraints has to be a list containing upper and lower bounds')
								pass
							else:
								raise InputError('cashocs._optimal_control.optimal_control_problem.OptimalControlProblem', 'control_constraints',
														 'control_constraints has to be a list containing upper and lower bounds')
						self.control_constraints = control_constraints
					elif (type(control_constraints[0]) in [float, int] or (control_constraints[0].__module__ == 'dolfin.function.function' and type(control_constraints[0]).__name__=='Function')) \
						and (type(control_constraints[1]) in [float, int] or (control_constraints[1].__module__ == 'dolfin.function.function' and type(control_constraints[1]).__name__=='Function')):

						self.control_constraints = [control_constraints]
					else:
						raise InputError('cashocs._optimal_control.optimal_control_problem.OptimalControlProblem', 'control_constraints',
														 'control_constraints has to be a list containing upper and lower bounds')

			except:
				raise InputError('cashocs._optimal_control.optimal_control_problem.OptimalControlProblem', 'control_constraints',
														 'control_constraints has to be a list containing upper and lower bounds')

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
				raise InputError('cashocs._optimal_control.optimal_control_problem.OptimalControlProblem', 'control_constraints', 'Wrong type for the control constraints')

			if type(pair[1]) in [float, int]:
				upper_bound = fenics.Function(self.controls[idx].function_space())
				upper_bound.vector()[:] = pair[1]
			elif pair[1].__module__ == 'dolfin.function.function' and type(pair[1]).__name__ == 'Function':
				upper_bound = pair[1]
			else:
				raise InputError('cashocs._optimal_control.optimal_control_problem.OptimalControlProblem', 'control_constraints', 'Wrong type for the control constraints')

			self.control_constraints.append([lower_bound, upper_bound])

		### Check whether the control constraints are feasible, and whether they are actually present
		self.require_control_constraints = [False for i in range(self.control_dim)]
		for idx, pair in enumerate(self.control_constraints):
			if not np.alltrue(pair[0].vector()[:] < pair[1].vector()[:]):
				raise InputError('cashocs._optimal_control.optimal_control_problem.OptimalControlProblem', 'control_constraints',
								 'The lower bound must always be smaller than the upper bound for the control_constraints.')

			if np.max(pair[0].vector()[:]) == float('-inf') and np.min(pair[1].vector()[:]) == float('inf'):
				# no control constraint for this component
				pass
			else:
				self.require_control_constraints[idx] = True
				
				control_element = self.controls[idx].ufl_element()
				if control_element.family() == 'Mixed':
					for j in range(control_element.value_size()):
						sub_elem = control_element.extract_component(j)[1]
						if sub_elem.family() == 'Real' or (sub_elem.family() == 'Lagrange' and sub_elem.degree() == 1) \
								or (sub_elem.family() == 'Discontinuous Lagrange' and sub_elem.degree() == 0):
							pass
						else:
							raise InputError('cashocs._optimal_control.optimal_control_problem.OptimalControlProblem', 'controls',
									 'Control constraints are only implemented for linear Lagrange, constant Discontinuous Lagrange, and Real elements.')

				else:
					if control_element.family() == 'Real' or (control_element.family() == 'Lagrange' and control_element.degree() == 1) \
							or (control_element.family() == 'Discontinuous Lagrange' and control_element.degree() == 0):
						pass
					else:
						raise InputError('cashocs._optimal_control.optimal_control_problem.OptimalControlProblem', 'controls',
									 'Control constraints are only implemented for linear Lagrange, constant Discontinuous Lagrange, and Real elements.')

		if not len(self.riesz_scalar_products) == self.control_dim:
			raise InputError('cashocs._optimal_control.optimal_control_problem.OptimalControlProblem', 'riesz_scalar_products', 'Length of controls does not match')
		if not len(self.control_constraints) == self.control_dim:
			raise InputError('cashocs._optimal_control.optimal_control_problem.OptimalControlProblem', 'control_constraints', 'Length of controls does not match')
		### end overloading
		
		self.form_handler = ControlFormHandler(self.lagrangian, self.bcs_list, self.states, self.controls, self.adjoints, self.config,
											   self.riesz_scalar_products, self.control_constraints, self.ksp_options, self.adjoint_ksp_options,
											   self.require_control_constraints)

		self.state_spaces = self.form_handler.state_spaces
		self.control_spaces = self.form_handler.control_spaces
		self.adjoint_spaces = self.form_handler.adjoint_spaces

		self.projected_difference = [fenics.Function(V) for V in self.control_spaces]

		self.state_problem = StateProblem(self.form_handler, self.initial_guess)
		self.adjoint_problem = AdjointProblem(self.form_handler, self.state_problem)
		self.gradient_problem = GradientProblem(self.form_handler, self.state_problem, self.adjoint_problem)

		self.algorithm = _optimization_algorithm_configuration(self.config)

		self.reduced_cost_functional = ReducedCostFunctional(self.form_handler, self.state_problem)

		self.gradients = self.gradient_problem.gradients
		self.objective_value = 1.0



	def _stationary_measure_squared(self):
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



	def _erase_pde_memory(self):
		"""Resets the memory of the PDE problems so that new solutions are computed.

		This sets the value of has_solution to False for all relevant PDE problems,
		where memory is stored.

		Returns
		-------
		None
		"""

		self.state_problem.has_solution = False
		self.adjoint_problem.has_solution = False
		self.gradient_problem.has_solution = False



	def solve(self, algorithm=None, rtol=None, atol=None, max_iter=None):
		r"""Solves the optimization problem by the method specified in the config file.

		Updates / overwrites states, controls, and adjoints according
		to the optimization method, i.e., the user-input :py:func:`fenics.Function` objects.

		Parameters
		----------
		algorithm : str or None, optional
			Selects the optimization algorithm. Valid choices are
			``'gradient_descent'`` or ``'gd'`` for a gradient descent method,
			``'conjugate_gradient'``, ``'nonlinear_cg'``, ``'ncg'`` or ``'cg'``
			for nonlinear conjugate gradient methods, ``'lbfgs'`` or ``'bfgs'`` for
			limited memory BFGS methods, ``'newton'`` for a truncated Newton method,
			and ``'pdas'`` or ``'primal_dual_active_set'`` for a
			primal dual active set method. This overwrites
			the value specified in the config file. If this is ``None``,
			then the value in the config file is used. Default is
			``None``.
		rtol : float or None, optional
			The relative tolerance used for the termination criterion.
			Overwrites the value specified in the config file. If this
			is ``None``, the value from the config file is taken. Default
			is ``None``.
		atol : float or None, optional
			The absolute tolerance used for the termination criterion.
			Overwrites the value specified in the config file. If this
			is ``None``, the value from the config file is taken. Default
			is ``None``.
		max_iter : int or None, optional
			The maximum number of iterations the optimization algorithm
			can carry out before it is terminated. Overwrites the value
			specified in the config file. If this is ``None``, the value from
			the config file is taken. Default is ``None``.

		Returns
		-------
		None

		Notes
		-----
		If either ``rtol`` or ``atol`` are specified as arguments to the solve
		call, the termination criterion changes to:

		  - a purely relative one (if only ``rtol`` is specified), i.e.,

		  .. math:: || \nabla J(u_k) || \leq \texttt{rtol} || \nabla J(u_0) ||.

		  - a purely absolute one (if only ``atol`` is specified), i.e.,

		  .. math:: || \nabla J(u_k) || \leq \texttt{atol}.

		  - a combined one if both ``rtol`` and ``atol`` are specified, i.e.,

		  .. math:: || \nabla J(u_k) || \leq \texttt{atol} + \texttt{rtol} || \nabla J(u_0) ||.
		"""
		
		self.algorithm = _optimization_algorithm_configuration(self.config, algorithm)

		if self.algorithm == 'newton' or \
				(self.algorithm == 'pdas' and self.config.get('AlgoPDAS', 'inner_pdas') == 'newton'):
			self.form_handler._ControlFormHandler__compute_newton_forms()

		if self.algorithm == 'newton':
			self.hessian_problem = HessianProblem(self.form_handler, self.gradient_problem)
		if self.algorithm == 'pdas':
			self.unconstrained_hessian = UnconstrainedHessianProblem(self.form_handler, self.gradient_problem)

		if (rtol is not None) and (atol is None):
			self.config.set('OptimizationRoutine', 'rtol', str(rtol))
			self.config.set('OptimizationRoutine', 'atol', str(0.0))
		elif (atol is not None) and (rtol is None):
			self.config.set('OptimizationRoutine', 'rtol', str(0.0))
			self.config.set('OptimizationRoutine', 'atol', str(atol))
		elif (atol is not None) and (rtol is not None):
			self.config.set('OptimizationRoutine', 'rtol', str(rtol))
			self.config.set('OptimizationRoutine', 'atol', str(atol))

		if max_iter is not None:
			self.config.set('OptimizationRoutine', 'maximum_iterations', str(max_iter))
		
		self._check_for_custom_forms()
		
		if self.algorithm == 'gradient_descent':
			self.solver = GradientDescent(self)
		elif self.algorithm == 'lbfgs':
			self.solver = LBFGS(self)
		elif self.algorithm == 'conjugate_gradient':
			self.solver = CG(self)
		elif self.algorithm == 'newton':
			self.solver = Newton(self)
		elif self.algorithm == 'pdas':
			self.solver = PDAS(self)
		elif self.algorithm == 'none':
			raise InputError('cashocs.OptimalControlProblem.solve', 'algorithm', 'You did not specify a solution algorithm in your config file. You have to specify one in the solve '
																				 'method. Needs to be one of'
																				 '\'gradient_descent\' (\'gd\'), \'lbfgs\' (\'bfgs\'), \'conjugate_gradient\' (\'cg\'), '
																				 '\'newton\', or \'primal_dual_active_set\' (\'pdas\').')
		else:
			raise ConfigError('OptimizationRoutine', 'algorithm', 'Not a valid input. Needs to be one '
							  'of \'gradient_descent\' (\'gd\'), \'lbfgs\' (\'bfgs\'), \'conjugate_gradient\' (\'cg\'), \'newton\', or \'primal_dual_active_set\' (\'pdas\').')

		self.solver.run()
		self.solver.post_processing()



	def compute_gradient(self):
		"""Solves the Riesz problem to determine the gradient.

		This can be used for debugging, or code validation.
		The necessary solutions of the state and adjoint systems
		are carried out automatically.

		Returns
		-------
		list[dolfin.function.function.Function]
			A list consisting of the (components) of the gradient.
		"""

		self.gradient_problem.solve()

		return self.gradients
	
	
	
	def supply_derivatives(self, derivatives):
		"""Overrides the derivatives of the reduced cost functional w.r.t. controls.
		
		This allows users to implement their own derivatives and use cashocs as a
		solver library only.
		
		Parameters
		----------
		derivatives : ufl.form.Form or list[ufl.form.Form]
			The derivatives of the reduced (!) cost functional w.r.t. controls.

		Returns
		-------
		None
		"""
		
		try:
			if type(derivatives) == list and len(derivatives) > 0:
				for i in range(len(derivatives)):
					if derivatives[i].__module__=='ufl.form' and type(derivatives[i]).__name__=='Form':
						pass
					else:
						raise InputError('cashocs._optimal_control.optimal_control_problem.OptimalControlProblem.supply_derivatives',
										 'derivatives', 'derivatives have to be ufl forms')
				mod_derivatives = derivatives
			elif derivatives.__module__ == 'ufl.form' and type(derivatives).__name__ == 'Form':
				mod_derivatives = [derivatives]
			else:
				raise InputError('cashocs._optimal_control.optimal_control_problem.OptimalControlProblem.supply_derivatives',
								 'derivatives', 'derivatives have to be ufl forms')
		except:
			raise InputError('cashocs._optimal_control.optimal_control_problem.OptimalControlProblem.supply_derivatives',
							 'derivatives', 'derivatives have to be ufl forms')
		
		for idx, form in enumerate(mod_derivatives):
			if len(form.arguments()) == 2:
				raise InputError('cashocs._optimal_control.optimal_control_problem.OptimalControlProblem.supply_derivatives',
								 'derivatives', 'Do not use TrialFunction for the derivatives.')
			elif len(form.arguments()) == 0:
				raise InputError('cashocs._optimal_control.optimal_control_problem.OptimalControlProblem.supply_derivatives',
								 'derivatives', 'The specified derivatives must include a TestFunction object from the control space.')
			
			if not form.arguments()[0].ufl_function_space() == self.form_handler.control_spaces[idx]:
				raise InputError('cashocs._optimal_control.optimal_control_problem.OptimalControlProblem.supply_derivatives',
								 'derivatives', 'The TestFunction has to be chosen from the same space as the corresponding adjoint.')
		
		if not len(mod_derivatives) == self.form_handler.control_dim:
			raise InputError('cashocs._optimal_control.optimal_control_problem.OptimalControlProblem.supply_derivatives',
								 'derivatives', 'Length of derivatives does not match number of controls.')
		
		self.form_handler.gradient_forms_rhs = mod_derivatives
		self.has_custom_derivative = True

		
	
	
	def supply_custom_forms(self, derivatives, adjoint_forms, adjoint_bcs_list):
		"""Overrides both adjoint system and derivatives with user input.
		
		This allows the user to specify both the derivatives of the reduced cost functional
		and the corresponding adjoint system, and allows them to use cashocs as a solver.
		
		See Also
		--------
		supply_derivatives
		supply_adjoint_forms
		
		Parameters
		----------
		derivatives : ufl.form.Form or list[ufl.form.Form]
			The derivatives of the reduced (!) cost functional w.r.t. controls.
		adjoint_forms : ufl.form.Form or list[ufl.form.Form]
			The UFL forms of the adjoint system(s).
		adjoint_bcs_list : list[dolfin.fem.dirichletbc.DirichletBC] or list[list[dolfin.fem.dirichletbc.DirichletBC]] or dolfin.fem.dirichletbc.DirichletBC or None
			The list of Dirichlet boundary conditions for the adjoint system(s).

		Returns
		-------
		None
		"""
		
		self.supply_derivatives(derivatives)
		self.supply_adjoint_forms(adjoint_forms, adjoint_bcs_list)
