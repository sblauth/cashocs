"""Contains the blueprint for optimal control and shape optimization problems.

This module is used to define the base class of optimization problems, which
has many shared parameters between optimal control and shape optimization
problems. Can be subclassed to generate custom optimization problems.
"""

import copy



class OptimizationProblem:
	"""A class representing abstract optimization problems.

	This is subclassed to create both optimal control and
	shape optimization problems, which share a lot of common
	parameters. Can be subclassed to generate custom
	optimization problems.
	"""

	def __init__(self, state_forms, bcs_list, cost_functional_form, states, adjoints, config,
				 initial_guess=None, ksp_options=None, adjoint_ksp_options=None):
		"""Initializes the optimization problem.

		If one uses a single PDE constraint, the inputs can be the objects
		(UFL forms, functions, etc.) directly. In case multiple PDE constraints
		are present the inputs have to be put into (ordered) lists. The order of
		the objects depends on the order of the state variables, so that
		state_forms[i] is the weak form of the PDE for state[i] with boundary
		conditions bcs_list[i] and corresponding adjoint state adjoints[i].

		Parameters
		----------
		state_forms : ufl.form.Form or list[ufl.form.Form]
			The weak form of the state equation. Can be either a UFL form
			or a list of UFL forms (if we have multiple equations).
		bcs_list : list[dolfin.fem.dirichletbc.DirichletBC] or list[list[dolfin.fem.dirichletbc.DirichletBC]]
				   or dolfin.fem.dirichletbc.DirichletBC or None
			The list of DirichletBC objects describing Dirichlet (essential) boundary conditions.
			If this is None, then no Dirichlet boundary conditions are imposed.
		cost_functional_form : ufl.form.Form
			The UFL form of the cost functional.
		states : dolfin.function.function.Function or list[dolfin.function.function.Function]
			The state variable(s), can either be a single fenics Function, or a (ordered) list of these.
		adjoints : dolfin.function.function.Function or list[dolfin.function.function.Function]
			The adjoint variable(s), can either be a single fenics Function, or a (ordered) list of these.
		config : configparser.ConfigParser
			The config file for the problem, generated via cestrel.create_config(path_to_config).
		initial_guess : list[dolfin.function.function.Function], optional
			A list of functions that act as initial guess for the state variables, should be valid
			input for fenics.assign. If this is None, then a zero initial guess is used
			(default is None).
		ksp_options : list[list[str]] or list[list[list[str]]] or None, optional
			A list of strings corresponding to command line options for PETSc,
			used to solve the state systems. If this is None, then the direct solver
			mumps is used (default is None).
		adjoint_ksp_options : list[list[str]] or list[list[list[str]]] or None
			A list of strings corresponding to command line options for PETSc,
			used to solve the adjoint systems. If this is None, then the same options
			as for the state systems are used (default is None).
		"""

		### Overloading, so that we do not have to use lists for a single state and a single control
		### state_forms
		try:
			if type(state_forms) == list and len(state_forms) > 0:
				for i in range(len(state_forms)):
					if state_forms[i].__module__=='ufl.form' and type(state_forms[i]).__name__=='Form':
						pass
					else:
						raise Exception('state_forms have to be ufl forms')
				self.state_forms = state_forms
			elif state_forms.__module__ == 'ufl.form' and type(state_forms).__name__ == 'Form':
				self.state_forms = [state_forms]
			else:
				raise Exception('State forms have to be ufl forms')
		except:
			raise Exception('Type of state_forms is wrong.')
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
							raise Exception('bcs_list has inconsistent types.')
					self.bcs_list = bcs_list

				elif bcs_list[0].__module__ == 'dolfin.fem.dirichletbc' and type(bcs_list[0]).__name__ == 'DirichletBC':
					for i in range(len(bcs_list)):
						if bcs_list[i].__module__=='dolfin.fem.dirichletbc' and type(bcs_list[i]).__name__=='DirichletBC':
							pass
						else:
							raise Exception('bcs_list has inconsistent types.')
					self.bcs_list = [bcs_list]
			elif bcs_list.__module__ == 'dolfin.fem.dirichletbc' and type(bcs_list).__name__ == 'DirichletBC':
				self.bcs_list = [[bcs_list]]
			else:
				raise Exception('Type of bcs_list is wrong.')
		except:
			raise Exception('Type of bcs_list is wrong.')

		### cost_functional_form
		try:
			if cost_functional_form.__module__ == 'ufl.form' and type(cost_functional_form).__name__ == 'Form':
				self.cost_functional_form = cost_functional_form
			else:
				raise Exception('cost_functional_form has to be a ufl form')
		except:
			raise Exception('Type of cost_functional_form is wrong.')

		### states
		try:
			if type(states) == list and len(states) > 0:
				for i in range(len(states)):
					if states[i].__module__ == 'dolfin.function.function' and type(states[i]).__name__ == 'Function':
						pass
					else:
						raise Exception('states have to be fenics Functions.')

				self.states = states

			elif states.__module__ == 'dolfin.function.function' and type(states).__name__ == 'Function':
				self.states = [states]
			else:
				raise Exception('Type of states is wrong.')
		except:
			raise Exception('Type of states is wrong.')

		### adjoints
		try:
			if type(adjoints) == list and len(adjoints) > 0:
				for i in range(len(adjoints)):
					if adjoints[i].__module__ == 'dolfin.function.function' and type(adjoints[i]).__name__ == 'Function':
						pass
					else:
						raise Exception('adjoints have to fenics Functions.')

				self.adjoints = adjoints

			elif adjoints.__module__ == 'dolfin.function.function' and type(adjoints).__name__ == 'Function':
				self.adjoints = [adjoints]
			else:
				raise Exception('Type of adjoints is wrong.')
		except:
			raise Exception('Type of adjoints is wrong.')

		### config
		if config.__module__ == 'configparser' and type(config).__name__ == 'ConfigParser':
			self.config = config
		else:
			raise Exception('config has to be of configparser.ConfigParser type')

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
				raise Exception('Initial guess has to be a list of functions')


		### ksp_options
		if ksp_options is None:
			self.ksp_options = []
			option = [
				['ksp_type', 'preonly'],
				['pc_type', 'lu'],
				['pc_factor_mat_solver_type', 'mumps'],
				['mat_mumps_icntl_24', 1]
			]

			for i in range(self.state_dim):
				self.ksp_options.append(option)

		elif type(ksp_options) == list and type(ksp_options[0]) == list and type(ksp_options[0][0]) == str:
			self.ksp_options = [ksp_options[:]]

		elif type(ksp_options) == list and type(ksp_options[0]) == list and type(ksp_options[0][0]) == list:
			self.ksp_options = ksp_options[:]

		else:
			raise Exception('Wrong input format for ksp_options.')



		### adjoint_ksp_options
		if adjoint_ksp_options is None:
			self.adjoint_ksp_options = self.ksp_options[:]

		elif type(adjoint_ksp_options) == list and type(adjoint_ksp_options[0]) == list and type(adjoint_ksp_options[0][0]) == str:
			self.adjoint_ksp_options = [adjoint_ksp_options[:]]

		elif type(adjoint_ksp_options) == list and type(adjoint_ksp_options[0]) == list and type(adjoint_ksp_options[0][0]) == list:
			self.adjoint_ksp_options = adjoint_ksp_options[:]

		else:
			raise Exception('Wrong input format for adjoint_ksp_options.')


		assert len(self.bcs_list) == self.state_dim, 'Length of states does not match'
		assert len(self.states) == self.state_dim, 'Length of states does not match'
		assert len(self.adjoints) == self.state_dim, 'Length of states does not match'
		if self.initial_guess is not None:
			assert len(self.initial_guess) == self.state_dim, 'Length of states does not match'
		assert len(self.ksp_options) == self.state_dim, 'Length of states does not match'
		assert len(self.adjoint_ksp_options) == self.state_dim, 'Length of states does not match'

		self.state_problem = None
		self.adjoint_problem = None



	def compute_state_variables(self):
		"""Solves the state system.

		This can be used for debugging purposes, to validate the solver
		and general behavior. Updates and overwrites the user input for
		the state variables.

		Returns
		-------
		None
		"""

		self.state_problem.solve()



	def compute_adjoint_variables(self):
		"""Solves the adjoint system.

		This can be used for debugging purposes, solver validation.
		Updates / overwrites the user input for the adjoint variables.
		The solve to the corresponding state system needed to determine
		the adjoints is carried out automatically.

		Returns
		-------
		None
		"""

		self.adjoint_problem.solve()
